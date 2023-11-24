import inspect
import math
import os
from functools import partial
from typing import Tuple

import bitsandbytes as bnb
import torch
import yaml
from datasets import load_dataset  # type: ignore
from torch import Tensor, nn  # type: ignore
from torch.backends import cuda, cudnn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type:ignore

import wandb
from llm import LLM
from torch_datasets import Pile, Sentence, WebData


def expoential_lr(beta: float = 0.95, min_factor: float = 0.01, step: int = 0):
    return max(beta**step, min_factor)


def save_checkpoint(
    checkpoint_path: str,
    model: LLM,
    opt: Optimizer,
    sched: LambdaLR,
    step: int,
):
    os.makedirs(checkpoint_path, exist_ok=True)
    model.eval()
    model.save_to_file(os.path.join(checkpoint_path, "model.pt"))
    opt_state = opt.state_dict()
    sched_state = sched.state_dict()
    torch.save(opt_state, os.path.join(checkpoint_path, "opt.pt"))
    torch.save(sched_state, os.path.join(checkpoint_path, "sched.pt"))
    yaml.dump({"step": step}, open(os.path.join(checkpoint_path, "log.yaml"), "w"))


def step_model(
    device: str,
    dl: DataLoader,
    model: nn.Module,
    opt: Optimizer,
    sched: LambdaLR,
    num_epochs: int,
    grad_accum: int,
    step: int,
    pbar: tqdm,
    enable_compiler: bool = True,
):
    loss = 0
    input_ids = None
    output_ids = None
    optimized_model = optimize_model(model, enable_compiler)
    for epoch in range(num_epochs):
        for i, batch in enumerate(dl):
            batch = batch.to(device)
            out = optimized_model(**batch, labels=batch.input_ids)
            out["loss"].backward()
            loss = out["loss"].item()
            input_ids = batch["input_ids"]
            output_ids = out["logits"]
            pbar.set_description(
                f"epoch: {epoch:3d}/{num_epochs:3d}, step: {step:8d}, loss: {loss:0.6f}, lr: {sched.get_last_lr()[0]:0.3e}"
            )
            pbar.update()
            if i % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                yield epoch, step, loss, input_ids, output_ids


def optimize_model(model: LLM, enabled: bool = True) -> nn.Module:
    proxy_model = model
    if enabled:
        proxy_model = torch.compile(
            model, fullgraph=True, dynamic=True, mode="max-autotune"
        )
    # proxy_model = mesh_model(model)
    # proxy_model = model
    return proxy_model


def num_params(model: nn.Module) -> str:
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def to_human_readable(n: int) -> str:
        if n < 1e3:
            return str(n)
        elif n < 1e6:
            return f"{n/1e3:.0f}K"
        elif n < 1e9:
            return f"{n/1e6:.0f}M"
        elif n < 1e12:
            return f"{n/1e9:.0f}B"
        elif n < 1e15:
            return f"{n/1e12:.0f}T"
        else:
            return f"{n/1e15:.0f}P"

    return to_human_readable(n_p)


def train(
    root: str | None = None,
    name: str = "default",
    data_name: str = "webtext",
    checkpoint_path: str = "models",
    lr: float = 2e-4,
    num_epochs: int = 10,
    save_every: int = 1000,
    grad_accum: int = 10,
    max_size: int = 512,
    batch_size: int = 2,
    num_workers: int = 16,
    model_config: dict = {
        "embedding_size": 8192,
        "hidden_size": 512,
        "num_layers": 80,
        "head_size": 128,
    },
    device: str = "cuda",
    dtype: str = "bfloat16",
    tokenizer_id: str = "meta-llama/Llama-2-7b-chat-hf",
    enable_device_mesh: bool = False,
    enable_compiler: bool = False,
):
    device = torch.device(device)
    dtype = getattr(torch, dtype)

    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(tokenizer.vocab_size, **model_config)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"total params: {num_params(model)}")
    try:
        model.eval()
        model.load_from_file(os.path.join(checkpoint_path, "model.pt"))
    except Exception as e:
        print(e)
    model.enable_gradient_checkpointing()
    model = model.to(device).to(dtype)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0, fused=True)

    sched = LambdaLR(opt, partial(expoential_lr, 0.999, 0.01))
    try:
        opt.load_state_dict(torch.load(os.path.join(checkpoint_path, "opt.pt")))
        sched.load_state_dict(torch.load(os.path.join(checkpoint_path, "sched.pt")))
    except Exception as e:
        print(e)
    data = None
    if data_name == "pile":
        data = Pile(root)
    elif data_name == "webtext":
        data = WebData()
    dataset = Sentence(data, max_size=max_size, tokenizer=tokenizer)

    def collate_fn(batch, max_size: int = max_size):
        text = [item["text"] for item in batch]
        inputs = tokenizer(
            text,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_size,
            add_special_tokens=True,
        )

        return inputs

    dl = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
    )

    step = 0
    try:
        log = yaml.full_load(open(os.path.join(checkpoint_path, "log.yaml")))
        step = log["step"]
    except Exception as e:
        print(e)
    hostname = os.uname().nodename
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm",
        name=f"llm-{hostname}",
        # track hyperparameters and run metadata
        id=f"llm-{hostname}",
        config={
            "grad_accum": grad_accum,
            "dtype": dtype,
            "max_size": max_size,
            "batch_size": batch_size,
            "model_config": model_config,
            "lr": lr,
            "tokenizer_id": tokenizer_id,
            "architecture": "LLM",
            "dataset": "Pile",
            "epochs": num_epochs,
            "num_params": num_params(model),
        },
    )
    total_tokens = 50000000 * 22
    num_samples = total_tokens // max_size
    num_batches = num_samples // batch_size
    pbar = tqdm(total=num_batches, dynamic_ncols=True)
    iteration = step_model(
        device,
        dl,
        model,
        opt,
        sched,
        num_epochs,
        grad_accum,
        step,
        pbar,
        enable_compiler,
    )

    for epoch, step, loss, input_ids, output_ids in iteration:
        in_text = tokenizer.decode(input_ids[0][:-1], skip_special_tokens=True)
        out_text = tokenizer.decode(
            output_ids[0].argmax(dim=-1)[1:], skip_special_tokens=True
        )
        pbar.write(f"IN : {in_text:64s}")
        pbar.write(f"OUT: {out_text:64s}")
        wandb.log(
            {
                "in_text": in_text,
                "out_text": out_text,
                "loss": loss,
                "lr": sched.get_last_lr()[0],
            },
            step=step,
        )
        if step % save_every == 0 and math.isfinite(loss):
            save_checkpoint(checkpoint_path, model, opt, sched, step)

    pbar.close()


if __name__ == "__main__":
    train()