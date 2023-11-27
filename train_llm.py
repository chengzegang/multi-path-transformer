from collections import OrderedDict
from datetime import datetime
import glob
import inspect
import math
import os
from functools import partial
from typing import Optional, Tuple
import torch.utils.data.datapipes as dp
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)  # type:ignore
import wandb
from llm import LLM, add_gradient_checkpoint
from torch_datasets import Pile, Sentence, WebData
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
import evaluate
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
import torch._dynamo.config

torch._dynamo.config.cache_size_limit = 256
cudnn.benchmark = True


class Evaluation:
    def __init__(self, model: LLM, tokenizer: AutoTokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.perplexity = evaluate.load("perplexity", module_type="metric")

    def step(self, step: int):
        with torch.inference_mode():
            self.model.eval()
            text = "This is a"
            output_str = " ".join(self.model.stream(self.tokenizer, text, 128))
            results = self.perplexity.compute(predictions=output_str, model_id="gpt2")
            wandb.log(results["mean_perplexity"], step=step)
            return results


def expoential_lr(
    initial_step: int = 0,
    warmup_steps=2000,
    beta: float = 0.95,
    min_factor: float = 0.01,
    step: int = 0,
):
    if step < initial_step + warmup_steps:
        return max(min_factor, step / (initial_step + warmup_steps))
    else:
        return max(beta ** (step - warmup_steps - initial_step), min_factor)


def step_model(
    device: str,
    dl: DataLoader,
    model: LLM,
    opt: Optimizer,
    sched: LambdaLR,
    num_epochs: int,
    grad_accum: int,
    step: int,
    pbar: tqdm,
    enable_compiler: bool = True,
    ddp: bool = False,
    ema: bool = True,
):
    loss = 0
    input_ids = None
    output_ids = None
    optimized_model = optimize_model(model, enable_compiler)
    if ddp:
        optimized_model = DDP(
            optimized_model,
            [device],
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    # first_descend_stage_ended = False
    avg_model = None
    if ema:
        avg_model = AveragedModel(
            model, device="cpu", avg_fn=get_ema_avg_fn(0.99), use_buffers=True
        )
    wandb.watch(
        (
            model.embed_tokens,
            model.decoder.layers[0],
            model.decoder.layers[len(model.decoder.layers) // 2],
            model.decoder.layers[-1],
        ),
        log_freq=100,
    )
    for epoch in range(num_epochs):
        for i, batch in enumerate(dl):
            optimized_model.train()
            batch = batch.to(device)
            out = optimized_model(batch.input_ids, labels=batch.input_ids)
            # with torch.autocast(
            #    "cuda", torch.float32, enabled=first_descend_stage_ended
            # ):
            (out["loss"] / grad_accum).backward()
            loss = out["loss"].item()
            # if not first_descend_stage_ended and loss < 4.0:
            #    first_descend_stage_ended = True
            input_ids = batch["input_ids"]
            output_ids = out["logits"]
            pbar.set_description(
                f"epoch: {epoch:3d}/{num_epochs:3d}, step: {step:8d}, loss: {loss:0.6f}, lr: {sched.get_last_lr()[0]:0.3e}"
            )
            pbar.update()
            if i % grad_accum == 0 and i > 0:
                # with torch.autocast(
                #    "cuda", torch.float32, enabled=first_descend_stage_ended
                # ):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if avg_model is not None:
                    avg_model.update_parameters(model)
                opt.zero_grad()
                sched.step(step)
                step += 1
                yield epoch, step, loss, input_ids, output_ids


def optimize_model(model: LLM, enabled: bool = True) -> nn.Module:
    proxy_model = model
    if enabled:
        proxy_model = torch.compile(model, mode="reduce-overhead")
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


def partial_load_state_dict(mod: nn.Module, state_dict: OrderedDict) -> nn.Module:
    mod_state_dict = mod.state_dict()
    for k, v in state_dict.items():
        if k in mod_state_dict and v.shape == mod_state_dict[k]:
            mod_state_dict[k] = v
    mod.load_state_dict(mod_state_dict)
    return mod

def train(
    root: Optional[str] = None,
    name: str = "default",
    data_name: str = "webtext",
    checkpoint_path: str = "models",
    lr: float = 2e-4,
    num_epochs: int = 10,
    save_every: int = 100,
    grad_accum: int = 2,
    max_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 16,
    model_config: dict = {
        "embedding_size": 4096,
        "hidden_size": 512,
        "num_layers": 32,
        "head_size": 64,
    },
    device: str = "cuda",
    dtype: str = "bfloat16",
    tokenizer_id: str = "meta-llama/Llama-2-7b-chat-hf",
    ddp: bool = False,
    enable_compiler: bool = False,
    warmup_steps: int = 2000,
    ema: bool = True,
):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, dtype)

    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(tokenizer.vocab_size, **model_config)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"total params: {num_params(model)}")
    step = 0

    try:
        ckpts = glob.glob("models/llm*.pt")
        ckpt = sorted(ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1]
        partial_load_state_dict(model, torch.load(ckpt, mmap=True))
        step = int(ckpt.split("-")[-1].split(".")[0])
    except Exception as e:
        print(f"fail to load ckpt {ckpt}, starting from scratch")
    model = add_gradient_checkpoint(model)
    model = model.to(device).to(dtype)
    opt = None
    if ddp:
        opt = ZRO(
            model.parameters(),
            AdamW,
            lr=lr,
            weight_decay=1e-2,
            betas=(0.9, 0.999),
            fused=True,
            parameters_as_bucket_view=True,
        )
    else:
        opt = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-2,
            fused=True,
            betas=(0.9, 0.999),
        )

    sched = LambdaLR(opt, partial(expoential_lr, step, warmup_steps, 0.9999, 0.1))
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

    try:
        log = yaml.full_load(open(os.path.join(checkpoint_path, "log.yaml")))
        step = log["step"]
    except Exception as e:
        print(e)
    hostname = os.uname().nodename
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm",
        name=f"llm-{hostname}",
        # track hyperparameters and run metadata
        id=f"llm-{hostname}-{date}",
        resume="allow",
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
    total_tokens = 50000000 * 22 // world_size
    num_samples = total_tokens // max_size
    num_batches = num_samples // batch_size
    pbar = tqdm(total=num_batches, dynamic_ncols=True)
    pbar.update(step)
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
        ddp,
        ema,
    )
    # evaluation = Evaluation(model, tokenizer, device)
    for epoch, step, loss, input_ids, output_ids in iteration:
        if local_rank == 0:
            in_text = tokenizer.decode(input_ids[0][:-1], skip_special_tokens=True)
            out_text = tokenizer.decode(
                output_ids[0].argmax(dim=-1)[1:], skip_special_tokens=True
            )
            pbar.write(f"IN : {in_text[:256]}...")
            pbar.write(f"OUT: {out_text[:256]}...")
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
                # results = evaluation.step(step)
                # pbar.write(repr(results["mean_perplexity"]))
                ckpts = glob.glob("models/llm*.pt")
                if len(ckpts) > 3:
                    os.remove(
                        sorted(
                            ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0])
                        )[0]
                    )
                model.eval()
                torch.save(model.state_dict(), f"models/llm-{step}.pt")
                torch.cuda.empty_cache()

    pbar.close()


if __name__ == "__main__":
    import torch._dynamo.config  # type: ignore

    DAVID_100M = {
        "bunch_size": 2,
        "hidden_size": 512,
        "num_layers": 24,
        "num_heads": 16,
        "head_size": 64,
    }
    DAVID_500M = {
        "bunch_size": 8,
        "hidden_size": 512,
        "num_layers": 80,
        "num_heads": 16,
        "head_size": 64,
    }
    DAVID_3B = {
        "bunch_size": 16,
        "hidden_size": 1024,
        "num_layers": 96,
        "num_heads": 16,
        "head_size": 128,
    }
    greene_config = {
        "root": "/scratch/work/public/ml-datasets/pile/train/",
        "name": "greene",
        "data_name": "pile",
        "max_size": 4096,
        "grad_accum": 32,
        "save_every": 10,
        "batch_size": 1,
        "model_config": DAVID_500M,
        "warmup_steps": 100,
        "ddp": False,
        "lr": 2e-4,
    }

    local_config = {
        "root": "/home/caleb/data/pile/train/",
        "name": "local",
        "data_name": "webtext",
        "max_size": 1024,
        "grad_accum": 8,
        "save_every": 10,
        "batch_size": 1,
        "model_config": DAVID_500M,
        "ddp": False,
        "warmup_steps": 0,
        "ema": False,
    }
    host = os.uname().nodename
    config = None
    if host == "LPC":
        config = local_config
    else:
        config = greene_config
    train(**config)
