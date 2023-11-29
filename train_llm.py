from collections import OrderedDict
from datetime import datetime
import glob
import inspect
import math
import os
from functools import partial
from typing import Optional, Tuple
import torch.utils.data.datapipes as dp

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
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor  # type: ignore
from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel  # type: ignore
from llm import LLM, add_gradient_checkpoint
from llm_datasets import Pile, Sentence, WebData
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
import evaluate
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
import torch._dynamo.config
from torch.distributed.pipeline.sync import Pipe  # type: ignore

torch._dynamo.config.cache_size_limit = 256
cudnn.benchmark = True
cudnn.allow_tf32 = True
cuda.matmul.allow_bf16_reduced_precision_reduction = True
cuda.matmul.allow_tf32 = True


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
    warmup_steps=2000,
    beta: float = 0.95,
    min_factor: float = 0.01,
    step: int = 0,
):
    if step < warmup_steps:
        return max(1e-8, step / (warmup_steps))
    else:
        return max(beta ** (step - warmup_steps), min_factor)


def grad_accumulation_scheduler(
    step: int, init_accum_steps: int = 0, last_accum_steps: int = 0, rate: float = 0.01
):
    curr_steps = min(last_accum_steps, math.ceil(init_accum_steps + (1 + rate) ** step))
    return curr_steps


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
    distributed: bool = False,
    ema: bool = False,
    num_tokens_per_batch: int = 0,
):
    # loss = 0
    input_ids = None
    output_ids = None
    optimized_model = optimize_model(model, enable_compiler)
    if distributed:
        # optimized_model = DDP(
        #    optimized_model,
        #    [device],
        #    gradient_as_bucket_view=True,
        #    static_graph=True,
        # )
        nnodes = int(os.getenv("NNODES", 1))
        nproc_per_node = int(os.getenv("NPROC_PER_NODE", 1))
        total_gpus = nnodes * nproc_per_node
        mesh_assign = torch.arange(total_gpus).reshape(nnodes, nproc_per_node).tolist()
        mesh = DeviceMesh(device_type="cuda", mesh=mesh_assign)
        optimized_model = parallelize_module(
            optimized_model, mesh, parallelize_plan=PairwiseParallel()
        )
    avg_model = None
    if ema:
        avg_model = AveragedModel(
            model, device="cpu", avg_fn=get_ema_avg_fn(0.9), use_buffers=True
        )
    if os.getenv("LOCAL_RANK", "0") == "0":
        wandb.watch(
            (
                model.embed_tokens,
                model.decoder.layers[0],
                model.decoder.layers[len(model.decoder.layers) // 2],
                model.decoder.layers[-1],
            ),
        )
    target_num_tokens_per_batch = 1024 * 512
    target_grad_accum = target_num_tokens_per_batch // num_tokens_per_batch
    schedule_grad_accum = partial(
        grad_accumulation_scheduler,
        init_accum_steps=grad_accum,
        last_accum_steps=target_grad_accum,
    )
    for epoch in range(num_epochs):
        accum_loss = 0
        for i, batch in enumerate(dl):
            curr_grad_accum = schedule_grad_accum(step)
            optimized_model.train()
            batch = batch.to(device)
            out = optimized_model(batch.input_ids, labels=batch.input_ids)

            # (out["loss"] / curr_grad_accum).backward()
            accum_loss += out["loss"] / curr_grad_accum

            input_ids = batch["input_ids"]
            output_ids = out["logits"]
            if pbar is not None:
                pbar.set_description(
                    f"epoch: {epoch:3d}/{num_epochs:3d}, step: {step:8d}, loss: {out['loss'].item():0.6f}, lr: {sched.get_last_lr()[0]:0.3e}, grad_accum: {i % curr_grad_accum:3d}/{curr_grad_accum}"
                )
                pbar.update()
            if i % curr_grad_accum == 0 and i > 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if avg_model is not None:
                    avg_model.update_parameters(model)
                opt.zero_grad()
                sched.step(step)
                step += 1
                yield epoch, step, accum_loss, input_ids, output_ids
                accum_loss = 0


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
    batch_size: int = 4,
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
    distributed: bool = False,
    enable_compiler: bool = False,
    warmup_steps: int = 100,
    ema: bool = True,
):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, dtype)

    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(tokenizer.vocab_size, **model_config)
    tokenizer.save_pretrained(checkpoint_path)
    total_params = num_params(model)
    print(f"total params: {total_params}")
    step = 0

    try:
        ckpts = glob.glob("models/llm*.pt")
        ckpt = sorted(ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1]
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        except Exception as e:
            print(e)
            partial_load_state_dict(model, torch.load(ckpt, mmap=True))
        #step = int(ckpt.split("-")[-1].split(".")[0])
    except Exception:
        print("fail to load a checkpoint, starting from scratch")
    model = add_gradient_checkpoint(model)
    model = model.to(device).to(dtype)
    opt = None
    if distributed:
        opt = ZRO(
            model.parameters(),
            AdamW,
            lr=lr,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            fused=True,
            parameters_as_bucket_view=True,
        )
    else:
        opt = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-3,
            fused=True,
            betas=(0.9, 0.999),
        )

    sched = LambdaLR(opt, partial(expoential_lr, warmup_steps, 0.999, 0.1))
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

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    if os.getenv("LOCAL_RANK", "0") == "0":
        wandb.init(
            # set the wandb project where this run will be logged
            project="llm",
            name=f"llm-{total_params}-{name}-{date}",
            # track hyperparameters and run metadata
            id=f"llm-{total_params}-{name}-{date}",
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
                "num_params": total_params,
            },
        )

    total_tokens = 5000000000000 // world_size
    num_samples = total_tokens // max_size
    num_batches = num_samples // batch_size
    pbar = None
    num_tokens_per_batch = batch_size * max_size
    if local_rank == 0:
        pbar = tqdm(total=num_batches, dynamic_ncols=True, unit_scale=True)
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
        distributed,
        ema,
        num_tokens_per_batch,
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

    pbar.close()


def save_checkpoint(
    path: str,
    model: LLM,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    step: int,
    **kwargs,
):
    os.makedirs(path, exist_ok=True)
    state = {
        "model_config": model.config,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "optimizer_config": optimizer.defaults,
        "scheduler_config": scheduler.state_dict(),
        "step": step,
        "created_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
        **kwargs,
    }
    with open(os.path.join(path, "checkpoint.yaml"), "w") as f:
        yaml.dump(state, f)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))


def check_mappings(d1: dict, d2: dict) -> bool:
    if len(d1) != len(d2):
        return False

    if set(d1.keys()) != set(d2.keys()):
        return False

    for k, v in d1.items():
        if k not in d2:
            return False
        if isinstance(v, dict):
            return check_mappings(v, d2[k])
        else:
            if v != d2[k]:
                return False
    return True


def load_checkpoint(
    path: str, model: LLM, optimizer: Optimizer, scheduler: LambdaLR
) -> Tuple[LLM, Optimizer, LambdaLR, dict]:
    state = yaml.full_load(open(os.path.join(path, "checkpoint.yaml")))

    model = LLM(**state["model_config"])
    try:
        partial_load_state_dict(model, torch.load(os.path.join(path, "model.pt")))
    except Exception:
        pass
    if check_mappings(state["optimizer_config"], optimizer.defaults):
        try:
            optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))
        except Exception:
            pass
    if check_mappings(state["scheduler_config"], scheduler.state_dict()):
        try:
            scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt")))
        except Exception:
            pass
    return model, optimizer, scheduler, state


if __name__ == "__main__":
    import torch._dynamo.config  # type: ignore

    DAVID_100M = {
        "bunch_size": 4,
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
        "lr": 2e-4,
    }

    local_config = {
        "root": "/home/caleb/data/pile/train/",
        "name": "local",
        "data_name": "webtext",
        "max_size": 256,
        "grad_accum": 8,
        "save_every": 5,
        "batch_size": 1,
        "model_config": DAVID_100M,
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
