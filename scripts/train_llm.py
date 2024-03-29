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
from torch.distributed.tensor.parallel.ddp import pre_dp_module_transform
import tempfile
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
    AutoModelForTokenClassification,
)  # type:ignore
import wandb
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor  # type: ignore
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    PairwiseParallel,
)
from multipath.nn.llm import LLM, add_gradient_checkpoint
from multipath.torch_datasets.llm_datasets import Pile, Sentence, WebData
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
import evaluate
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
import torch._dynamo.config
from torch.distributed.pipeline.sync import Pipe  # type: ignore
from torch.distributed import rpc
import torch._dynamo
import warnings


warnings.simplefilter("ignore")
torch._dynamo.config.suppress_errors = True
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


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + math.cos(step / total_steps * math.pi)
    )


def grad_accumulation_scheduler(
    step: int, init_accum_steps: int = 0, last_accum_steps: int = 0, rate: float = 0.01
):
    curr_steps = min(last_accum_steps, math.ceil(init_accum_steps + (1 + rate) ** step))
    return curr_steps


def step_model(
    device: str,
    dl: DataLoader,
    proxy_model: LLM,
    opt: Optimizer,
    sched: LambdaLR,
    num_epochs: int,
    grad_accum: int,
    step: int,
    tokens: int,
    pbar: tqdm,
    enable_compiler: bool = False,
    distributed: bool = False,
    ema: bool = False,
    num_tokens_per_batch: int = 0,
):
    # loss = 0
    input_ids = None
    output_ids = None

    avg_model = None
    if ema:
        avg_model = AveragedModel(
            proxy_model, device="cpu", avg_fn=get_ema_avg_fn(0.99), use_buffers=True
        )
    if os.getenv("LOCAL_RANK", "0") == "0":
        wandb.watch(proxy_model)
    target_num_tokens_per_batch = 512 * 512
    world_size = int(os.getenv("WORLD_SIZE", 1))
    target_grad_accum = (
        target_num_tokens_per_batch // num_tokens_per_batch // world_size
    )

    schedule_grad_accum = partial(
        grad_accumulation_scheduler,
        init_accum_steps=grad_accum,
        last_accum_steps=target_grad_accum,
    )

    eval_loss = 0
    curr_grad_accum = schedule_grad_accum(step)

    for epoch in range(num_epochs):
        accum_loss = []

        for i, batch in enumerate(dl):
            proxy_model.train()
            input_ids = batch["input_ids"].to(device)
            out = proxy_model(input_ids, labels=input_ids)
            logits = out["logits"]

            if len(accum_loss) % curr_grad_accum == 0 or not isinstance(
                proxy_model, DDP
            ):
                (out["loss"] / curr_grad_accum).backward()
            else:
                with proxy_model.no_sync():
                    (out["loss"] / curr_grad_accum).backward()
            accum_loss.append(out["loss"].item())
            output_ids = out["logits"]
            if os.getenv("LOCAL_RANK", "0") == "0":
                pbar.set_description(
                    f"epoch: {epoch:3d}/{num_epochs:3d}, step: {step:8d}, loss: {out['loss'].item():0.6f}, eval_loss: {eval_loss:0.6f}, lr: {sched.get_last_lr()[0]:0.3e}, grad_accum: {len(accum_loss):3d}/{curr_grad_accum}"
                )
                pbar.update(torch.numel(input_ids) * world_size)
            if len(accum_loss) % curr_grad_accum == 0:
                nn.utils.clip_grad_value_(proxy_model.parameters(), 1.0)
                opt.step()
                if avg_model is not None:
                    avg_model.update_parameters(proxy_model)

                sched.step(step)
                opt.zero_grad()

                step += 1
                tokens += torch.numel(input_ids) * world_size
                proxy_model.eval()
                avg_loss = sum(accum_loss) / len(accum_loss)
                accum_loss = []
                yield epoch, step, tokens, avg_loss, input_ids, output_ids
                curr_grad_accum = schedule_grad_accum(step)
        yield epoch, step, tokens, accum_loss, input_ids, output_ids


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
    grad_checkpoints: int = 4,
    weight_decay: float = 1e-1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-5,
):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 1))
    num_nodes = world_size // local_world_size
    num_workers = int(os.getenv("NUM_WORKERS", num_workers or 4)) // local_world_size

    print(
        f"world_size: {world_size}, local_world_size: {local_world_size}, num_nodes: {num_nodes}, num_workers: {num_workers}"
    )

    if distributed:
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        dist.init_process_group("nccl")

    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, dtype)

    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(tokenizer.vocab_size, **model_config).to(dtype)
    tokenizer.save_pretrained(checkpoint_path)
    total_params = num_params(model)
    print(f"total params: {total_params}")
    step = 0
    tokens = 0
    try:
        ckpts = glob.glob("models/llm*.pt")
        ckpt = sorted(ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1]
        try:
            model.load_state_dict(
                torch.load(ckpt, map_location="cpu", mmap=True), strict=False
            )
            model.to(dtype)
        except Exception as e:
            print(e)
            partial_load_state_dict(model, torch.load(ckpt, mmap=True))
        tags = os.path.basename(ckpt).split(".")[0].split("-")
        step = int(tags[1])
        tokens = int(tags[2])
    except Exception as e:
        print(e)
        print("fail to load a checkpoint, starting from scratch")
    model = add_gradient_checkpoint(model, grad_checkpoints)
    proxy_model = None
    if distributed:
        model = model.to(device)
        proxy_model = DDP(model, gradient_as_bucket_view=True, static_graph=True)
    else:
        proxy_model = model.to(device)

    opt = None
    if distributed:
        opt = ZRO(
            proxy_model.parameters(),
            AdamW,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            fused=True,
            parameters_as_bucket_view=True,
            eps=eps,
        )
    else:
        opt = bnb.optim.AdamW8bit(
            proxy_model.parameters(),
            lr=1e-5,
            weight_decay=weight_decay,
            # fused=True,
            betas=betas,
            eps=eps,
            optim_bits=8,
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
        return {
            "input_ids": inputs["input_ids"],
        }

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers // world_size,
    )

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    if os.getenv("LOCAL_RANK", "0") == "0":
        wandb.init(
            # set the wandb project where this run will be logged
            project="multi-path-llm",
            name=f"{total_params}-{name}-{date}",
            # track hyperparameters and run metadata
            id=f"{total_params}-{name}-{date}",
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

    total_tokens = 5000000000000
    pbar = None
    num_tokens_per_batch = batch_size * max_size
    if local_rank == 0:
        pbar = tqdm(total=total_tokens, dynamic_ncols=True, unit_scale=True)
        pbar.update(tokens)
    iteration = step_model(
        device,
        dl,
        proxy_model,
        opt,
        sched,
        num_epochs,
        grad_accum,
        step,
        tokens,
        pbar,
        enable_compiler,
        distributed,
        ema,
        num_tokens_per_batch,
    )

    for epoch, step, tokens, loss, input_ids, output_ids in iteration:
        if local_rank == 0:
            in_text = tokenizer.decode(input_ids[0][:-1], skip_special_tokens=True)
            train_out_text = tokenizer.decode(
                output_ids[0].argmax(dim=-1)[1:], skip_special_tokens=True
            )
            pbar.write("=" * 64)
            pbar.write(f"INPUT       : {repr(in_text[:64])}...")
            pbar.write(f"TRAIN OUTPUT: {repr(train_out_text[:64])}...")
            pbar.write("=" * 64)
            wandb.log(
                {
                    "loss": loss,
                    "lr": sched.get_last_lr()[0],
                    "tokens": tokens,
                },
                step=step,
            )
            if step % save_every == 0 and math.isfinite(loss):
                ckpts = glob.glob("models/llm*.pt")
                if len(ckpts) > 3:
                    os.remove(
                        sorted(
                            ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0])
                        )[0]
                    )
                model.eval()
                torch.save(
                    model.state_dict(),
                    f"models/llm-{step}-{tokens}.pt",
                )

    pbar.close()
