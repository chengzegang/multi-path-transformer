import glob
import math
from functools import partial

import colossalai
import torch
import torch._dynamo.config
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, TorchDDPPlugin
from torch.backends import cuda, cudnn
from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm.auto import tqdm
from transformers import AutoTokenizer  # type:ignore
from colossalai.nn.optimizer.fused_adam import FusedAdam
from multipath.nn.llm import LLM
from multipath.torch_datasets.llm_datasets import Pile, Sentence, WebData
from scripts.train_llm import num_params

torch._dynamo.config.cache_size_limit = 256
cudnn.benchmark = True
cudnn.allow_tf32 = True
cuda.matmul.allow_bf16_reduced_precision_reduction = True
cuda.matmul.allow_tf32 = True


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + math.cos(step / total_steps * math.pi)
    )


def train(
    rank: int = 0,
    world_size: int = 1,
    port: int = 27900,
    max_size=256,
    batch_size=1,
    num_workers=1,
    save_path="./models/colossalai/",
):
    # launch colossalai
    tokenizer_id = "meta-llama/Llama-2-7b-chat-hf"
    model_config = {
        "bunch_size": 8,
        "hidden_size": 512,
        "num_layers": 80,
        "num_heads": 16,
        "head_size": 64,
    }
    dtype = torch.bfloat16
    colossalai.launch(
        config=dict(), rank=rank, world_size=world_size, port=port, host="localhost"
    )

    # create plugin and objects for training
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(tokenizer.vocab_size, **model_config).to(dtype).to(rank)

    total_params = num_params(model)
    print(f"total params: {total_params}")

    plugin = GeminiPlugin(
        enable_gradient_accumulation=True,
        precision="bf16",
        max_norm=1.0,
        master_weights=False,
        hidden_dim=512,
    )
    booster = Booster(plugin=plugin)
    optimizer = FusedAdam(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.98),
        eps=1e-5,
    )
    scheduler = LambdaLR(
        optimizer, partial(cosine_lr, total_steps=100000, lr_max=1.0, lr_min=0.1)
    )

    def criterion(input_ids, logits):
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].view(-1, logits.size(-1)), input_ids[:, 1:].view(-1)
        )

    data = None
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
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers // world_size,
    )

    # use booster.boost to wrap the training objects
    model, optimizer, criterion, dl, scheduler = booster.boost(
        model, optimizer, criterion, dl, scheduler
    )

    try:
        booster.load_model(model, save_path)
    except Exception as e:
        print(e)
        print("Starting from scratch")

    # do training as normal, except that the backward should be called by booster
    step = 0
    gradient_accumulation_steps = 512 * 512 // world_size // batch_size // max_size
    model.cpu()
    for i, batch in enumerate(pbar := tqdm(dl)):
        input_ids = batch["input_ids"]
        model.train()
        outputs = model(input_ids)
        loss = criterion(input_ids, outputs["logits"])
        booster.backward(loss, optimizer)
        if i % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        pbar.set_description(
            f"loss {loss.item():.4f} lr: {scheduler.get_last_lr()[0]}, step: {step}"
        )
        if step % 100 == 0:
            # checkpointing using booster api
            booster.save_model(model, save_path, shard=False, use_safetensors=True)


if __name__ == "__main__":
    import torch._dynamo

    torch._dynamo.reset()
    train()
