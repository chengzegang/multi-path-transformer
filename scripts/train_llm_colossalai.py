import glob
from functools import partial

import colossalai
import torch
import torch._dynamo.config
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from torch.backends import cuda, cudnn
from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm.auto import tqdm
from transformers import AutoTokenizer  # type:ignore

from multipath.nn.llm import LLM
from multipath.torch_datasets.llm_datasets import Pile, Sentence, WebData
from scripts.train_llm import num_params

torch._dynamo.config.cache_size_limit = 256
cudnn.benchmark = True
cudnn.allow_tf32 = True
cuda.matmul.allow_bf16_reduced_precision_reduction = True
cuda.matmul.allow_tf32 = True


def train(
    rank: int = 0,
    world_size: int = 1,
    port: int = 27900,
    max_size=1024,
    batch_size=1,
    num_workers=1,
    save_path="./model/colossalai/",
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

    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.98),
        eps=1e-5,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    def criterion(input_ids, logits):
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].view(-1, logits.size(-1)), input_ids[:, 1:].view(-1)
        )

    # use booster.boost to wrap the training objects
    model, optimizer, criterion, _, scheduler = booster.boost(
        model, optimizer, criterion, lr_scheduler=scheduler
    )
    data = None
    data = WebData()
    dataset = Sentence(data, max_size=max_size, tokenizer=tokenizer)

    try:
        booster.load_model(model, save_path)
    except Exception as e:
        print(e)
        print("Starting from scratch")

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
    # do training as normal, except that the backward should be called by booster
    step = 0
    for i, batch in enumerate(pbar := tqdm(dl)):
        input_ids = batch["input_ids"].to(rank)
        outputs = model(input_ids)
        loss = criterion(input_ids, outputs["logits"])
        booster.backward(loss, optimizer)
        optimizer.clip_grad_by_norm(1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        pbar.set_description(f"loss {loss.item():.4f}")
        if step % 100 == 0:
            # checkpointing using booster api
            booster.save_model(
                model, save_path, shard=True, size_per_shard=10, use_safetensors=True
            )


if __name__ == "__main__":
    train()
