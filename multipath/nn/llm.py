from collections import OrderedDict
from functools import partial
import math
import random
from typing import Any, List, Mapping, Optional, Tuple, Union
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch import Tensor, nn  # type: ignore
from torch.backends import cuda, cudnn
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm  # type: ignore
import matplotlib.pyplot as plt
from .modules import Attention, Decoder, MSNorm
import matplotlib
from transformers import AutoTokenizer
from enum import Enum

matplotlib.use("Agg")


class LLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        bunch_size: int = 8,
        hidden_size: int = 512,
        num_layers: int = 80,
        num_heads: int = 8,
        head_size: int = 64,
        padding_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.bunch_size = bunch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.padding_idx = padding_idx

        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size * bunch_size,
            padding_idx=padding_idx,
            dtype=torch.bfloat16,
        )
        self.decoder = Decoder(hidden_size, num_layers, num_heads, head_size)
        self.lm_head = nn.Linear(
            bunch_size * hidden_size, vocab_size, dtype=torch.bfloat16
        )

    def _forward(self, input_ids: Tensor) -> Tensor:
        input_embeds = self.embed_tokens(input_ids)
        pred_logits, _ = self.decoder(input_embeds)
        pred_logits = self.lm_head(pred_logits)
        return pred_logits

    @property
    def model_config(self):
        return {
            "vocab_size": self.vocab_size,
            "bunch_size": self.bunch_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "padding_idx": self.padding_idx,
        }

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[
            List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
        ] = None,
    ):
        pred_logits = None
        loss = None
        if labels is not None:
            pred_logits = self._forward(input_ids)
            if labels is not None:
                target = labels[:, 1:].reshape(-1)
                pred = pred_logits[:, :-1].flatten(0, 1)

                pred_id_samples = torch.multinomial(
                    pred.detach().float().softmax(dim=-1), 8, replacement=True
                )
                luckyhits = (pred_id_samples == target.unsqueeze(-1)).sum(dim=-1)
                weight = 1 - torch.sqrt(luckyhits.float() / 8)

                loss = F.cross_entropy(pred, target, reduction="none")
                loss = (loss * weight).mean()
        else:
            input_embeds = self.embed_tokens(input_ids)
            pred_logits, past_key_values = self.decoder(
                input_embeds, key_value_states=past_key_values
            )
            pred_logits = self.lm_head(pred_logits)
        return {"logits": pred_logits, "loss": loss, "past_key_values": past_key_values}

    def generate(self, input_ids: Tensor, max_length: int = 512):
        key_value_states = None
        input_embeds = self.embed_tokens(input_ids)
        pred_ids = input_ids
        with torch.inference_mode():
            for _ in tqdm(range(max_length)):
                pred_logits, key_value_states = self.decoder(
                    input_embeds, key_value_states
                )
                pred_logits = self.lm_head(pred_logits)
                pred_ids = torch.cat(
                    [pred_ids, pred_logits.argmax(dim=-1)[:, -1:]], dim=1
                )
                input_embeds = self.embed_tokens(pred_ids[:, -1:])
        return pred_ids

    @torch.inference_mode()
    def stream(self, tokenizer: AutoTokenizer, query: str, device: str = "cuda"):
        self.eval()
        input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
        past_key_values = None
        for _ in range(512):
            outputs = self(input_ids, past_key_values=past_key_values)
            pred_logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            pred_logits = pred_logits[:, -1:].view(-1)
            topk = 10
            probs, token_ids = pred_logits.topk(topk, dim=-1, largest=True, sorted=True)
            probs = (probs * topk).softmax(dim=-1)[probs > 0.01]
            if probs.numel() > 3:
                sample_id = torch.multinomial(probs, 3)
                sample_id = sample_id[random.randint(0, sample_id.numel() - 1)]
            else:
                sample_id = 0
            sample_token_id = token_ids[sample_id]
            pred_strings = tokenizer.decode(sample_token_id)
            if sample_token_id == tokenizer.eos_token_id:
                return pred_strings
            yield pred_strings
            input_ids = torch.as_tensor(sample_token_id).view(-1, 1)


def _wrapped_forward(mod: nn.Module, *args, **kwargs):
    if mod.training:
        return checkpoint(mod._org_forward, *args, use_reentrant=True, **kwargs)
    else:
        return mod._org_forward(*args, **kwargs)


def add_gradient_checkpoint(model: LLM, splits: int = 4):
    for i in range(0, len(model.decoder.layers), len(model.decoder.layers) // splits):
        layer = model.decoder.layers[i]
        layer._org_forward = layer.forward
        layer.forward = partial(_wrapped_forward, layer)
    return model


class CasualModel(Enum):
    DAVID_100M = partial(
        LLM,
        **{
            "bunch_size": 2,
            "hidden_size": 512,
            "num_layers": 24,
            "num_heads": 16,
            "head_size": 64,
        },
    )
    DAVID_500M = partial(
        LLM,
        **{
            "bunch_size": 8,
            "hidden_size": 512,
            "num_layers": 80,
            "num_heads": 16,
            "head_size": 64,
        },
    )
    DAVID_3B = partial(
        LLM,
        **{
            "bunch_size": 16,
            "hidden_size": 1024,
            "num_layers": 96,
            "num_heads": 16,
            "head_size": 128,
        },
    )
