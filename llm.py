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
from modules import Attention, Decoder, MSNorm
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

    def decode(
        self, input_embeds: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        states = self.decoder(input_embeds, attn_mask=attn_mask)
        return states

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[
            List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
        ] = None,
    ):
        input_embeds = self.embed_tokens(input_ids)
        pred_logits, past_key_values = self.decoder(
            input_embeds, key_value_states=past_key_values
        )
        pred_logits = self.lm_head(pred_logits)

        loss = None
        if labels is not None:
            target = labels[:, 1:].reshape(-1)
            pred = pred_logits[:, :-1].flatten(0, 1)
            loss = F.cross_entropy(pred, target)
            loss.backward()
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

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # return super().load_state_dict(state_dict, strict, assign)
        migrated_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "q_proj.linear" in k:
                k = k.replace("q_proj.linear", "q_proj")
            if "k_proj.linear" in k:
                k = k.replace("k_proj.linear", "k_proj")
            if "v_proj.linear" in k:
                k = k.replace("v_proj.linear", "v_proj")
            if "w_proj.linear" in k:
                k = k.replace("w_proj.linear", "w_proj")
            if "out_proj.linear" in k:
                k = k.replace("out_proj.linear", "out_proj")

            if "pre_outer_norm" in k:
                k = k.replace(
                    "pre_outer_norm",
                    "outer.norm",
                )
            if "outer_attention" in k:
                k = k.replace("outer_attention", "outer.attention")
            if "pre_inter_norm" in k:
                k = k.replace(
                    "pre_inter_norm",
                    "inter.norm",
                )
            if "inter_attention" in k:
                k = k.replace("inter_attention", "inter.attention")
            if "lm_head_norm" in k:
                k = k.replace("lm_head_norm", "decoder.out_norm")
            migrated_state_dict[k] = v

        super().load_state_dict(migrated_state_dict, strict, assign)


def _wrapped_forward(mod: nn.Module, *args, **kwargs):
    if mod.training:
        return checkpoint(mod._org_forward, *args, use_reentrant=True, **kwargs)
    else:
        return mod._org_forward(*args, **kwargs)


def add_gradient_checkpoint(model: LLM):
    for layer in model.decoder.layers:
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
