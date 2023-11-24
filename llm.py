from functools import partial
from typing import Optional, Union

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch import Tensor, nn  # type: ignore
from torch.backends import cuda, cudnn
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm  # type: ignore

from modules import Decoder, MHRMSNorm, MHLinear, MHLinearMean

cudnn.benchmark = True
cuda.matmul.allow_tf32 = True


class LLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        head_size: int = 128,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx, dtype=torch.bfloat16
        )
        self.embed_norm = MHRMSNorm(hidden_size)
        self.decoder = Decoder(hidden_size, num_layers, head_size)
        self.lm_head = MHLinearMean(hidden_size, vocab_size, dtype=torch.bfloat16)
        self.token_seen = 0

    def save_to_file(self, path: str):
        state_dict = self.state_dict()
        snapshot = {
            "state_dict": state_dict,
            "model_config": {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "head_size": self.head_size,
                "padding_idx": self.padding_idx,
                "token_seen": self.token_seen,
            },
        }
        torch.save(snapshot, path)

    def enable_gradient_checkpointing(self):
        self.decoder.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.decoder.disable_gradient_checkpointing()

    @classmethod
    def load_from_file(cls, path: str):
        snapshot = torch.load(path, mmap=True, map_location="cpu")
        token_seen = snapshot["model_config"].pop("token_seen", 0)
        obj = cls(**snapshot["model_config"])
        obj.load_state_dict(snapshot["state_dict"])
        obj.token_seen = token_seen
        return obj

    def build_attention_mask(
        self, seq_length: int, device: Optional[torch.device] = None
    ):
        mask = torch.tril(
            torch.ones(
                seq_length,
                seq_length,
                dtype=torch.bool,
                device=device,
            )
        )
        return mask

    def decode(
        self, input_embeds: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        states = self.decoder(input_embeds, attn_mask=attn_mask)
        return states

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
    ):
        attn_mask = self.build_attention_mask(
            input_ids.shape[1], device=input_ids.device
        )
        input_embeds = self.embed_tokens(input_ids)
        input_embeds = self.embed_norm(input_embeds)
        pred_logits, _ = self.decoder(input_embeds, attn_mask=attn_mask)
        pred_logits = self.lm_head(pred_logits)
        print(pred_logits.shape, labels.shape)
        loss = F.cross_entropy(
            pred_logits[:, :-1].flatten(0, 1),
            labels[:, 1:].reshape(-1),
        )
        if self.training:
            self.token_seen += input_ids.numel()
        return {"logits": pred_logits, "loss": loss}

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


def add_gradient_checkpoint(model: LLM):
    for layer in model.decoder.layers:
        layer._org_forward = layer.forward
        layer.forward = partial(checkpoint, layer._org_forward, use_reentrant=False)
    return model
