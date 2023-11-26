from functools import partial
from typing import List, Optional, Tuple, Union

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch import Tensor, nn  # type: ignore
from torch.backends import cuda, cudnn
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm  # type: ignore
import matplotlib.pyplot as plt
from modules import Decoder, MSNorm, MPLinear
import matplotlib

matplotlib.use("Agg")
cudnn.benchmark = True
cuda.matmul.allow_tf32 = True


class LLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        bunch_size: int = 8,
        hidden_size: int = 512,
        num_layers: int = 80,
        head_size: int = 64,
        padding_idx: int = 0,
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
        self.embed_norm = MSNorm(hidden_size)
        self.decoder = Decoder(hidden_size, num_layers, head_size)
        self.lm_head_norm = MSNorm(hidden_size)
        self.lm_head = MPLinear(hidden_size, vocab_size, dtype=torch.bfloat16)

    def enable_gradient_checkpointing(self):
        self.decoder.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.decoder.disable_gradient_checkpointing()

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
        input_embeds = self.embed_norm(input_embeds)
        pred_logits, past_key_values = self.decoder(
            input_embeds, key_value_states=past_key_values
        )
        pred_logits = self.lm_head_norm(pred_logits)
        pred_logits = self.lm_head(pred_logits)
        pred_logits = pred_logits.view(
            pred_logits.shape[0], pred_logits.shape[1], self.bunch_size, self.vocab_size
        )
        pred_logits = torch.logsumexp(pred_logits, dim=-2)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                pred_logits[:, :-1].flatten(0, 1),
                labels[:, 1:].reshape(-1),
            )
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


def add_gradient_checkpoint(model: LLM):
    for layer in model.decoder.layers:
        layer._org_forward = layer.forward
        layer.forward = partial(checkpoint, layer._org_forward, use_reentrant=False)
    return model
