import torch
import torch.nn.functional as F
from torch import Tensor, nn  # type: ignore
from typing import Optional, Tuple, Union


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[..., : x.shape[-2], :]
    sin = sin[..., : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model
        # Generate and save the inverse frequency buffer (non trainable)
        max_seq_length = 10000
        inv_freq = 1.0 / (
            max_seq_length
            ** (
                torch.arange(0, dim_model, 2, dtype=torch.bfloat16, requires_grad=False)
                / dim_model
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        _cos_cached, s_sin_cached = self._update_cos_sin_tables(max_seq_length)
        self.register_buffer("_cos_cached", _cos_cached, persistent=False)
        self.register_buffer("_sin_cached", s_sin_cached, persistent=False)

    def _update_cos_sin_tables(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.bfloat16, requires_grad=False)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        _cos_cached = emb.cos()[None, None, None, :, :]
        _sin_cached = emb.sin()[None, None, None, :, :]

        return _cos_cached, _sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class BIKVAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_size: int = 64,
        num_kv: int = 65536,
        index_size: int = 64,
    ):
        super().__init__()
        self.num_kv = num_kv
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.index_size = index_size

        self.i_proj = nn.Linear(
            hidden_size, index_size, dtype=torch.bfloat16, bias=False
        )
        self.q_proj = nn.Linear(
            hidden_size, hidden_size, dtype=torch.bfloat16, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, hidden_size, dtype=torch.bfloat16, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, hidden_size, dtype=torch.bfloat16, bias=False
        )

        self.out_proj = nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16)

        self.rotary = RotaryEmbedding(head_size)

        self.indices = nn.Parameter(
            torch.randn(
                num_kv,
                index_size,
            )
        )
        self.keys = nn.Parameter(
            torch.randn(
                num_kv,
                hidden_size,
            )
        )
        self.values = nn.Parameter(
            torch.randn(
                num_kv,
                hidden_size,
            )
        )

    def forward(self, input_embeds: Tensor, is_causal: bool = True) -> Tensor:
        indices = F.sigmoid(self.i_proj(input_embeds))
        with torch.no_grad():
            cached_indices = F.sigmoid(self.i_proj(self.indices))
            choices = torch.matmul(indices, self.indices.t()).argmax(-1)
        chosen_indices = torch.index_select(cached_indices, 0, choices)
        index_weights = torch.matmul(indices, chosen_indices.t())
        chosen_keys = torch.index_select(self.keys, 0, choices)
        chosen_values = torch.index_select(self.values, 0, choices)
        q = self.q_proj(input_embeds)
        k = self.k_proj(chosen_keys)
        v = self.v_proj(chosen_values)

        q = q.view(q.shape[0], q.shape[1], -1, self.head_size)
        k = k.view(k.shape[0], k.shape[1], -1, self.head_size)
        v = v.view(v.shape[0], v.shape[1], -1, self.head_size)

        q, k = self.rotary(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        o = F.scaled_dot_product_attention(q, k, v, index_weights, is_causal=is_causal)
        o = o.transpose(1, 2)
        o = o.view(o.shape[0], o.shape[1], -1)
        o = self.out_proj(o)

        return o
