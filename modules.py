import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.utils.checkpoint import checkpoint


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.linear.weight.data.normal_(mean=0.0, std=math.sqrt(1 / in_features))
        self.linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        mhx = x.view(*x.shape[:-1], -1, self.in_features)
        mhx: Tensor = self.linear(mhx)
        x = mhx.view(*x.shape[:-1], -1)
        return x


class MSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mhx = x.view(-1, self.hidden_size)
        mhx = (
            mhx
            * torch.rsqrt((mhx**2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )
        x = mhx.view_as(x)
        return x


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features

        self.w1 = Linear(in_features, hidden_features, bias=bias, dtype=torch.bfloat16)
        self.w2 = Linear(in_features, hidden_features, bias=bias, dtype=torch.bfloat16)
        self.w3 = Linear(hidden_features, in_features, bias=bias, dtype=torch.bfloat16)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes :attr:`swiglu` with the module's weights

        Args:
            x (torch.Tensor): A Tensor of shape ``[..., in_features]``

        Returns:
            torch.Tensor: A Tensor of shape ``[..., out_features]``
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model
        super().__init__()
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

    def _update_cos_sin_tables(self, seq_len) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.bfloat16, requires_grad=False)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        _cos_cached = emb.cos()[None, None, :, :]
        _sin_cached = emb.sin()[None, None, :, :]

        return _cos_cached, _sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class MonteCarloDropout(nn.Module):
    def __init__(self, p: float = 0.1, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        x = F.dropout(x, self.p, True, self.inplace)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        orient: str = "outer",
        dropout: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.orient = orient
        self.dropout = MonteCarloDropout(dropout)
        self.q_proj = Linear(hidden_size, num_heads * head_size, dtype=torch.bfloat16)
        self.k_proj = Linear(hidden_size, num_heads * head_size, dtype=torch.bfloat16)
        self.v_proj = Linear(hidden_size, num_heads * head_size, dtype=torch.bfloat16)
        self.w_proj = Linear(hidden_size, num_heads * head_size, dtype=torch.bfloat16)
        self.out_proj = Linear(num_heads * head_size, hidden_size, dtype=torch.bfloat16)
        self.rotary = RotaryEmbedding(head_size)
        self.nonlinear = nn.SiLU(True)

    def _reshape_qkv(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            -1,
            self.head_size,
        )
        if self.orient == "outer":
            return hidden_states
        else:
            return hidden_states.transpose(1, 2)

    def _reshape_scores(self, scores: Tensor) -> Tensor:
        if self.orient == "outer":
            return scores.transpose(1, 2).flatten(-2)
        else:
            return scores.flatten(-2)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hidden_states = self.dropout(hidden_states)
        q: Tensor = self.q_proj(hidden_states)
        k: Tensor = self.k_proj(hidden_states)
        v: Tensor = self.v_proj(hidden_states)
        w: Tensor = self.w_proj(hidden_states)
        if key_value_states is not None:
            k_cache, v_cache = key_value_states
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        assert q is not None and k is not None and v is not None
        qh = self._reshape_qkv(q)
        kh = self._reshape_qkv(k)
        vh = self._reshape_qkv(v)

        qh, kh = self.rotary(qh, kh)
        attn_scores = F.scaled_dot_product_attention(
            qh.transpose(1, 2),
            kh.transpose(1, 2),
            vh.transpose(1, 2),
            is_causal=key_value_states is None and self.orient == "outer",
        )

        attn_scores = self._reshape_scores(attn_scores)

        out = self.out_proj(self.nonlinear(w) * attn_scores)

        return out, (k, v)


class DecoderLayer(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, head_size: int, dropout: float = 0.01
    ):
        super().__init__()
        self.pre_outer_norm = MSNorm(hidden_size)
        self.outer_attention = Attention(
            hidden_size, num_heads, head_size, "outer", dropout
        )

        self.pre_inter_norm = MSNorm(hidden_size)
        self.inter_attention = Attention(
            hidden_size, num_heads, head_size, "inner", dropout
        )

    def _outer_forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[
            Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        ] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        residual = hidden_states
        hidden_states = self.pre_outer_norm(residual)
        hidden_states, inter_key_value_states = self.outer_attention(
            hidden_states,
            None if key_value_states is None else key_value_states[0],
        )
        residual = residual + hidden_states
        return residual, inter_key_value_states

    def _inter_forward(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        residual = hidden_states
        hidden_states = self.pre_inter_norm(residual)
        hidden_states, outer_key_value_states = self.inter_attention(
            hidden_states, None
        )
        residual = residual + hidden_states
        return residual, outer_key_value_states

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[
            Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        ] = None,
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
        inter_key_value_states, outer_key_value_states = None, None
        residual, inter_key_value_states = self._outer_forward(
            hidden_states, key_value_states
        )
        residual, outer_key_value_states = self._inter_forward(residual)
        key_value_states = (inter_key_value_states, outer_key_value_states)
        return residual, key_value_states  # type: ignore


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 32,
        num_heads: int = 8,
        head_size: int = 128,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def _pipeline_forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
                for i, hs in enumerate(hidden_states):
                    hs = hs.to(layer.pre_outer_norm.weight.device)
                    hs, _ = layer(hs, None)
                    hidden_states[i] = hs
        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[
            List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
        ] = None,
    ) -> Tuple[Tensor, List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]]:
        new_key_value_states: List[
            Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        ] = []
        if key_value_states is not None:
            for i, layer in enumerate(self.layers):
                kvs = key_value_states[i]
                hidden_states, new_kvs = layer(
                    hidden_states,
                    kvs,
                )
                new_key_value_states.append(new_kvs)
            return hidden_states, new_key_value_states
        else:
            for layer in self.layers:
                hidden_states, new_kvs = layer(hidden_states, None)
                new_key_value_states.append(new_kvs)
            return hidden_states, new_key_value_states
