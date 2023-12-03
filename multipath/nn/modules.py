import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.utils.checkpoint import checkpoint
import torch._dynamo

torch._dynamo.config.suppress_errors = True
_KVCT = Tuple[
    Optional[Tuple[Tensor, Tensor]],
    Optional[Tuple[Tensor, Tensor]],
    Optional[Tuple[Tensor, Tensor]],
]


# @torch.compile(dynamic=False, mode="max-autotune")
@torch.jit.script
def fused_msnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps) * weight
    return x


class MSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, x: Tensor) -> Tensor:
        return fused_msnorm(x, self.weight, self.eps)


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

        self.w1 = nn.Linear(
            in_features, hidden_features, bias=bias, dtype=torch.bfloat16
        )
        self.w2 = nn.Linear(
            in_features, hidden_features, bias=bias, dtype=torch.bfloat16
        )
        self.w3 = nn.Linear(
            hidden_features, in_features, bias=bias, dtype=torch.bfloat16
        )

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: Tensor) -> Tensor:
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


class MonteCarloDropout(nn.Module):
    def __init__(self, p: float = 0.1, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        x = F.dropout(x, self.p, True, self.inplace)
        return x


@torch.jit.script
def fused_rotary_attention(
    dim: int,
    is_causal: bool,
    head_size: int,
    x: Tensor,
    qw: Tensor,
    qb: Optional[Tensor],
    kw: Tensor,
    kb: Optional[Tensor],
    vw: Tensor,
    vb: Optional[Tensor],
    ow: Tensor,
    ob: Optional[Tensor],
    rotery_cos: Tensor,
    rotery_sin: Tensor,
    dropout: float = 0.01,
) -> Tensor:
    x = F.dropout(x, dropout, True)
    q = F.linear(x, qw, qb)
    k = F.linear(x, kw, kb)
    v = F.linear(x, vw, vb)
    q = q.view(q.shape[0], q.shape[1], q.shape[2], -1, head_size).transpose(dim, -2)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], -1, head_size).transpose(dim, -2)
    v = v.view(v.shape[0], v.shape[1], v.shape[2], -1, head_size).transpose(dim, -2)
    q = apply_rotary_pos_emb(q, rotery_cos, rotery_sin)
    k = apply_rotary_pos_emb(k, rotery_cos, rotery_sin)
    q = q.unsqueeze(dim=dim + 1)
    k = k.unsqueeze(dim=dim)
    v = v.unsqueeze(dim=dim)

    o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    o = o.mean(dim=dim + 1).transpose(dim, -2).flatten(-2)
    o = F.linear(o, ow, ob)
    return o


@torch.jit.script
def fused_decoder_layer(
    norm_weight: Tensor,
    dim: int,
    is_causal: bool,
    head_size: int,
    x: Tensor,
    qw: Tensor,
    qb: Optional[Tensor],
    kw: Tensor,
    kb: Optional[Tensor],
    vw: Tensor,
    vb: Optional[Tensor],
    ow: Tensor,
    ob: Optional[Tensor],
    rotery_cos: Tensor,
    rotery_sin: Tensor,
    eps: float = 1e-6,
    dropout: float = 0.01,
) -> Tensor:
    residual = x
    x = fused_msnorm(x, norm_weight, eps)
    x = fused_rotary_attention(
        dim,
        is_causal,
        head_size,
        x,
        qw,
        qb,
        kw,
        kb,
        vw,
        vb,
        ow,
        ob,
        rotery_cos,
        rotery_sin,
        dropout,
    )
    x = residual + x
    return x


@torch.jit.script
def fused_kvcache_rotary_attention(
    dim: int,
    head_size: int,
    cache_k: Optional[Tensor],
    cache_v: Optional[Tensor],
    x: Tensor,
    qw: Tensor,
    qb: Optional[Tensor],
    kw: Tensor,
    kb: Optional[Tensor],
    vw: Tensor,
    vb: Optional[Tensor],
    ow: Tensor,
    ob: Optional[Tensor],
    rotery_cos: Tensor,
    rotery_sin: Tensor,
    dropout: float = 0.01,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x = F.dropout(x, dropout, True)
    q = F.linear(x, qw, qb)
    k = F.linear(x, kw, kb)
    v = F.linear(x, vw, vb)
    if cache_k is not None and cache_v is not None:
        k = torch.cat([cache_k, k], dim=1)
        v = torch.cat([cache_v, v], dim=1)
    cache_k = k
    cache_v = v
    q = q.view(q.shape[0], q.shape[1], q.shape[2], -1, head_size).transpose(dim, -2)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], -1, head_size).transpose(dim, -2)
    v = v.view(v.shape[0], v.shape[1], v.shape[2], -1, head_size).transpose(dim, -2)
    q = apply_rotary_pos_emb(q, rotery_cos, rotery_sin)
    k = apply_rotary_pos_emb(k, rotery_cos, rotery_sin)
    q = q.unsqueeze(dim=dim + 1)
    k = k.unsqueeze(dim=dim)
    v = v.unsqueeze(dim=dim)

    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    o = o.mean(dim=dim + 1).transpose(dim, -2).flatten(-2)
    o = F.linear(o, ow, ob)
    return o, (cache_k.detach(), cache_v.detach())


@torch.jit.script
def fused_kvcache_decoder_layer(
    norm_weight: Tensor,
    dim: int,
    head_size: int,
    cache_k: Optional[Tensor],
    cache_v: Optional[Tensor],
    x: Tensor,
    qw: Tensor,
    qb: Optional[Tensor],
    kw: Tensor,
    kb: Optional[Tensor],
    vw: Tensor,
    vb: Optional[Tensor],
    ow: Tensor,
    ob: Optional[Tensor],
    rotery_cos: Tensor,
    rotery_sin: Tensor,
    eps: float = 1e-6,
    dropout: float = 0.01,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    residual = x
    x = fused_msnorm(x, norm_weight, eps)
    x, kv_cache = fused_kvcache_rotary_attention(
        dim,
        head_size,
        cache_k,
        cache_v,
        x,
        qw,
        qb,
        kw,
        kb,
        vw,
        vb,
        ow,
        ob,
        rotery_cos,
        rotery_sin,
        dropout,
    )
    x = residual + x
    return x, kv_cache


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
        self.q_proj = nn.Linear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16, bias=False
        )
        self.out_proj = nn.Linear(
            num_heads * head_size, hidden_size, dtype=torch.bfloat16, bias=False
        )
        self.rotary = RotaryEmbedding(head_size)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        tdim = None
        is_causal = False

        if self.orient == "outer":
            tdim = 1
            is_causal = True
        elif self.orient == "inter":
            tdim = 2
        else:
            tdim = 3
        if self.training:
            return (
                fused_rotary_attention(
                    tdim,
                    is_causal,
                    self.head_size,
                    hidden_states,
                    self.q_proj.weight,
                    self.q_proj.bias,
                    self.k_proj.weight,
                    self.k_proj.bias,
                    self.v_proj.weight,
                    self.v_proj.bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.rotary._cos_cached,
                    self.rotary._sin_cached,
                    self.dropout.p,
                ),
                None,
            )
        else:
            return fused_kvcache_rotary_attention(
                tdim,
                self.head_size,
                key_value_states[0],
                key_value_states[1],
                hidden_states,
                self.q_proj.weight,
                self.q_proj.bias,
                self.k_proj.weight,
                self.k_proj.bias,
                self.v_proj.weight,
                self.v_proj.bias,
                self.out_proj.weight,
                self.out_proj.bias,
                self.rotary._cos_cached,
                self.rotary._sin_cached,
                self.dropout.p,
            )


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        dropout: float = 0.01,
        orient: str = "outer",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout = dropout
        self.orient = orient
        self.norm = MSNorm(hidden_size)
        self.attention = Attention(hidden_size, num_heads, head_size, orient, dropout)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        if key_value_states is None:
            return (
                fused_decoder_layer(
                    self.norm.weight,
                    1 if self.orient == "outer" else 2 if self.orient == "inter" else 3,
                    True if self.orient == "outer" else False,
                    self.head_size,
                    hidden_states,
                    self.attention.q_proj.weight,
                    self.attention.q_proj.bias,
                    self.attention.k_proj.weight,
                    self.attention.k_proj.bias,
                    self.attention.v_proj.weight,
                    self.attention.v_proj.bias,
                    self.attention.out_proj.weight,
                    self.attention.out_proj.bias,
                    self.attention.rotary._cos_cached,
                    self.attention.rotary._sin_cached,
                    self.norm.eps,
                    self.attention.dropout.p,
                ),
                None,
            )
        else:
            return fused_kvcache_decoder_layer(
                self.norm.weight,
                1 if self.orient == "outer" else 2 if self.orient == "inter" else 3,
                True if self.orient == "outer" else False,
                self.head_size,
                key_value_states[0],
                key_value_states[1],
                hidden_states,
                self.attention.q_proj.weight,
                self.attention.q_proj.bias,
                self.attention.k_proj.weight,
                self.attention.k_proj.bias,
                self.attention.v_proj.weight,
                self.attention.v_proj.bias,
                self.attention.out_proj.weight,
                self.attention.out_proj.bias,
                self.attention.rotary._cos_cached,
                self.attention.rotary._sin_cached,
                self.norm.eps,
                self.attention.dropout.p,
            )


class DecoderOuterLayer(_DecoderLayer):
    def __init__(
        self, hidden_size: int, num_heads: int, head_size: int, dropout: float = 0.01
    ):
        super().__init__(hidden_size, num_heads, head_size, dropout, "outer")


class DecoderInterLayer(_DecoderLayer):
    def __init__(
        self, hidden_size: int, num_heads: int, head_size: int, dropout: float = 0.01
    ):
        super().__init__(hidden_size, num_heads, head_size, dropout, "inter")


class DecoderInnerLayer(_DecoderLayer):
    def __init__(
        self, hidden_size: int, num_heads: int, head_size: int, dropout: float = 0.01
    ):
        super().__init__(hidden_size, num_heads, head_size, dropout, "inner")


class DecoderLayer(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, head_size: int, dropout: float = 0.01
    ):
        super().__init__()
        self.outer = DecoderOuterLayer(hidden_size, num_heads, head_size, dropout)
        self.inter = DecoderInterLayer(hidden_size, num_heads, head_size, dropout)
        self.inner = DecoderInnerLayer(hidden_size, num_heads, head_size, dropout)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[_KVCT] = None,
    ) -> Tuple[Tensor, _KVCT]:
        if key_value_states is None:
            key_value_states = [None, None, None]
        residual, kvc1 = self.outer(hidden_states, key_value_states[0])
        residual, kvc2 = self.inter(residual, key_value_states[1])
        residual, kvc3 = self.inner(residual, key_value_states[2])
        return residual, (kvc1, kvc2, kvc3)


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
        self.out_norm = MSNorm(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[List[_KVCT]] = None,
    ) -> Tuple[Tensor, List[_KVCT]]:
        new_key_value_states: List[_KVCT] = []
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1, self.hidden_size
        ).contiguous()
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
            hidden_states = self.out_norm(hidden_states)
            hidden_states = hidden_states.flatten(-2)
            return hidden_states, new_key_value_states
