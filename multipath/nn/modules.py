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
def fused_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps) * weight
    return x


class MultiPathExcitedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        groups: int = 8,
        dtype=torch.bfloat16,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w1 = nn.Parameter(
            torch.randn(
                (out_features, out_features // groups), dtype=dtype, device=device
            )
        )
        self.w2 = nn.Parameter(
            torch.randn(
                (out_features // groups, in_features), dtype=dtype, device=device
            )
        )
        self.bias = (
            None
            if not bias
            else nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        )

    def forward(self, x: Tensor) -> Tensor:
        return fused_excited_linear(x, self.w1, self.w2, self.bias)


class MultiPathExcitedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, x: Tensor) -> Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


class MultiPathExcitedSwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features

        self.w1 = MultiPathExcitedLinear(
            in_features, hidden_features, dtype=torch.bfloat16
        )
        self.w2 = MultiPathExcitedLinear(
            in_features, hidden_features, dtype=torch.bfloat16
        )
        self.w3 = MultiPathExcitedLinear(
            hidden_features, out_features, dtype=torch.bfloat16
        )
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: Tensor) -> Tensor:
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


class MultiPathExcitedRotaryEmbedding(torch.nn.Module):
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
def fused_excited_linear(
    x: Tensor, w1: Tensor, w2: Tensor, b: Optional[Tensor] = None
) -> Tensor:
    if b is None:
        return F.silu(x @ w2.t()) @ w1.t()
    else:
        return F.silu(x @ w2.t()) @ w1.t() + b


@torch.jit.script
def fused_pd_rotary_attention(
    dim: int,
    is_causal: bool,
    head_size: int,
    x: Tensor,
    qw1: Tensor,
    qw2: Tensor,
    qb: Optional[Tensor],
    kw1: Tensor,
    kw2: Tensor,
    kb: Optional[Tensor],
    vw1: Tensor,
    vw2: Tensor,
    vb: Optional[Tensor],
    ow1: Tensor,
    ow2: Tensor,
    ob: Optional[Tensor],
    rotery_cos: Tensor,
    rotery_sin: Tensor,
    dropout: float = 0.01,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x = F.dropout(x, dropout, True)
    q = fused_excited_linear(x, qw1, qw2, qb)
    k = fused_excited_linear(x, kw1, kw2, kb)
    v = fused_excited_linear(x, vw1, vw2, vb)

    q = q.view(q.shape[0], q.shape[1], q.shape[2], -1, head_size).transpose(dim, -2)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], -1, head_size).transpose(dim, -2)
    v = v.view(v.shape[0], v.shape[1], v.shape[2], -1, head_size).transpose(dim, -2)
    q = apply_rotary_pos_emb(q, rotery_cos, rotery_sin)
    k = apply_rotary_pos_emb(k, rotery_cos, rotery_sin)

    o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    o = o.transpose(dim, -2).flatten(-2)

    o = fused_excited_linear(x, ow1, ow2, ob)
    return o, (k.detach(), v.detach())


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
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x = F.dropout(x, dropout, True)
    q = F.linear(x, qw, qb)
    k = F.linear(x, kw, kb)
    v = F.linear(x, vw, vb)

    q = q.view(q.shape[0], q.shape[1], q.shape[2], -1, head_size).transpose(dim, -2)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], -1, head_size).transpose(dim, -2)
    v = v.view(v.shape[0], v.shape[1], v.shape[2], -1, head_size).transpose(dim, -2)
    q = apply_rotary_pos_emb(q, rotery_cos, rotery_sin)
    k = apply_rotary_pos_emb(k, rotery_cos, rotery_sin)

    o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    o = o.transpose(dim, -2).flatten(-2)

    o = F.linear(o, ow, ob)
    return o, (k.detach(), v.detach())


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
    eps: float = 1e-5,
    dropout: float = 0.01,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    residual = x
    x = fused_rmsnorm(x, norm_weight, eps)
    x, kv_cache = fused_rotary_attention(
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
    return x, kv_cache


@torch.jit.script
def fused_pd_kvcache_rotary_attention(
    dim: int,
    head_size: int,
    cache_k: Optional[Tensor],
    cache_v: Optional[Tensor],
    x: Tensor,
    qw1: Tensor,
    qw2: Tensor,
    qb: Optional[Tensor],
    kw1: Tensor,
    kw2: Tensor,
    kb: Optional[Tensor],
    vw1: Tensor,
    vw2: Tensor,
    vb: Optional[Tensor],
    ow1: Tensor,
    ow2: Tensor,
    ob: Optional[Tensor],
    rotery_cos: Tensor,
    rotery_sin: Tensor,
    dropout: float = 0.01,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x = F.dropout(x, dropout, True)
    q = fused_excited_linear(x, qw1, qw2, qb)
    k = fused_excited_linear(x, kw1, kw2, kb)
    v = fused_excited_linear(x, vw1, vw2, vb)
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

    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    o = o.transpose(dim, -2).flatten(-2)
    o = fused_excited_linear(x, ow1, ow2, ob)
    return o, (cache_k.detach(), cache_v.detach())


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

    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    o = o.transpose(dim, -2).flatten(-2)
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
    eps: float = 1e-5,
    dropout: float = 0.01,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    residual = x
    x = fused_rmsnorm(x, norm_weight, eps)
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


class MultiPathExcitedAttention(nn.Module):
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
        self.dropout = dropout
        self.num_heads = num_heads
        self.dropout = MonteCarloDropout(dropout)
        self.q_proj = MultiPathExcitedLinear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16
        )
        self.k_proj = MultiPathExcitedLinear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16
        )
        self.v_proj = MultiPathExcitedLinear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16
        )
        self.out_proj = MultiPathExcitedLinear(
            num_heads * head_size, hidden_size, dtype=torch.bfloat16
        )
        self.rotary = MultiPathExcitedRotaryEmbedding(head_size)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        tdim = None

        if self.orient == "outer":
            tdim = 1
        elif self.orient == "inter":
            tdim = 2
            is_causal = False
        else:
            raise ValueError(f"Invalid attention orientation: {self.orient}")
        if key_value_states is None:
            return fused_pd_rotary_attention(
                tdim,
                is_causal,
                self.head_size,
                hidden_states,
                self.q_proj.w1,
                self.q_proj.w2,
                self.q_proj.bias,
                self.k_proj.w1,
                self.k_proj.w2,
                self.k_proj.bias,
                self.v_proj.w1,
                self.v_proj.w2,
                self.v_proj.bias,
                self.out_proj.w1,
                self.out_proj.w2,
                self.out_proj.bias,
                self.rotary._cos_cached,
                self.rotary._sin_cached,
                self.dropout.p,
            )
        else:
            return fused_pd_kvcache_rotary_attention(
                tdim,
                self.head_size,
                key_value_states[0],
                key_value_states[1],
                hidden_states,
                self.q_proj.w1,
                self.q_proj.w2,
                self.q_proj.bias,
                self.k_proj.w1,
                self.k_proj.w2,
                self.k_proj.bias,
                self.v_proj.w1,
                self.v_proj.w2,
                self.v_proj.bias,
                self.out_proj.w1,
                self.out_proj.w2,
                self.out_proj.bias,
                self.rotary._cos_cached,
                self.rotary._sin_cached,
                self.dropout.p,
            )


class MultiPathExcitedDecoderAttention(nn.Module):
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
        self.norm = MultiPathExcitedRMSNorm(hidden_size)
        self.attention = MultiPathExcitedAttention(
            hidden_size, num_heads, head_size, orient
        )

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states, key_value_states = self.attention(
            hidden_states, key_value_states, is_causal
        )
        hidden_states = residual + hidden_states
        return hidden_states, key_value_states


class MultiPathExcitedDecoderMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.norm = MultiPathExcitedRMSNorm(hidden_size)
        self.mlp = MultiPathExcitedSwiGLU(
            hidden_size, hidden_size * 8 // 3, hidden_size
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.mlp(self.norm(hidden_states)) + hidden_states


class MultiPathExcitedDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.outer = MultiPathExcitedDecoderAttention(
            hidden_size,
            num_heads,
            head_size,
            dropout,
        )
        self.inter = MultiPathExcitedDecoderAttention(
            hidden_size, num_heads, head_size, dropout, "inter"
        )
        self.mlp = MultiPathExcitedDecoderMLP(
            hidden_size,
            num_heads,
            head_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[_KVCT] = None,
        is_causal: bool = True,
    ) -> Tuple[Tensor, _KVCT]:
        if key_value_states is None:
            key_value_states = [None, None]
        residual, kvc1 = self.outer(hidden_states, key_value_states[0], is_causal)
        residual, kvc2 = self.inter(residual, key_value_states[1], is_causal)
        residual = self.mlp(residual)
        return residual, (kvc1, kvc2)


class MultiPathExcitedTransformerDecoder(nn.Module):
    def __init__(
        self,
        total_size: int = 16384,
        path_size: int = 2048,
        num_layers: int = 32,
        num_heads: int = 32,
        head_size: int = 64,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.total_size = total_size
        self.path_size = path_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.in_norm = nn.LayerNorm(total_size)
        self.in_proj = nn.Linear(total_size, total_size)
        self.layers = nn.ModuleList(
            [
                MultiPathExcitedDecoderLayer(
                    hidden_size=path_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = MultiPathExcitedRMSNorm(path_size)
        self.out_proj = nn.Linear(total_size, total_size)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[List[_KVCT]] = None,
        is_causal: bool = True,
    ) -> Tuple[Tensor, List[_KVCT]]:
        new_key_value_states: List[_KVCT] = []
        hidden_states = self.in_norm(hidden_states)
        hidden_states = F.silu(hidden_states, True)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], hidden_states.shape[1], -1, self.path_size
        )
        if key_value_states is not None:
            for i, layer in enumerate(self.layers):
                kvs = key_value_states[i]
                hidden_states, new_kvs = layer(
                    hidden_states,
                    kvs,
                    is_causal,
                )
                new_key_value_states.append(new_kvs)
        else:
            for layer in self.layers:
                hidden_states, new_kvs = layer(hidden_states, None, is_causal)
                new_key_value_states.append(new_kvs)

        hidden_states = self.out_norm(hidden_states)
        hidden_states = F.silu(hidden_states, True)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states, new_key_value_states
