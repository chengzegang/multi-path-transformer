import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.utils.checkpoint import checkpoint
import torch._dynamo

torch._dynamo.config.suppress_errors = True


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
def factorize_head(q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    num_heads = q.shape[-2]
    q = q.repeat_interleave(num_heads, dim=-2)
    k = k.repeat(1, 1, 1, num_heads, 1)
    v = v.repeat(1, 1, 1, num_heads, 1)
    return q, k, v


@torch.jit.script
def defactorize_head(o: Tensor) -> Tensor:
    num_heads = int(math.sqrt(o.shape[-2]))
    return o.view(o.shape[0], o.shape[1], o.shape[2], -1, num_heads, o.shape[-1]).mean(
        dim=-2
    )


# @torch.compile(dynamic=False, mode="max-autotune")
@torch.jit.script
def fused_outer_rotary_attention(
    head_size: int,
    x: Tensor,
    qw: Tensor,
    qb: Tensor,
    kw: Tensor,
    kb: Tensor,
    vw: Tensor,
    vb: Tensor,
    ow: Tensor,
    ob: Tensor,
    rotery_cos: Tensor,
    rotery_sin: Tensor,
) -> Tensor:
    q = F.linear(x, qw, qb)
    k = F.linear(x, kw, kb)
    v = F.linear(x, vw, vb)
    q = q.view(q.shape[0], q.shape[1], q.shape[2], -1, head_size).transpose(1, -2)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], -1, head_size).transpose(1, -2)
    v = v.view(v.shape[0], v.shape[1], v.shape[2], -1, head_size).transpose(1, -2)
    q, k, v = factorize_head(q, k, v)
    q = apply_rotary_pos_emb(q, rotery_cos, rotery_sin)
    k = apply_rotary_pos_emb(k, rotery_cos, rotery_sin)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = defactorize_head(o)
    o = o.transpose(1, -2).flatten(-2)
    o = F.linear(o, ow, ob)
    return o


# @torch.compile(dynamic=False, mode="max-autotune")
@torch.jit.script
def fused_inter_rotary_attention(
    head_size: int,
    x: Tensor,
    qw: Tensor,
    qb: Tensor,
    kw: Tensor,
    kb: Tensor,
    vw: Tensor,
    vb: Tensor,
    ow: Tensor,
    ob: Tensor,
    rotery_cos: Tensor,
    rotery_sin: Tensor,
) -> Tensor:
    q = F.linear(x, qw, qb)
    k = F.linear(x, kw, kb)
    v = F.linear(x, vw, vb)
    q = q.view(q.shape[0], q.shape[1], q.shape[2], -1, head_size).transpose(2, -2)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], -1, head_size).transpose(2, -2)
    v = v.view(v.shape[0], v.shape[1], v.shape[2], -1, head_size).transpose(2, -2)
    q, k, v = factorize_head(q, k, v)
    q = apply_rotary_pos_emb(q, rotery_cos, rotery_sin)
    k = apply_rotary_pos_emb(k, rotery_cos, rotery_sin)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    o = defactorize_head(o)
    o = o.transpose(2, -2).flatten(-2)
    o = F.linear(o, ow, ob)
    return o


# @torch.compile(dynamic=False, mode="max-autotune")
@torch.jit.script
def fused_inner_rotary_attention(
    head_size: int,
    x: Tensor,
    qw: Tensor,
    qb: Tensor,
    kw: Tensor,
    kb: Tensor,
    vw: Tensor,
    vb: Tensor,
    ow: Tensor,
    ob: Tensor,
    rotery_cos: Tensor,
    rotery_sin: Tensor,
) -> Tensor:
    q = F.linear(x, qw, qb)
    k = F.linear(x, kw, kb)
    v = F.linear(x, vw, vb)

    q = q.view(q.shape[0], q.shape[1], q.shape[2], -1, head_size)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], -1, head_size)
    v = v.view(v.shape[0], v.shape[1], v.shape[2], -1, head_size)
    q, k, v = factorize_head(q, k, v)
    q = apply_rotary_pos_emb(q, rotery_cos, rotery_sin)
    k = apply_rotary_pos_emb(k, rotery_cos, rotery_sin)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    o = defactorize_head(o)
    o = o.flatten(-2)
    o = F.linear(o, ow, ob)
    return o


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
            hidden_size, num_heads * head_size, dtype=torch.bfloat16
        )
        self.k_proj = nn.Linear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16
        )
        self.v_proj = nn.Linear(
            hidden_size, num_heads * head_size, dtype=torch.bfloat16
        )
        self.out_proj = nn.Linear(
            num_heads * head_size, hidden_size, dtype=torch.bfloat16
        )
        self.rotary = RotaryEmbedding(head_size)

    def _reshape_qkv(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.view(
            hidden_states.shape[0],  # batch
            hidden_states.shape[1],  # seq
            hidden_states.shape[2],  # path
            -1,
            self.head_size,
        )

    def _pre_attention_permute_helper(self, hidden_states: Tensor) -> Tensor:
        if self.orient == "outer":
            return hidden_states.transpose(1, -2)
        else:
            return hidden_states.transpose(2, -2)

    def _pre_attention_permute(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self._pre_attention_permute_helper(q),
            self._pre_attention_permute_helper(k),
            self._pre_attention_permute_helper(v),
        )

    def _post_attention_permute(self, hidden_states: Tensor) -> Tensor:
        if self.orient == "outer":
            return hidden_states.transpose(1, -2).flatten(-2)
        else:
            return hidden_states.transpose(2, -2).flatten(-2)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]] | Tensor:
        hidden_states = self.dropout(hidden_states)
        if self.training:
            if self.orient == "outer":
                return fused_outer_rotary_attention(
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
                ), (None, None)
            elif self.orient == "inter":
                return fused_inter_rotary_attention(
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
                ), (None, None)
            else:
                return fused_inner_rotary_attention(
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
                ), (None, None)

        q: Tensor = self.q_proj(hidden_states)
        k: Tensor = self.k_proj(hidden_states)
        v: Tensor = self.v_proj(hidden_states)

        if key_value_states is not None:
            k_cache, v_cache = key_value_states
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        assert q is not None and k is not None and v is not None
        qh = self._reshape_qkv(q)
        kh = self._reshape_qkv(k)
        vh = self._reshape_qkv(v)

        qh, kh = self.rotary(qh, kh)

        qh, kh, vh = self._pre_attention_permute(qh, kh, vh)

        attn_scores = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            is_causal=key_value_states is None and self.orient == "outer",
        )

        attn_scores = self._post_attention_permute(attn_scores)

        out = self.out_proj(attn_scores)

        return out, (k.detach(), v.detach())


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
        self.norm = MSNorm(hidden_size)
        self.attention = Attention(hidden_size, num_heads, head_size, "inter", dropout)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        residual = hidden_states
        hidden_states = self.norm(residual)
        hidden_states, key_value_states = self.attention(
            hidden_states,
            None if key_value_states is None else key_value_states,
        )
        residual = residual + hidden_states
        return residual, key_value_states


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
        key_value_states: Optional[
            Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        ] = None,
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
        inter_key_value_states, outer_key_value_states = None, None
        residual, outer_key_value_states = self.outer(
            hidden_states, None if key_value_states is None else key_value_states[0]
        )
        residual, inter_key_value_states = self.inter(
            residual, None if key_value_states is None else key_value_states[1]
        )
        residual, _ = self.inner(residual, None)
        key_value_states = (inter_key_value_states, outer_key_value_states)
        return residual, key_value_states


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
        key_value_states: Optional[
            List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
        ] = None,
    ) -> Tuple[Tensor, List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]]:
        new_key_value_states: List[
            Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        ] = []
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
