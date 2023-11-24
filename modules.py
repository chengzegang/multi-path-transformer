import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.utils.checkpoint import checkpoint
import bitsandbytes as bnb


class MPLinear(nn.Module):
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
        x = x.view(*x.shape[:-1], -1, self.in_features)
        x = self.linear(x)
        x = x.flatten(-2)
        return x


class Mixin(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        bias=True,
        dtype=torch.bfloat16,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        self.out_features = out_features

        self.q_proj = nn.Linear(
            in_features, out_features, bias=bias, dtype=dtype, device=device
        )
        self.k_proj = nn.Linear(
            in_features, out_features, bias=bias, dtype=dtype, device=device
        )
        self.v_proj = nn.Linear(
            in_features, out_features, bias=bias, dtype=dtype, device=device
        )
        self.o_proj = nn.Linear(
            out_features, out_features, bias=bias, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(out_features)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=1.0 / math.sqrt(p.shape[1]))
            else:
                nn.init.zeros_(p)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(*x.shape[:-1], -1, self.in_features)

        x = torch.softmax(x + 1e-6, dim=-2) * x.mean(dim=-2, keepdim=True)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k = self.rotary(q, k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = self.o_proj(out)
        out = out.flatten(-2)
        return out


class MSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        out_shape = x.shape
        x = x.reshape(-1, self.dim)
        x = (
            x
            * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )
        x = x.view(out_shape)
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

        self.w1 = MPLinear(
            in_features, hidden_features, bias=bias, dtype=torch.bfloat16
        )
        self.w2 = MPLinear(
            in_features, hidden_features, bias=bias, dtype=torch.bfloat16
        )
        self.w3 = MPLinear(
            hidden_features, in_features, bias=bias, dtype=torch.bfloat16
        )

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


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.q_proj = MPLinear(hidden_size, hidden_size, dtype=torch.bfloat16)
        self.k_proj = MPLinear(hidden_size, hidden_size, dtype=torch.bfloat16)
        self.v_proj = MPLinear(hidden_size, hidden_size, dtype=torch.bfloat16)
        self.out_proj = MPLinear(hidden_size, hidden_size, dtype=torch.bfloat16)
        self.rotary = RotaryEmbedding(head_size)

    def _reshape_qkv(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            -1,
            self.head_size,
        )

    def _reshape_scores(self, scores: Tensor) -> Tensor:
        return scores.contiguous().transpose(1, 2).flatten(-2)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        q, k, v = None, None, None
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
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
            is_causal=key_value_states is None,
        )

        attn_scores = self._reshape_scores(attn_scores)
        out = self.out_proj(attn_scores)

        return out, (k, v)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, head_size: int, mlp_dim: int):
        super().__init__()
        self.pre_attention_norm = MSNorm(hidden_size)
        self.attention = Attention(hidden_size, head_size)

        self.pre_mlp_norm = MSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, mlp_dim)

        self.mixin = Mixin(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        residual = hidden_states
        hidden_states = self.pre_attention_norm(residual)
        hidden_states, key_value_states = self.attention(
            hidden_states, key_value_states, attn_mask
        )

        residual = residual + hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        residual = residual + hidden_states
        hidden_states = self.mixin(hidden_states)

        residual = residual + hidden_states

        return residual, key_value_states  # type: ignore


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 32,
        head_size: int = 128,
        mlp_dim: int = 1366,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size=hidden_size,
                    head_size=head_size,
                    mlp_dim=mlp_dim,
                )
                for i in range(num_layers)
            ]
        )
        self._grad_checkpoint = False

    def enable_gradient_checkpointing(self):
        self._grad_checkpoint = True

    def disable_gradient_checkpointing(self):
        self._grad_checkpoint = False

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[List[Tuple[Tensor, Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        new_key_value_states: List[Tuple[Tensor, Tensor]] = []
        if key_value_states is not None:
            for i, layer in enumerate(self.layers):
                kvs = key_value_states[i]
                if self._grad_checkpoint:
                    hidden_states, new_kvs = checkpoint(
                        layer,
                        hidden_states,
                        kvs,
                        attn_mask,
                        use_reentrant=True,
                    )
                else:
                    hidden_states, new_kvs = layer(
                        hidden_states,
                        kvs,
                        attn_mask,
                    )
                new_key_value_states.append(new_kvs)
            return hidden_states, new_key_value_states
        else:
            for layer in self.layers:
                if self._grad_checkpoint:
                    hidden_states, new_kvs = checkpoint(
                        layer,
                        hidden_states,
                        None,
                        attn_mask,
                        use_reentrant=True,
                    )
                else:
                    hidden_states, new_kvs = layer(hidden_states, None, attn_mask)
                new_key_value_states.append(new_kvs)
            return hidden_states, new_key_value_states
