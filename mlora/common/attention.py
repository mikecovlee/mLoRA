import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .cache import Cache, StaticCache


def prepare_4d_causal_attention_mask(
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
) -> torch.Tensor:
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    using_static_cache = isinstance(past_key_values, StaticCache)

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(
        -1, 1
    )
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)

    return causal_mask


def precompute_rope_angle(
    dim: int, seq_len: int, theta: float = 10000.0, device: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float32
            )
            / dim
        )
    )
    t = torch.arange(seq_len, dtype=torch.int64).to(device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    emb.requires_grad_(False)

    # cos(angle), sin(angle)
    return (emb.cos(), emb.sin())


@torch.jit.script
def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.jit.script
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_len: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:seq_len].to(xq.dtype)
    sin = sin[:seq_len].to(xq.dtype)

    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


def get_unpad_data(attention_mask: torch.Tensor):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


@torch.jit.script
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attention_score = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
        query.size(-1)
    )
    if attention_mask is not None:
        attention_score = attention_score + attention_mask
    attention_score = F.softmax(attention_score, dim=-1, dtype=torch.float32).to(
        value.dtype
    )
    attention_score = torch.matmul(attention_score, value)
    attention_score = attention_score.transpose(1, 2).contiguous()
    return attention_score
