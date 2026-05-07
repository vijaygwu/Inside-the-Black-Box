"""LLM Generation Utilities (Chapter 30).

This module implements KV caching and autoregressive generation from
"Inside the Black Box: Transformers and Attention".
"""

import torch
import torch.nn.functional as F
from typing import Optional


def append_kv_cache(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one layer's new K/V tensors to its decode-time cache."""
    if past_kv is None:
        return k_new, v_new

    k_past, v_past = past_kv
    k_all = torch.cat([k_past, k_new], dim=-2)
    v_all = torch.cat([v_past, v_new], dim=-2)
    return k_all, v_all


def cache_aware_attention_step(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    scale: Optional[float] = None
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One-token decode step: append K/V, attend current Q over cache."""
    if q_new.size(-2) != 1:
        raise ValueError("q_new must contain exactly one decode position")

    if scale is None:
        scale = q_new.size(-1) ** 0.5

    k_all, v_all = append_kv_cache(k_new, v_new, past_kv)
    scores = torch.matmul(q_new, k_all.transpose(-2, -1)) / scale
    weights = F.softmax(scores, dim=-1)
    context = torch.matmul(weights, v_all)
    return context, (k_all, v_all)


def generate_with_kv_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Autoregressive generation with cache reuse.

    Args:
        model: A transformer model whose forward pass accepts past_key_values,
            use_cache, and position_ids. Call model.eval() before using this
            helper for inference.
        input_ids: Input token IDs of shape (batch_size, seq_len).
            This compact helper assumes unpadded prompt batches. Padded batched
            generation needs pad-aware position IDs, attention masks, and, for
            right padding, a gather of each row's last non-pad logit.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (must be positive).

    Returns:
        Generated token IDs of shape (batch_size, seq_len + max_new_tokens).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    generated = input_ids.clone()
    past_kv = None
    batch_size = generated.size(0)

    for _ in range(max_new_tokens):
        # Once the cache is warm, only the newest token needs fresh projections.
        if past_kv is not None:
            input_slice = generated[:, -1:]
            start_pos = generated.size(1) - 1
        else:
            input_slice = generated
            start_pos = 0
        step_len = input_slice.size(1)
        positions = torch.arange(
            start_pos,
            start_pos + step_len,
            device=generated.device
        )
        position_ids = positions.unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            outputs = model(
                input_slice,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values

        # Sample next token
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    return generated
