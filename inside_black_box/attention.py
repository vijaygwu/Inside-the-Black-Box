"""Scaled Dot-Product Attention (Chapter 26).

This module implements the fundamental attention mechanism from
"Inside the Black Box: Transformers and Attention".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""

    def __init__(self, d_k: int, dropout: float = 0.0):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,  # (batch, n_queries, d_k)
        K: torch.Tensor,  # (batch, n_keys, d_k)
        V: torch.Tensor,  # (batch, n_keys, d_v)
        mask: Optional[torch.Tensor] = None  # bool/int allow-mask or float additive mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A boolean/integer mask uses 1/True for visible keys.
        A floating-point additive mask uses 0 for visible keys and -inf or a
        large negative value for blocked keys.

        Returns:
            output: (batch, n_queries, d_v)
            attention_weights: (batch, n_queries, n_keys)
            Fully masked query rows return zero attention and zero output.
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            if torch.is_floating_point(mask):
                additive_mask = torch.broadcast_to(
                    mask.to(device=scores.device, dtype=scores.dtype),
                    scores.shape
                )
                scores = scores + additive_mask
                allow = additive_mask == 0
            else:
                allow = torch.broadcast_to(
                    mask.to(device=scores.device, dtype=torch.bool),
                    scores.shape
                )
                masked_value = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~allow, masked_value)
            query_has_keys = allow.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~query_has_keys, 0.0)
        else:
            query_has_keys = None

        # Softmax to get normalized attention weights
        attn_weights = F.softmax(scores, dim=-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~allow, 0.0)
        attn_for_values = self.dropout(attn_weights)

        # Weighted sum of values
        output = torch.matmul(attn_for_values, V)
        if query_has_keys is not None:
            output = output.masked_fill(~query_has_keys, 0.0)

        return output, attn_weights


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create lower triangular mask for causal attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask  # (seq_len, seq_len)
