"""Multi-Head Attention (Chapter 27).

This module implements multi-head attention from
"Inside the Black Box: Transformers and Attention".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias in projections
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Per-head dimension
        # Projection matrices (combined for efficiency)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, query_len, d_model)
            key: (batch, key_len, d_model)
            value: (batch, key_len, d_model)
            mask: Optional mask broadcastable to
                  (batch, n_heads, query_len, key_len).
                  Boolean/integer masks use 1/True for visible keys.
                  Floating-point additive masks use 0 for visible keys and -inf or
                  a large negative value for blocked keys.

        Returns:
            output: (batch, query_len, d_model)
            attention_weights: (batch, n_heads, query_len, key_len)
            Fully masked query rows return zero attention and zero output.
        """
        batch_size = query.size(0)
        # Step 1: Linear projections
        Q = self.W_q(query)  # (batch, query_len, d_model)
        K = self.W_k(key)    # (batch, key_len, d_model)
        V = self.W_v(value)  # (batch, key_len, d_model)
        # Step 2: Reshape for multi-head
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Step 3: Scaled dot-product attention
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
            head_has_keys = allow.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~head_has_keys, 0.0)
            query_has_keys = head_has_keys.squeeze(-1).any(dim=1)
        else:
            query_has_keys = None
        # Softmax over keys
        attn_weights = F.softmax(scores, dim=-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~allow, 0.0)
        attn_for_values = self.dropout(attn_weights)
        # Weighted sum of values
        context = torch.matmul(attn_for_values, V)
        # Step 4: Concatenate heads
        # (batch, n_heads, query_len, d_k) -> (batch, query_len, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        # Step 5: Output projection
        output = self.W_o(context)
        if query_has_keys is not None:
            output = output.masked_fill(~query_has_keys.unsqueeze(-1), 0.0)
        return output, attn_weights


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None):
    """Create causal (autoregressive) attention mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
