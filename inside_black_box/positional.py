"""Positional Encodings (Chapter 28).

This module implements sinusoidal and rotary positional encodings from
"Inside the Black Box: Transformers and Attention".
"""

import torch
import math


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Precompute sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension (must be even)

    Returns:
        Positional encoding tensor of shape (max_len, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError(
            f"d_model must be even for sine/cosine pairs, got {d_model}"
        )

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()

    # Compute division term
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    # Apply sin to even indices, cos to odd
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, n_heads, seq_len, head_dim)
        position_ids: Position indices (batch, seq_len)

    Returns:
        Rotated tensor of same shape as input
    """
    batch, n_heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE pairs, got {head_dim}")

    # Compute phases in float32, then cast trig results back to x.dtype.
    compute_dtype = torch.float32
    freqs = 1.0 / (
        10000 ** (
            torch.arange(0, head_dim, 2, device=x.device, dtype=compute_dtype)
            / head_dim
        )
    )

    # Compute angles: (batch, seq_len, head_dim/2)
    angles = (
        position_ids.to(device=x.device, dtype=compute_dtype).unsqueeze(-1)
        * freqs
    )

    # Expand across heads to match the attention tensor layout
    cos = torch.cos(angles).to(dtype=x.dtype).unsqueeze(1)
    sin = torch.sin(angles).to(dtype=x.dtype).unsqueeze(1)

    # Split x into pairs
    x1 = x[..., 0::2]  # Even dimensions
    x2 = x[..., 1::2]  # Odd dimensions

    # Apply rotation
    rot_even = x1 * cos - x2 * sin
    rot_odd = x2 * cos + x1 * sin
    x_rot = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

    return x_rot
