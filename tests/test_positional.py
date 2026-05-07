"""Tests for positional encoding module (Chapter 28)."""

import torch
import pytest
import math
from inside_black_box.positional import sinusoidal_encoding, apply_rope


class TestSinusoidalEncoding:
    def test_output_shape(self):
        """Test output shape."""
        max_len, d_model = 100, 512
        pe = sinusoidal_encoding(max_len, d_model)
        assert pe.shape == (max_len, d_model)

    def test_odd_dimension_raises(self):
        """Test that odd d_model raises error."""
        with pytest.raises(ValueError):
            sinusoidal_encoding(100, 511)

    def test_bounded_values(self):
        """Test that values are in [-1, 1]."""
        pe = sinusoidal_encoding(100, 256)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_different_frequencies(self):
        """Test that different dimensions have different frequencies."""
        pe = sinusoidal_encoding(100, 8)
        # First two dimensions should have higher frequency than last two
        diff_low = (pe[1:, 0] - pe[:-1, 0]).abs().mean()
        diff_high = (pe[1:, 6] - pe[:-1, 6]).abs().mean()
        assert diff_low > diff_high

    def test_position_zero(self):
        """Test encoding at position 0."""
        pe = sinusoidal_encoding(10, 8)
        # sin(0) = 0, cos(0) = 1
        assert torch.allclose(pe[0, 0::2], torch.zeros(4), atol=1e-6)
        assert torch.allclose(pe[0, 1::2], torch.ones(4), atol=1e-6)


class TestApplyRope:
    def test_output_shape(self):
        """Test that RoPE preserves shape."""
        batch, n_heads, seq_len, head_dim = 2, 8, 10, 64
        x = torch.randn(batch, n_heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        x_rot = apply_rope(x, position_ids)

        assert x_rot.shape == x.shape

    def test_odd_head_dim_raises(self):
        """Test that odd head_dim raises error."""
        x = torch.randn(2, 4, 10, 63)  # Odd head_dim
        position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)

        with pytest.raises(ValueError):
            apply_rope(x, position_ids)

    def test_dtype_preserved(self):
        """Test that output dtype matches input."""
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(2, 4, 10, 64, dtype=dtype)
            position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
            x_rot = apply_rope(x, position_ids)
            assert x_rot.dtype == dtype

    def test_relative_position_property(self):
        """Test that RoPE enables relative position computation.

        For RoPE: q_m^T @ k_n depends only on (m - n) when q and k
        are rotated versions of the same base vector.
        """
        head_dim = 64
        base = torch.randn(1, 1, 1, head_dim)

        # Create two copies at different positions
        pos_m = torch.tensor([[5]])
        pos_n = torch.tensor([[3]])

        q_m = apply_rope(base, pos_m)
        k_n = apply_rope(base, pos_n)

        # And another pair with same relative distance
        pos_m2 = torch.tensor([[10]])
        pos_n2 = torch.tensor([[8]])  # Same diff: 10-8 = 5-3 = 2

        q_m2 = apply_rope(base, pos_m2)
        k_n2 = apply_rope(base, pos_n2)

        # Dot products should be equal (same relative distance)
        dot1 = (q_m * k_n).sum()
        dot2 = (q_m2 * k_n2).sum()

        assert torch.allclose(dot1, dot2, atol=1e-5)
