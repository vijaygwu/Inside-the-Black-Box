"""Tests for attention module (Chapter 26)."""

import torch
import pytest
from inside_black_box.attention import ScaledDotProductAttention, create_causal_mask


class TestScaledDotProductAttention:
    def test_output_shape(self):
        """Test that output shape matches expected dimensions."""
        batch, n_queries, n_keys, d_k, d_v = 2, 10, 15, 64, 64
        attn = ScaledDotProductAttention(d_k)
        Q = torch.randn(batch, n_queries, d_k)
        K = torch.randn(batch, n_keys, d_k)
        V = torch.randn(batch, n_keys, d_v)

        output, weights = attn(Q, K, V)

        assert output.shape == (batch, n_queries, d_v)
        assert weights.shape == (batch, n_queries, n_keys)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along key dimension."""
        batch, seq_len, d_k = 2, 10, 64
        attn = ScaledDotProductAttention(d_k)
        Q = torch.randn(batch, seq_len, d_k)
        K = torch.randn(batch, seq_len, d_k)
        V = torch.randn(batch, seq_len, d_k)

        _, weights = attn(Q, K, V)
        weight_sums = weights.sum(dim=-1)

        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_causal_mask(self):
        """Test that causal mask is lower triangular."""
        seq_len = 10
        mask = create_causal_mask(seq_len)

        assert mask.shape == (seq_len, seq_len)
        # Check lower triangular
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == True
                else:
                    assert mask[i, j] == False

    def test_masked_attention(self):
        """Test that masked positions get zero attention weight."""
        batch, seq_len, d_k = 1, 5, 32
        attn = ScaledDotProductAttention(d_k)
        Q = torch.randn(batch, seq_len, d_k)
        K = torch.randn(batch, seq_len, d_k)
        V = torch.randn(batch, seq_len, d_k)
        mask = create_causal_mask(seq_len)

        _, weights = attn(Q, K, V, mask=mask)

        # Check that upper triangular positions have zero weight
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, i, j] == 0.0


class TestCreateCausalMask:
    def test_shape(self):
        """Test mask shape."""
        mask = create_causal_mask(10)
        assert mask.shape == (10, 10)

    def test_dtype(self):
        """Test mask is boolean."""
        mask = create_causal_mask(10)
        assert mask.dtype == torch.bool

    def test_device(self):
        """Test mask can be created on specified device."""
        if torch.cuda.is_available():
            mask = create_causal_mask(10, device=torch.device('cuda'))
            assert mask.device.type == 'cuda'
