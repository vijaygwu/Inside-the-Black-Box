"""Tests for multi-head attention module (Chapter 27)."""

import torch
import pytest
from inside_black_box.multi_head import MultiHeadAttention, create_causal_mask


class TestMultiHeadAttention:
    def test_output_shape(self):
        """Test that output shape matches input."""
        batch, seq_len, d_model, n_heads = 2, 10, 512, 8
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)

        output, attn = mha(x, x, x)

        assert output.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, n_heads, seq_len, seq_len)

    def test_d_model_divisibility(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=512, n_heads=7)

    def test_self_attention(self):
        """Test self-attention (Q=K=V from same input)."""
        batch, seq_len, d_model, n_heads = 2, 10, 256, 4
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)

        output, attn = mha(x, x, x)

        assert output.shape == x.shape
        # Attention weights should sum to 1 per head per query
        weight_sums = attn.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_cross_attention(self):
        """Test cross-attention (different Q and K/V sources)."""
        batch, q_len, kv_len, d_model, n_heads = 2, 10, 20, 256, 4
        mha = MultiHeadAttention(d_model, n_heads)
        query = torch.randn(batch, q_len, d_model)
        key = torch.randn(batch, kv_len, d_model)
        value = torch.randn(batch, kv_len, d_model)

        output, attn = mha(query, key, value)

        assert output.shape == (batch, q_len, d_model)
        assert attn.shape == (batch, n_heads, q_len, kv_len)

    def test_masked_attention(self):
        """Test that causal mask blocks future positions."""
        batch, seq_len, d_model, n_heads = 1, 5, 64, 2
        mha = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)
        mask = create_causal_mask(seq_len)

        _, attn = mha(x, x, x, mask=mask)

        # Check upper triangular has zero weight
        for h in range(n_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attn[0, h, i, j] == 0.0

    def test_no_dropout_in_eval(self):
        """Test deterministic output in eval mode."""
        batch, seq_len, d_model, n_heads = 2, 10, 128, 4
        mha = MultiHeadAttention(d_model, n_heads, dropout=0.5)
        mha.eval()
        x = torch.randn(batch, seq_len, d_model)

        out1, _ = mha(x, x, x)
        out2, _ = mha(x, x, x)

        assert torch.allclose(out1, out2)
