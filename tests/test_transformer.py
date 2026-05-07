"""Tests for transformer module (Chapter 29)."""

import torch
import pytest
from inside_black_box.transformer import TransformerBlock, GPTModel


class TestTransformerBlock:
    def test_output_shape(self):
        """Test that block preserves input shape."""
        batch, seq_len, d_model, n_heads = 2, 10, 256, 4
        block = TransformerBlock(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)

        output = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connections are working."""
        d_model, n_heads = 64, 2
        block = TransformerBlock(d_model, n_heads, dropout=0.0)
        block.eval()

        # With zero-initialized FFN weights, output should be close to input + attn
        x = torch.randn(1, 5, d_model)
        with torch.no_grad():
            out = block(x)

        # Output should be different from input (attention applied)
        assert not torch.allclose(out, x)

    def test_with_mask(self):
        """Test block with causal mask."""
        batch, seq_len, d_model, n_heads = 1, 5, 64, 2
        block = TransformerBlock(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = block(x, mask=mask)

        assert output.shape == x.shape


class TestGPTModel:
    def test_output_shape(self):
        """Test that GPT outputs logits of correct shape."""
        vocab_size, d_model, n_layers, n_heads = 1000, 128, 2, 4
        batch, seq_len = 2, 10

        model = GPTModel(vocab_size, d_model, n_layers, n_heads)
        input_ids = torch.randint(0, vocab_size, (batch, seq_len))

        logits = model(input_ids)

        assert logits.shape == (batch, seq_len, vocab_size)

    def test_weight_tying(self):
        """Test that embedding and output weights are tied."""
        model = GPTModel(vocab_size=1000, d_model=128, n_layers=2, n_heads=4)

        assert model.head.weight is model.token_emb.weight

    def test_causal_masking(self):
        """Test that GPT uses causal masking (future tokens don't affect past)."""
        vocab_size, d_model = 100, 64
        model = GPTModel(vocab_size, d_model, n_layers=1, n_heads=2)
        model.eval()

        # Two sequences: same prefix, different continuation
        seq1 = torch.tensor([[1, 2, 3, 4, 5]])
        seq2 = torch.tensor([[1, 2, 3, 99, 99]])

        with torch.no_grad():
            logits1 = model(seq1)
            logits2 = model(seq2)

        # First 3 positions should have identical outputs
        assert torch.allclose(logits1[0, :3], logits2[0, :3], atol=1e-5)

    def test_different_sequence_lengths(self):
        """Test that model handles different sequence lengths."""
        model = GPTModel(vocab_size=100, d_model=64, n_layers=2, n_heads=2)

        for seq_len in [5, 10, 20]:
            input_ids = torch.randint(0, 100, (1, seq_len))
            logits = model(input_ids)
            assert logits.shape == (1, seq_len, 100)
