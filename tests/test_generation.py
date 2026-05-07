"""Tests for generation module (Chapter 30)."""

import torch
import pytest
from inside_black_box.generation import append_kv_cache, cache_aware_attention_step


class TestAppendKvCache:
    def test_no_cache(self):
        """Test that None cache returns new tensors."""
        k_new = torch.randn(2, 4, 1, 64)
        v_new = torch.randn(2, 4, 1, 64)

        k_out, v_out = append_kv_cache(k_new, v_new, None)

        assert torch.equal(k_out, k_new)
        assert torch.equal(v_out, v_new)

    def test_with_cache(self):
        """Test that cache concatenation works."""
        batch, heads, head_dim = 2, 4, 64
        cache_len = 10

        k_past = torch.randn(batch, heads, cache_len, head_dim)
        v_past = torch.randn(batch, heads, cache_len, head_dim)
        k_new = torch.randn(batch, heads, 1, head_dim)
        v_new = torch.randn(batch, heads, 1, head_dim)

        k_out, v_out = append_kv_cache(k_new, v_new, (k_past, v_past))

        assert k_out.shape == (batch, heads, cache_len + 1, head_dim)
        assert v_out.shape == (batch, heads, cache_len + 1, head_dim)
        # Check that past is preserved
        assert torch.equal(k_out[:, :, :cache_len], k_past)
        assert torch.equal(v_out[:, :, :cache_len], v_past)
        # Check that new is appended
        assert torch.equal(k_out[:, :, -1:], k_new)
        assert torch.equal(v_out[:, :, -1:], v_new)


class TestCacheAwareAttentionStep:
    def test_single_query(self):
        """Test attention step with single query position."""
        batch, heads, head_dim = 2, 4, 64
        cache_len = 10

        q_new = torch.randn(batch, heads, 1, head_dim)
        k_new = torch.randn(batch, heads, 1, head_dim)
        v_new = torch.randn(batch, heads, 1, head_dim)

        context, (k_cache, v_cache) = cache_aware_attention_step(
            q_new, k_new, v_new, past_kv=None
        )

        assert context.shape == (batch, heads, 1, head_dim)
        assert k_cache.shape == (batch, heads, 1, head_dim)

    def test_with_cache(self):
        """Test attention step with existing cache."""
        batch, heads, head_dim = 2, 4, 64
        cache_len = 10

        k_past = torch.randn(batch, heads, cache_len, head_dim)
        v_past = torch.randn(batch, heads, cache_len, head_dim)
        q_new = torch.randn(batch, heads, 1, head_dim)
        k_new = torch.randn(batch, heads, 1, head_dim)
        v_new = torch.randn(batch, heads, 1, head_dim)

        context, (k_cache, v_cache) = cache_aware_attention_step(
            q_new, k_new, v_new, past_kv=(k_past, v_past)
        )

        assert context.shape == (batch, heads, 1, head_dim)
        assert k_cache.shape == (batch, heads, cache_len + 1, head_dim)

    def test_multi_query_raises(self):
        """Test that multiple query positions raise error."""
        q_new = torch.randn(2, 4, 3, 64)  # 3 query positions
        k_new = torch.randn(2, 4, 3, 64)
        v_new = torch.randn(2, 4, 3, 64)

        with pytest.raises(ValueError):
            cache_aware_attention_step(q_new, k_new, v_new)

    def test_custom_scale(self):
        """Test attention with custom scale factor."""
        q_new = torch.randn(1, 2, 1, 64)
        k_new = torch.randn(1, 2, 1, 64)
        v_new = torch.randn(1, 2, 1, 64)

        # Should not raise with custom scale
        context, _ = cache_aware_attention_step(
            q_new, k_new, v_new, scale=10.0
        )

        assert context.shape == (1, 2, 1, 64)
