"""Inside the Black Box - Companion code for Book 4.

This package provides implementations of transformer components
from "Inside the Black Box: Transformers and Attention".
"""

from .attention import ScaledDotProductAttention, create_causal_mask
from .multi_head import MultiHeadAttention
from .positional import sinusoidal_encoding, apply_rope
from .transformer import TransformerBlock, GPTModel
from .generation import (
    append_kv_cache,
    cache_aware_attention_step,
    generate_with_kv_cache,
)

__version__ = "1.0.0"
__all__ = [
    "ScaledDotProductAttention",
    "create_causal_mask",
    "MultiHeadAttention",
    "sinusoidal_encoding",
    "apply_rope",
    "TransformerBlock",
    "GPTModel",
    "append_kv_cache",
    "cache_aware_attention_step",
    "generate_with_kv_cache",
]
