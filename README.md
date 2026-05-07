# Inside the Black Box - Companion Code

Companion code for **"Inside the Black Box: Transformers and Attention"** (Book 4 of The AI Engineer's Library).

## Installation

```bash
pip install -r requirements.txt
```

## Package Structure

```
inside_black_box/
├── attention.py      # Chapter 26: Scaled Dot-Product Attention
├── multi_head.py     # Chapter 27: Multi-Head Attention
├── positional.py     # Chapter 28: Positional Encodings (sinusoidal, RoPE)
├── transformer.py    # Chapter 29: Transformer Architecture (GPT)
└── generation.py     # Chapter 30: KV Caching and Generation
```

## Quick Start

```python
import torch
from inside_black_box import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    sinusoidal_encoding,
    apply_rope,
    TransformerBlock,
    GPTModel,
)

# Scaled dot-product attention (Chapter 26)
attn = ScaledDotProductAttention(d_k=64)
Q = torch.randn(2, 10, 64)
K = torch.randn(2, 15, 64)
V = torch.randn(2, 15, 64)
output, weights = attn(Q, K, V)

# Multi-head attention (Chapter 27)
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)
output, attn_weights = mha(x, x, x)  # Self-attention

# Sinusoidal positional encoding (Chapter 28)
pe = sinusoidal_encoding(max_len=1000, d_model=512)

# GPT model (Chapter 29)
model = GPTModel(vocab_size=50000, d_model=768, n_layers=12, n_heads=12)
input_ids = torch.randint(0, 50000, (2, 100))
logits = model(input_ids)
```

## Running Tests

```bash
pytest -v
```

## License

This code accompanies the book and is provided for educational purposes.
