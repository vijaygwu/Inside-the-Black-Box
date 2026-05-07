"""Transformer Architecture (Chapter 29).

This module implements the full transformer architecture from
"Inside the Black Box: Transformers and Attention".
"""

import torch
import torch.nn as nn
from typing import Optional

from .multi_head import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Single Transformer decoder block with Pre-Norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_ff = d_ff or 4 * d_model
        # Layer norms (Pre-Norm)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # Multi-head attention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, mask=mask)
        x = x + self.dropout(attn_out)
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """GPT-style decoder-only Transformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Output projection (weight tied with embeddings)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)
        # Enforce causality on every call.
        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.head(self.ln_f(x))
