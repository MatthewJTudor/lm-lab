from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from lm_lab.core.embedding import TokenEmbedding, EmbeddingConfig
from lm_lab.core.position import PositionEmbedding, PositionEmbeddingConfig
from lm_lab.core.block import TransformerBlock, TransformerBlockConfig

@dataclass(frozen=True)
class TransformerLMConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    pos_mode: Literal["learned", "sinusoidal"] = "learned"

    # --- attention ---
    n_heads: int = 1
    attn_bias: bool = False
    attn_impl: Literal["naive"] = "naive"  # future-proof slot

    # --- MLP ---
    mlp_hidden_mult: int = 4
    activation: Literal["gelu", "relu"] = "gelu"

    # --- normalization / residual style ---
    norm_mode: Literal["pre", "post"] = "pre"
    layer_norm_eps: float = 1e-5

    # --- regularization ---
    dropout: float = 0.0

    # --- embeddings ---
    tie_embeddings: bool = True

class TransformerLM(nn.Module):
    """
    Minimalistic GPT-style language model.
    """

    def __init__(self, cfg: TransformerLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = TokenEmbedding(
            EmbeddingConfig(vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model)
        )

        self.pos_emb = PositionEmbedding(
            PositionEmbeddingConfig(
                max_seq_len=self.cfg.max_seq_len,
                d_model=self.cfg.d_model,
                mode=cfg.pos_mode,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(d_model=cfg.d_model,
                                           n_heads=cfg.n_heads,
                                           attn_bias=cfg.attn_bias,
                                           mlp_hidden_mult=cfg.mlp_hidden_mult,
                                           activation=cfg.activation,
                                           )
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.token_emb.embedding.weight

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) integer tokens.
        returns: (B, T, vocab_size) logits
        """
        B, T = idx.shape

        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length.")

        tok = self.token_emb(idx)
        pos = self.pos_emb(idx)

        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits

