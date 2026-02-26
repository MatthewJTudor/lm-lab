from __future__ import annotations

from dataclasses import dataclass

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
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(d_model=cfg.d_model)
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

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

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits

