from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass(frozen=True)
class EmbeddingConfig:
    vocab_size: int
    d_model : int

class TokenEmbedding(nn.Module):
    """
    Simple token embedding layer.

    Input shape:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)