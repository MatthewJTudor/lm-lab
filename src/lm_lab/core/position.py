from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass(frozen=True)
class PositionEmbeddingConfig:
    max_seq_len: int
    d_model: int

class PositionEmbedding(nn.Module):
    """
    Learned positional embedding

    Input:
        x shape: (batch_size, seq_len)

    Output:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, cfg: PositionEmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=cfg.max_seq_len,
            embedding_dim=cfg.d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is only used for shape inference
        """
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        return self.embedding(positions)