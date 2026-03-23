from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for the token embedding layer.

    Attributes:
        vocab_size: Number of discrete tokens in the vocabulary.
        d_model: Embedding dimension (residual stream width).
    """

    vocab_size: int
    d_model: int


class TokenEmbedding(nn.Module):
    """
    Token embedding layer mapping token IDs to dense vectors.

    Shape contract:
        input:  (B, T)       integer token IDs
        output: (B, T, C)    embedding vectors

    Notes:
        - This layer is purely a lookup; no contextual mixing occurs here.
        - The output feeds directly into the residual stream.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs into dense vectors.

        Args:
            x: Token IDs of shape (B, T).

        Returns:
            Embedding tensor of shape (B, T, C).
        """
        return self.embedding(x)