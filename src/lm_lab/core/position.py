from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PositionEmbeddingConfig:
    """
    Configuration for positional embeddings.

    Attributes:
        max_seq_len: Maximum supported sequence length.
        d_model: Embedding dimension (residual stream width).
        mode: Positional embedding mode.
    """

    max_seq_len: int
    d_model: int
    mode: Literal["learned", "sinusoidal"] = "learned"


class PositionEmbedding(nn.Module):
    """
    Positional embedding layer with selectable mode.

    Supported modes:
        - learned: trainable position embedding table
        - sinusoidal: fixed sinusoidal encoding

    Shape contract:
        input:  (B, T)       token IDs used only for shape/device reference
        output: (B, T, C)    positional embedding vectors

    Notes:
        - Positional embeddings are added to token embeddings in the model.
        - ``pos_offset`` supports cache-aware decoding by shifting the position range.
    """

    def __init__(self, cfg: PositionEmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.mode == "learned":
            self.embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("_pe", torch.empty(0), persistent=False)

        elif cfg.mode == "sinusoidal":
            self.embedding = None
            pe = self._build_sinusoidal(cfg.max_seq_len, cfg.d_model)
            # Fixed sinusoidal table is stored as a buffer so it moves with the module
            # while remaining outside the trainable parameter set.
            self.register_buffer("_pe", pe, persistent=True)

        else:
            raise ValueError(f"Unknown PositionEmbedding mode: {cfg.mode}")

    @staticmethod
    def _build_sinusoidal(max_seq_len: int, d_model: int) -> torch.Tensor:
        """
        Build a sinusoidal positional encoding table.

        Args:
            max_seq_len: Maximum number of supported positions.
            d_model: Embedding dimension.

        Returns:
            Positional encoding table of shape (max_seq_len, d_model).
        """
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model / 2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe  # (L, d)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """
        Return positional embeddings for the provided sequence shape.

        Args:
            x: Token ID tensor of shape (B, T). Token values are not used directly;
               only the batch size, sequence length, and device are used.
            pos_offset: Starting position offset, used during cache-aware decoding.

        Returns:
            Positional embeddings of shape (B, T, C).

        Raises:
            ValueError: If sequence length or position range exceeds max_seq_len,
                or if pos_offset is negative.
        """
        B, T = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")

        if pos_offset < 0:
            raise ValueError("pos_offset must be >= 0")

        if self.cfg.mode == "learned":
            # Learned mode materializes explicit position IDs before embedding lookup.
            positions = (torch.arange(T, device=x.device) + pos_offset).unsqueeze(0).expand(B, T)
            if positions.max().item() >= self.cfg.max_seq_len:
                raise ValueError("Position index exceeds max_seq_len (increase max_seq_len).")
            return self.embedding(positions)

        end = pos_offset + T
        if end > self.cfg.max_seq_len:
            raise ValueError("Position index exceeds max_seq_len (increase max_seq_len).")

        # Sinusoidal mode slices the precomputed table for the requested position range.
        pe = self._pe[pos_offset:end].to(device=x.device)
        return pe.unsqueeze(0).expand(B, -1, -1)