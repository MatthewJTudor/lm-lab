from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import torch
import torch.nn as nn

@dataclass(frozen=True)
class PositionEmbeddingConfig:
    max_seq_len: int
    d_model: int
    mode: Literal["learned", "sinusoidal"] = "learned"

class PositionEmbedding(nn.Module):
    """
    Positional embedding with selectable mode:
      - learned: trainable embedding table
      - sinusoidal: fixed sinusoidal encoding (Vaswani et al., 2017)

    Input:
        x shape: (batch_size, seq_len)

    Output:
        (batch_size, seq_len, d_model)
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
            # buffer: moves with device, not trainable
            self.register_buffer("_pe", pe, persistent=True)

        else:
            raise ValueError(f"Unknown PositionEmbedding mode: {cfg.mode}")

    @staticmethod
    def _build_sinusoidal(max_seq_len: int, d_model: int) -> torch.Tensor:
        """
        Returns (max_seq_len, d_model) sinusoidal positional encodings
        """
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1) # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        ) # (d_model/2, d)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe # (L, d)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        B, T = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")

        if pos_offset < 0:
            raise ValueError("pos_offset must be >= 0")

        if self.cfg.mode == "learned":
            positions = (torch.arange(T, device=x.device) + pos_offset).unsqueeze(0).expand(B, T)
            if positions.max().item() >= self.cfg.max_seq_len:
                raise ValueError("Position index exceeds max_seq_len (increase max_seq_len).")
            return self.embedding(positions)

        end = pos_offset + T
        if end > self.cfg.max_seq_len:
            raise ValueError("Position index exceeds max_seq_len (increase max_seq_len).")
        pe = self._pe[pos_offset:end].to(device=x.device)
        return pe.unsqueeze(0).expand(B, -1, -1)