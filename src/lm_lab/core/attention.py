from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class AttentionConfig:
    d_model: int

class SelfAttention(nn.Module):
    """
    Single head causal self-attention
    """

    def __init__(self, cfg: AttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        B, T, d = x.shape

        Q = self.W_q(x) # (B, T, d)
        K = self.W_k(x) # (B, T, d)
        V = self.W_v(x) # (B, T, d)

        # Compute scaled dot-product attention scores
        att = Q @ K.transpose(-2, -1) # (B, T, T)
        att = att / math.sqrt(d)

        weights = F.softmax(att, dim=-1) # (B, T, T)

        out = weights @ V # (B, T, d)

        return out