from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from lm_lab.core.attention import SelfAttention, AttentionConfig

@dataclass(frozen=True)
class TransformerBlockConfig:
    d_model: int

class TransformerBlock(nn.Module):
    """
    Minimal transformer block:
        x -> x + Attention(LN(x))
    """

    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(AttentionConfig(d_model=cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.ln(x))