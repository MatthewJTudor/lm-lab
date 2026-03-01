from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from lm_lab.core.attention import SelfAttention, AttentionConfig
from lm_lab.core.attention import KVCache

@dataclass(frozen=True)
class TransformerBlockConfig:
    d_model: int
    n_heads: int = 1
    attn_bias: bool = False

    # --- MLP ---
    mlp_hidden_mult: int = 4
    activation: Literal["gelu", "relu"] = "gelu"

    dropout: float = 0.0

class TransformerBlock(nn.Module):
    """
    Minimal transformer block (Pre-LN):
        x -> x + Attn(LN1(x))
        x -> x + MLP(LN2(x))
    """

    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(
            AttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                attn_bias=cfg.attn_bias,
                dropout=cfg.dropout,
            )
        )

        self.ln2 = nn.LayerNorm(cfg.d_model)
        d_ff = cfg.mlp_hidden_mult * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, cfg.d_model)

        if cfg.activation == "gelu":
            self.act = nn.GELU()
        elif cfg.activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {cfg.activation}")

        self.drop = nn.Dropout(cfg.dropout)

    def forward_kv(
            self,
            x: torch.Tensor,
            past_kv: KVCache | None = None,
            use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present = self.attn.forward_kv(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.drop(self.fc2(self.act(self.fc1(self.ln2(x)))))
        return x, present

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.drop(self.fc2(self.act(self.fc1(self.ln2(x)))))
        return x