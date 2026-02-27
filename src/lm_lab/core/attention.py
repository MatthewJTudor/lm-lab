from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class AttentionConfig:
    d_model: int
    n_heads: int = 1
    attn_bias: bool = False
    dropout: float = 0.0


class SelfAttention(nn.Module):
    """
    Multi-head causal self-attention (n_heads=1 reduces to single-head).
    """

    def __init__(self, cfg: AttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d_model = cfg.d_model
        n_heads = cfg.n_heads

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One projection for QKV is the standard clean approach
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=cfg.attn_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=cfg.attn_bias)

        # Causal mask cache (registered buffer so it moves with the module)
        self.register_buffer("_causal_mask", torch.empty(0), persistent=False)

        self.drop = nn.Dropout(cfg.dropout)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns a (1, 1, T, T) boolean mask where True means "allowed".
        """
        if self._causal_mask.numel() == 0 or self._causal_mask.size(-1) < T or self._causal_mask.device != device:
            mask = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
            self._causal_mask = mask  # (T, T)
        return self._causal_mask[:T, :T].view(1, 1, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        B, T, C = x.shape
        H = self.n_heads
        D = self.d_head

        # Project once, then reshape to heads
        qkv = self.W_qkv(x) # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, H, D)
        q, k, v = qkv.unbind(dim=2) # each (B, T, H, D)

        # Move heads forward: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention: (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        # Causal mask
        mask = self._get_causal_mask(T, x.device) # (1, 1, T, T)
        att = att.masked_fill(~mask, float('-inf'))

        weights = F.softmax(att, dim=-1) # (B, H, T, T)

        out = weights @ v # (B, H, T, D)

        # Back to (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.W_o(out)
        out = self.drop(out)
        return out