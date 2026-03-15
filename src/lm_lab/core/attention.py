from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class KVCache:
    k: torch.Tensor  # (B, H, T_cache, D)
    v: torch.Tensor  # (B, H, T_cache, D)

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

    def forward_kv(
        self,
        x: torch.Tensor,              # (B, T, C)
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """
        If past_kv is provided, expected x has T=1 for fast incremental decode.
        Returns (out, present_kv) if use_cache else (out, None).
        """
        B, T, C = x.shape

        # Contract: KV-cache path only supports incremental decoding when cache is provided.
        # If you want to support chunked decoding later, you must apply a causal mask over (past+new).
        if past_kv is not None:
            assert T == 1, "forward_kv with past_kv requires T==1 (incremental decode)."

        H = self.n_heads
        D = self.d_head

        qkv = self.W_qkv(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, H, D)
        q, k, v = qkv.unbind(dim=2)  # (B, T, H, D)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_kv is not None:
            # Incremental decode path (recommended T=1)
            k_cat = torch.cat([past_kv.k, k], dim=2)  # (B,H,T_cache+T,D)
            v_cat = torch.cat([past_kv.v, v], dim=2)
            k_used, v_used = k_cat, v_cat
        else:
            k_used, v_used = k, v

        # Attention scores: (B, H, T, T_k)
        att = (q @ k_used.transpose(-2, -1)) / math.sqrt(D)

        if past_kv is None:
            # Only need causal mask when doing full-seq attention
            mask = self._get_causal_mask(T, x.device)  # (1,1,T,T)
            att = att.masked_fill(~mask, float("-inf"))
        # else: cached decode has no "future" tokens, so no mask needed

        weights = F.softmax(att, dim=-1)
        out = weights @ v_used  # (B,H,T,D)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        out = self.drop(out)

        present = None
        if use_cache:
            present = KVCache(k=k_used, v=v_used)

        return out, present

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_kv(x, past_kv=None, use_cache=False)
        return out