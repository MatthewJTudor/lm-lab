from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    """
    Per-layer key/value cache for incremental autoregressive decoding.

    Attributes:
        k: Cached keys of shape (B, H, T_cache, D).
        v: Cached values of shape (B, H, T_cache, D).
    """

    k: torch.Tensor  # (B, H, T_cache, D)
    v: torch.Tensor  # (B, H, T_cache, D)


@dataclass(frozen=True)
class AttentionConfig:
    """
    Configuration for causal self-attention.

    Attributes:
        d_model: Residual stream width.
        n_heads: Number of attention heads.
        attn_bias: Whether the QKV and output projections use bias terms.
        dropout: Dropout probability applied after the output projection.
    """

    d_model: int
    n_heads: int = 1
    attn_bias: bool = False
    dropout: float = 0.0


class SelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Notes:
        - ``n_heads=1`` reduces to single-head attention.
        - Input and output shapes are both (B, T, C).
        - The cache path is an optimization for incremental decode and must not
          change the semantic result for equivalent context.
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

        # Standard GPT-style projection layout: one linear layer produces Q, K, and V.
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=cfg.attn_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=cfg.attn_bias)

        # Cache the causal mask as a non-persistent buffer so it follows device moves
        # without becoming part of the saved model state.
        self.register_buffer("_causal_mask", torch.empty(0), persistent=False)

        self.drop = nn.Dropout(cfg.dropout)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Return a cached causal mask of shape (1, 1, T, T).

        A value of True means the query position is allowed to attend to the
        corresponding key position.
        """
        if self._causal_mask.numel() == 0 or self._causal_mask.size(-1) < T or self._causal_mask.device != device:
            mask = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
            self._causal_mask = mask  # (T, T)
        return self._causal_mask[:T, :T].view(1, 1, T, T)

    def forward_kv(
        self,
        x: torch.Tensor,  # (B, T, C)
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """
        Forward pass with optional KV-cache support.

        Behavior:
            - If ``past_kv`` is None, this is a standard full-sequence causal pass.
            - If ``past_kv`` is provided, this is an incremental decode pass and
              the current implementation expects ``T == 1``.

        Args:
            x: Input tensor of shape (B, T, C).
            past_kv: Previously cached keys and values for this layer.
            use_cache: Whether to return the updated cache.

        Returns:
            A tuple of:
                - attention output of shape (B, T, C)
                - updated KV cache, or None if caching is disabled
        """
        B, T, C = x.shape

        # Current cache contract only supports incremental decode once history exists.
        # Chunked cached decoding would require masking over (past + new) positions.
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
            # Append the new token state to cached history during incremental decode.
            k_cat = torch.cat([past_kv.k, k], dim=2)  # (B, H, T_cache + T, D)
            v_cat = torch.cat([past_kv.v, v], dim=2)
            k_used, v_used = k_cat, v_cat
        else:
            k_used, v_used = k, v

        # Attention scores: (B, H, T, T_k)
        att = (q @ k_used.transpose(-2, -1)) / math.sqrt(D)

        if past_kv is None:
            # Full-sequence path requires a causal mask to block future positions.
            mask = self._get_causal_mask(T, x.device)  # (1, 1, T, T)
            att = att.masked_fill(~mask, float("-inf"))
        # Cached incremental decode has no future positions relative to the current token.

        weights = F.softmax(att, dim=-1)
        out = weights @ v_used  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        out = self.drop(out)

        present = None
        if use_cache:
            present = KVCache(k=k_used, v=v_used)

        return out, present

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass without cache usage.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Attention output of shape (B, T, C).
        """
        out, _ = self.forward_kv(x, past_kv=None, use_cache=False)
        return out