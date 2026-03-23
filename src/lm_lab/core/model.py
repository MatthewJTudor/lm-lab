from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from lm_lab.capture.events import CaptureContext
from lm_lab.core.embedding import TokenEmbedding, EmbeddingConfig
from lm_lab.core.position import PositionEmbedding, PositionEmbeddingConfig
from lm_lab.core.block import TransformerBlock, TransformerBlockConfig
from lm_lab.core.attention import KVCache
from lm_lab.hooks.manager import HookManager


@dataclass(frozen=True)
class TransformerLMConfig:
    """
    Configuration for the TransformerLM model.

    Attributes:
        vocab_size: Vocabulary size used by the token embedding and output head.
        max_seq_len: Maximum supported sequence length.
        d_model: Residual stream width.
        n_layers: Number of transformer blocks.
        pos_mode: Positional embedding mode.

        n_heads: Number of attention heads per block.
        attn_bias: Whether attention projection layers use bias terms.
        attn_impl: Reserved slot for attention implementation selection.

        mlp_hidden_mult: Expansion factor for each block MLP hidden dimension.
        activation: Nonlinearity used inside each block MLP.

        norm_mode: Reserved slot for normalization/residual style selection.
        layer_norm_eps: Epsilon value for layer normalization.

        dropout: Dropout probability applied in model sublayers.
        tie_embeddings: Whether to tie token embedding weights to the output head.
    """

    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    pos_mode: Literal["learned", "sinusoidal"] = "learned"

    # --- attention ---
    n_heads: int = 1
    attn_bias: bool = False
    attn_impl: Literal["naive"] = "naive"  # future-proof slot

    # --- MLP ---
    mlp_hidden_mult: int = 4
    activation: Literal["gelu", "relu"] = "gelu"

    # --- normalization / residual style ---
    norm_mode: Literal["pre", "post"] = "pre"
    layer_norm_eps: float = 1e-5

    # --- regularization ---
    dropout: float = 0.0

    # --- embeddings ---
    tie_embeddings: bool = True


class TransformerLM(nn.Module):
    """
    Minimal GPT-style autoregressive language model.

    High-level flow:
        idx -> token embedding + positional embedding
            -> transformer blocks
            -> final layer norm
            -> vocabulary projection

    Shape contract:
        input:  (B, T)
        output: (B, T, V)

    Notes:
        - The cache path is an optimization for generation, not a semantic change.
        - Hooks are observational only and must not alter forward behavior.
        - Sequence length must satisfy T <= max_seq_len.
    """

    def __init__(
        self,
        cfg: TransformerLMConfig,
        hook_manager: HookManager | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hook_manager = hook_manager

        self.token_emb = TokenEmbedding(
            EmbeddingConfig(vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model)
        )

        self.pos_emb = PositionEmbedding(
            PositionEmbeddingConfig(
                max_seq_len=self.cfg.max_seq_len,
                d_model=self.cfg.d_model,
                mode=cfg.pos_mode,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_model=cfg.d_model,
                        n_heads=cfg.n_heads,
                        attn_bias=cfg.attn_bias,
                        mlp_hidden_mult=cfg.mlp_hidden_mult,
                        activation=cfg.activation,
                        layer_norm_eps=cfg.layer_norm_eps,
                        dropout=cfg.dropout,
                    ),
                    block_idx=i,
                    hook_manager=self.hook_manager,
                )
                for i in range(cfg.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.token_emb.embedding.weight

        self.drop = nn.Dropout(cfg.dropout)

    def _crop_past_kvs(
        self,
        past_kvs: list[KVCache | None],
        keep: int,
    ) -> list[KVCache | None]:
        """
        Crop cached keys and values to keep only the most recent positions.

        This preserves the model's max_seq_len contract during incremental
        decoding when cached context would otherwise grow too large.

        Args:
            past_kvs: Per-layer KV caches.
            keep: Number of cached positions to retain.

        Returns:
            A per-layer KV cache list cropped to the requested history length.
        """
        if keep <= 0:
            return [None] * len(past_kvs)

        out: list[KVCache | None] = []
        for kv in past_kvs:
            if kv is None:
                out.append(None)
                continue
            if kv.k.size(2) > keep:
                out.append(
                    KVCache(
                        k=kv.k[:, :, -keep:, :].contiguous(),
                        v=kv.v[:, :, -keep:, :].contiguous(),
                    )
                )
            else:
                out.append(kv)
        return out

    def forward_kv(
        self,
        idx: torch.Tensor,
        past_kvs: list[KVCache | None] | None = None,
        use_cache: bool = True,
        context: CaptureContext | None = None,
    ) -> tuple[torch.Tensor, list[KVCache | None]]:
        """
        Cache-aware forward pass for autoregressive generation.

        Behavior:
            - If past_kvs is None, the call is treated as a prompt/warmup pass.
            - If past_kvs is provided, the call is treated as an incremental
              decode pass, typically with T=1.

        Args:
            idx: Token IDs of shape (B, T).
            past_kvs: Optional per-layer cache from a previous forward_kv call.
            use_cache: Whether to return updated KV caches.
            context: Optional capture metadata for observational hooks.

        Returns:
            A tuple of:
                - logits of shape (B, T, V)
                - updated per-layer KV caches
        """
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length.")

        if past_kvs is not None:
            keep = self.cfg.max_seq_len - T
            if keep < 0:
                raise ValueError("T exceeds max_seq_len in forward_kv incremental path.")
            # Crop cached history so cached + current tokens remain within the
            # configured context window.
            past_kvs = self._crop_past_kvs(past_kvs, keep=keep)

        tok = self.token_emb(idx)

        pos_offset = 0
        if past_kvs is not None and past_kvs[0] is not None:
            pos_offset = past_kvs[0].k.size(2)

        pos = self.pos_emb(idx, pos_offset=pos_offset)
        x = self.drop(tok + pos)

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        if len(past_kvs) != len(self.blocks):
            raise ValueError("past_kvs must match number of layers")

        new_kvs: list[KVCache | None] = []
        for i, block in enumerate(self.blocks):
            x, present = block.forward_kv(
                x,
                past_kv=past_kvs[i],
                use_cache=use_cache,
                context=context,
            )
            new_kvs.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kvs

    def forward(
        self,
        idx: torch.Tensor,
        context: CaptureContext | None = None,
    ) -> torch.Tensor:
        """
        Standard forward pass without cache usage.

        Args:
            idx: Token IDs of shape (B, T).
            context: Optional capture metadata for observational hooks.

        Returns:
            Logits of shape (B, T, vocab_size).
        """
        B, T = idx.shape

        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length.")

        tok = self.token_emb(idx)
        pos = self.pos_emb(idx)

        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x, context=context)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits