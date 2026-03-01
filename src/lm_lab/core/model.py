from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from lm_lab.core.embedding import TokenEmbedding, EmbeddingConfig
from lm_lab.core.position import PositionEmbedding, PositionEmbeddingConfig
from lm_lab.core.block import TransformerBlock, TransformerBlockConfig
from lm_lab.core.attention import KVCache

@dataclass(frozen=True)
class TransformerLMConfig:
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
    Minimalistic GPT-style language model.
    """

    def __init__(self, cfg: TransformerLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

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
                    TransformerBlockConfig(d_model=cfg.d_model,
                                           n_heads=cfg.n_heads,
                                           attn_bias=cfg.attn_bias,
                                           mlp_hidden_mult=cfg.mlp_hidden_mult,
                                           activation=cfg.activation,
                                           )
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.token_emb.embedding.weight

        self.drop = nn.Dropout(cfg.dropout)

    from lm_lab.core.attention import KVCache

    def _crop_past_kvs(
            self,
            past_kvs: list[KVCache | None],
            keep: int,
    ) -> list[KVCache | None]:
        if keep <= 0:
            return [None] * len(past_kvs)

        out: list[KVCache | None] = []
        for kv in past_kvs:
            if kv is None:
                out.append(None)
                continue
            # kv.k: (B,H,T,D)
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
            idx: torch.Tensor,  # (B,T)
            past_kvs: list[KVCache | None] | None = None,
            use_cache: bool = True,
    ) -> tuple[torch.Tensor, list[KVCache | None]]:
        """
        Cache-aware forward for generation.
        If past_kvs is None: full prompt pass; caches are built.
        If past_kvs is provided: recommended idx has T=1.
        Returns (logits, new_kvs).
        """
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length.")

        if past_kvs is not None:
            # We are about to append T new tokens; keep only last (max_seq_len - T)
            keep = self.cfg.max_seq_len - T
            if keep < 0:
                raise ValueError("T exceeds max_seq_len in forward_kv incremental path.")
            past_kvs = self._crop_past_kvs(past_kvs, keep=keep)

        tok = self.token_emb(idx)

        # position offset handled below (see PositionEmbedding patch)
        # For warmup (past_kvs is None): offset=0
        # For incremental (past_kvs not None, recommended T=1): offset = cached_len (or total_len - T)
        pos_offset = 0
        if past_kvs is not None and past_kvs[0] is not None:
            # cached length so far (per layer cache length should match)
            pos_offset = past_kvs[0].k.size(2)

        pos = self.pos_emb(idx, pos_offset=pos_offset)
        x = self.drop(tok + pos)

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        if len(past_kvs) != len(self.blocks):
            raise ValueError("past_kvs must match number of layers")

        new_kvs: list[KVCache | None] = []
        for i, block in enumerate(self.blocks):
            x, present = block.forward_kv(x, past_kv=past_kvs[i], use_cache=use_cache)
            new_kvs.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kvs

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) integer tokens.
        returns: (B, T, vocab_size) logits
        """
        B, T = idx.shape

        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length.")

        tok = self.token_emb(idx)
        pos = self.pos_emb(idx)

        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits

