from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from lm_lab.capture.events import CaptureMetadata
from lm_lab.core.attention import SelfAttention, AttentionConfig
from lm_lab.core.attention import KVCache
from lm_lab.hooks.manager import HookManager


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

    def __init__(
        self,
        cfg: TransformerBlockConfig,
        block_idx: int = 0,
        hook_manager: HookManager | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_idx = block_idx
        self.hook_manager = hook_manager

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

    def _tap(
            self,
            tap_name: str,
            tensor: torch.Tensor,
            metadata: CaptureMetadata | None,
    ) -> None:
        if self.hook_manager is None or metadata is None:
            return

        full_name = f"blocks.{self.block_idx}.{tap_name}"
        tap_meta = CaptureMetadata(
            run_id=metadata.run_id,
            phase=metadata.phase,
            step=metadata.step,
            seed=metadata.seed,
            layer=f"blocks.{self.block_idx}",
            tap_name=tap_name,
            dtype="",  # filled by HookManager.emit
            device="",  # filled by HookManager.emit
            timestamp_s=0.0,  # filled by HookManager.emit
            sample_id=metadata.sample_id,
            prompt_id=metadata.prompt_id,
            regime_label=metadata.regime_label,
            knob_name=metadata.knob_name,
            knob_value=metadata.knob_value,
        )
        self.hook_manager.emit(full_name, tensor, tap_meta)

    def forward_kv(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
        metadata: CaptureMetadata | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present = self.attn.forward_kv(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        self._tap("post_attn_residual", x, metadata)

        x = x + self.drop(self.fc2(self.act(self.fc1(self.ln2(x)))))
        return x, present

    def forward(
        self,
        x: torch.Tensor,
        metadata: CaptureMetadata | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        self._tap("post_attn_residual", x, metadata)

        x = x + self.drop(self.fc2(self.act(self.fc1(self.ln2(x)))))
        return x