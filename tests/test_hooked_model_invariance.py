from __future__ import annotations

import torch

from lm_lab.capture.events import CaptureMetadata
from lm_lab.core.model import TransformerLM, TransformerLMConfig
from lm_lab.hooks.manager import HookManager
from lm_lab.utils.seed import SeedConfig, seed_everything


def test_hooks_do_not_change_forward_logits() -> None:
    seed_everything(SeedConfig(seed=123))

    cfg = TransformerLMConfig(
        vocab_size=50,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        tie_embeddings=False,
    )

    idx = torch.randint(0, 50, (2, 8))

    model_plain = TransformerLM(cfg)
    state = model_plain.state_dict()

    model_hooked = TransformerLM(cfg, hook_manager=HookManager(enabled=True))
    model_hooked.load_state_dict(state)

    events = []

    def cb(event) -> None:
        events.append(event)

    model_hooked.hook_manager.register("blocks.0.post_attn_residual", cb)

    meta = CaptureMetadata(
        run_id=None,
        phase="train",
        step=0,
        seed=123,
        layer="",
        tap_name="",
        dtype="",
        device="",
        timestamp_s=0.0,
    )

    logits_plain = model_plain(idx)
    logits_hooked = model_hooked(idx, metadata=meta)

    assert torch.allclose(logits_plain, logits_hooked)
    assert len(events) > 0