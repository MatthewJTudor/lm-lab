from __future__ import annotations

import torch

from lm_lab.capture.events import CaptureMetadata
from lm_lab.hooks.manager import HookManager


def test_hook_manager_emits_registered_event() -> None:
    manager = HookManager(enabled=True)
    seen = []

    def cb(event) -> None:
        seen.append(event)

    manager.register("blocks.0.post_attn_residual", cb)

    meta = CaptureMetadata(
        run_id=None,
        phase="train",
        step=1,
        seed=42,
        layer="blocks.0",
        tap_name="post_attn_residual",
        dtype="",
        device="",
        timestamp_s=0.0,
    )

    x = torch.randn(2, 3, 4, requires_grad=True)
    manager.emit("blocks.0.post_attn_residual", x, meta)

    assert len(seen) == 1
    event = seen[0]

    assert event.name == "blocks.0.post_attn_residual"
    assert event.metadata.phase == "train"
    assert event.tensor.requires_grad is False
    assert event.shape == (2, 3, 4)
    assert event.metadata.dtype == str(event.tensor.dtype)
    assert event.metadata.device == str(event.tensor.device)
    assert event.metadata.timestamp_s > 0.0