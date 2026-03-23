from __future__ import annotations

import torch

from lm_lab.capture.events import CaptureContext
from lm_lab.hooks.manager import HookManager


def _base_context(**overrides) -> CaptureContext:
    data = dict(
        run_id=None,
        phase="train",
        global_step=1,
        decode_step=None,
        seed=42,
        layer="blocks.0",
        tap_name="post_attn_residual",
        sample_id=None,
        prompt_id=None,
        regime_label="baseline",
        knob_name=None,
        knob_value=None,
    )
    data.update(overrides)
    return CaptureContext(**data)


def test_hook_manager_emits_registered_event() -> None:
    manager = HookManager(enabled=True)
    seen = []

    def cb(event) -> None:
        seen.append(event)

    manager.register("blocks.0.post_attn_residual", cb)

    ctx = _base_context()
    x = torch.randn(2, 3, 4, requires_grad=True)

    manager.emit("blocks.0.post_attn_residual", x, ctx)

    assert len(seen) == 1
    event = seen[0]

    assert event.name == "blocks.0.post_attn_residual"
    assert event.metadata.phase == "train"
    assert event.metadata.global_step == 1
    assert event.metadata.decode_step is None
    assert event.metadata.layer == "blocks.0"
    assert event.metadata.tap_name == "post_attn_residual"

    assert event.tensor.requires_grad is False
    assert event.shape == (2, 3, 4)

    assert event.metadata.dtype == str(event.tensor.dtype)
    assert event.metadata.device == str(event.tensor.device)
    assert event.metadata.timestamp_s > 0.0


def test_hook_manager_disabled_emits_nothing() -> None:
    manager = HookManager(enabled=False)
    seen = []

    def cb(event) -> None:
        seen.append(event)

    manager.register("blocks.0.post_attn_residual", cb)

    x = torch.randn(2, 3, 4, requires_grad=True)
    manager.emit("blocks.0.post_attn_residual", x, _base_context())

    assert seen == []


def test_hook_manager_unregistered_tap_emits_nothing() -> None:
    manager = HookManager(enabled=True)
    x = torch.randn(2, 3, 4, requires_grad=True)

    # No callback registered for this name; should be a quiet no-op.
    manager.emit("blocks.0.post_attn_residual", x, _base_context())


def test_hook_manager_clone_tensors_true_returns_distinct_storage() -> None:
    manager = HookManager(enabled=True, clone_tensors=True)
    seen = []

    def cb(event) -> None:
        seen.append(event)

    manager.register("blocks.0.post_attn_residual", cb)

    x = torch.randn(2, 3, 4, requires_grad=True)
    manager.emit("blocks.0.post_attn_residual", x, _base_context())

    assert len(seen) == 1
    event = seen[0]

    assert event.tensor.requires_grad is False
    assert event.tensor.data_ptr() != x.data_ptr()


def test_hook_manager_multiple_callbacks_fire_in_registration_order() -> None:
    manager = HookManager(enabled=True)
    order: list[str] = []

    def cb1(event) -> None:
        order.append("cb1")

    def cb2(event) -> None:
        order.append("cb2")

    manager.register("blocks.0.post_attn_residual", cb1)
    manager.register("blocks.0.post_attn_residual", cb2)

    x = torch.randn(2, 3, 4)
    manager.emit("blocks.0.post_attn_residual", x, _base_context())

    assert order == ["cb1", "cb2"]

def test_hook_manager_preserves_full_composite_context_surface() -> None:
    manager = HookManager(enabled=True)
    seen = []

    def cb(event) -> None:
        seen.append(event)

    manager.register("blocks.3.post_attn_residual", cb)

    ctx = CaptureContext(
        run_id="run-123",
        phase="generate",
        global_step=None,
        decode_step=7,
        seed=999,
        layer="blocks.3",
        tap_name="post_attn_residual",
        sample_id="sample-42",
        prompt_id="prompt-abc",
        regime_label="baseline",
        knob_name="temperature",
        knob_value=0.7,
    )

    x = torch.randn(2, 5, 8, requires_grad=True)
    manager.emit("blocks.3.post_attn_residual", x, ctx)

    assert len(seen) == 1
    event = seen[0]
    meta = event.metadata

    # Preserved composite context fields
    assert meta.run_id == "run-123"
    assert meta.phase == "generate"
    assert meta.global_step is None
    assert meta.decode_step == 7
    assert meta.seed == 999
    assert meta.layer == "blocks.3"
    assert meta.tap_name == "post_attn_residual"
    assert meta.sample_id == "sample-42"
    assert meta.prompt_id == "prompt-abc"
    assert meta.regime_label == "baseline"
    assert meta.knob_name == "temperature"
    assert meta.knob_value == 0.7

    # Reserved orchestration-owned field: present in schema, not populated yet
    assert meta.event_id is None

    # Manager-enriched runtime fields
    assert meta.dtype == str(event.tensor.dtype)
    assert meta.device == str(event.tensor.device)
    assert meta.timestamp_s > 0.0

    # Event payload integrity
    assert event.name == "blocks.3.post_attn_residual"
    assert event.shape == (2, 5, 8)
    assert event.tensor.requires_grad is False