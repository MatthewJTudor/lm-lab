from __future__ import annotations

import torch

from lm_lab.capture.events import CaptureContext
from lm_lab.core.model import TransformerLM, TransformerLMConfig
from lm_lab.hooks.manager import HookManager
from lm_lab.utils.seed import SeedConfig, seed_everything


def _train_context() -> CaptureContext:
    return CaptureContext(
        run_id=None,
        phase="train",
        global_step=0,
        decode_step=None,
        seed=123,
        layer="",
        tap_name="",
        sample_id=None,
        prompt_id=None,
        regime_label="baseline",
        knob_name=None,
        knob_value=None,
    )


def _generate_context(decode_step: int) -> CaptureContext:
    return CaptureContext(
        run_id=None,
        phase="generate",
        global_step=None,
        decode_step=decode_step,
        seed=123,
        layer="",
        tap_name="",
        sample_id=None,
        prompt_id="prompt-0",
        regime_label="baseline",
        knob_name=None,
        knob_value=None,
    )


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

    ctx = _train_context()

    logits_plain = model_plain(idx)
    logits_hooked = model_hooked(idx, context=ctx)

    assert torch.allclose(logits_plain, logits_hooked)
    assert len(events) > 0

    event = events[0]
    assert event.name == "blocks.0.post_attn_residual"
    assert event.metadata.phase == "train"
    assert event.metadata.global_step == 0
    assert event.metadata.decode_step is None
    assert event.metadata.layer == "blocks.0"
    assert event.metadata.tap_name == "post_attn_residual"


def test_hooks_with_no_context_emit_nothing_and_do_not_change_forward() -> None:
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

    logits_plain = model_plain(idx)
    logits_hooked = model_hooked(idx, context=None)

    assert torch.allclose(logits_plain, logits_hooked)
    assert events == []


def test_forward_kv_propagates_generate_decode_step_context() -> None:
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

    model = TransformerLM(cfg, hook_manager=HookManager(enabled=True))
    model.eval()

    events = []

    def cb(event) -> None:
        events.append(event)

    model.hook_manager.register("blocks.0.post_attn_residual", cb)

    # Warmup prompt pass
    idx = torch.randint(0, 50, (1, 5))
    logits, past_kvs = model.forward_kv(
        idx,
        past_kvs=None,
        use_cache=True,
        context=_generate_context(decode_step=0),
    )

    assert logits.shape[:2] == (1, 5)
    assert len(events) > 0

    first_event = events[0]
    assert first_event.name == "blocks.0.post_attn_residual"
    assert first_event.metadata.phase == "generate"
    assert first_event.metadata.global_step is None
    assert first_event.metadata.decode_step == 0

    # Incremental decode step
    events.clear()
    next_idx = torch.randint(0, 50, (1, 1))
    logits, past_kvs = model.forward_kv(
        next_idx,
        past_kvs=past_kvs,
        use_cache=True,
        context=_generate_context(decode_step=1),
    )

    assert logits.shape[:2] == (1, 1)
    assert len(events) > 0

    next_event = events[0]
    assert next_event.metadata.phase == "generate"
    assert next_event.metadata.global_step is None
    assert next_event.metadata.decode_step == 1
    assert next_event.metadata.layer == "blocks.0"
    assert next_event.metadata.tap_name == "post_attn_residual"