from __future__ import annotations

from collections import defaultdict
from typing import Callable
import time

import torch

from lm_lab.capture.events import CaptureContext, CaptureEvent, CaptureMetadata


HookCallback = Callable[[CaptureEvent], None]


class HookManager:
    """
    Explicit name-addressable hook manager.

    Hooks are registered by tap name.
    Emitted tensors are detached before callbacks receive them.
    """

    def __init__(self, enabled: bool = True, clone_tensors: bool = False) -> None:
        self.enabled = enabled
        self.clone_tensors = clone_tensors
        self._callbacks: dict[str, list[HookCallback]] = defaultdict(list)

    def register(self, name: str, fn: HookCallback) -> None:
        self._callbacks[name].append(fn)

    def clear(self) -> None:
        self._callbacks.clear()

    def emit(self, name: str, tensor: torch.Tensor, context: CaptureContext) -> None:
        if not self.enabled:
            return

        callbacks = self._callbacks.get(name)
        if not callbacks:
            return

        captured = tensor.detach()
        if self.clone_tensors:
            captured = captured.clone()

        event_meta = CaptureMetadata(
            run_id=context.run_id,
            phase=context.phase,
            global_step=context.global_step,
            decode_step=context.decode_step,
            seed=context.seed,
            layer=context.layer,
            tap_name=context.tap_name,
            dtype=str(captured.dtype),
            device=str(captured.device),
            timestamp_s=time.time(),
            sample_id=context.sample_id,
            prompt_id=context.prompt_id,
            regime_label=context.regime_label,
            knob_name=context.knob_name,
            knob_value=context.knob_value,
        )

        event = CaptureEvent(
            name=name,
            tensor=captured,
            shape=tuple(captured.shape),
            metadata=event_meta,
        )

        for fn in callbacks:
            fn(event)