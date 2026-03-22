from __future__ import annotations

from collections import defaultdict
from typing import Callable
import time

import torch

from lm_lab.capture.events import CaptureEvent, CaptureMetadata


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

    def emit(self, name: str, tensor: torch.Tensor, metadata: CaptureMetadata) -> None:
        if not self.enabled:
            return

        callbacks = self._callbacks.get(name)
        if not callbacks:
            return

        captured = tensor.detach()
        if self.clone_tensors:
            captured = captured.clone()

        event_meta = CaptureMetadata(
            run_id=metadata.run_id,
            phase=metadata.phase,
            step=metadata.step,
            seed=metadata.seed,
            layer=metadata.layer,
            tap_name=metadata.tap_name,
            dtype=str(captured.dtype),
            device=str(captured.device),
            timestamp_s=time.time(),
            sample_id=metadata.sample_id,
            prompt_id=metadata.prompt_id,
            regime_label=metadata.regime_label,
            knob_name=metadata.knob_name,
            knob_value=metadata.knob_value,
        )

        event = CaptureEvent(
            name=name,
            tensor=captured,
            shape=tuple(captured.shape),
            metadata=event_meta,
        )

        for fn in callbacks:
            fn(event)