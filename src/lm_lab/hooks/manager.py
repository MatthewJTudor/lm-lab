from __future__ import annotations

from collections import defaultdict
from typing import Callable
import time

import torch

from lm_lab.capture.events import CaptureContext, CaptureEvent, CaptureMetadata


HookCallback = Callable[[CaptureEvent], None]


class HookManager:
    """
    Explicit name-addressable hook dispatcher for observational tensor capture.

    Hooks are registered by stable tap name. When a tap is emitted, the manager:
        - checks whether hooks are enabled
        - detaches the tensor from autograd
        - optionally clones the tensor
        - constructs finalized capture metadata
        - dispatches a CaptureEvent to registered callbacks

    Notes:
        - This manager is observational only and must not alter model semantics.
        - Event identity is not yet minted here; ``event_id`` remains reserved.
    """

    def __init__(self, enabled: bool = True, clone_tensors: bool = False) -> None:
        """
        Initialize the hook manager.

        Args:
            enabled: Whether hook emission is active.
            clone_tensors: Whether to clone detached tensors before dispatch.
                This can protect against later in-place mutations at the cost of
                additional memory.
        """
        self.enabled = enabled
        self.clone_tensors = clone_tensors
        self._callbacks: dict[str, list[HookCallback]] = defaultdict(list)

    def register(self, name: str, fn: HookCallback) -> None:
        """
        Register a callback for a named tap.

        Args:
            name: Fully qualified tap name.
            fn: Callback invoked with the resulting CaptureEvent.
        """
        self._callbacks[name].append(fn)

    def clear(self) -> None:
        """
        Remove all registered callbacks.
        """
        self._callbacks.clear()

    def emit(self, name: str, tensor: torch.Tensor, context: CaptureContext) -> None:
        """
        Emit a captured tensor event to all callbacks registered for a tap.

        Args:
            name: Fully qualified tap name.
            tensor: Tensor to capture.
            context: Semantic capture context supplied by the tap site.
        """
        if not self.enabled:
            return

        callbacks = self._callbacks.get(name)
        if not callbacks:
            return

        # Detach before dispatch so observational consumers cannot interfere
        # with autograd or forward semantics.
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