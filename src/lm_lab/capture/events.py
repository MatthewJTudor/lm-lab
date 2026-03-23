from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CaptureContext:
    """
    Semantic capture context supplied at the tap site.

    This is the lightweight context passed from model code into the hook/capture
    layer before final event metadata is constructed.

    Attributes:
        run_id: Optional run identifier.
        phase: Execution phase, such as ``train``, ``eval``, or ``generate``.
        global_step: Training/evaluation step index when applicable.
        decode_step: Autoregressive generation step index when applicable.
        seed: Seed associated with the current execution path.
        layer: Structural layer identifier for the emitting site.
        tap_name: Stable tap name at the emitting site.
        sample_id: Optional sample identifier.
        prompt_id: Optional prompt identifier.
        regime_label: Experimental regime label.
        knob_name: Optional name of the varied knob.
        knob_value: Optional value of the varied knob.
    """

    run_id: str | None
    phase: str
    global_step: int | None
    decode_step: int | None
    seed: int
    layer: str
    tap_name: str
    sample_id: Optional[str] = None
    prompt_id: Optional[str] = None
    regime_label: str = "baseline"
    knob_name: Optional[str] = None
    knob_value: Optional[float] = None


@dataclass(frozen=True)
class CaptureMetadata:
    """
    Finalized metadata attached to a captured tensor event.

    This record extends the semantic capture context with implementation-side
    details such as dtype, device, timestamp, and reserved event identity.

    Attributes:
        run_id: Optional run identifier.
        phase: Execution phase, such as ``train``, ``eval``, or ``generate``.
        global_step: Training/evaluation step index when applicable.
        decode_step: Autoregressive generation step index when applicable.
        seed: Seed associated with the current execution path.
        layer: Structural layer identifier for the emitting site.
        tap_name: Stable tap name at the emitting site.
        event_id: Optional unique event identifier.
        dtype: Tensor dtype string.
        device: Tensor device string.
        timestamp_s: Informational wall-clock timestamp in seconds.
        sample_id: Optional sample identifier.
        prompt_id: Optional prompt identifier.
        regime_label: Experimental regime label.
        knob_name: Optional name of the varied knob.
        knob_value: Optional value of the varied knob.
    """

    run_id: str | None
    phase: str
    global_step: int | None
    decode_step: int | None
    seed: int
    layer: str
    tap_name: str
    event_id: str | None = None
    dtype: str = ""
    device: str = ""
    timestamp_s: float = 0.0
    sample_id: Optional[str] = None
    prompt_id: Optional[str] = None
    regime_label: str = "baseline"
    knob_name: Optional[str] = None
    knob_value: Optional[float] = None


@dataclass(frozen=True)
class CaptureEvent:
    """
    Captured tensor event emitted by the hook manager.

    Attributes:
        name: Fully qualified tap name.
        tensor: Captured tensor payload.
        shape: Materialized tensor shape for convenience and serialization support.
        metadata: Finalized capture metadata for the event.
    """

    name: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    metadata: CaptureMetadata