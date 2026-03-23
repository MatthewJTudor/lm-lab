from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CaptureContext:
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
    name: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    metadata: CaptureMetadata