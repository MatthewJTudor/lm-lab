from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

@dataclass
class LMMetricRecord:
    run_id: str | None = None
    phase: str = "train"
    step: int = 0
    seed: int = 0
    tokenizer_mode: str = ""

    regime_label: str = "baseline"
    knob_name: Optional[str] = None
    knob_value: Optional[float] = None
    prompt_id: Optional[str] = None
    sample_id: Optional[str] = None

    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

def format_metric_record(record: LMMetricRecord, metric_order: list[str] | None = None) -> str:
    prefix = (
        f"run_id={record.run_id or 'none'} | "
        f"phase={record.phase} | "
        f"step={record.step} | "
        f"seed={record.seed} | "
        f"tok={record.tokenizer_mode} | "
        f"regime={record.regime_label}"
    )

    items = record.metrics.items()
    if metric_order is not None:
        items = [(k, record.metrics[k]) for k in metric_order if k in record.metrics]

    metric_parts: list[str] = []
    for k, v in items:
        if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-4):
            metric_parts.append(f"{k}={v:.4e}")
        else:
            metric_parts.append(f"{k}={v:.6f}")

    return prefix + " | " + " | ".join(metric_parts)