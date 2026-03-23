from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class LMMetricRecord:
    """
    Structured metric record for train, eval, or generation observations.

    Attributes:
        run_id: Optional run identifier for grouping related records.
        phase: Execution phase, typically ``train``, ``eval``, or ``generate``.
        global_step: Training/evaluation step index when applicable.
        decode_step: Autoregressive generation step index when applicable.
        seed: Seed associated with the run or generation path.
        tokenizer_mode: Tokenizer mode used for the run.

        regime_label: Experimental regime label, such as ``baseline``.
        knob_name: Optional name of the knob being varied.
        knob_value: Optional value of the knob being varied.
        prompt_id: Optional identifier for the input prompt.
        sample_id: Optional identifier for the current sample.

        metrics: Scalar metric payload stored as name -> value pairs.
    """

    run_id: str | None = None
    phase: str = "train"
    global_step: int | None = None
    decode_step: int | None = None
    seed: int = 0
    tokenizer_mode: str = ""

    regime_label: str = "baseline"
    knob_name: Optional[str] = None
    knob_value: Optional[float] = None
    prompt_id: Optional[str] = None
    sample_id: Optional[str] = None

    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert the record into a plain dictionary.

        Returns:
            JSON-serializable dictionary representation of the record.
        """
        return asdict(self)


def format_metric_record(
    record: LMMetricRecord,
    metric_order: list[str] | None = None,
) -> str:
    """
    Format a metric record as a compact human-readable log line.

    Args:
        record: Structured metric record to format.
        metric_order: Optional metric display order. Metrics not present in the
            record are skipped.

    Returns:
        Formatted log string.
    """
    step_part = (
        f"global_step={record.global_step}"
        if record.global_step is not None
        else f"decode_step={record.decode_step}"
    )

    prefix = (
        f"run_id={record.run_id or 'none'} | "
        f"phase={record.phase} | "
        f"{step_part} | "
        f"seed={record.seed} | "
        f"tok={record.tokenizer_mode} | "
        f"regime={record.regime_label}"
    )

    items = record.metrics.items()
    if metric_order is not None:
        items = [(k, record.metrics[k]) for k in metric_order if k in record.metrics]

    metric_parts: list[str] = []
    for k, v in items:
        # Use scientific notation for very large or very small magnitudes to keep
        # log output compact and readable.
        if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-4):
            metric_parts.append(f"{k}={v:.4e}")
        else:
            metric_parts.append(f"{k}={v:.6f}")

    return prefix + " | " + " | ".join(metric_parts)