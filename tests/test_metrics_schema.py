from __future__ import annotations

from lm_lab.metrics.schema import LMMetricRecord, format_metric_record


def test_lm_metric_record_defaults_are_valid() -> None:
    rec = LMMetricRecord()
    assert rec.run_id is None
    assert rec.phase == "train"
    assert rec.global_step is None
    assert rec.decode_step is None
    assert rec.metrics == {}


def test_lm_metric_record_to_dict_contains_expected_fields() -> None:
    rec = LMMetricRecord(
        run_id="abc",
        phase="eval",
        global_step=10,
        decode_step=None,
        seed=42,
        tokenizer_mode="bpe",
        metrics={"eval_loss": 1.23},
    )
    d = rec.to_dict()
    assert d["run_id"] == "abc"
    assert d["phase"] == "eval"
    assert d["global_step"] == 10
    assert d["decode_step"] is None
    assert d["metrics"]["eval_loss"] == 1.23


def test_format_metric_record_includes_phase_and_train_step_metrics() -> None:
    rec = LMMetricRecord(
        run_id=None,
        phase="train",
        global_step=5,
        decode_step=None,
        seed=42,
        tokenizer_mode="bpe",
        metrics={"train_loss": 1.0, "perplexity": 2.0},
    )
    s = format_metric_record(rec, ["train_loss", "perplexity"])
    assert "run_id=none" in s
    assert "phase=train" in s
    assert "global_step=5" in s
    assert "train_loss=1.000000" in s
    assert "perplexity=2.000000" in s


def test_format_metric_record_includes_generate_decode_step() -> None:
    rec = LMMetricRecord(
        run_id="gen-1",
        phase="generate",
        global_step=None,
        decode_step=7,
        seed=42,
        tokenizer_mode="bpe",
        metrics={"generation_entropy_mean": 0.5},
    )
    s = format_metric_record(rec, ["generation_entropy_mean"])
    assert "run_id=gen-1" in s
    assert "phase=generate" in s
    assert "decode_step=7" in s
    assert "generation_entropy_mean=0.500000" in s


def test_format_metric_record_uses_scientific_notation_for_extremes() -> None:
    rec = LMMetricRecord(
        run_id=None,
        phase="train",
        global_step=0,
        decode_step=None,
        seed=42,
        tokenizer_mode="bpe",
        metrics={"perplexity": 3.1063e37},
    )
    s = format_metric_record(rec, ["perplexity"])
    assert "global_step=0" in s
    assert "e+" in s or "e-" in s