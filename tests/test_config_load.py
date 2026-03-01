from __future__ import annotations

from pathlib import Path

from lm_lab.config.load import load_run_config


def test_load_run_config_resolves_repo_root_corpus_path() -> None:
    # assumes tests are run from repo root (recommended)
    cfg = load_run_config("configs/run.toml")

    assert cfg.data_text, "data_text should not be empty"
    assert cfg.data_text_path is not None, "data_text_path should be set when corpus_path is used"

    p = Path(cfg.data_text_path)
    # normalize for windows
    assert p.name == "corpus.txt"
    assert p.parent.name == "data"


def test_load_run_config_gen_fields_present() -> None:
    cfg = load_run_config("configs/run.toml")
    assert cfg.gen is not None
    assert isinstance(cfg.gen.top_p, float)
    assert isinstance(cfg.gen.top_k, int)