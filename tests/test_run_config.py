from pathlib import Path

from lm_lab.config.load import load_run_config


def test_load_run_config(tmp_path: Path) -> None:
    toml = """
[seed]
seed = 123

[data]
corpus = "abc\\n"
block_size = 8

[model]
d_model = 64
n_layers = 2
max_seq_len = 8
pos_mode = "sinusoidal"

[train]
steps = 300
lr = 1e-3
log_every = 50
""".lstrip()

    p = tmp_path / "run.toml"
    p.write_text(toml, encoding="utf-8")

    cfg = load_run_config(p)

    assert cfg.seed.seed == 123
    assert cfg.data_text == "abc\n"
    assert cfg.data.block_size == 8

    assert cfg.model.d_model == 64
    assert cfg.model.n_layers == 2
    assert cfg.model.max_seq_len == 8
    assert cfg.model.pos_mode == "sinusoidal"

    assert cfg.train.steps == 300
    assert cfg.train.lr == 1e-3
    assert cfg.train.log_every == 50