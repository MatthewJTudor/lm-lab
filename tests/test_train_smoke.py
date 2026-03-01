from __future__ import annotations

from dataclasses import replace
import numpy as np

import torch
import torch.nn.functional as F

from lm_lab.config.load import load_run_config
from lm_lab.data.sequence_dataset import SequenceDataset
from lm_lab.tokenization.char import CharTokenizer
from lm_lab.utils.seed import seed_everything
from lm_lab.core.model import TransformerLM


def _full_batch(ds: SequenceDataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    x_batch = torch.from_numpy(np.stack(xs)).long()
    y_batch = torch.from_numpy(np.stack(ys)).long()
    return x_batch, y_batch


def _loss(model: TransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)  # (B, T, V)
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y.view(B * T))

def _loss_eval(model: TransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(x)  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
    if was_training:
        model.train()
    return loss


def _run_smoke(pos_mode: str) -> tuple[float, float]:
    cfg = load_run_config("configs/run.toml")

    # Override corpus to keep smoke test fast + deterministic
    tiny = "hello world\n" * 50

    cfg = replace(
        cfg,
        data_text=tiny,
        data=replace(cfg.data, block_size=32),
        model=replace(cfg.model, pos_mode=pos_mode, d_model=32, n_layers=1, max_seq_len=32),
        train=replace(cfg.train, lr=1e-3),
    )

    seed_everything(cfg.seed)

    tok = CharTokenizer.build(cfg.data_text)
    tokens = tok.encode(cfg.data_text)
    ds = SequenceDataset(tokens, cfg.data)

    # Mini-batch sample instead of full batch
    rng = np.random.default_rng(cfg.seed.seed)
    bs = 64
    idxs = rng.integers(0, len(ds), size=bs, dtype=np.int64)

    xs, ys = zip(*(ds[i] for i in idxs))
    x = torch.from_numpy(np.stack(xs)).long()
    y = torch.from_numpy(np.stack(ys)).long()

    model_cfg = replace(cfg.model, vocab_size=tok.vocab_size)
    model = TransformerLM(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    start = float(_loss_eval(model, x, y).item())

    model.train()
    steps = 10
    for _ in range(steps):
        loss = _loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    end = float(_loss_eval(model, x, y).item())
    return start, end


def test_train_smoke_learned_decreases_loss():
    start, end = _run_smoke("learned")
    assert end < start, (start, end)


def test_train_smoke_sinusoidal_decreases_loss():
    start, end = _run_smoke("sinusoidal")
    assert end < start, (start, end)