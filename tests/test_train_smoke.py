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
    cfg = replace(cfg, model=replace(cfg.model, pos_mode=pos_mode))

    seed_everything(cfg.seed)
    torch.manual_seed(cfg.seed.seed)

    tok = CharTokenizer.build(cfg.data_text)
    tokens = tok.encode(cfg.data_text)
    ds = SequenceDataset(tokens, cfg.data)
    x, y = _full_batch(ds)

    model_cfg = replace(cfg.model, vocab_size=tok.vocab_size)
    model = TransformerLM(model_cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    # Measure in eval mode (dropout off)
    start = float(_loss_eval(model, x, y).item())

    # Train in train mode (dropout on), grads enabled
    model.train()
    steps = 30
    for _ in range(steps):
        loss = _loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Measure in eval mode again
    end = float(_loss_eval(model, x, y).item())
    return start, end


def test_train_smoke_learned_decreases_loss():
    start, end = _run_smoke("learned")
    assert end < start * 0.75, (start, end)


def test_train_smoke_sinusoidal_decreases_loss():
    start, end = _run_smoke("sinusoidal")
    assert end < start * 0.75, (start, end)