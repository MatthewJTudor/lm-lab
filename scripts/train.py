from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lm_lab.config.load import load_run_config
from lm_lab.data.sequence_dataset import SequenceDataset
from lm_lab.tokenization.char import CharTokenizer
from lm_lab.utils.seed import seed_everything


def build_full_batch(ds: SequenceDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full-batch over the entire dataset.

    Returns:
        x: (B, T) long
        y: (B, T) long
    """
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for i in range(len(ds)):
        x, y = ds[i]
        x_list.append(x)
        y_list.append(y)

    x_batch = torch.from_numpy(np.stack(x_list)).long()
    y_batch = torch.from_numpy(np.stack(y_list)).long()
    return x_batch, y_batch

def build_batch(ds: SequenceDataset, idxs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for i in idxs:
        x, y = ds[int(i)]
        x_list.append(x)
        y_list.append(y)
    x_batch = torch.from_numpy(np.stack(x_list)).long()
    y_batch = torch.from_numpy(np.stack(y_list)).long()
    return x_batch, y_batch


def make_run_dir(runs_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def eval_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
    if was_training:
        model.train()
    return float(loss.item())

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal deterministic LM training loop")
    parser.add_argument("--config", type=str, required=True, help="Path to run.toml")
    parser.add_argument("--save", action="store_true", help="Save run artifacts to runs/<timestamp>/")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Base output directory for --save")
    args = parser.parse_args()

    cfg = load_run_config(args.config)

    # Determinism (CPU baseline)
    seed_everything(cfg.seed)
    torch.manual_seed(cfg.seed.seed)

    # Tokenizer + encode corpus
    tok = CharTokenizer.build(cfg.data_text)
    tokens = tok.encode(cfg.data_text)

    # Dataset
    ds = SequenceDataset(tokens, cfg.data)

    # Invariant: current model enforces T <= max_seq_len, and dataset uses block_size windows.
    if cfg.model.max_seq_len != cfg.data.block_size:
        raise ValueError(
            f"Config mismatch: model.max_seq_len={cfg.model.max_seq_len} "
            f"!= data.block_size={cfg.data.block_size}. "
            "For now these must be equal."
        )



    # Model (fill vocab_size after tokenizer build)
    from lm_lab.core.model import TransformerLM  # local import keeps script boundary clean
    model_cfg = replace(cfg.model, vocab_size=tok.vocab_size)
    model = TransformerLM(model_cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    N = len(ds)
    bs = getattr(cfg.train, "batch_size", 0) or 0
    use_full_batch = (bs == 0) or (bs >= N)

    rng = np.random.default_rng(cfg.seed.seed)

    # Train
    for step in range(cfg.train.steps):
        if use_full_batch:
            idxs = np.arange(N, dtype=np.int64)
        else:
            idxs = rng.integers(low=0, high=N, size=bs, dtype=np.int64)

        x_batch, y_batch = build_batch(ds, idxs)

        logits = model(x_batch)  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            y_batch.reshape(B * T),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.train.log_every == 0:
            train_loss = float(loss.item())
            snap = eval_loss(model, x_batch, y_batch)
            print(f"step: {step} | train_loss: {train_loss:.6f} | eval_loss: {snap:.6f}")

    # Optional save (simple, LM-layer only)
    if args.save:
        run_dir = make_run_dir(Path(args.runs_dir))
        shutil.copy2(args.config, run_dir / "config.toml")
        torch.save(model.state_dict(), run_dir / "final.pt")
        print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()