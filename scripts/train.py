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
from lm_lab.metrics.basic import token_accuracy, perplexity, grad_norm_total
from lm_lab.metrics.logits import (
    next_token_rank_mean,
    confidence_margin_mean,
    max_probability_mean,
    logit_entropy_mean,
)
from lm_lab.metrics.schema import LMMetricRecord, format_metric_record, fmt
from lm_lab.tokenization.build import build_tokenizer
from lm_lab.tokenization.io import save_tokenizer
from lm_lab.utils.seed import seed_everything


def build_batch(ds: SequenceDataset, idxs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a batch from a selected set of dataset indices.

    Args:
        ds: Sequence dataset providing (x, y) next-token windows.
        idxs: Dataset indices to materialize into the batch.

    Returns:
        A tuple of:
            - x batch of shape (B, T) with dtype long
            - y batch of shape (B, T) with dtype long
    """
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
    """
    Create a timestamped run directory under the provided runs root.

    Args:
        runs_dir: Base output directory for saved runs.

    Returns:
        Newly created run directory path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def eval_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Evaluate loss on a provided batch.

    The model is temporarily switched to eval mode and restored to its prior
    training state afterward. Inputs are moved to the model device before the
    forward pass.

    Args:
        model: Model to evaluate.
        x: Input token IDs of shape (B, T).
        y: Target token IDs of shape (B, T).

    Returns:
        Scalar cross-entropy loss for the batch.
    """
    was_training = model.training
    model.eval()

    dev = next(model.parameters()).device
    x = x.to(dev)
    y = y.to(dev)

    with torch.no_grad():
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

    if was_training:
        model.train()
    return float(loss.item())


def eval_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """
    Evaluate both loss and logits on a provided batch.

    This helper is used when evaluation metrics need access to the output
    distribution in addition to the scalar loss.

    Args:
        model: Model to evaluate.
        x: Input token IDs of shape (B, T).
        y: Target token IDs of shape (B, T).

    Returns:
        A tuple of:
            - scalar cross-entropy loss
            - logits of shape (B, T, V)
    """
    was_training = model.training
    model.eval()

    dev = next(model.parameters()).device
    x = x.to(dev)
    y = y.to(dev)

    with torch.no_grad():
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

    if was_training:
        model.train()

    return float(loss.item()), logits

def format_compact_line(train_record, eval_record):
    if train_record.global_step is not None:
        step = train_record.global_step
    else:
        step = train_record.decode_step

    train_part = f"train | {fmt(train_record)}"
    eval_part = f"eval | {fmt(eval_record)}"

    return f"step {step} | {train_part} | {eval_part}"


def main() -> None:
    """
    Run the minimal deterministic LM training loop.

    Responsibilities:
        - load typed run configuration
        - seed all relevant RNGs
        - build tokenizer, dataset, model, and optimizer
        - execute the training loop
        - emit structured train/eval metric records
        - optionally save config and checkpoint artifacts

    Notes:
        - This script is orchestration-only; core model semantics live elsewhere.
        - Metric emission is observational and must not alter model behavior.
    """
    parser = argparse.ArgumentParser(description="Minimal deterministic LM training loop")
    parser.add_argument("--config", type=str, required=True, help="Path to run.toml")
    parser.add_argument("--save", action="store_true", help="Save run artifacts to runs/<timestamp>/")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Base output directory for --save")
    parser.add_argument(
        "--print_mode",
        type=str,
        default="compact",
        choices=["full", "compact", "off"],
        help="Console logging mode",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)

    if cfg.train.shuffle:
        print("[warn] train.shuffle is currently unused (sampling regime is RNG-window based).")

    # Establish deterministic baseline before any model or sampling objects are created.
    seed_everything(cfg.seed)

    # Resolve execution device from config.
    device = torch.device(cfg.train.device)

    # Build tokenizer from the configured corpus, then encode the training text.
    tok = build_tokenizer(cfg.tokenizer, cfg.data_text)
    tokens = tok.encode(cfg.data_text)

    # Build next-token training windows from the flat token stream.
    ds = SequenceDataset(tokens, cfg.data)

    # Current invariant: dataset window length must match model context length.
    if cfg.model.max_seq_len != cfg.data.block_size:
        raise ValueError(
            f"Config mismatch: model.max_seq_len={cfg.model.max_seq_len} "
            f"!= data.block_size={cfg.data.block_size}. "
            "For now these must be equal."
        )

    # Fill vocab_size after tokenizer build so model dimensions match the actual tokenizer.
    from lm_lab.core.model import TransformerLM  # local import keeps script boundary clean
    model_cfg = replace(cfg.model, vocab_size=tok.vocab_size)
    model = TransformerLM(model_cfg).to(device)
    model.train()

    # Optimizer selection is intentionally narrow for now to keep the training path simple.
    opt_name = getattr(cfg.train, "optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer!r} (only 'adamw' supported)")

    N = len(ds)
    bs = getattr(cfg.train, "batch_size", 0) or 0
    use_full_batch = (bs == 0) or (bs >= N)

    # Separate NumPy RNG keeps batch-index sampling deterministic under a fixed seed.
    rng = np.random.default_rng(cfg.seed.seed)

    run_id: str | None = None
    run_dir: Path | None = None

    if args.save:
        run_dir = make_run_dir(Path(args.runs_dir))
        run_id = run_dir.name

    # Main optimization loop.
    for step in range(cfg.train.steps):
        if use_full_batch:
            idxs = np.arange(N, dtype=np.int64)
        else:
            idxs = rng.integers(low=0, high=N, size=bs, dtype=np.int64)

        x_batch, y_batch = build_batch(ds, idxs)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            y_batch.reshape(B * T),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.train.grad_clip and cfg.train.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)

        optimizer.step()

        # Emit metrics at the configured logging cadence rather than every optimization step.
        if step % cfg.train.log_every == 0:
            train_loss = float(loss.item())

            eval_loss_value, eval_logits = eval_step(model, x_batch, y_batch)

            train_metrics = {
                "train_loss": train_loss,
                "perplexity": perplexity(train_loss),
                "grad_norm_total": grad_norm_total(model),
                "token_accuracy": token_accuracy(logits, y_batch),
                "logit_entropy_mean": logit_entropy_mean(logits),
                "max_probability_mean": max_probability_mean(logits),
                "confidence_margin_mean": confidence_margin_mean(logits),
                "next_token_rank_mean": next_token_rank_mean(logits, y_batch),
            }

            eval_metrics = {
                "eval_loss": eval_loss_value,
                "perplexity": perplexity(eval_loss_value),
                "token_accuracy": token_accuracy(eval_logits, y_batch),
                "logit_entropy_mean": logit_entropy_mean(eval_logits),
                "max_probability_mean": max_probability_mean(eval_logits),
                "confidence_margin_mean": confidence_margin_mean(eval_logits),
                "next_token_rank_mean": next_token_rank_mean(eval_logits, y_batch),
            }

            train_record = LMMetricRecord(
                run_id=run_id,
                phase="train",
                global_step=step,
                seed=cfg.seed.seed,
                tokenizer_mode=cfg.tokenizer.mode,
                regime_label="baseline",
                metrics=train_metrics,
            )

            eval_record = LMMetricRecord(
                run_id=run_id,
                phase="eval",
                global_step=step,
                seed=cfg.seed.seed,
                tokenizer_mode=cfg.tokenizer.mode,
                regime_label="baseline",
                metrics=eval_metrics,
            )

            train_metric_order = [
                "train_loss",
                "perplexity",
                "grad_norm_total",
                "token_accuracy",
                "logit_entropy_mean",
                "max_probability_mean",
                "confidence_margin_mean",
                "next_token_rank_mean",
            ]
            eval_metric_order = [
                "eval_loss",
                "perplexity",
                "token_accuracy",
                "logit_entropy_mean",
                "max_probability_mean",
                "confidence_margin_mean",
                "next_token_rank_mean",
            ]

            if args.print_mode == "full":
                print(format_metric_record(train_record, train_metric_order))
                print(format_metric_record(eval_record, eval_metric_order))

            elif args.print_mode == "compact":
                print(format_compact_line(train_record, eval_record))

            elif args.print_mode == "off":
                pass

    # Optional save keeps artifacts local to the script boundary.
    if args.save:
        assert run_dir is not None
        shutil.copy2(args.config, run_dir / "config.toml")
        save_tokenizer(tok, run_dir / "tokenizer.json")
        torch.save(model.state_dict(), run_dir / "final.pt")
        print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()