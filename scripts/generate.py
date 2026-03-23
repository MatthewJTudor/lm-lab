from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from lm_lab.config.load import load_run_config
from lm_lab.tokenization.build import build_tokenizer
from lm_lab.utils.seed import seed_everything
from lm_lab.inference.sampling import sample_next_token


def _find_latest_checkpoint(runs_dir: Path) -> Path:
    """
    Find the most recent saved checkpoint under the runs directory.

    Expected layout:
        runs/<timestamp>/final.pt

    Timestamp directories are assumed to sort lexicographically when formatted
    as ``YYYYMMDD_HHMMSS``.

    Args:
        runs_dir: Base directory containing saved run subdirectories.

    Returns:
        Path to the newest available ``final.pt`` checkpoint.

    Raises:
        FileNotFoundError: If the runs directory does not exist or no checkpoint
            files are found.
    """
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir does not exist: {runs_dir}")

    candidates: list[Path] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        ckpt = p / "final.pt"
        if ckpt.exists():
            candidates.append(ckpt)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in runs_dir: {runs_dir}. "
            "Run: python scripts/train.py --config configs/run.toml --save"
        )

    # Timestamp folder names sort correctly under the standard run-dir format.
    candidates.sort(key=lambda x: x.parent.name)
    return candidates[-1]


def main() -> None:
    """
    Run autoregressive text generation from a saved checkpoint.

    Responsibilities:
        - load typed run configuration
        - resolve generation defaults from config and CLI overrides
        - rebuild the tokenizer used for training
        - load model weights from an explicit or inferred checkpoint
        - generate text with cached or uncached decoding

    Notes:
        - This script is orchestration-only; model semantics live in core/.
        - The KV-cache path is an optimization for generation speed, not a
          semantic change to the model.
    """
    parser = argparse.ArgumentParser(description="Autoregressive text generation")
    parser.add_argument("--config", type=str, required=True, help="Path to run.toml")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to checkpoint (default: latest in runs_dir)",
    )
    parser.add_argument("--runs_dir", type=str, default="runs", help="Where to look for latest checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text to continue from")
    parser.add_argument("--max_new_tokens", type=int, default=-1, help="How many new tokens to generate")
    parser.add_argument(
        "--temperature",
        type=float,
        default=-1.0,
        help="0 => greedy; >0 => sampling; -1 => use config if present",
    )
    parser.add_argument("--top_k", type=int, default=-1, help="0 => off; -1 => use config if present")
    parser.add_argument("--top_p", type=float, default=-1.0, help="Nucleus sampling p in (0,1]; -1 uses config")
    parser.add_argument("--seed", type=int, default=-1, help="-1 => use config seed; else override")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV cache for faster generation")
    args = parser.parse_args()

    cfg = load_run_config(args.config)

    # Rebuild the tokenizer from the configured corpus so token IDs match training.
    tok = build_tokenizer(cfg.tokenizer, cfg.data_text)

    # Resolve generation defaults with precedence:
    # CLI override > config.gen > local hard default.
    gen_temp: float = 0.0
    gen_top_k: int = 0
    gen_top_p: float = 1.0
    gen_seed: int = cfg.seed.seed

    if not args.prompt:
        args.prompt = "Alice "

    if cfg.gen is not None:
        gen_temp = float(cfg.gen.temperature)
        gen_top_k = int(cfg.gen.top_k)
        gen_top_p = float(cfg.gen.top_p)
        gen_seed = int(cfg.gen.seed)

    if args.temperature != -1.0:
        gen_temp = float(args.temperature)
    if args.top_k != -1:
        gen_top_k = int(args.top_k)
    if args.top_p != -1.0:
        gen_top_p = float(args.top_p)
    if args.seed != -1:
        gen_seed = int(args.seed)

    max_new = args.max_new_tokens if args.max_new_tokens != -1 else (
        cfg.gen.max_new_tokens if cfg.gen else 100
    )

    # Establish deterministic generation behavior under the resolved seed.
    seed_everything(replace(cfg.seed, seed=gen_seed))

    # Build the model after tokenizer reconstruction so vocab_size matches the checkpoint.
    from lm_lab.core.model import TransformerLM  # keep script boundary clean

    model_cfg = replace(cfg.model, vocab_size=tok.vocab_size)
    model = TransformerLM(model_cfg)

    ckpt_path = Path(args.ckpt) if args.ckpt else _find_latest_checkpoint(Path(args.runs_dir))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Encode the prompt into the model's token space.
    prompt_ids = tok.encode(args.prompt)
    if len(prompt_ids) == 0:
        # Fallback so generation always has at least one starting token.
        prompt_ids = [tok.stoi[tok.unk_token]]

    idx = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # (1, T)

    past_kvs = None
    with torch.inference_mode():
        if args.use_kv_cache:
            # Warm up the cache on the visible prompt window.
            idx_cond = idx[:, -cfg.model.max_seq_len:].contiguous()
            logits, past_kvs = model.forward_kv(idx_cond, past_kvs=None, use_cache=True)

            for _ in range(max_new):
                # If total context has outgrown the model window, rebuild cache from
                # the most recent visible span.
                if idx.size(1) > cfg.model.max_seq_len:
                    idx_cond = idx[:, -cfg.model.max_seq_len:].contiguous()
                    logits, past_kvs = model.forward_kv(idx_cond, past_kvs=None, use_cache=True)
                else:
                    # Incremental decode path: feed only the newest token and append
                    # to the cached history.
                    last = idx[:, -1:].contiguous()
                    logits, past_kvs = model.forward_kv(last, past_kvs=past_kvs, use_cache=True)

                next_logits = logits[0, -1, :]
                next_id = sample_next_token(
                    next_logits,
                    temperature=gen_temp,
                    top_k=gen_top_k,
                    top_p=gen_top_p,
                )
                idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
        else:
            for _ in range(max_new):
                # Uncached path recomputes over the currently visible context window.
                idx_cond = idx[:, -cfg.model.max_seq_len:]
                logits = model(idx_cond)
                next_logits = logits[0, -1, :]
                next_id = sample_next_token(
                    next_logits,
                    temperature=gen_temp,
                    top_k=gen_top_k,
                    top_p=gen_top_p,
                )
                idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

    out = tok.decode(idx[0].tolist())
    print(f"[ckpt] {ckpt_path}")
    print(out)


if __name__ == "__main__":
    main()