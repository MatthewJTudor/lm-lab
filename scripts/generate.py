from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from lm_lab.config.load import load_run_config
from lm_lab.tokenization.char import CharTokenizer
from lm_lab.utils.seed import seed_everything

def _find_latest_checkpoint(runs_dir: Path) -> Path:
    """
    Find runs/<timestamp>/final.pt with the newest timestamp-like folder name.
    """
    if not runs_dir.exists():
        raise FileNotFoundError(f'runs_dir does not exist: {runs_dir}')

    candidates: list[Path] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        ckpt = p / 'final.pt'
        if ckpt.exists():
            candidates.append(ckpt)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in runs_dir: {runs_dir}. "
            "Run: python scripts/train.py --config configs/run.toml --save"
        )

    # Timestamps sort lexicographically if formatted like YYYYMMDD_HHMMSS
    candidates.sort(key=lambda x: x.parent.name)
    return candidates[-1]

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    logits: (V,)
    Keep only top-k logits; set the rest to -inf.
    """
    if k <= 0  or k >= logits.numel():
        return logits

    v, _ = torch.topk(logits, k)
    kth = v[-1]
    neg_inf = torch.full_like(logits, float("-inf"))
    return torch.where(logits >= kth, logits, neg_inf)

def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering.
    Keeps the smallest set of tokens with cumulative prob >= p.
    logits: (V,)
    returns: (V,) logits with filtered tokens set to -inf
    """
    if p <= 0.0 or p >= 1.0:
        return logits  # no filtering

    # Convert to probs
    probs = F.softmax(logits, dim=-1)

    # Sort descending
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    cum = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens while cum prob <= p, but always keep at least 1 token
    keep = cum <= p
    keep[0] = True
    # shift right so we include the first above-threshold token
    keep = torch.cat([torch.tensor([True], device=keep.device, dtype=torch.bool), keep[:-1]])

    # Map keep mask back to original index space
    keep_idx = sorted_idx[keep]

    filtered = torch.full_like(logits, float("-inf"))
    filtered[keep_idx] = logits[keep_idx]
    return filtered

def _sample_next_token(
    logits: torch.Tensor,   # (V,)
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    if logits.dim() != 1:
        raise ValueError(f"Expected logits shape (V,), got {tuple(logits.shape)}")

    # Greedy
    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")

    if top_p < 0.0 or top_p > 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")

    V = logits.numel()
    if top_k < 0 or top_k > V:
        raise ValueError("top_k must be between 0 and vocab size")

    if 0.0 < temperature < 1e-6:
        temperature = 1e-6

    scaled = logits / temperature

    if top_k and top_k > 0:
        scaled = _top_k_filter(scaled, k=top_k)

    if top_p and top_p < 1.0:
        scaled = _top_p_filter(scaled, p=top_p)

    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoregressive text generation (no KV-cache yet)")
    parser.add_argument("--config", type=str, required=True, help="Path to run.toml")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Path to checkpoint (default: latest in runs_dir)"
                        )
    parser.add_argument("--runs_dir", type=str, default="runs", help="Where to look for latest checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text (char-level)")
    parser.add_argument("--max_new_tokens", type=int, default=-1, help="How many new tokens to generate")
    parser.add_argument("--temperature", type=float, default=-1.0,
                        help="0 => greedy; >0 => sampling; -1 => use config if present")
    parser.add_argument("--top_k", type=int, default=-1, help="0 => off; -1 => use config if present")
    parser.add_argument("--top_p", type=float, default=-1.0, help="Nucleus sampling p in (0,1]; -1 uses config")
    parser.add_argument("--seed", type=int, default=-1, help="-1 => use config seed; else override")
    args = parser.parse_args()

    cfg = load_run_config(args.config)

    # Tokenizer must match training corpus for this v0 setup
    tok = CharTokenizer.build(cfg.data_text)

    # Resolve gen defaults: CLI > config.gen > hard defaults
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

    # Determinism (CPU baseline)
    seed_everything(replace(cfg.seed, seed=gen_seed))
    torch.manual_seed(gen_seed)

    # Build model, load weights
    from lm_lab.core.model import TransformerLM  # keep script boundary clean

    model_cfg = replace(cfg.model, vocab_size=tok.vocab_size)
    model = TransformerLM(model_cfg)

    ckpt_path = Path(args.ckpt) if args.ckpt else _find_latest_checkpoint(Path(args.runs_dir))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Encode prompt
    prompt_ids = tok.encode(args.prompt)
    if len(prompt_ids) == 0:
        # If empty prompt, start from a single <unk> (id=1) just to have a token.
        prompt_ids = [tok.stoi[tok.UNK]]

    idx = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # (1, T)

    # Generate
    with torch.inference_mode():
        for _ in range(max_new):
            # Crop context to model max
            idx_cond = idx[:, -cfg.model.max_seq_len:]

            logits = model(idx_cond)  # (1, T, V)
            next_logits = logits[0, -1, :]  # (V,)

            next_id = _sample_next_token(
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