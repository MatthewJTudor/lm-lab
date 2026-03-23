from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Optional

from lm_lab.utils.seed import SeedConfig
from lm_lab.data.sequence_dataset import SequenceDatasetConfig
from lm_lab.core.model import TransformerLMConfig


@dataclass(frozen=True)
class GenConfig:
    """
    Generation-time decoding configuration.

    Attributes:
        temperature: Sampling temperature. A value of 0.0 is treated as greedy decode.
        top_k: Top-k truncation for next-token sampling.
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum number of tokens to generate.
        seed: RNG seed used for generation-time determinism.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 100
    seed: int = 1337


@dataclass(frozen=True)
class TrainConfig:
    """
    Training loop configuration.

    Attributes:
        steps: Number of training steps.
        lr: Optimizer learning rate.
        log_every: Interval for training/evaluation logging.

        optimizer: Optimizer name.
        weight_decay: Weight decay applied by the optimizer.

        batch_size: Batch size. A value of 0 selects full-batch training.
        shuffle: Reserved training-order flag for future batching regimes.

        grad_clip: Maximum gradient norm for clipping. A value of 0.0 disables clipping.
        device: Device string used for training execution.
    """

    steps: int
    lr: float
    log_every: int

    optimizer: str = "adamw"
    weight_decay: float = 0.01

    batch_size: int = 0  # 0 => full batch (today)
    shuffle: bool = True

    grad_clip: float = 0.0
    device: str = "cpu"


@dataclass(frozen=True)
class TokenizerConfig:
    """
    Tokenizer construction configuration.

    Attributes:
        mode: Tokenizer mode to build.
        bpe_vocab_size: Target BPE vocabulary size when mode == "bpe".
    """

    mode: str = "char"
    bpe_vocab_size: int = 512


@dataclass(frozen=True)
class RunConfig:
    """
    Fully resolved runtime configuration for a training/generation run.

    Attributes:
        seed: Global seeding configuration.
        data_text: Resolved training corpus text.
        data: Dataset/windowing configuration.
        model: Transformer model configuration.
        train: Training loop configuration.
        tokenizer: Tokenizer construction configuration.
        gen: Optional generation configuration.
        data_text_path: Resolved corpus path when text was loaded from disk, else None.
    """

    seed: SeedConfig
    data_text: str
    data: SequenceDatasetConfig
    model: TransformerLMConfig
    train: TrainConfig
    tokenizer: TokenizerConfig
    gen: Optional[GenConfig] = None
    data_text_path: Optional[str] = None


def load_run_config(path: str | Path) -> RunConfig:
    """
    Load a TOML run configuration and resolve it into typed runtime dataclasses.

    Resolution behavior:
        - Parses the TOML file at ``path``.
        - Resolves corpus text from either ``[data].corpus_path`` or ``[data].corpus``.
        - Prefers corpus paths relative to the inferred repo root, with a fallback
          to the config file directory.
        - Leaves model vocab_size at 0 so it can be filled after tokenizer build.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        A fully resolved RunConfig instance.

    Raises:
        ValueError: If no corpus source is provided.
    """
    p = Path(path).resolve()
    cfg_dir = p.parent

    # Heuristic: configs/ is one level under repo root.
    repo_root = cfg_dir.parent
    raw = tomllib.loads(p.read_text(encoding="utf-8"))

    seed_cfg = SeedConfig(seed=int(raw["seed"]["seed"]))

    data_raw = raw["data"]

    corpus_path = str(data_raw.get("corpus_path", "")).strip()

    corpus_inline = data_raw.get("corpus", "")
    if corpus_inline is None:
        corpus_inline = ""
    # Preserve inline newlines exactly as provided in the TOML.
    corpus_inline = str(corpus_inline)

    if corpus_path:
        corpus_rel = Path(corpus_path)
        if corpus_rel.is_absolute():
            corpus_p = corpus_rel
        else:
            # Prefer repo-root-relative paths to match the standard project layout.
            corpus_p = (repo_root / corpus_rel).resolve()
            if not corpus_p.exists():
                # Fallback: resolve relative to the config file directory.
                corpus_p = (cfg_dir / corpus_rel).resolve()

        data_text = corpus_p.read_text(encoding="utf-8")
        data_text_path = str(corpus_p)
    elif corpus_inline:
        data_text = corpus_inline
        data_text_path = None
    else:
        raise ValueError("No corpus provided. Set [data].corpus_path or [data].corpus in run.toml.")

    data_cfg = SequenceDatasetConfig(block_size=int(raw["data"]["block_size"]))

    model_raw = raw["model"]
    model_cfg = TransformerLMConfig(
        vocab_size=0,  # Filled after tokenizer build.
        max_seq_len=int(model_raw["max_seq_len"]),
        d_model=int(model_raw["d_model"]),
        n_layers=int(model_raw["n_layers"]),
        pos_mode=model_raw.get("pos_mode", "learned"),

        # --- attention ---
        n_heads=int(model_raw.get("n_heads", 1)),
        attn_bias=bool(model_raw.get("attn_bias", False)),
        attn_impl=model_raw.get("attn_impl", "naive"),

        # --- MLP ---
        mlp_hidden_mult=int(model_raw.get("mlp_hidden_mult", 4)),
        activation=model_raw.get("activation", "gelu"),

        # --- normalization / residual style ---
        norm_mode=model_raw.get("norm_mode", "pre"),
        layer_norm_eps=float(model_raw.get("layer_norm_eps", 1e-5)),

        # --- regularization ---
        dropout=float(model_raw.get("dropout", 0.0)),

        # --- embeddings ---
        tie_embeddings=bool(model_raw.get("tie_embeddings", True)),
    )

    train_raw = raw["train"]
    train_cfg = TrainConfig(
        steps=int(train_raw["steps"]),
        lr=float(train_raw["lr"]),
        log_every=int(train_raw["log_every"]),

        optimizer=str(train_raw.get("optimizer", "adamw")),
        weight_decay=float(train_raw.get("weight_decay", 0.01)),
        batch_size=int(train_raw.get("batch_size", 0)),
        shuffle=bool(train_raw.get("shuffle", True)),
        grad_clip=float(train_raw.get("grad_clip", 0.0)),
        device=str(train_raw.get("device", "cpu")),
    )

    tok_raw = raw.get("tokenizer", {})
    tok_cfg = TokenizerConfig(
        mode=str(tok_raw.get("mode", "char")),
        bpe_vocab_size=int(tok_raw.get("bpe_vocab_size", 512)),
    )

    gen_cfg = None
    if "gen" in raw:
        gen_raw = raw["gen"]
        gen_cfg = GenConfig(
            temperature=float(gen_raw.get("temperature", 1.0)),
            top_k=int(gen_raw.get("top_k", 0)),
            top_p=float(gen_raw.get("top_p", 1.0)),
            max_new_tokens=int(gen_raw.get("max_new_tokens", 100)),
            seed=int(gen_raw.get("seed", 1337)),
        )

    return RunConfig(
        seed=seed_cfg,
        data_text=data_text,
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        tokenizer=tok_cfg,
        gen=gen_cfg,
        data_text_path=data_text_path,
    )