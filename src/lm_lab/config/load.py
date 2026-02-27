from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Literal, Optional

from lm_lab.utils.seed import SeedConfig
from lm_lab.data.sequence_dataset import SequenceDatasetConfig
from lm_lab.core.model import TransformerLMConfig

@dataclass(frozen=True)
class GenConfig:
    temperature: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 100
    seed: int = 1337

@dataclass(frozen=True)
class TrainConfig:
    steps: int
    lr: float
    log_every: int

    optimizer: str = "adamw"
    weight_decay: float = 0.01

    batch_size: int = 0  # 0 => full batch (today)
    shuffle: bool = False

    grad_clip: float = 0.0
    device: str = "cpu"

@dataclass(frozen=True)
class RunConfig:
    seed: SeedConfig
    data_text: str
    data: SequenceDatasetConfig
    model: TransformerLMConfig
    train: TrainConfig
    gen: Optional[GenConfig] = None


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))

    seed_cfg = SeedConfig(seed=int(raw["seed"]["seed"]))

    data_text = str(raw["data"]["corpus"])
    data_cfg = SequenceDatasetConfig(block_size=int(raw["data"]["block_size"]))

    # extend TransformerLMConfig to include pos_mode if you haven’t already
    model_raw = raw["model"]
    model_cfg = TransformerLMConfig(
        vocab_size=0,  # filled after tokenizer build
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
        shuffle=bool(train_raw.get("shuffle", False)),
        grad_clip=float(train_raw.get("grad_clip", 0.0)),
        device=str(train_raw.get("device", "cpu")),
    )

    gen_cfg = None
    if "gen" in raw:
        gen_raw = raw["gen"]
        gen_cfg = GenConfig(
            temperature=float(gen_raw.get("temperature", 1.0)),
            top_k=int(gen_raw.get("top_k", 0)),
            max_new_tokens=int(gen_raw.get("max_new_tokens", 100)),
            seed=int(gen_raw.get("seed", 1337)),
        )

    return RunConfig(
        seed=seed_cfg,
        data_text=data_text,
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        gen=gen_cfg,
    )