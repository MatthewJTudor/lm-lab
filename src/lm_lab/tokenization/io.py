from __future__ import annotations

"""
Tokenizer serialization utilities.

This module provides a stable, inspectable mechanism for saving and loading
tokenizer state as part of a training run artifact.

Design goals:
- Deterministic round-trip (save → load produces identical behavior)
- JSON-based (human-readable, portable)
- Explicit tokenizer typing (no implicit inference)
- No side effects (pure serialization/deserialization)

The tokenizer artifact is considered part of the run state, alongside:
    - model checkpoint (final.pt)
    - config.toml

This ensures:
    - reproducible generation
    - consistent tokenization between train and generate
    - compatibility with experiment orchestration layers
"""

import json
from pathlib import Path
from typing import Any

from lm_lab.tokenization.char import CharTokenizer, CharTokenizerConfig
from lm_lab.tokenization.word import WordTokenizer
from lm_lab.tokenization.bpe import BPETokenizer


def save_tokenizer(tokenizer: object, path: str | Path) -> None:
    """
    Serialize a tokenizer to JSON.

    Args:
        tokenizer:
            Tokenizer instance (CharTokenizer, WordTokenizer, or BPETokenizer).

        path:
            Output file path. Will be overwritten if it exists.

    Notes:
        - Output is JSON for readability and portability.
        - This function does not validate tokenizer correctness beyond type.
    """
    p = Path(path)
    payload = _tokenizer_to_dict(tokenizer)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tokenizer(path: str | Path) -> object:
    """
    Load a tokenizer from a JSON artifact.

    Args:
        path:
            Path to tokenizer JSON file.

    Returns:
        Reconstructed tokenizer instance.

    Raises:
        ValueError:
            If tokenizer type is unknown or payload is malformed.
    """
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return _tokenizer_from_dict(payload)


def _tokenizer_to_dict(tokenizer: object) -> dict[str, Any]:
    """
    Convert a tokenizer instance into a JSON-serializable dictionary.

    This function is intentionally explicit by tokenizer type to ensure
    stable, predictable serialization.
    """
    if isinstance(tokenizer, CharTokenizer):
        return {
            "type": "char",
            "cfg": {
                "add_bos": tokenizer.cfg.add_bos,
                "add_eos": tokenizer.cfg.add_eos,
            },
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        }

    if isinstance(tokenizer, WordTokenizer):
        return {
            "type": "word",
            "stoi": tokenizer.stoi,
            # JSON requires string keys → convert int keys explicitly
            "itos": {str(k): v for k, v in tokenizer.itos.items()},
            "unk_token": tokenizer.unk_token,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
        }

    if isinstance(tokenizer, BPETokenizer):
        return {
            "type": "bpe",
            "stoi": tokenizer.stoi,
            "itos": {str(k): v for k, v in tokenizer.itos.items()},
            # store merges explicitly as list of pairs
            "merges": [[left, right] for left, right in tokenizer.merges],
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
        }

    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)!r}")


def _tokenizer_from_dict(payload: dict[str, Any]) -> object:
    """
    Reconstruct a tokenizer instance from serialized dictionary form.

    This is the inverse of `_tokenizer_to_dict`.

    Important:
        - JSON keys are strings → integer keys must be restored explicitly
        - No inference is performed; type must be present in payload
    """
    tok_type = payload["type"]

    if tok_type == "char":
        cfg_raw = payload["cfg"]
        cfg = CharTokenizerConfig(
            add_bos=bool(cfg_raw.get("add_bos", False)),
            add_eos=bool(cfg_raw.get("add_eos", False)),
        )
        return CharTokenizer(
            cfg=cfg,
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos=[str(x) for x in payload["itos"]],
        )

    if tok_type == "word":
        return WordTokenizer(
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos={int(k): str(v) for k, v in payload["itos"].items()},
            unk_token=str(payload.get("unk_token", "<unk>")),
            bos_token=str(payload.get("bos_token", "<bos>")),
            eos_token=str(payload.get("eos_token", "<eos>")),
        )

    if tok_type == "bpe":
        merges = [(str(left), str(right)) for left, right in payload["merges"]]
        return BPETokenizer(
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos={int(k): str(v) for k, v in payload["itos"].items()},
            merges=merges,
            bos_token=str(payload.get("bos_token", "<bos>")),
            eos_token=str(payload.get("eos_token", "<eos>")),
        )

    raise ValueError(f"Unknown tokenizer type: {tok_type!r}")