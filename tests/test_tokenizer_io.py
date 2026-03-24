from __future__ import annotations

"""
Tests for tokenizer serialization.

Primary invariant:
    tokenizer → save → load → tokenizer'

must preserve:
    - encode() behavior
    - decode() behavior
    - vocab_size

This ensures tokenizer artifacts are safe for:
    - generation reuse
    - experiment reproducibility
"""

from pathlib import Path
import tempfile

from lm_lab.tokenization.char import CharTokenizer
from lm_lab.tokenization.word import WordTokenizer
from lm_lab.tokenization.bpe import BPETokenizer

from lm_lab.tokenization.io import save_tokenizer, load_tokenizer


_SAMPLE_TEXT = "Hello world! Watson's test.\nAnother line."


def _roundtrip(tokenizer, tmp_path: Path):
    path = tmp_path / "tokenizer.json"

    save_tokenizer(tokenizer, path)
    loaded = load_tokenizer(path)

    # Core invariants
    assert tokenizer.vocab_size == loaded.vocab_size

    encoded_orig = tokenizer.encode(_SAMPLE_TEXT)
    encoded_new = loaded.encode(_SAMPLE_TEXT)

    assert encoded_orig == encoded_new

    decoded_orig = tokenizer.decode(encoded_orig)
    decoded_new = loaded.decode(encoded_new)

    assert decoded_orig == decoded_new


def test_char_tokenizer_roundtrip(tmp_path: Path):
    tok = CharTokenizer.build(_SAMPLE_TEXT)
    _roundtrip(tok, tmp_path)


def test_word_tokenizer_roundtrip(tmp_path: Path):
    tok = WordTokenizer.build(_SAMPLE_TEXT)
    _roundtrip(tok, tmp_path)


def test_bpe_tokenizer_roundtrip(tmp_path: Path):
    tok = BPETokenizer.build(_SAMPLE_TEXT, vocab_size=64)
    _roundtrip(tok, tmp_path)