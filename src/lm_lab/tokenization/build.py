from __future__ import annotations

from lm_lab.config.load import TokenizerConfig
from lm_lab.tokenization.char import CharTokenizer
from lm_lab.tokenization.protocol import TokenizerProtocol
from lm_lab.tokenization.word import WordTokenizer


def build_tokenizer(cfg: TokenizerConfig, text: str) -> TokenizerProtocol:
    mode = cfg.mode.strip().lower()

    if mode == "char":
        return CharTokenizer.build(text)

    if mode == "word":
        return WordTokenizer.build(text)

    raise ValueError(
        f"Unsupported tokenizer mode: {cfg.mode!r}. "
        "Supported modes today: ['char', 'word']"
    )