from __future__ import annotations

from lm_lab.config.load import TokenizerConfig
from lm_lab.tokenization.char import CharTokenizer
from lm_lab.tokenization.protocol import TokenizerProtocol
from lm_lab.tokenization.word import WordTokenizer
from lm_lab.tokenization.bpe import BPETokenizer


def build_tokenizer(cfg: TokenizerConfig, text: str) -> TokenizerProtocol:
    mode = cfg.mode.strip().lower()

    if mode == "char":
        return CharTokenizer.build(text)

    if mode == "word":
        return WordTokenizer.build(text)

    if mode == "bpe":
        return BPETokenizer.build(text, vocab_size=cfg.bpe_vocab_size)

    raise ValueError(
        f"Unsupported tokenizer mode: {cfg.mode!r}. "
        "Supported modes today: ['char', 'word', 'bpe']"
    )