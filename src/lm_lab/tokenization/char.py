from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class CharTokenizerConfig:
    """
    Configuration for the character-level tokenizer.

    Attributes:
        add_bos: Whether to prepend a beginning-of-sequence token.
        add_eos: Whether to append an end-of-sequence token.
    """
    add_bos: bool = False
    add_eos: bool = False


class CharTokenizer:
    """
    Deterministic character-level tokenizer.

    Vocab layout:
        0: <pad>
        1: <unk>
        2: <bos>
        3: <eos>
        4+: sorted unique characters from the corpus

    Notes:
        - Vocabulary construction is deterministic via sorted unique characters.
        - Unknown characters map to <unk>.
        - Special tokens are omitted during decode by default.
    """

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, cfg: CharTokenizerConfig, stoi: Dict[str, int], itos: List[str]) -> None:
        self.cfg = cfg
        self.stoi = stoi
        self.itos = itos

        # Sanity checks: ensure special token indices are consistent
        assert self.itos[self.stoi[self.PAD]] == self.PAD
        assert self.itos[self.stoi[self.UNK]] == self.UNK
        assert self.itos[self.stoi[self.BOS]] == self.BOS
        assert self.itos[self.stoi[self.EOS]] == self.EOS

    @classmethod
    def build(cls, text: str, cfg: CharTokenizerConfig | None = None) -> "CharTokenizer":
        """
        Build a character-level tokenizer from corpus text.

        Args:
            text: Corpus used to construct the vocabulary.
            cfg: Optional tokenizer configuration.

        Returns:
            A fully constructed CharTokenizer.

        Notes:
            - Vocabulary is deterministic: characters are sorted.
            - Special tokens are prepended to the vocabulary.
        """
        if cfg is None:
            cfg = CharTokenizerConfig()

        special = [cls.PAD, cls.UNK, cls.BOS, cls.EOS]
        chars = sorted(set(text))
        itos = special + chars
        stoi = {s: i for i, s in enumerate(itos)}
        return cls(cfg=cfg, stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        """
        Encode a string into character token IDs.

        Args:
            s: Input string.

        Returns:
            List of token IDs.
        """
        ids: List[int] = []

        if self.cfg.add_bos:
            ids.append(self.stoi[self.BOS])

        unk_id = self.stoi[self.UNK]
        for ch in s:
            ids.append(self.stoi.get(ch, unk_id))

        if self.cfg.add_eos:
            ids.append(self.stoi[self.EOS])

        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode token IDs back into a string.

        Args:
            ids: Iterable of token IDs.

        Returns:
            Decoded string.

        Notes:
            - <pad>, <bos>, and <eos> tokens are skipped.
            - <unk> is rendered as the replacement character "�".
        """
        chars: List[str] = []
        for i in ids:
            if not (0 <= i < len(self.itos)):
                raise ValueError(f"Token ID out of range: {i}")

            tok = self.itos[i]

            # Skip structural tokens during decode
            if tok in (self.PAD, self.BOS, self.EOS):
                continue

            if tok == self.UNK:
                chars.append("�")
            else:
                chars.append(tok)

        return "".join(chars)