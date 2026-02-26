from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

@dataclass(frozen=True)
class CharTokenizerConfig:
    add_bos: bool = False
    add_eos: bool = False

class CharTokenizer:
    """
    A deterministic character-level tokenizer.

    Vocab layout:
      0: <pad>
      1: <unk>
      2: <bos>
      3: <eos>
      4+: sorted unique characters
    """

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, cfg: CharTokenizerConfig, stoi: Dict[str, int], itos: List[str]) -> None:
        self.cfg = cfg
        self.stoi = stoi
        self.itos = itos

        # Small sanity checks
        assert self.itos[self.stoi[self.PAD]] == self.PAD
        assert self.itos[self.stoi[self.UNK]] == self.UNK
        assert self.itos[self.stoi[self.BOS]] == self.BOS
        assert self.itos[self.stoi[self.EOS]] == self.EOS

    @classmethod
    def build(cls, text: str, cfg: CharTokenizerConfig | None = None) -> "CharTokenizer":
        """
        Build vocab from a corpus string.
        Deterministic by construction: characters are sorted.
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
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
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
        chars: List[str] = []
        for i in ids:
            if not (0 <= i < len(self.itos)):
                raise ValueError(f"Token ID out of range: {i}")
            tok = self.itos[i]
            # Skip special tokens on decode (common behavior; we can make configurable later)
            if tok in (self.PAD, self.BOS, self.EOS):
                continue
            if tok == self.UNK:
                chars.append("�")
            else:
                chars.append(tok)
        return "".join(chars)