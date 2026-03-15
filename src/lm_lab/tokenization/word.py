from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_WORD_RE = re.compile(r"\n|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class WordTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def build(cls, text: str) -> "WordTokenizer":
        specials = ["<unk>", "<bos>", "<eos>"]
        pieces = cls._tokenize_text(text)

        vocab = sorted(set(pieces))
        all_tokens = specials + vocab

        stoi = {tok: i for i, tok in enumerate(all_tokens)}
        itos = {i: tok for tok, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        return _WORD_RE.findall(text)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        pieces = self._tokenize_text(s)
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(piece, unk_id) for piece in pieces]

    def decode(self, ids: Iterable[int]) -> str:
        tokens: list[str] = []
        specials = {self.unk_token, self.bos_token, self.eos_token}

        for idx in ids:
            tok = self.itos[int(idx)]
            if tok in specials:
                continue
            tokens.append(tok)

        out: list[str] = []
        no_space_before = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
        no_space_after = {"(", "[", "{", "\n"}

        for tok in tokens:
            if tok == "\n":
                out.append("\n")
                continue

            if not out:
                out.append(tok)
                continue

            prev = out[-1]

            if prev == "\n":
                out.append(tok)
            elif tok in no_space_before:
                out.append(tok)
            elif prev in no_space_after:
                out.append(tok)
            else:
                out.append(" " + tok)

        return "".join(out)