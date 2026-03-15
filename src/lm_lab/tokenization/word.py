from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


# Normalize common Unicode punctuation to simpler ASCII-ish forms.
def _normalize_text(text: str) -> str:
    return (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("—", "--")
            .replace("–", "-")
    )


# Token pattern:
# - newline as its own token
# - word with optional internal apostrophe chunk: don't, I've, we'll
# - numbers
# - punctuation / symbols as separate tokens
_WORD_RE = re.compile(
    r"\n|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]",
    re.UNICODE,
)


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
        text = _normalize_text(text)
        return _WORD_RE.findall(text)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        pieces = self._tokenize_text(s)
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(piece, unk_id) for piece in pieces]

    def decode(self, ids: Iterable[int]) -> str:
        specials = {self.unk_token, self.bos_token, self.eos_token}
        tokens: list[str] = []

        for idx in ids:
            tok = self.itos[int(idx)]
            if tok in specials:
                continue
            tokens.append(tok)

        no_space_before = {
            ".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "'",
        }
        no_space_after = {
            "(", "[", "{",
        }

        out: list[str] = []
        prev_tok: str | None = None
        quote_open = False

        for tok in tokens:
            if tok == "\n":
                out.append("\n")
                prev_tok = "\n"
                continue

            if tok == '"':
                if not out or prev_tok == "\n":
                    # opening quote at start of text/line
                    out.append('"')
                    quote_open = True
                elif quote_open:
                    # closing quote attaches to previous token
                    out.append('"')
                    quote_open = False
                else:
                    # opening quote after normal text gets a leading space
                    out.append(' "')
                    quote_open = True

                prev_tok = tok
                continue

            if prev_tok is None:
                out.append(tok)
            elif prev_tok == "\n":
                out.append(tok)
            elif tok in no_space_before:
                out.append(tok)
            elif prev_tok in no_space_after:
                out.append(tok)
            elif prev_tok == '"':
                # token after opening quote attaches directly
                out.append(tok)
            else:
                out.append(" " + tok)

            prev_tok = tok

        text = "".join(out)
        text = text.replace("\n ", "\n")
        return text