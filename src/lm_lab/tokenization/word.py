from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


def _normalize_text(text: str) -> str:
    """
    Normalize common Unicode punctuation to simpler ASCII-like forms.

    This keeps tokenization behavior more stable across corpora that mix
    typographic and plain-text punctuation.

    Args:
        text: Raw input text.

    Returns:
        Normalized text.
    """
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
    """
    Deterministic word-level tokenizer with simple punctuation-aware splitting.

    Notes:
        - Text is normalized before tokenization.
        - Vocabulary is built from sorted unique token pieces.
        - Unknown pieces map to ``<unk>`` during encoding.
        - Special tokens are skipped during decode.
    """

    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def build(cls, text: str) -> "WordTokenizer":
        """
        Build a word tokenizer from corpus text.

        Args:
            text: Corpus used to construct the vocabulary.

        Returns:
            A fully constructed WordTokenizer.
        """
        specials = ["<unk>", "<bos>", "<eos>"]
        pieces = cls._tokenize_text(text)

        vocab = sorted(set(pieces))
        all_tokens = specials + vocab

        stoi = {tok: i for i, tok in enumerate(all_tokens)}
        itos = {i: tok for tok, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        """
        Split normalized text into word-level token pieces.

        Args:
            text: Raw input text.

        Returns:
            Token piece sequence.
        """
        text = _normalize_text(text)
        return _WORD_RE.findall(text)

    @property
    def vocab_size(self) -> int:
        """Return the size of the tokenizer vocabulary."""
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        """
        Encode text into token IDs.

        Args:
            s: Input text.

        Returns:
            Token ID sequence.
        """
        pieces = self._tokenize_text(s)
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(piece, unk_id) for piece in pieces]

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode token IDs back into text using simple spacing rules.

        Args:
            ids: Token ID sequence.

        Returns:
            Reconstructed text.

        Notes:
            - Special tokens are omitted from the output.
            - Spacing is reconstructed heuristically around punctuation, quotes,
              brackets, and newlines.
            - Decode aims for readable text, not exact byte-for-byte inversion
              of the original source string.
        """
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
                    # Opening quote at the start of text or line.
                    out.append('"')
                    quote_open = True
                elif quote_open:
                    # Closing quote attaches to the preceding token.
                    out.append('"')
                    quote_open = False
                else:
                    # Opening quote after normal text receives a leading space.
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
                # Token after an opening quote attaches directly.
                out.append(tok)
            else:
                out.append(" " + tok)

            prev_tok = tok

        text = "".join(out)
        text = text.replace("\n ", "\n")
        return text