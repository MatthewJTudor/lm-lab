from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


# Simple GPT-like chunking:
# - words optionally prefixed by spaces
# - numbers optionally prefixed by spaces
# - punctuation optionally prefixed by spaces
# - newline runs
# - remaining whitespace runs
_CHUNK_RE = re.compile(
    r"""
    \n+                             |   # newline runs
    [ ]+[^\W\d_]+                  |   # space-prefixed unicode words
    [^\W\d_]+                      |   # unicode words at start
    [ ]+\d+                        |   # space-prefixed numbers
    \d+                            |   # numbers at start
    [ ]+[^\w\s]+                   |   # space-prefixed punctuation/symbols
    [^\w\s]+                       |   # punctuation/symbols at start
    [ ]+                               # leftover spaces
    """,
    re.VERBOSE | re.UNICODE,
)


def _chunk_text(text: str) -> list[str]:
    return _CHUNK_RE.findall(text)


def _chunk_to_byte_tokens(chunk: str) -> list[bytes]:
    return [bytes([b]) for b in chunk.encode("utf-8")]


def _chunk_text_to_tokens(text: str) -> list[list[bytes]]:
    return [_chunk_to_byte_tokens(chunk) for chunk in _chunk_text(text)]


def _get_pair_counts(chunks: list[list[bytes]]) -> dict[tuple[bytes, bytes], int]:
    counts: dict[tuple[bytes, bytes], int] = {}
    for tokens in chunks:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair_in_chunk(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    merged: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def _merge_pair_in_chunks(
    chunks: list[list[bytes]],
    pair: tuple[bytes, bytes],
) -> list[list[bytes]]:
    return [_merge_pair_in_chunk(tokens, pair) for tokens in chunks]


@dataclass(frozen=True)
class BPETokenizer:
    stoi: dict[bytes, int]
    itos: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    bos_token: bytes = b"<bos>"
    eos_token: bytes = b"<eos>"

    @classmethod
    def build(cls, text: str, vocab_size: int = 512) -> "BPETokenizer":
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")

        chunks = _chunk_text_to_tokens(text)
        vocab: set[bytes] = set()

        for tokens in chunks:
            vocab.update(tokens)

        merges: list[tuple[bytes, bytes]] = []

        while len(vocab) < vocab_size:
            pair_counts = _get_pair_counts(chunks)
            if not pair_counts:
                break

            # deterministic tie-break: highest count, then lexicographic pair
            best_pair, best_count = max(
                pair_counts.items(),
                key=lambda kv: (kv[1], kv[0]),
            )

            if best_count < 2:
                break

            chunks = _merge_pair_in_chunks(chunks, best_pair)
            merged_tok = best_pair[0] + best_pair[1]

            vocab.add(merged_tok)
            merges.append(best_pair)

        specials = [cls.bos_token, cls.eos_token]
        all_tokens = specials + sorted(vocab)

        stoi = {tok: i for i, tok in enumerate(all_tokens)}
        itos = {i: tok for tok, i in stoi.items()}

        return cls(
            stoi=stoi,
            itos=itos,
            merges=merges,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        chunks = _chunk_text_to_tokens(s)

        for pair in self.merges:
            chunks = _merge_pair_in_chunks(chunks, pair)

        flat_tokens: list[bytes] = []
        for tokens in chunks:
            flat_tokens.extend(tokens)

        return [self.stoi[tok] for tok in flat_tokens]

    def decode(self, ids: Iterable[int]) -> str:
        chunks: list[bytes] = []

        for idx in ids:
            tok = self.itos[int(idx)]
            if tok in {self.bos_token, self.eos_token}:
                continue
            chunks.append(tok)

        data = b"".join(chunks)
        return data.decode("utf-8", errors="replace")