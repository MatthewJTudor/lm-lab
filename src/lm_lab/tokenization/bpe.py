from __future__ import annotations

from collections import Counter
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

TokenSymbol = bytes | str

def bytes_to_unicode() -> dict[int, str]:
    """
    Reversible byte -> unicode mapping in the style of GPT-2.

    Maps each byte value 0..255 to a unique unicode character so that:
    - every byte is representable as a single unicode symbol
    - the mapping is reversible
    - common visible characters stay readable where possible
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {b: chr(c) for b, c in zip(bs, cs)}

def unicode_to_bytes() -> dict[str, int]:
    """
    Reverse mapping for bytes_to_unicode().
    """
    enc = bytes_to_unicode()
    return {ch: b for b, ch in enc.items()}

def _chunk_text(text: str) -> list[str]:
    return _CHUNK_RE.findall(text)

def _chunk_to_mapped_tokens(chunk: str) -> list[str]:
    """
    Convert a text chunk into GPT-style mapped unicode symbols,
    one symbol per UTF-8 byte.
    """
    byte_encoder = bytes_to_unicode()
    return [byte_encoder[b] for b in chunk.encode("utf-8")]


def _chunk_text_to_mapped_tokens(text: str) -> list[list[str]]:
    return [_chunk_to_mapped_tokens(chunk) for chunk in _chunk_text(text)]

def _get_pair_counts(
    chunks: list[list[TokenSymbol]],
) -> dict[tuple[TokenSymbol, TokenSymbol], int]:
    counts: dict[tuple[TokenSymbol, TokenSymbol], int] = {}

    for tokens in chunks:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] = counts.get(pair, 0) + 1

    return counts

def _merge_pair_in_chunk(
    tokens: list[TokenSymbol],
    pair: tuple[TokenSymbol, TokenSymbol],
) -> list[TokenSymbol]:
    merged: list[TokenSymbol] = []
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
    chunks: list[list[TokenSymbol]],
    pair: tuple[TokenSymbol, TokenSymbol],
) -> list[list[TokenSymbol]]:
    return [_merge_pair_in_chunk(tokens, pair) for tokens in chunks]

def inspect_chunks(text: str) -> list[str]:
    return _chunk_text(text)

@dataclass(frozen=True)
class MergeStat:
    rank: int
    left: str
    right: str
    merged: str

@dataclass(frozen=True)
class TokenStat:
    token: str
    count: int
    byte_length: int

@dataclass(frozen=True)
class BPETokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    merges: list[tuple[str, str]]
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def build(cls, text: str, vocab_size: int = 512) -> "BPETokenizer":
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")

        chunks = _chunk_text_to_mapped_tokens(text)
        vocab: set[str] = set()

        for tokens in chunks:
            vocab.update(tokens)

        merges: list[tuple[str, str]] = []

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
        chunks = _chunk_text_to_mapped_tokens(s)

        for pair in self.merges:
            chunks = _merge_pair_in_chunks(chunks, pair)

        flat_tokens: list[str] = []
        for tokens in chunks:
            flat_tokens.extend(tokens)

        return [self.stoi[tok] for tok in flat_tokens]

    def decode(self, ids: Iterable[int]) -> str:
        byte_decoder = unicode_to_bytes()
        symbols: list[str] = []

        for idx in ids:
            tok = self.itos[int(idx)]
            if tok in {self.bos_token, self.eos_token}:
                continue
            symbols.append(tok)

        data = bytes(byte_decoder[ch] for tok in symbols for ch in tok)
        return data.decode("utf-8", errors="replace")

    def inspect_merges(self, top_n: int = 20) -> list[MergeStat]:
        if top_n < 0:
            raise ValueError("top_n must be >= 0")

        out: list[MergeStat] = []
        for i, (left, right) in enumerate(self.merges[:top_n]):
            out.append(
                MergeStat(
                    rank=i,
                    left=left,
                    right=right,
                    merged=left + right,
                )
            )
        return out

    def inspect_token_frequencies(self, text: str, top_n: int = 20) -> list[TokenStat]:
        if top_n < 0:
            raise ValueError("top_n must be >= 0")

        ids = self.encode(text)
        counts = Counter(ids)

        stats: list[TokenStat] = []
        for tok_id, count in counts.most_common(top_n):
            tok = self.itos[tok_id]
            stats.append(
                TokenStat(
                    token=tok,
                    count=count,
                    byte_length=len(tok),
                )
            )
        return stats

    def inspect_vocab_token_lengths(self) -> list[tuple[str, int]]:
        toks: list[tuple[str, int]] = []
        for tok_id in sorted(self.itos):
            tok = self.itos[tok_id]
            if tok in {self.bos_token, self.eos_token}:
                continue
            toks.append((tok, len(tok)))
        return toks