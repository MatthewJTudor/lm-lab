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
    \n+                            |   # newline runs
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
    Build a reversible byte-to-unicode mapping in the style of GPT-2.

    Each byte value 0..255 is mapped to a unique unicode character so that:
        - every byte is representable as a single symbol
        - the mapping is reversible
        - common visible characters remain readable where possible

    Returns:
        Mapping from byte value to unicode symbol.
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


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {ch: b for b, ch in _BYTE_ENCODER.items()}


def unicode_to_bytes() -> dict[str, int]:
    """
    Return the reverse unicode-to-byte mapping.

    Returns:
        Mapping from unicode symbol back to byte value.
    """
    return _BYTE_DECODER.copy()


def _chunk_text(text: str) -> list[str]:
    """
    Split text into GPT-style pre-BPE chunks.

    Args:
        text: Raw input text.

    Returns:
        Chunk sequence preserving whitespace-sensitive boundaries.
    """
    return _CHUNK_RE.findall(text)


def _chunk_to_mapped_tokens(chunk: str) -> list[str]:
    """
    Convert a text chunk into mapped unicode byte symbols.

    Each UTF-8 byte becomes one mapped unicode symbol so BPE can operate over a
    reversible symbol stream.

    Args:
        chunk: Input text chunk.

    Returns:
        Sequence of mapped unicode symbols.
    """
    return [_BYTE_ENCODER[b] for b in chunk.encode("utf-8")]


def _chunk_text_to_mapped_tokens(text: str) -> list[list[str]]:
    """
    Chunk text and convert each chunk into mapped unicode byte symbols.

    Args:
        text: Raw input text.

    Returns:
        List of token-symbol chunks.
    """
    return [_chunk_to_mapped_tokens(chunk) for chunk in _chunk_text(text)]


def _get_pair_counts(
    chunks: list[list[TokenSymbol]],
) -> dict[tuple[TokenSymbol, TokenSymbol], int]:
    """
    Count adjacent token-symbol pairs across all chunks.

    Args:
        chunks: Token-symbol chunks.

    Returns:
        Pair-frequency mapping.
    """
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
    """
    Merge one target pair wherever it appears in a single chunk.

    Args:
        tokens: Token-symbol sequence for one chunk.
        pair: Adjacent symbol pair to merge.

    Returns:
        New chunk with the pair merged where applicable.
    """
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
    """
    Apply one pair merge across all chunks.

    Args:
        chunks: Token-symbol chunks.
        pair: Adjacent symbol pair to merge.

    Returns:
        Updated chunk list.
    """
    return [_merge_pair_in_chunk(tokens, pair) for tokens in chunks]


def inspect_chunks(text: str) -> list[str]:
    """
    Expose the initial GPT-style chunking for inspection/debugging.

    Args:
        text: Raw input text.

    Returns:
        Chunk sequence produced before byte-symbol mapping.
    """
    return _chunk_text(text)


@dataclass(frozen=True)
class MergeStat:
    """
    Inspection record for one learned BPE merge.
    """

    rank: int
    left: str
    right: str
    merged: str


@dataclass(frozen=True)
class TokenStat:
    """
    Inspection record for one token-frequency summary entry.
    """

    token: str
    count: int
    byte_length: int


@dataclass(frozen=True)
class BPETokenizer:
    """
    Deterministic byte-pair tokenizer with GPT-style byte-to-unicode mapping.

    Notes:
        - Text is first chunked into whitespace-sensitive pieces.
        - Chunks are converted into reversible byte-level unicode symbols.
        - BPE merges are learned deterministically from pair frequencies.
        - Special tokens are excluded during decode.
    """

    stoi: dict[str, int]
    itos: dict[int, str]
    merges: list[tuple[str, str]]
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def build(cls, text: str, vocab_size: int = 512) -> "BPETokenizer":
        """
        Build a BPE tokenizer from corpus text.

        Args:
            text: Corpus used to learn the merge rules.
            vocab_size: Target vocabulary size including learned merged symbols.

        Returns:
            A fully constructed BPETokenizer.

        Raises:
            ValueError: If vocab_size is less than 2.
        """
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

            # Deterministic tie-break: highest count, then lexicographic pair.
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
        """Return the size of the tokenizer vocabulary."""
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        """
        Encode text into BPE token IDs.

        Args:
            s: Input text.

        Returns:
            Token ID sequence.
        """
        chunks = _chunk_text_to_mapped_tokens(s)

        for pair in self.merges:
            chunks = _merge_pair_in_chunks(chunks, pair)

        flat_tokens: list[str] = []
        for tokens in chunks:
            flat_tokens.extend(tokens)

        return [self.stoi[tok] for tok in flat_tokens]

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode BPE token IDs back into text.

        Args:
            ids: Token ID sequence.

        Returns:
            Decoded UTF-8 text.

        Notes:
            - Special tokens are omitted from the output.
            - Invalid UTF-8 byte sequences are decoded with replacement.
        """
        symbols: list[str] = []

        for idx in ids:
            tok = self.itos[int(idx)]
            if tok in {self.bos_token, self.eos_token}:
                continue
            symbols.append(tok)

        data = bytes(_BYTE_DECODER[ch] for tok in symbols for ch in tok)
        return data.decode("utf-8", errors="replace")

    def inspect_merges(self, top_n: int = 20) -> list[MergeStat]:
        """
        Return the first learned merges in rank order.

        Args:
            top_n: Maximum number of merge records to return.

        Returns:
            Merge summary records.

        Raises:
            ValueError: If top_n is negative.
        """
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
        """
        Summarize the most frequent encoded tokens for a given text sample.

        Args:
            text: Input text to analyze.
            top_n: Maximum number of token records to return.

        Returns:
            Token frequency summary records.

        Raises:
            ValueError: If top_n is negative.
        """
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
        """
        Return vocabulary entries and their symbol lengths.

        Returns:
            List of (token, length) tuples for non-special vocabulary items.
        """
        toks: list[tuple[str, int]] = []
        for tok_id in sorted(self.itos):
            tok = self.itos[tok_id]
            if tok in {self.bos_token, self.eos_token}:
                continue
            toks.append((tok, len(tok)))
        return toks