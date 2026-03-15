from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _utf8_bytes(text: str) -> list[bytes]:
    return [bytes([b]) for b in text.encode("utf-8")]


def _get_pair_counts(tokens: list[bytes]) -> dict[tuple[bytes, bytes], int]:
    counts: dict[tuple[bytes, bytes], int] = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
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

        tokens = _utf8_bytes(text)
        vocab: set[bytes] = set(tokens)
        merges: list[tuple[bytes, bytes]] = []

        while len(vocab) < vocab_size:
            pair_counts = _get_pair_counts(tokens)
            if not pair_counts:
                break

            # deterministic tie-break: highest count, then lexicographic pair
            best_pair, best_count = max(
                pair_counts.items(),
                key=lambda kv: (kv[1], kv[0]),
            )

            if best_count < 2:
                break

            tokens = _merge_pair(tokens, best_pair)
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
        tokens = _utf8_bytes(s)

        for pair in self.merges:
            tokens = _merge_pair(tokens, pair)

        return [self.stoi[tok] for tok in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        chunks: list[bytes] = []

        for idx in ids:
            tok = self.itos[int(idx)]
            if tok in {self.bos_token, self.eos_token}:
                continue
            chunks.append(tok)

        data = b"".join(chunks)
        return data.decode("utf-8", errors="replace")