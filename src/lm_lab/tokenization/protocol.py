from __future__ import annotations

from typing import Iterable, Protocol


class TokenizerProtocol(Protocol):
    @property
    def vocab_size(self) -> int:
        ...

    def encode(self, s: str) -> list[int]:
        ...

    def decode(self, ids: Iterable[int]) -> str:
        ...