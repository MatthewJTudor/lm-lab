from __future__ import annotations

from typing import Iterable, Protocol


class TokenizerProtocol(Protocol):
    """
    Structural interface for tokenizer implementations used in LM-Lab.

    All tokenizer modes must provide:
        - a fixed vocabulary size
        - text -> token ID encoding
        - token ID -> text decoding

    This protocol allows the training and generation pipeline to work against a
    common tokenizer surface without depending on a specific tokenizer class.
    """

    @property
    def vocab_size(self) -> int:
        """Return the size of the tokenizer vocabulary."""
        ...

    def encode(self, s: str) -> list[int]:
        """
        Encode a string into token IDs.

        Args:
            s: Input text.

        Returns:
            Token ID sequence.
        """
        ...

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode token IDs back into text.

        Args:
            ids: Token ID sequence.

        Returns:
            Decoded text.
        """
        ...