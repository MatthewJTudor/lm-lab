from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SequenceDatasetConfig:
    """
    Configuration for next-token sequence windowing.

    Attributes:
        block_size: Length of each input sequence window.
    """
    block_size: int


class SequenceDataset:
    """
    Turn a flat token sequence into next-token prediction windows.

    Given tokens:
        [t0, t1, ..., tN]

    Produces samples:
        x = [ti, ..., ti+block_size-1]
        y = [ti+1, ..., ti+block_size]

    Notes:
        - Each sample is a sliding window over the flat token stream.
        - ``x`` and ``y`` are aligned for next-token prediction.
        - The dataset is deterministic for a fixed token sequence.
    """

    def __init__(self, tokens: List[int], cfg: SequenceDatasetConfig) -> None:
        self.tokens = np.array(tokens, dtype=np.int64)
        self.cfg = cfg

        if len(self.tokens) <= cfg.block_size:
            raise ValueError("Token sequence must be longer than block_size")

        # Number of valid sliding windows of length block_size.
        self.length = len(self.tokens) - cfg.block_size

    def __len__(self) -> int:
        """
        Return the number of valid sequence windows in the dataset.
        """
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return one next-token training sample.

        Args:
            idx: Starting index of the sliding window.

        Returns:
            A tuple of:
                - x: input token window of shape (block_size,)
                - y: next-token target window of shape (block_size,)

        Raises:
            IndexError: If idx is outside the valid dataset range.
        """
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        block_size = self.cfg.block_size

        x = self.tokens[idx:idx + block_size]
        y = self.tokens[idx + 1: idx + block_size + 1]

        return x, y