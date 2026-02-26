from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

@dataclass(frozen=True)
class SequenceDatasetConfig:
    block_size: int

class SequenceDataset:
    """
    Turns a flat token sequence into (x, y) training pairs
    for next token prediction

    Given tokens:
        x = [t0, t1, ..., tN]

    Produces samples:
        x = [ti, ..., ti+block_size-1]
        y = [ti+1, ..., ti+block_size]
    """

    def __init__(self, tokens: List[int], cfg: SequenceDatasetConfig) -> None:
        self.tokens = np.array(tokens, dtype=np.int64)
        self.cfg = cfg

        if len(self.tokens) <= cfg.block_size:
            raise ValueError("Token sequence must be longer than block_size")

        # Number of valid windows
        self.length = len(self.tokens) - cfg.block_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if not (0 <= idx < self.length):
            raise IndexError(idx)

        block_size = self.cfg.block_size

        x = self.tokens[idx:idx + block_size]
        y = self.tokens[idx + 1 : idx + block_size + 1]

        return x, y