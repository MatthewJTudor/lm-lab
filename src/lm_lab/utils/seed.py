from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedConfig:
    """
    Configuration for reproducible seeding across supported RNG sources.

    Attributes:
        seed: Base seed value applied across Python, NumPy, and PyTorch.
        python_hash_seed: Whether to set ``PYTHONHASHSEED`` for stable hash-based
            iteration behavior across processes.
    """

    seed: int = 1337
    # Python hash randomization affects dict/set iteration order across processes.
    # Setting PYTHONHASHSEED makes this stable.
    python_hash_seed: bool = True


def seed_everything(cfg: SeedConfig) -> None:
    """
    Seed all supported random number generators for reproducible execution.

    This function attempts to seed:
        - Python's ``random`` module
        - NumPy
        - PyTorch
        - CUDA RNGs when CUDA is available

    Notes:
        - This establishes the project's standard deterministic baseline.
        - Full PyTorch deterministic algorithm mode is optional and only enabled
          if the config exposes ``deterministic_torch=True``.
        - Missing optional dependencies are ignored.
    """
    if cfg.python_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    random.seed(cfg.seed)

    try:
        import numpy as np
        np.random.seed(cfg.seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(cfg.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        # Optional strict PyTorch determinism. This may reduce performance but
        # can improve reproducibility when supported by the selected operations.
        if getattr(cfg, "deterministic_torch", False):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    except Exception:
        pass