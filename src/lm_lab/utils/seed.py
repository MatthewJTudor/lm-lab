from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 1337
    # Python hash randomization affects dict/set iteration order across processes.
    # Setting PYTHONHASHSEED makes this stable.
    python_hash_seed: bool = True


def seed_everything(cfg: SeedConfig) -> None:
    """
    Seed all RNGs we control (python, numpy, torch).
    Safe baseline for CPU. CUDA determinism optional.
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

        # Optional: full determinism (slower but strict)
        if getattr(cfg, "deterministic_torch", False):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    except Exception:
        pass

