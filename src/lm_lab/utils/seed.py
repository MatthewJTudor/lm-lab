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
    Seed all RNGs we control.

    This is intentionally minimal right now (no torch yet).
    We'll extend it later with torch + cuda determinism settings.
    """
    if cfg.python_hash_seed:
        # Must be set before Python starts to fully guarantee hash stability,
        # but setting it here still helps for subprocesses and makes intent explicit.
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    random.seed(cfg.seed)

    try:
        import numpy as np  # local import to avoid hard dependency assumptions
    except Exception:
        np = None

    if np is not None:
        np.random.seed(cfg.seed)