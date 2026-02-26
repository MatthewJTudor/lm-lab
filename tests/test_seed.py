from lm_lab.utils.seed import SeedConfig, seed_everything
import random
import numpy as np


def test_seed_everything_reproducible() -> None:
    seed_everything(SeedConfig(seed=123))

    a1 = random.random()
    b1 = np.random.rand()

    seed_everything(SeedConfig(seed=123))

    a2 = random.random()
    b2 = np.random.rand()

    assert a1 == a2
    assert b1 == b2