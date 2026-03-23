from __future__ import annotations

import pytest
import torch

from lm_lab.core.block import TransformerBlock, TransformerBlockConfig
from lm_lab.utils.seed import SeedConfig, seed_everything


def test_block_shape() -> None:
    cfg = TransformerBlockConfig(d_model=16)
    block = TransformerBlock(cfg)

    x = torch.randn(4, 5, 16)
    y = block(x)

    assert y.shape == (4, 5, 16)


def test_block_changes_values() -> None:
    x = torch.randn(2, 5, 16)
    block = TransformerBlock(TransformerBlockConfig(d_model=16))
    y = block(x)
    assert not torch.allclose(x, y)


def test_block_invalid_head_partition_raises() -> None:
    with pytest.raises(ValueError):
        TransformerBlock(TransformerBlockConfig(d_model=15, n_heads=2))


def test_block_deterministic_init_and_forward_under_fixed_seed() -> None:
    seed_everything(SeedConfig(seed=123))
    block1 = TransformerBlock(
        TransformerBlockConfig(d_model=16, n_heads=4, dropout=0.0)
    )

    seed_everything(SeedConfig(seed=123))
    block2 = TransformerBlock(
        TransformerBlockConfig(d_model=16, n_heads=4, dropout=0.0)
    )

    x = torch.randn(2, 5, 16)

    y1 = block1(x)
    y2 = block2(x)

    assert torch.allclose(y1, y2)


def test_block_forward_kv_matches_forward_when_no_cache() -> None:
    seed_everything(SeedConfig(seed=123))

    block = TransformerBlock(
        TransformerBlockConfig(d_model=16, n_heads=4, dropout=0.0)
    )
    x = torch.randn(2, 5, 16)

    y_full = block(x)
    y_kv, present = block.forward_kv(x, past_kv=None, use_cache=False)

    assert present is None
    assert y_kv.shape == y_full.shape
    assert torch.allclose(y_kv, y_full)