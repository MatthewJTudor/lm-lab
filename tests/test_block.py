import torch

from lm_lab.core.block import TransformerBlock, TransformerBlockConfig


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