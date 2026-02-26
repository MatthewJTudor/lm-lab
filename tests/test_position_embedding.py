import torch

from lm_lab.core.position import PositionEmbedding, PositionEmbeddingConfig


def test_position_embedding_shape() -> None:
    cfg = PositionEmbeddingConfig(max_seq_len=10, d_model=16)
    pos = PositionEmbedding(cfg)

    x = torch.zeros((4, 5), dtype=torch.long)
    y = pos(x)

    assert y.shape == (4, 5, 16)