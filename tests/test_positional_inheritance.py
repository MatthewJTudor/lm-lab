import torch

from lm_lab.core.position import PositionEmbeddingConfig, PositionEmbedding

def test_pos_embedding_token_invariant():
    cfg = PositionEmbeddingConfig(max_seq_len=16, d_model=8, mode="learned")
    pos = PositionEmbedding(cfg)

    idx1 = torch.zeros(2, 5, dtype=torch.long)
    idx2 = torch.randint(0, 10, (2, 5), dtype=torch.long)

    out1 = pos(idx1)
    out2 = pos(idx2)

    assert torch.allclose(out1, out2)

def test_pos_embedding_sinusoid_invariant():
    cfg = PositionEmbeddingConfig(max_seq_len=16, d_model=8, mode="sinusoidal")
    pos = PositionEmbedding(cfg)

    idx1 = torch.zeros(2, 5, dtype=torch.long)
    idx2 = torch.randint(0, 10, (2, 5), dtype=torch.long)

    out1 = pos(idx1)
    out2 = pos(idx2)

    assert torch.allclose(out1, out2)

def test_pos_embedding_varies_over_time():
    cfg = PositionEmbeddingConfig(max_seq_len=16, d_model=8, mode="learned")
    pos = PositionEmbedding(cfg)

    idx = torch.zeros(1, 5, dtype=torch.long)
    out = pos(idx)  # (1, 5, 8)

    assert not torch.allclose(out[:, 0, :], out[:, 1, :])

def test_pos_embedding_varies_over_time_sinusoid():
    cfg = PositionEmbeddingConfig(max_seq_len=16, d_model=8, mode="sinusoidal")
    pos = PositionEmbedding(cfg)

    idx = torch.zeros(1, 5, dtype=torch.long)
    out = pos(idx)  # (1, 5, 8)

    assert not torch.allclose(out[:, 0, :], out[:, 1, :])