import torch

from lm_lab.core.embedding import TokenEmbedding, EmbeddingConfig


def test_embedding_shape() -> None:
    cfg = EmbeddingConfig(vocab_size=20, d_model=16)
    emb = TokenEmbedding(cfg)

    x = torch.randint(0, 20, (4, 5))  # (batch=4, seq_len=5)
    y = emb(x)

    assert y.shape == (4, 5, 16)