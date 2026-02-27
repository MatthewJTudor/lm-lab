import torch
import pytest

from lm_lab.core.position import PositionEmbedding, PositionEmbeddingConfig


@pytest.mark.parametrize("mode", ["learned", "sinusoidal"])
def test_position_embedding_shape_device(mode: str) -> None:
    B, T, d = 2, 8, 32
    pe = PositionEmbedding(PositionEmbeddingConfig(max_seq_len=16, d_model=d, mode=mode))

    x = torch.zeros((B, T), dtype=torch.long)
    out = pe(x)

    assert out.shape == (B, T, d)
    assert out.device == x.device
    assert out.dtype == torch.float32  # your sinusoidal buffer is float32; learned emb returns float32 too


def test_sinusoidal_changes_with_position() -> None:
    B, T, d = 1, 8, 32
    pe = PositionEmbedding(PositionEmbeddingConfig(max_seq_len=16, d_model=d, mode="sinusoidal"))

    x = torch.zeros((B, T), dtype=torch.long)
    out = pe(x)[0]  # (T, d)

    # Different positions should not be identical vectors
    assert not torch.allclose(out[0], out[1])

def test_learned_has_parameters_sinusoidal_does_not() -> None:
    learned = PositionEmbedding(PositionEmbeddingConfig(max_seq_len=16, d_model=32, mode="learned"))
    sinus = PositionEmbedding(PositionEmbeddingConfig(max_seq_len=16, d_model=32, mode="sinusoidal"))

    learned_params = [n for n, _ in learned.named_parameters()]
    sinus_params = [n for n, _ in sinus.named_parameters()]

    assert any("embedding" in n for n in learned_params)
    assert all("embedding" not in n for n in sinus_params)

    # sinusoidal should have a buffer named _pe
    buffers = dict(sinus.named_buffers())
    assert "_pe" in buffers
    assert buffers["_pe"].shape == (16, 32)