import torch
import pytest

from lm_lab.core.model import TransformerLM, TransformerLMConfig


@pytest.mark.parametrize("pos_mode", ["learned", "sinusoidal"])
def test_model_forward_shape_with_pos_mode(pos_mode: str) -> None:
    B, T = 2, 8
    vocab = 20

    cfg = TransformerLMConfig(
        vocab_size=vocab,
        max_seq_len=T,
        d_model=32,
        n_layers=2,
        pos_mode=pos_mode,
    )
    model = TransformerLM(cfg)

    idx = torch.randint(0, vocab, (B, T), dtype=torch.long)
    logits = model(idx)

    assert logits.shape == (B, T, vocab)