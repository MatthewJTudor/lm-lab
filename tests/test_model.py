import torch

from lm_lab.core.model import TransformerLM, TransformerLMConfig


def test_model_forward_shape() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
    )

    model = TransformerLM(cfg)

    idx = torch.randint(0, 30, (4, 10))  # (B=4, T=10)

    logits = model(idx)

    assert logits.shape == (4, 10, 30)