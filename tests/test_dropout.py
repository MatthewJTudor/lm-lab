import torch

from lm_lab.core.model import TransformerLM, TransformerLMConfig


def test_dropout_changes_output_in_train_mode() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=8,
        d_model=32,
        n_layers=1,
        dropout=0.2,
    )
    m = TransformerLM(cfg)
    m.train()

    x = torch.randint(0, 30, (2, 8))

    y1 = m(x)
    y2 = m(x)

    assert not torch.allclose(y1, y2)


def test_dropout_disabled_in_eval_mode() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=8,
        d_model=32,
        n_layers=1,
        dropout=0.2,
    )
    m = TransformerLM(cfg)
    m.eval()

    x = torch.randint(0, 30, (2, 8))

    y1 = m(x)
    y2 = m(x)

    assert torch.allclose(y1, y2)