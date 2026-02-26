import torch

from lm_lab.core.attention import SelfAttention, AttentionConfig


def test_attention_shape() -> None:
    cfg = AttentionConfig(d_model=16)
    attn = SelfAttention(cfg)

    x = torch.randn(4, 5, 16)
    y = attn(x)

    assert y.shape == (4, 5, 16)