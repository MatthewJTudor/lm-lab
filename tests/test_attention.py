import torch
import pytest

from lm_lab.core.attention import SelfAttention, AttentionConfig


def test_attention_shape_single_head() -> None:
    cfg = AttentionConfig(d_model=16, n_heads=1)
    attn = SelfAttention(cfg)

    x = torch.randn(4, 5, 16)
    y = attn(x)

    assert y.shape == (4, 5, 16)


def test_attention_shape_multi_head() -> None:
    cfg = AttentionConfig(d_model=32, n_heads=4)
    attn = SelfAttention(cfg)

    x = torch.randn(2, 6, 32)
    y = attn(x)

    assert y.shape == (2, 6, 32)


def test_attention_is_causal() -> None:
    """
    If attention is causal, modifying a future token
    must not affect earlier outputs.
    """
    cfg = AttentionConfig(d_model=8, n_heads=1, dropout=0.0)
    attn = SelfAttention(cfg)
    attn.eval()

    x1 = torch.zeros(1, 4, 8)
    x2 = x1.clone()

    # Modify only the last token (future)
    x2[0, 3, :] = 10.0

    y1 = attn(x1)
    y2 = attn(x2)

    # Earlier positions (0,1,2) must match
    assert torch.allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-5)


def test_attention_head_divisibility_error() -> None:
    with pytest.raises(ValueError):
        SelfAttention(AttentionConfig(d_model=30, n_heads=4))