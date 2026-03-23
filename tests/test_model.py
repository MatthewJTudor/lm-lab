from __future__ import annotations

import pytest
import torch

from lm_lab.core.model import TransformerLM, TransformerLMConfig
from lm_lab.utils.seed import SeedConfig, seed_everything


def test_model_forward_shape() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
    )

    model = TransformerLM(cfg)

    idx = torch.randint(0, 30, (4, 10))

    logits = model(idx)

    assert logits.shape == (4, 10, 30)


def test_model_forward_raises_when_seq_exceeds_max_seq_len() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
    )
    model = TransformerLM(cfg)

    idx = torch.randint(0, 30, (2, 17))

    with pytest.raises(ValueError):
        model(idx)


def test_model_forward_kv_warmup_returns_cache_per_layer() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=3,
        n_heads=4,
        dropout=0.0,
        tie_embeddings=False,
    )
    model = TransformerLM(cfg)

    idx = torch.randint(0, 30, (2, 10))
    logits, past_kvs = model.forward_kv(idx, past_kvs=None, use_cache=True)

    assert logits.shape == (2, 10, 30)
    assert len(past_kvs) == 3
    assert all(kv is not None for kv in past_kvs)


def test_model_forward_kv_incremental_decode_shape() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        tie_embeddings=False,
    )
    model = TransformerLM(cfg)

    prompt = torch.randint(0, 30, (1, 6))
    _, past_kvs = model.forward_kv(prompt, past_kvs=None, use_cache=True)

    next_idx = torch.randint(0, 30, (1, 1))
    logits, new_kvs = model.forward_kv(next_idx, past_kvs=past_kvs, use_cache=True)

    assert logits.shape == (1, 1, 30)
    assert len(new_kvs) == 2
    assert all(kv is not None for kv in new_kvs)


def test_model_tie_embeddings_shares_weight_object() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        tie_embeddings=True,
    )
    model = TransformerLM(cfg)

    assert model.head.weight is model.token_emb.embedding.weight


def test_model_untied_embeddings_do_not_share_weight_object() -> None:
    cfg = TransformerLMConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        tie_embeddings=False,
    )
    model = TransformerLM(cfg)

    assert model.head.weight is not model.token_emb.embedding.weight


def test_model_deterministic_init_and_forward_under_fixed_seed() -> None:
    seed_everything(SeedConfig(seed=123))
    model1 = TransformerLM(
        TransformerLMConfig(
            vocab_size=30,
            max_seq_len=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            dropout=0.0,
            tie_embeddings=False,
        )
    )

    seed_everything(SeedConfig(seed=123))
    model2 = TransformerLM(
        TransformerLMConfig(
            vocab_size=30,
            max_seq_len=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            dropout=0.0,
            tie_embeddings=False,
        )
    )

    idx = torch.randint(0, 30, (2, 8))
    y1 = model1(idx)
    y2 = model2(idx)

    assert torch.allclose(y1, y2)