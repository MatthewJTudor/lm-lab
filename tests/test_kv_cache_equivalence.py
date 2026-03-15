from __future__ import annotations

import pytest
import torch

from lm_lab.core.model import TransformerLM, TransformerLMConfig
from lm_lab.utils.seed import SeedConfig
from lm_lab.utils.seed import seed_everything


def _greedy_step_full(model: TransformerLM, idx: torch.Tensor) -> int:
    logits = model(idx)
    return int(torch.argmax(logits[0, -1, :]).item())


def _greedy_step_kv(model: TransformerLM, idx_last: torch.Tensor, past_kvs):
    logits, new_kvs = model.forward_kv(idx_last, past_kvs=past_kvs, use_cache=True)
    next_id = int(torch.argmax(logits[0, -1, :]).item())
    return next_id, new_kvs


def _greedy_from_logits(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits[0, -1, :]).item())


def test_kv_cache_matches_uncached_greedy() -> None:
    seed_everything(SeedConfig(seed=123))

    cfg = TransformerLMConfig(
        vocab_size=50,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        pos_mode="learned",
        n_heads=2,
        dropout=0.0,
        tie_embeddings=False,
    )
    model = TransformerLM(cfg)
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    # Warm cache on full prompt and take first next-token from that logits.
    logits, past = model.forward_kv(prompt, past_kvs=None, use_cache=True)

    # Keep within context window to avoid needing cache cropping in the test.
    steps = cfg.max_seq_len - prompt.size(1)

    idx_full = prompt.clone()
    idx_kv = prompt.clone()

    for _ in range(steps):
        # Uncached: run full context window and pick next id
        cond = idx_full[:, -cfg.max_seq_len:]
        nid_full = _greedy_step_full(model, cond)
        idx_full = torch.cat([idx_full, torch.tensor([[nid_full]])], dim=1)

        # Cached: pick next id from most recent logits, then feed only last token
        nid_kv = _greedy_from_logits(logits)
        idx_kv = torch.cat([idx_kv, torch.tensor([[nid_kv]])], dim=1)
        logits, past = model.forward_kv(idx_kv[:, -1:], past_kvs=past, use_cache=True)

        assert nid_full == nid_kv

    # Optional: stronger end condition
    assert torch.equal(idx_full, idx_kv)

def test_kv_cache_requires_incremental_decode_when_past_present() -> None:
    seed_everything(SeedConfig(seed=123))

    cfg = TransformerLMConfig(
        vocab_size=50,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        pos_mode="learned",
        n_heads=2,
        dropout=0.0,
        tie_embeddings=False,
    )
    model = TransformerLM(cfg)
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    _, past = model.forward_kv(prompt, past_kvs=None, use_cache=True)

    # Illegal: feeding more than one token while also providing a past cache
    two_tokens = torch.tensor([[6, 7]], dtype=torch.long)

    with pytest.raises(AssertionError):
        _ = model.forward_kv(two_tokens, past_kvs=past, use_cache=True)