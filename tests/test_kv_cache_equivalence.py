from __future__ import annotations

import torch
from dataclasses import replace

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

    idx = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    # warmup
    logits_warm, past = model.forward_kv(idx, past_kvs=None, use_cache=True)

    # keep within context window for now
    N = cfg.max_seq_len - idx.shape[1]

    idx_full = idx.clone()
    idx_kv = idx.clone()

    for t in range(N):
        # uncached
        cond = idx_full[:, -cfg.max_seq_len:]
        nid_full = _greedy_step_full(model, cond)
        idx_full = torch.cat([idx_full, torch.tensor([[nid_full]])], dim=1)

        # cached
        if t == 0:
            # FIRST token comes from warmup logits (last prompt position)
            nid_kv = int(torch.argmax(logits_warm[0, -1, :]).item())
        else:
            # subsequent tokens: feed only the last generated token
            nid_kv, past = _greedy_step_kv(model, idx_kv[:, -1:], past)

        idx_kv = torch.cat([idx_kv, torch.tensor([[nid_kv]])], dim=1)

        assert nid_full == nid_kv