from __future__ import annotations

import math
import torch

from lm_lab.inference.sampling import top_k_filter, top_p_filter, sample_next_token


def test_top_k_filter_keeps_exactly_k_when_unique() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])  # unique
    out = top_k_filter(logits, k=2)

    kept = torch.isfinite(out)
    assert kept.sum().item() == 2
    assert math.isinf(out[0].item()) and out[0].item() < 0  # masked -> -inf
    assert torch.isfinite(out[4])  # top logit kept


def test_top_p_filter_keeps_at_least_one_token() -> None:
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])  # extremely peaked
    out = top_p_filter(logits, p=0.1)
    kept = torch.isfinite(out)
    assert kept.sum().item() >= 1


def test_top_p_filter_masks_some_tokens_for_reasonable_p() -> None:
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
    out = top_p_filter(logits, p=0.6)
    kept = torch.isfinite(out)
    assert kept.sum().item() < logits.numel()


def test_sample_next_token_greedy() -> None:
    logits = torch.tensor([0.0, 1.0, 0.5])
    tid = sample_next_token(logits, temperature=0.0, top_k=0, top_p=1.0)
    assert tid == 1


def test_sample_next_token_param_validation() -> None:
    logits = torch.tensor([0.0, 1.0, 0.5])

    try:
        sample_next_token(logits, temperature=-0.1)
        assert False, "Expected ValueError for negative temperature"
    except ValueError:
        pass

    try:
        sample_next_token(logits, temperature=0.8, top_p=1.5)
        assert False, "Expected ValueError for top_p > 1"
    except ValueError:
        pass

    try:
        sample_next_token(logits, temperature=0.8, top_k=999)
        assert False, "Expected ValueError for top_k > vocab size"
    except ValueError:
        pass