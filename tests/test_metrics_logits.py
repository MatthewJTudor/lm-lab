from __future__ import annotations

import torch

from lm_lab.metrics.logits import (
    logit_entropy_mean,
    max_probability_mean,
    confidence_margin_mean,
    next_token_rank_mean,
)


def test_logit_entropy_mean_higher_for_uniform_logits() -> None:
    uniform = torch.zeros(1, 2, 4)
    peaky = torch.tensor([[[10.0, 0.0, 0.0, 0.0],
                           [10.0, 0.0, 0.0, 0.0]]])
    assert logit_entropy_mean(uniform) > logit_entropy_mean(peaky)


def test_max_probability_mean_higher_for_peaky_logits() -> None:
    uniform = torch.zeros(1, 2, 4)
    peaky = torch.tensor([[[10.0, 0.0, 0.0, 0.0],
                           [10.0, 0.0, 0.0, 0.0]]])
    assert max_probability_mean(peaky) > max_probability_mean(uniform)


def test_confidence_margin_mean_higher_for_peaky_logits() -> None:
    tied = torch.zeros(1, 2, 4)
    peaky = torch.tensor([[[10.0, 0.0, 0.0, 0.0],
                           [10.0, 0.0, 0.0, 0.0]]])
    assert confidence_margin_mean(peaky) > confidence_margin_mean(tied)


def test_next_token_rank_mean_is_one_when_target_is_top() -> None:
    logits = torch.tensor([[[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0]]])
    targets = torch.tensor([[0, 1]])
    assert next_token_rank_mean(logits, targets) == 1.0


def test_next_token_rank_mean_worse_when_target_not_top() -> None:
    logits = torch.tensor([[[0.0, 10.0, 5.0]]])
    targets = torch.tensor([[2]])
    assert next_token_rank_mean(logits, targets) > 1.0