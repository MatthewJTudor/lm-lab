from __future__ import annotations

import math

import torch
import torch.nn as nn

from lm_lab.metrics.basic import perplexity, token_accuracy, grad_norm_total


def test_perplexity_zero_loss_is_one() -> None:
    assert perplexity(0.0) == 1.0


def test_perplexity_increases_with_loss() -> None:
    assert perplexity(2.0) > perplexity(1.0)


def test_token_accuracy_perfect_prediction() -> None:
    logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    targets = torch.tensor([[0, 1]])
    assert token_accuracy(logits, targets) == 1.0


def test_token_accuracy_all_wrong() -> None:
    logits = torch.tensor([[[0.0, 10.0], [10.0, 0.0]]])
    targets = torch.tensor([[0, 1]])
    assert token_accuracy(logits, targets) == 0.0


def test_grad_norm_total_zero_when_no_grads() -> None:
    model = nn.Linear(4, 2)
    assert grad_norm_total(model) == 0.0


def test_grad_norm_total_positive_when_grads_present() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = model(x).sum()
    y.backward()
    assert grad_norm_total(model) > 0.0