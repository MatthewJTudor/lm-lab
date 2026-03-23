from __future__ import annotations

import math
import torch
from torch import nn


def perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: Scalar loss value.

    Returns:
        Perplexity = exp(loss).
    """
    return math.exp(loss)


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute next-token prediction accuracy.

    Args:
        logits: Model output of shape (B, T, V).
        targets: Ground-truth token IDs of shape (B, T).

    Returns:
        Mean token-level accuracy.
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return float(correct.mean().item())


def grad_norm_total(model: nn.Module) -> float:
    """
    Compute total L2 norm of model gradients.

    Args:
        model: Model with gradients already computed.

    Returns:
        Scalar L2 norm across all parameter gradients.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5