from __future__ import annotations

import math
import torch
from torch import nn

def perplexity(loss: float) -> float:
    return math.exp(loss)

def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return float(correct.mean().item())

def grad_norm_total(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5