from __future__ import annotations

import torch

def logit_entropy_mean(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.mean().item())

def max_probability_mean(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return float(probs.max(dim=-1).values.mean().item())

def confidence_margin_mean(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    return float(margin.mean().item())

def next_token_rank_mean(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    sorted_idx = torch.argsort(probs, dim=-1, descending=True)
    ranks = (sorted_idx == targets.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
    return float(ranks.float().mean().item())