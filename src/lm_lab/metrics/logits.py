from __future__ import annotations

import torch


def logit_entropy_mean(logits: torch.Tensor) -> float:
    """
    Compute mean token-level entropy of the model output distribution.

    Args:
        logits: Model output of shape (B, T, V).

    Returns:
        Mean entropy across all tokens.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.mean().item())


def max_probability_mean(logits: torch.Tensor) -> float:
    """
    Compute mean maximum probability across tokens.

    Args:
        logits: Model output of shape (B, T, V).

    Returns:
        Mean of the highest predicted probability per token.
    """
    probs = torch.softmax(logits, dim=-1)
    return float(probs.max(dim=-1).values.mean().item())


def confidence_margin_mean(logits: torch.Tensor) -> float:
    """
    Compute mean confidence margin between top-1 and top-2 probabilities.

    Args:
        logits: Model output of shape (B, T, V).

    Returns:
        Mean difference between the highest and second-highest probabilities.
    """
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    return float(margin.mean().item())


def next_token_rank_mean(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute mean rank of the true next token under the predicted distribution.

    Args:
        logits: Model output of shape (B, T, V).
        targets: Ground-truth token IDs of shape (B, T).

    Returns:
        Mean rank (1 = best) of the true token across all positions.
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_idx = torch.argsort(probs, dim=-1, descending=True)
    ranks = (sorted_idx == targets.unsqueeze(-1)).nonzero(as_tuple=False)[:, -1] + 1
    return float(ranks.float().mean().item())