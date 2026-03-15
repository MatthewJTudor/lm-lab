from __future__ import annotations

import torch
import torch.nn.functional as F

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    logits: (V,)
    Keep only top-k logits; set the rest to -inf.
    """
    if k <= 0  or k >= logits.numel():
        return logits

    v, _ = torch.topk(logits, k)
    kth = v[-1]
    neg_inf = torch.full_like(logits, float("-inf"))
    return torch.where(logits >= kth, logits, neg_inf)

def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering.
    Keeps the smallest set of tokens with cumulative prob >= p.
    logits: (V,)
    returns: (V,) logits with filtered tokens set to -inf
    """
    if p <= 0.0 or p >= 1.0:
        return logits  # no filtering

    # Convert to probs
    probs = F.softmax(logits, dim=-1)

    # Sort descending
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    cum = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens while cum prob <= p, but always keep at least 1 token
    keep = cum <= p
    keep[0] = True
    # shift right so we include the first above-threshold token
    prefix = keep.new_ones((1,), dtype=torch.bool)  # same device
    keep = torch.cat([prefix, keep[:-1]])

    # Map keep mask back to original index space
    keep_idx = sorted_idx[keep]

    filtered = torch.full_like(logits, float("-inf"))
    filtered[keep_idx] = logits[keep_idx]
    return filtered

def sample_next_token(
    logits: torch.Tensor,   # (V,)
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    if logits.dim() != 1:
        raise ValueError(f"Expected logits shape (V,), got {tuple(logits.shape)}")

    # Greedy
    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")

    if top_p < 0.0 or top_p > 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")

    V = logits.numel()
    if top_k < 0 or top_k > V:
        raise ValueError(f"top_k must be between 0 and vocab size (got {top_k}, vocab={V})")

    if 0.0 < temperature < 1e-6:
        temperature = 1e-6

    scaled = logits / temperature

    if top_k and top_k > 0:
        scaled = top_k_filter(scaled, k=top_k)

    if top_p and top_p < 1.0:
        scaled = top_p_filter(scaled, p=top_p)

    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())