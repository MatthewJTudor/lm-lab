from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply top-k filtering to a 1D logit vector.

    Args:
        logits: Logits of shape (V,).
        k: Number of highest-logit entries to keep.

    Returns:
        Logits of shape (V,) with all non-top-k entries set to ``-inf``.

    Notes:
        - If ``k <= 0`` or ``k >= V``, the input logits are returned unchanged.
        - This function does not sample; it only masks logits.
    """
    if k <= 0 or k >= logits.numel():
        return logits

    v, _ = torch.topk(logits, k)
    kth = v[-1]
    neg_inf = torch.full_like(logits, float("-inf"))
    return torch.where(logits >= kth, logits, neg_inf)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to a 1D logit vector.

    Args:
        logits: Logits of shape (V,).
        p: Cumulative probability threshold.

    Returns:
        Logits of shape (V,) with filtered entries set to ``-inf``.

    Notes:
        - Keeps the smallest set of tokens whose cumulative probability mass
          reaches or exceeds ``p``.
        - If ``p <= 0.0`` or ``p >= 1.0``, the input logits are returned
          unchanged.
    """
    if p <= 0.0 or p >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)

    # Sort by probability so cumulative mass can be thresholded.
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    # Keep all tokens up to the threshold and also include the first token
    # that crosses the threshold, ensuring at least one token survives.
    keep = cum <= p
    keep[0] = True
    prefix = keep.new_ones((1,), dtype=torch.bool)
    keep = torch.cat([prefix, keep[:-1]])

    # Map the keep mask back to the original vocabulary index space.
    keep_idx = sorted_idx[keep]

    filtered = torch.full_like(logits, float("-inf"))
    filtered[keep_idx] = logits[keep_idx]
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample or select the next token from a 1D logit vector.

    Args:
        logits: Logits of shape (V,).
        temperature: Sampling temperature. ``0.0`` selects greedy decode.
        top_k: Optional top-k filtering parameter.
        top_p: Optional nucleus sampling threshold.

    Returns:
        Selected token ID as an integer.

    Raises:
        ValueError: If logits shape or sampling arguments are invalid.

    Notes:
        - ``temperature == 0.0`` performs greedy decoding.
        - For ``temperature > 0.0``, logits are scaled and optionally filtered
          before multinomial sampling.
    """
    if logits.dim() != 1:
        raise ValueError(f"Expected logits shape (V,), got {tuple(logits.shape)}")

    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")

    if top_p < 0.0 or top_p > 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")

    V = logits.numel()
    if top_k < 0 or top_k > V:
        raise ValueError(f"top_k must be between 0 and vocab size (got {top_k}, vocab={V})")

    # Clamp extremely small positive temperatures to avoid unstable scaling.
    if 0.0 < temperature < 1e-6:
        temperature = 1e-6

    scaled = logits / temperature

    if top_k and top_k > 0:
        scaled = top_k_filter(scaled, k=top_k)

    if top_p and top_p < 1.0:
        scaled = top_p_filter(scaled, p=top_p)

    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())