from __future__ import annotations

import torch
from lm_lab.inference.sampling import sample_next_token


def test_sampling_reproducible_with_fixed_torch_seed() -> None:
    logits = torch.tensor([0.0, 1.0, 0.5])

    torch.manual_seed(123)
    out1 = [sample_next_token(logits, temperature=0.8, top_k=0, top_p=1.0) for _ in range(50)]

    torch.manual_seed(123)
    out2 = [sample_next_token(logits, temperature=0.8, top_k=0, top_p=1.0) for _ in range(50)]

    assert out1 == out2