from __future__ import annotations

import torch
import torch.nn.functional as F

import numpy as np

from lm_lab.utils.seed import seed_everything, SeedConfig
from lm_lab.tokenization.char import CharTokenizer
from lm_lab.data.sequence_dataset import SequenceDataset, SequenceDatasetConfig
from lm_lab.core.model import TransformerLM, TransformerLMConfig

def main() -> None:
    seed_everything(SeedConfig(seed=42))

    # Tiny corpus
    text = "hello world\n"

    # Tokenizer
    tok = CharTokenizer.build(text)
    tokens = tok.encode(text)

    # Dataset
    block_size = 8
    ds = SequenceDataset(tokens, SequenceDatasetConfig(block_size=block_size))

    print("Token length:", len(tokens))
    print("Dataset length:", len(ds))

    # Model config
    cfg = TransformerLMConfig(
        vocab_size=tok.vocab_size,
        max_seq_len=block_size,
        d_model=64,
        n_layers=2,
    )

    model = TransformerLM(cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Simple full-batch training
    for step in range(300):
        # Use all windows as one batch (overfitting)
        x_list = []
        y_list = []

        for i in range(len(ds)):
            x, y = ds[i]
            x_list.append(x)
            y_list.append(y)

        x_batch = torch.from_numpy(np.stack(x_list)).long()
        y_batch = torch.from_numpy(np.stack(y_list)).long()

        logits = model(x_batch)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            y_batch.view(B * T),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f'step: {step} | loss: {loss.item():.4f}')

if __name__ == '__main__':
    main()