from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lm_lab.config.load import TokenizerConfig
from lm_lab.core.model import TransformerLM, TransformerLMConfig
from lm_lab.data.sequence_dataset import SequenceDataset, SequenceDatasetConfig
from lm_lab.tokenization.build import build_tokenizer
from lm_lab.utils.seed import SeedConfig, seed_everything


def main() -> None:
    seed_everything(SeedConfig(seed=42))

    # Tiny corpus
    text = (
        "hello world\n"
        "hello there world\n"
        "hello world again\n"
        "world says hello\n"
    )

    # Tokenizer
    tok_cfg = TokenizerConfig(mode="word")
    tok = build_tokenizer(tok_cfg, text)
    tokens = tok.encode(text)

    # Dataset
    block_size = 8
    ds = SequenceDataset(tokens, SequenceDatasetConfig(block_size=block_size))

    print("tokenizer mode:", tok_cfg.mode)
    print("Vocab size:", tok.vocab_size)
    print("Token length:", len(tokens))
    print("Dataset length:", len(ds))
    print("Tokens:", tokens)

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
        x_list = []
        y_list = []

        for i in range(len(ds)):
            x, y = ds[i]
            x_list.append(x)
            y_list.append(y)

        x_batch = torch.from_numpy(np.stack(x_list)).long()
        y_batch = torch.from_numpy(np.stack(y_list)).long()

        logits = model(x_batch)
        bsz, seqlen, vocab = logits.shape

        loss = F.cross_entropy(
            logits.view(bsz * seqlen, vocab),
            y_batch.view(bsz * seqlen),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step: {step} | loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()