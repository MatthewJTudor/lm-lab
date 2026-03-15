# UNDER CONSTRUCTION
# LM-Lab

A minimal, deterministic GPT-style Transformer language model implemented from first principles in PyTorch.

This project focuses on mechanical clarity, architectural correctness, and reproducible training behavior. It is intentionally reductionist: components are implemented incrementally and validated with explicit invariants and tests.

The goal is to understand how modern transformer language models work internally while maintaining a clean separation between the core model and future experimental frameworks.

---

## Design Philosophy

- **Reductionist build order** — Each component is introduced independently and validated before integration.
    
- **Deterministic baseline** — CPU training with controlled seeds for reproducibility.
    
- **Explicit invariants** — Causality, positional correctness, and shape contracts are enforced via tests.
    
- **Clear architectural boundaries** — The language model core contains no experimental or orchestration logic.
    
- **Test-driven hardening** — New features are accompanied by invariants and smoke tests.
    

---

## Implemented Features

- Token embedding layer
    
- Learned positional embeddings
    
- Sinusoidal positional encodings (Vaswani et al., 2017)
    
- Multi-head causal self-attention
    
- Pre-LN transformer blocks
    
- Feedforward MLP (GELU/ReLU)
    
- Optional dropout
    
- Weight tying (embedding ↔ output projection)
    
- Deterministic full-batch training loop
    
- Eval-mode loss snapshot logging
    
- 29 pytest-based correctness tests
    

---

## Repository Structure

```
lm_lab/  
core/ # Transformer components  
config/ # Structured config loading (TOML)  
data/ # Sequence dataset utilities  
tokenization/ # Character tokenizer  
utils/ # Seeding & reproducibility  
scripts/  
train.py # Minimal training entrypoint  
tests/ # Invariant and smoke tests
```

## Running

Install dependencies (Python 3.10+ recommended):

```
pip install -r requirements.txt
```

Run tests:

```
pytest -q
```

Train the model:

```
python scripts/train.py --config configs/run.toml
```

## Determinism & Validation

The training loop uses:

- Explicit seeding
    
- Controlled evaluation snapshots
    
- Causal masking tests
    
- Positional embedding invariants
    
- Smoke tests verifying loss decreases in both learned and sinusoidal modes
    

The model is validated not just for execution but for structural correctness.

---

## Current Status

Core GPT-style architecture is stable.

Next phase:

- Autoregressive generation (greedy + sampling)
    
- KV-cache incremental decoding
    
- Extended sampling controls
    

---

## Educational Purpose

This repository is intended as a learning and engineering exercise in building transformer language models from first principles while maintaining production-quality discipline.