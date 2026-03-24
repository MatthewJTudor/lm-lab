# LM-Lab

A minimal, deterministic GPT-style Transformer language model built from first principles in PyTorch.

This project focuses on **mechanical clarity**, **reproducibility**, and **architectural correctness**. Every component is implemented incrementally, validated with explicit invariants, and kept strictly separate from experimental logic.

It serves as both:

- a **clean reference implementation** of a transformer LM
    
- a **stable foundation** for advanced experimentation (hooks, structured capture, ARF)
    

---

## Why This Project Exists

Most transformer repos optimize for scale or convenience.

This one optimizes for:

- **understanding** (every component is explicit)
    
- **control** (deterministic behavior)
    
- **verification** (test-backed invariants)
    

If something works, you should be able to explain _why_.

---

## Design Principles

- **Reductionist construction**  
    Build → validate → integrate
    
- **Deterministic by default**  
    Reproducible runs across seeds, configs, and environments
    
- **Explicit invariants**  
    Causality, KV-cache correctness, and positional behavior are enforced
    
- **Strict separation of concerns**  
    Core LM ≠ experimentation layer
    
- **Test-first hardening**  
    Features are only complete when proven
    

---

## Features

### Transformer Core

- Token embeddings
    
- Learned + sinusoidal positional encodings
    
- Multi-head causal self-attention
    
- KV-cache (incremental decoding)
    
- Pre-LN transformer blocks
    
- Feedforward MLP (GELU / ReLU)
    
- Residual connections + LayerNorm
    
- Weight tying (optional)
    

### Generation

- Greedy decoding
    
- Temperature scaling
    
- Top-k sampling
    
- Top-p (nucleus) sampling
    

### Tokenization

- Character tokenizer (deterministic baseline)
    
- Word tokenizer (regex-based)
    
- Byte Pair Encoding (BPE)
    

### Training System

- Cross-entropy objective
    
- AdamW optimizer
    
- Config-driven runs (TOML → dataclasses)
    
- Deterministic batching
    
- Eval loss snapshot logging
    
- Checkpoint + config persistence
    

### Reproducibility

- Global seed control (Python / NumPy / Torch)
    
- Deterministic dataset construction
    
- Fully reproducible runs
    

### Testing

- **115 pytest tests**
    
    - attention correctness
        
    - KV-cache equivalence
        
    - positional invariants
        
    - tokenizer validation
        
    - sampling correctness
        
    - training smoke tests
        

---

## Repository Structure

```
lm-lab/
├── configs/
├── data/
├── docs/
├── runs/
├── scripts/
├── src/lm_lab/
│   ├── core/
│   ├── tokenization/
│   ├── data/
│   ├── inference/
│   ├── config/
│   ├── utils/
│   ├── capture/     # structured tensor capture (observational only)
│   └── hooks/       # forward-pass observation (non-intrusive)
├── tests/
└── requirements.txt
```

Full structure: `docs/repo_structure.md`

---

## Quick Start

### Install

```Bash
pip install -r requirements.txt
```

**Run tests**

```Bash
pytest -q
```

**Train**

```Bash
python scripts/train.py --config configs/run.toml --save
```

### Generate

```python 
scripts/generate.py \  
  --config configs/run.toml \  
  --prompt "Alice " \  
  --temperature 0.7 \  
  --top_p 0.9
```

---

## Configuration

All runs are defined via `configs/run.toml`.

Example:

```TOML
[model]  
d_model = 128  
n_layers = 4  
n_heads = 4  
max_seq_len = 128  
  
[tokenizer]  
mode = "bpe"  
  
[train]  
steps = 4000  
lr = 1e-3  
batch_size = 64
```

Configs are parsed into typed dataclasses for safety and clarity.

---

## Determinism & Validation

Determinism is enforced as a **system property**, not a best effort:

- Fixed RNG seeds
    
- Deterministic data slicing
    
- No hidden randomness
    

Correctness is verified structurally:

- causal masking
    
- KV-cache vs full forward equivalence
    
- positional encoding invariants
    
- sampling behavior validation
    
- loss convergence checks
    

---

## Current Status

**Stable baseline complete**

- Transformer core validated
    
- KV-cache verified
    
- Sampling implemented (top-k, top-p)
    
- Tokenization system complete
    
- Full test suite passing (100+ tests)
    

---

## Roadmap

### Near-term

- Hook system (non-intrusive forward observation)
    
- Activation capture pipeline
    
- Structured snapshot storage

- Capture records full tensors at named tap points; all slicing and analysis occur downstream.
    

### Experimental (separate layer)

- Internal tensor diagnostics
    
- Resonance-based metrics (ARF)
    
- Model behavior analysis tooling
    

---

## Educational Value

This project is designed to make transformer systems:

- **inspectable**
    
- **predictable**
    
- **understandable**
    

It prioritizes **clarity over abstraction** and **correctness over convenience**.

---

## Guiding Principle

> If a system cannot be explained, it cannot be trusted.  
> If it cannot be reproduced, it cannot be validated.