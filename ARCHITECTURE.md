# ARCHITECTURE

## 1. Purpose

This document defines the **structural, computational, and behavioral contracts** of the LM-Lab transformer implementation.

It exists to ensure:

- architectural clarity
    
- invariant preservation
    
- safe extension of the system
    

This is a **reference specification**, not a tutorial.

---

## 2. Design Priorities

1. **Determinism**
    
    - Identical inputs + config + seed ⇒ identical outputs
        
2. **Reductionism**
    
    - Each component is independently understandable and testable
        
3. **Separation of Concerns**
    
    - Core model contains no experimental or observational logic
        
4. **Explicit Contracts**
    
    - Tensor shapes, flows, and invariants are defined and enforced
        
5. **Extensibility Without Contamination**
    
    - Hooks and capture systems must not modify core behavior
        

---

## 3. High-Level System Layout

```text
raw text  
→ tokenizer  
→ token ids (B, T)  
→ sequence dataset (windows)  
→ TransformerLM  
→ logits (B, T, V)  
→ loss (training) OR sampling (inference)
```

---

## 4. Module Boundaries

### `core/`

Owns all model computation:

- embeddings
    
- attention
    
- transformer blocks
    
- forward pass
    
- KV-cache
    

**Constraints:**

- no logging
    
- no capture logic
    
- no mutation outside forward computation
    

---

### `tokenization/`

Owns text → token mapping:

- char
    
- word
    
- BPE
    

**Constraints:**

- deterministic vocabulary construction
    
- no model dependencies
    

---

### `data/`

Owns sequence construction:

- sliding windows
    
- batching
    

**Constraints:**

- deterministic slicing
    
- no randomness outside controlled seed
    

---

### `inference/`

Owns decoding policy:

- greedy
    
- temperature
    
- top-k
    
- top-p
    

**Constraints:**

- operates only on logits
    
- does not modify model state
    

---

### `config/`

Owns structured configuration:

- TOML → dataclasses
    

---

### `scripts/`

Own orchestration only:

- training loop
    
- generation entrypoint
    

---

### `hooks/` (reserved)

Future:

- forward observation only
    

---

### `capture/` (reserved)

Future:

- activation storage
    

---

## 5. Tensor Contracts

### Input

- `idx`: `(B, T)`  
    Integer token IDs
    

**Constraints:**

- `T ≤ max_seq_len`
    

---

### Embeddings

- token embedding: `(B, T, C)`
    
- positional embedding: `(B, T, C)`
    

Combined:

```python
x = tok_emb(idx) + pos_emb(idx)
```

---

### Transformer Hidden State

- shape: `(B, T, C)`
    

Invariant:

- preserved across all blocks
    

---

### Attention

Input:

- `(B, T, C)`
    

Internal:

- queries/keys/values: `(B, n_heads, T, head_dim)`
    

Mask:

- causal (no access to future tokens)
    

Output:

- `(B, T, C)`
    

---

### MLP

Input:

- `(B, T, C)`
    

Output:

- `(B, T, C)`
    

---

### Logits

- `(B, T, V)`
    

Where:

- `V = vocab_size`
    

---

## 6. Transformer Block Contract

Each block implements:

```text
x  
→ LayerNorm  
→ causal self-attention  
→ residual add  
→ LayerNorm  
→ MLP  
→ residual add
```

### Invariants

- shape is preserved: `(B, T, C)`
    
- residual path always exists
    
- attention is strictly causal
    

---

## 7. TransformerLM Contract

### Forward

```python
logits = model(idx)
```

Returns:

- `(B, T, V)`
    

---

### Forward with Cache

```python
logits, past_kvs = model.forward_kv(idx, past_kvs, use_cache=True)
```

Returns:

- logits: `(B, T, V)`
    
- updated KV cache
    

---

### Invariants

- forward_kv must match forward for equivalent context
    
- cache path is an optimization, not a semantic change
    

---

## 8. KV-Cache Design

### Purpose

Avoid recomputing attention over previous tokens during generation.

---

### Behavior

1. **Warmup**
    
    - full forward pass builds initial cache
        
2. **Incremental decoding**
    
    - each step appends one token
        
    - only new token is processed
        
3. **Context overflow**
    
    - if sequence exceeds `max_seq_len`:
        
        - cache is discarded
            
        - recomputed from truncated context
            

---

### Invariants

- cached and uncached outputs must match
    
- cache must not change model semantics
    
- cache growth must be monotonic until reset
    

---

## 9. Tokenization Architecture

### Modes

- `char`
    
- `word`
    
- `bpe`
    

---

### Contracts

- deterministic encode/decode
    
- vocabulary fixed after construction
    
- no runtime mutation
    

---

## 10. Training Architecture

### Pipeline

```text
dataset → batch → model → logits → cross-entropy loss
```

---

### Components

- optimizer: AdamW
    
- objective: next-token prediction
    
- batching: deterministic
    

---

### Invariants

- loss decreases under training (smoke test)
    
- no stochastic behavior outside seeded RNG
    

---

## 11. Generation Architecture

### Loop

```text
prompt → model → logits → sampling → next token → repeat
```

---

### Sampling Methods

- greedy
    
- temperature
    
- top-k
    
- top-p
    

---

### Constraints

- operates only on logits
    
- deterministic when sampling disabled or seed fixed
    

---

## 12. Determinism Guarantees

Determinism is enforced across:

- tokenization
    
- dataset construction
    
- model initialization
    
- training loop
    
- generation (under fixed seed)
    

---

### Requirement

Given:

- same config
    
- same seed
    
- same checkpoint
    

Result:

- identical outputs
    

---

## 13. Testing Strategy

Tests validate **structure, not just execution**.

Coverage includes:

- attention correctness
    
- KV-cache equivalence
    
- positional encoding behavior
    
- tokenizer consistency
    
- sampling correctness
    
- training convergence
    

---

## 14. Reserved Extension Points

### Hooks

- forward-pass observation
    
- must not modify tensors
    

---

### Capture

- raw tensor snapshots
    
- no transformation at capture time
    

---

### Future Systems

- diagnostic pipelines
    
- analysis layers (external to core)
    

---

## 15. Non-Goals

The core model explicitly does **not**:

- include experimental logic
    
- perform analysis during forward pass
    
- modify tensors via hooks
    
- depend on external frameworks (e.g., Hydra)
    
- optimize for performance over clarity
    

---

## 16. Guiding Principle

> The architecture must remain:
> 
> - understandable
>     
> - testable
>     
> - deterministic
>     
> - extensible without mutation
>