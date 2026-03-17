# CONTRIBUTING

## Purpose

This document defines how changes are allowed to enter the LM-Lab system.

The goal is not just collaboration—it is **preserving architectural integrity while enabling controlled system evolution**.

All contributions must maintain:

- determinism
    
- architectural clarity
    
- separation of concerns
    
- explicit, testable behavior
    

---

## Core Principle

> If a change cannot be clearly explained in terms of tensor flow, invariants, and system boundaries, it should not be merged.

The language model core is intentionally minimal and must remain free of logic that alters its semantic behavior outside its defined scope.

---

## Contribution Scope

### Allowed Contributions

- Bug fixes
    
- Refactoring that preserves behavior
    
- Test additions or improvements
    
- Documentation improvements
    
- Sampling method extensions (`inference/`)
    
- Tokenizer improvements (`tokenization/`)
    
- Performance improvements that do not alter semantics
    
- Diagnostic and evaluation systems **outside core model computation**
    

---

### Disallowed Contributions

The following are **not permitted in the core model (`core/`)**:

- Logging or print statements inside model computation
    
- Embedding diagnostic or analytical computation into the forward path
    
- Experimental logic mixed into model execution
    
- Mutation of tensors through hooks
    
- Hidden side effects or implicit state changes
    
- Breaking determinism (directly or indirectly)
    

---

## Architectural Boundaries

All contributions must respect module ownership:

- `core/` → model computation only
    
- `tokenization/` → text ↔ token mapping
    
- `data/` → dataset construction
    
- `inference/` → sampling logic
    
- `scripts/` → orchestration only
    

Diagnostic, analytical, and experimental systems must be implemented **outside the semantic model core**, unless explicitly designed as non-intrusive observational infrastructure.

---

## Determinism Requirements

Determinism is a **system-level guarantee**, not a preference.

Changes must preserve:

- reproducible outputs under fixed seed
    
- deterministic dataset construction
    
- stable tokenizer behavior
    
- identical results for identical config + checkpoint
    

If stochastic behavior is introduced, it must be:

- explicit
    
- configurable
    
- test-covered
    

---

## Testing Requirements

All contributions must:

- pass the full test suite (`pytest -q`)
    
- include new tests when behavior is added or modified
    

### Required Test Coverage by Area

- **Attention changes**
    
    - must preserve causality
        
    - must maintain shape contracts
        
- **KV-cache changes**
    
    - must include equivalence tests vs uncached forward
        
- **Tokenization changes**
    
    - must preserve encode/decode consistency
        
- **Sampling changes**
    
    - must validate probability constraints and behavior
        
- **Training changes**
    
    - must pass loss-decreasing smoke tests
        

---

## Diagnostics and Evaluation

LM-Lab is designed to support **diagnostic and evaluation systems in later phases**.

These may include:

- standard language model evaluation metrics (loss curves, perplexity, etc.)
    
- internal activation statistics
    
- training stability and gradient diagnostics
    
- generation quality metrics
    
- structured tensor capture pipelines
    
- ARF-based resonance and coherence metrics
    

### Constraints

- diagnostics must not alter logits, gradients, cache behavior, or training semantics
    
- diagnostics must not be embedded inside `core/` unless strictly observational and non-semantic
    
- capture systems must store **raw, untransformed tensors**
    
- all diagnostic systems must preserve determinism when run under fixed conditions
    

### Alignment Principle

When performing analysis:

> Standard metrics and ARF-based metrics should be collected on aligned runs whenever possible  
> (same model, checkpoint, seed, inputs, and conditions).

This ensures meaningful comparison and validation.

---

## Code Guidelines

- Prefer clarity over abstraction
    
- Avoid hidden state
    
- Keep functions single-purpose
    
- Use explicit naming when tensor behavior is non-obvious
    

---

## Commit Guidelines

- Keep commits small and focused
    
- One logical change per commit
    
- Use clear, descriptive messages
    

Preferred style:

```text
feat: add top-p sampling  
fix: correct KV-cache shape handling  
test: add KV-cache equivalence test  
refactor: simplify attention projection
```

---

## Extension Rules

Future systems must follow these constraints:

### Hooks

- observational only
    
- must not modify tensors or gradients
    

### Capture

- store raw tensors only
    
- no transformation during capture
    

### Experimental Systems

- must live outside `core/`
    
- must not alter base model semantics
    

---

## Development Phases

The system evolves in structured phases:

1. **Core LM**
    
    - deterministic, validated transformer baseline
        
2. **Inference & Sampling**
    
    - controlled generation behavior
        
3. **Diagnostics & Capture**
    
    - standard metrics and tensor observation
        
4. **ARF Integration**
    
    - resonance-based diagnostics and comparative analysis
        

Each phase builds on a stable foundation and must not compromise prior guarantees.

---

## Non-Goals

This repository does **not** aim to:

- optimize for production-scale performance
    
- include full training frameworks or orchestration systems
    
- embed analysis directly into model computation
    
- abstract away transformer mechanics
    

---

## Final Note

This project prioritizes:

- understanding over convenience
    
- correctness over speed
    
- structure over feature accumulation
    

> The system must remain simple enough to reason about,  
> yet structured enough to support deep analysis.