# LM-Lab Knob Triage v0

## Purpose

This document classifies the knobs already present in LM-Lab by their experimental role.

The goal is to separate:

- **baseline-defining knobs**
- **regime-inducing knobs**
- **deferred knobs**

so the project can evolve in a deliberate order without overbuilding observability or contaminating the core model.

This triage is for **experimental planning**, not for changing core architecture. It follows the current LM-Lab design priorities of determinism, clarity, and separation of concerns.

---

## Guiding Principle

A knob should be promoted only when there is a clear reason to vary it.

The main question is not:

> what can be toggled?

The main question is:

> which existing knobs are most likely to create diagnostically meaningful behavioral differences while preserving interpretability?

This keeps the system aligned with the metrics-first sequencing already established in the design docs.

---

## Classification Categories

### 1. Baseline-Defining Knobs

These define the identity of the current LM instance.

Changing them is usually better interpreted as creating a **different baseline model**, not a simple regime variation within the same model family.

These are important, but they should usually be held fixed during early regime analysis.

### 2. Regime-Inducing Knobs

These are the best candidates for controlled experiments.

They are the knobs most likely to create observable changes in training dynamics or generation behavior while keeping causal interpretation reasonably clean.

These should be the first knobs used once the metrics layer is active.

### 3. Deferred Knobs

These knobs may become useful later, but they are lower priority for the present phase.

Reasons to defer include:

- weaker diagnostic value
- harder interpretation
- greater complexity than current needs justify
- risk of unnecessary surface-area growth

---

## Current Knob Inventory

Current knobs already exposed in config / scripts include:

- tokenizer mode
- BPE vocab size
- block size / max sequence length
- model width / depth
- positional mode
- attention head count
- attention bias
- MLP hidden multiplier
- activation
- dropout
- embedding tying
- optimizer / learning rate / weight decay
- batch size
- gradient clipping
- generation temperature
- generation top-k
- generation top-p
- max new tokens
- seed.

---

# Triage

## A. Baseline-Defining Knobs

## Tokenizer mode

Values currently supported:

- `char`
- `word`
- `bpe`.

**Classification:** baseline-defining

**Why:**  
This changes the representational object itself. Sequence length, vocabulary size, and semantic granularity all change, so downstream comparisons are less about regime and more about model family.

**Use:**  
Compare baselines, not first-line regime induction.

---

## BPE vocab size

**Classification:** baseline-defining

**Why:**  
This changes token segmentation behavior and therefore input statistics. It is meaningful, but early interpretation becomes muddier because both the sequence geometry and vocabulary geometry are moving together.

**Use:**  
Tokenizer baseline comparison after core metric flow is stable.

---

## `block_size` / `max_seq_len`

Current code currently requires these to match.

**Classification:** baseline-defining

**Why:**  
Context size materially changes the model’s operating frame. This is more like selecting a different training/inference context envelope than inducing a regime within a fixed system.

**Use:**  
Baseline architecture comparison.

---

## `d_model`

**Classification:** baseline-defining

**Why:**  
This changes representational capacity and internal geometry too broadly to treat as an early “knob” in the regime sense.

**Use:**  
Model family comparison.

---

## `n_layers`

**Classification:** baseline-defining

**Why:**  
Depth changes the computation graph in a major way. Valuable later, but not ideal as an early regime knob.

**Use:**  
Architecture comparison.

---

## `n_heads`

**Classification:** baseline-defining

**Why:**  
Important structurally, but again closer to defining the model than perturbing a fixed model.

**Use:**  
Architecture sweep later.

---

## `pos_mode`

Current values:

- `learned`
- `sinusoidal`.

**Classification:** baseline-defining

**Why:**  
This changes how positional information is represented across the whole model. Very meaningful, but best treated as a baseline decision first.

---

## `tie_embeddings`

**Classification:** baseline-defining

**Why:**  
This alters the relationship between input and output token representations. It matters, but it is a structural identity choice more than a convenient regime knob.

---

## B. First-Priority Regime-Inducing Knobs

These are the best initial knobs to study.

## `temperature`

Used in generation sampling.

**Classification:** regime-inducing

**Why:**  
Directly affects output sharpness and stochasticity without retraining. This is a very clean way to produce behavioral changes that should show up in generation metrics.

**Why first:**  
Low effort, high interpretability, no training loop changes.

**Expected effects:**

- entropy shift
- confidence shift
- repetition / degeneration changes
- diversity changes

---

## `top_k`

Used in generation sampling.

**Classification:** regime-inducing

**Why:**  
A controlled truncation knob on the output distribution. Easy to reason about and cheap to vary.

**Expected effects:**

- narrowing of candidate space
- possible fixation / repetition behavior at low values
- confidence profile changes

---

## `top_p`

Used in generation sampling.

**Classification:** regime-inducing

**Why:**  
Another clean generation-side knob. Useful because it truncates adaptively rather than at a fixed rank.

**Expected effects:**

- altered diversity
- altered entropy
- altered confidence concentration

---

## `lr`

Training learning rate.

**Classification:** regime-inducing

**Why:**  
One of the strongest and cleanest training-side knobs for creating stable vs unstable optimization behavior.

**Why first among train knobs:**  
Because its causal role is well understood and its effect is often visible in loss and gradient-related metrics.

**Expected effects:**

- convergence speed changes
- instability / divergence risk
- gradient norm behavior changes

---

## `grad_clip`

Training gradient clipping.

**Classification:** regime-inducing

**Why:**  
Useful as a stabilizer knob paired with learning rate. This is a strong candidate for later controlled comparisons such as “same LR, clipped vs unclipped.”

**Expected effects:**

- gradient norm ceiling behavior
- reduced instability in aggressive regimes

---

## `batch_size`

Training batch size.

**Classification:** regime-inducing

**Why:**  
This can alter optimization noise and training behavior without redefining the model itself.

**Caution:**  
Interpretation is a little less clean than LR, but still very worthwhile.

---

## C. Second-Priority Regime-Inducing Knobs

Useful later, but not my first recommendation.

## `weight_decay`

**Classification:** regime-inducing, second priority

**Why:**  
Can matter for optimization behavior, but its effect is usually less immediately legible than learning rate or clipping in small early experiments.

---

## `max_new_tokens`

**Classification:** regime-inducing, second priority

**Why:**  
Technically this changes the observation horizon more than the model behavior itself, but longer horizons can expose generation pathologies.

**Use:**  
Helpful for measurement protocol, but not the first behavioral knob.

---

## `seed`

**Classification:** protocol knob, not a primary experimental knob

**Why:**  
Needed for determinism and controlled variation, but not ideal as the main source of behavioral regime differentiation.

**Use:**  
Replication, not primary induction.

---

## D. Deferred Knobs

## `attn_bias`

**Classification:** deferred

**Why:**  
Low current value-to-complexity ratio for your present objective.

---

## `activation`

Current values supported in block config:

- `gelu`
- `relu`.

**Classification:** deferred

**Why:**  
Interesting, but not a first-pass regime lever. Better handled as a structured baseline comparison later.

---

## `dropout`

**Classification:** deferred

**Why:**  
It is a real knob, but since this project strongly prioritizes determinism and clarity, it is not the best first-line control variable for early diagnosis work. Current default is already `0.0`.

---

## `optimizer`

Currently only `adamw` is supported in train.

**Classification:** deferred

**Why:**  
Not really a live knob yet, since support is intentionally narrow.

---

# Recommended Experimental Order

## Phase 1: generation-only knob study

Start with:

- temperature
- top_k
- top_p

**Reason:**  
These are the cheapest, cleanest, and fastest way to study whether your metrics framework is seeing meaningful behavioral variation.

---

## Phase 2: training stability knob study

Then move to:

- lr
- grad_clip
- batch_size

**Reason:**  
These are strong candidates for later regime labels like instability or divergence, and they map well to standard LM metrics.

---

## Phase 3: baseline-family comparisons

After the above is stable, study:

- tokenizer mode
- BPE vocab size
- pos_mode
- d_model
- n_layers
- n_heads
- tie_embeddings

**Reason:**  
These are valuable, but they are closer to changing the model class than changing the regime of the same model.

---

# Implications for Taps and Capture

This triage supports a simple rule:

> Do not expose or activate more taps until a knob produces behavior worth inspecting.

That keeps observability demand-driven rather than speculative.

For now, existing taps can remain dormant. Their value becomes much clearer once a regime-inducing knob produces metric changes that need internal localization.

This is consistent with the current architecture, where hooks and capture are observational infrastructure rather than required runtime behavior.

---

# Practical Working Rule

For the current project phase:

- keep taps present
- keep capture off by default
- do not build toggle complexity prematurely
- use existing knobs to identify which behavioral differences are worth instrumenting

That gives you the cleanest path from:

**knob → metric shift → observational question → tap usage**

instead of:

**tap first → no clear question → bloat**

---

# Summary Decision Table

|Knob|Classification|Priority|
|---|---|---|
|temperature|regime-inducing|highest|
|top_k|regime-inducing|highest|
|top_p|regime-inducing|highest|
|lr|regime-inducing|high|
|grad_clip|regime-inducing|high|
|batch_size|regime-inducing|high|
|weight_decay|regime-inducing|medium|
|max_new_tokens|protocol / observation horizon|medium|
|seed|protocol / replication|medium|
|tokenizer mode|baseline-defining|later|
|bpe_vocab_size|baseline-defining|later|
|block_size / max_seq_len|baseline-defining|later|
|d_model|baseline-defining|later|
|n_layers|baseline-defining|later|
|n_heads|baseline-defining|later|
|pos_mode|baseline-defining|later|
|tie_embeddings|baseline-defining|later|
|attn_bias|deferred|low|
|activation|deferred|low|
|dropout|deferred|low|
|optimizer|deferred|low|

