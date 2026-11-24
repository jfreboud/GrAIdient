# Framework work for GrAIl's LLM interpretability slice

> **Driver**: this work is driven by GrAIl (the interp toolkit built on top of
> this fork). The overall design, model choices, and the GrAIl-side breakdown
> live in **`../GrAIl/Docs/LLM/LLM_SLICE.md`**. This document only lists the
> **differentiable primitives to add *here*** (rule: generic primitive → this
> repo; interp logic → GrAIl).
>
> Status: **specified, partly implemented.** Written 2026-07-30, updated
> 2026-09-24 (#1 done).

## Context

GrAIl is going to instrument modern SLMs (SmolLM3-3B first, then Phi-4-mini,
Qwen3, Gemma 3). This fork's LLM stack (`EmbeddingSeq`, `RMSNormSeq`,
`RoPESeq`, `QueryCausalSeq`/`SoftmaxCausalSeq`/`ValueCausalSeq` GQA,
`MultiplySeq`+SiLU for SwiGLU, `Constant2Seq`, GELU/GELUApprox) already covers
most of it — see `_buildModel` in `Tests/GrAIExamples/LLMExample.swift`. Only
the items below are missing.

Each new primitive follows the repo's conventions:
- Metal kernels as **Float/Half pairs**, registered in `Metal/MetalConfig.swift`;
- **reference gradient test** (XCTest, < 1% relative tolerance) in
  `Tests/GrAITests`, ideally backed by a PyTorch reference in
  `Tests/GrAITorchTests`;
- keep the diff vs upstream reviewable (this is a fork).

## Inventory of items

| # | Item | Requesting GrAIl model | Priority | Status |
|---|---|---|---|---|
| 1 | Configurable RoPE `theta` (+ per layer) | all | **immediate** (SmolLM3) | **done** |
| 2 | Per-head QK-RMSNorm | Qwen3, Gemma 3 | deferred | |
| 3 | Partial RoPE | Phi-4-mini | deferred | |
| 4 | Sliding-window causal mask | Gemma 3 | deferred | |
| 5 | Custom query scaling | Gemma 3 | deferred | |

GeGLU (Gemma 3) and NoPE (SmolLM3) require **nothing here**: `GELUApprox`
already exists; NoPE is a GrAIl-side construction decision (simply don't
insert `RoPESeq`).

---

### #1 — Configurable RoPE `theta`  *(the only one required for the 1st slice — done)*

**Problem**: `theta` was hardcoded to `10000`. Affected paths:
- CPU: `Sources/GrAIdient/LayerSeq/RoPESeq.swift` — `pow(10000.0, …)` in the 3
  paths (forward GC, forward, backward).
- GPU: `createRoPESeqMatrix{Float,Half}` kernel (cos/sin precomputation) in
  `Metal/Kernel/NLP{Float,Half}.metal`, registered in `MetalConfig.swift`.

Consequence: SmolLM3 wants `theta = 5e6`, Qwen3 `1e6`, Gemma 3 `1e6` (global) /
`10000` (local), etc. — impossible before this change. **This also confirmed
a real bug**: the existing Llama 3 and Mistral-7B-Instruct-v0.3 paths in
`LLMExample` were silently running with the wrong `theta` (the `10000`
default) instead of `500000` and `1000000` respectively.

**Done**:
- `theta: Double` is now a **required** parameter of `RoPESeq.init` (no
  default) — deliberately, so every call site states its value explicitly and
  the compiler flags call sites when a new model is wired in, instead of
  silently inheriting `10000`. The `Decoder` init still defaults to `10000.0`
  via `decodeIfPresent`, for backward compatibility with models already
  serialized to disk.
- it is serialized (Codable) and propagated through `clone`;
- it is passed via `setBytes` to the `createRoPESeqMatrix*` kernel and used in
  the 3 CPU paths;
- fixed the wrong `theta` at the `LLMExample`/`LLMExampleTests` call sites
  (Mistral-7B-Instruct-v0.3 → `1000000`, Llama 2 → `10000`, Llama 3 →
  `500000`, Gemma 2 → `10000`).

Not done: a **per-layer** `theta` (models with alternating local/global
attention, cf. Gemma 3, want two values) — nothing in the current API
prevents adding it later, since each `RoPESeq` instance already takes its own
`theta`.

**Test**: reference gradient test with at least two `theta` values (including
10000, to prove non-regression) — see `NLPTests.swift` cases `"RoPE"` /
`"RoPETheta"` — compared against a PyTorch RoPE implementation.

---

### #2 — Per-head QK-RMSNorm  *(deferred — Qwen3, Gemma 3)*

RMSNorm applied **per head**, over `head_dim`, to the Q and K projections
before RoPE. `RMSNormSeq` normalizes the full feature axis (`nbNeurons`), not
per-head sub-vectors → **new layer** (e.g. `RMSNormHeadSeq`, or an `nbHeads`
mode on `RMSNormSeq`): RMS reduction over `head_dim` inside each head, learned
gain of dimension `head_dim` shared across heads. Float/Half kernels + gradient
test. This is the only "real" new layer in the batch, shared by Qwen3 + Gemma 3.

### #3 — Partial RoPE  *(deferred — Phi-4-mini)*

Only rotate the first `k` dimensions of `head_dim` (Phi-4: 75%), the rest
passes through unchanged. Extension of `RoPESeq` (`rotaryDim`/`partialFactor`
parameter): the cos/sin matrix only covers `rotaryDim`, the remaining
dimensions are copied as-is in both forward *and* backward. Gradient test with
`rotaryDim < head_dim`.

### #4 — Sliding-window causal mask  *(deferred — Gemma 3)*

⚠️ The current `slidingWindow` on `QuerySeq` handles **KV-cache eviction**
(`cacheSeq >= cacheSeqMax`, see the TODO about instruct models and header
info) — this is **not** a windowed attention mask. Gemma 3 (5 local layers : 1
global, window ~1024) requires each query to only attend to the last `W` keys.
To do: windowed mask in the causal score computation
(`QueryCausalSeq`/`SoftmaxCausalSeq`) + clean interaction with the cache.
Low-to-medium effort but **needs validation** (mask + cache combined).

### #5 — Custom query scaling  *(deferred — Gemma 3)*

Gemma 3 replaces the standard `1/√head_dim` with a custom scalar factor.
Simple scalar parameter on `QueryCausalSeq` (default = current behavior).
Trivial.
