# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What GrAIdient is

Swift + Metal deep-learning framework (training *and* inference) for Apple Silicon, by Jean-François Reboud. This clone is JF's fork (`jfreboud/GrAIdient`, branch `main`), version 0.4.0+. It is the framework layer under **GrAIl** (interpretability toolkit at `../GrAIl`), which depends on it via `.package(path: "../GrAIdient")`.

## Role in the GrAIl workflow

- New framework work (layers, hooks) happens **here**, on branches, each with gradient tests.
- Decision rule: generic differentiable primitive → this repo; interpretability logic → GrAIl.
- Keep diffs vs. upstream reviewable — this is a fork, minimize gratuitous divergence.

## Layout

| Path | Contents |
|---|---|
| `Sources/GrAIdient/Core` | Layer/model core, optimizer, math functions (activation, normalization) |
| `Sources/GrAIdient/Layer1D` / `Layer2D` / `LayerSeq` | Layers by shape family; `LayerSeq` holds the LLM stack |
| `Sources/GrAIdient/Metal` | `MetalKernel` + `.metal` kernels (Float/Half pairs), registered in `MetalConfig.swift` |
| `Sources/GrAITestsUtils` | Trainer, gradient/flow-checking helpers |
| `Tests/GrAITests` | Pure-Swift unit tests, incl. `NLPTests.swift` |
| `Tests/GrAITorchTests` | Reference comparisons against PyTorch via PythonKit — needs a Python env (`Base/python_lib`) |
| `Tests/GrAIExamples` | End-to-end examples: CIFAR/VGG, Transformer, `LLMExample.swift` (Mistral 7B, Llama 2/3, Gemma2-2B; weight paths hardcoded to local disk, loaded via PythonKit) |

## LLM support already present (verified July 2026)

`EmbeddingSeq`, `RMSNormSeq`, `RoPESeq`, `QueryCausalSeq`/`ValueCausalSeq` (GQA via `nbHeadsKV`), `MultiplySeq` + SiLU (SwiGLU), sliding-window attention, prompt generation. Float16 compute via `GrAI.Precision.float16`; GPU via `GrAI.Opti.GPU = true` after touching `MetalKernel.get`.

**Framework work for GrAIl's LLM interp slice** (SmolLM3-3B first, then Phi-4-mini/Qwen3/Gemma 3) — spec in `Docs/Interp/LLM_LAYERS.md` (temporary doc, tracks cross-repo work with GrAIl; fold into it or delete once the slice lands). Done: RoPE `theta` is now a required parameter of `RoPESeq` (no default — forces every call site, including model-specific ones in `LLMExample`, to state it explicitly) instead of hardcoded to `10000`. Deferred: per-head QK-RMSNorm, partial RoPE, windowed causal mask (the existing `slidingWindow` is KV-cache eviction, not an attention mask), custom query scaling. Overall context: `../GrAIl/Docs/LLM/LLM_SLICE.md`.

## Building and testing

- swift-tools 5.7, macOS 13+, Swift 5 language mode. Tests use **XCTest** — follow that convention here (Swift Testing is the convention in GrAIl only).
- On this machine, command-line builds need the full Xcode toolchain: prefix with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` (default `xcode-select` points at CommandLineTools, which lacks some libraries).
- Fast check: `swift build`. Unit tests: `swift test --filter GrAITests`.
- `GrAITorchTests` and `GrAIExamples` require a configured Python env and local model weights — do not run them casually.
- New layers require gradient tests (reference-value comparison, < 1% relative tolerance); follow existing patterns in `Tests/GrAITests`.

## Conventions

- Swift, 4-space indent, Allman braces (opening brace on its own line).
- Metal kernels come in Float/Half pairs and must be registered in `Metal/MetalConfig.swift`.
- `CHANGELOG.md` is maintained — add entries under `[unreleased]` with the emoji style already in use.
