# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What GrAIdient is

Swift + Metal deep-learning framework (training *and* inference) for Apple Silicon, by Jean-François Reboud. This clone is JF's fork (`jfreboud/GrAIdient`, branch `main`), version 0.4.0+. It is the framework layer under **GrAIl** (interpretability toolkit at `../GrAIl`), which depends on it via `.package(path: "../GrAIdient")`.

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

## Building and testing

- swift-tools 5.7, macOS 13+, Swift 5 language mode. Tests use **XCTest** — follow that convention here (Swift Testing is the convention in GrAIl only).
- Command-line builds need the full Xcode toolchain: `xcode-select -p` must point at `/Applications/Xcode.app/Contents/Developer` (CommandLineTools lacks some libraries). If it does not, prefix commands with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Fast check: `swift build`. Unit tests: `swift test --filter GrAITests`.
- `GrAITorchTests` and `GrAIExamples` require a configured Python env and local model weights — do not run them casually.
- New layers require gradient tests (reference-value comparison, < 1% relative tolerance); follow existing patterns in `Tests/GrAITests`.

## Conventions

- Swift, 4-space indent, Allman braces (opening brace on its own line).
- Metal kernels come in Float/Half pairs and must be registered in `Metal/MetalConfig.swift`.
- Float16 compute via `GrAI.Precision.float16`; GPU via `GrAI.Opti.GPU = true` after touching `MetalKernel.get`.
- `RoPESeq` takes a **required** `theta` (no default): every call site, including model-specific ones in `LLMExample`, must state it explicitly.
- Where code goes: a generic differentiable primitive belongs in this repo, with its gradient tests; interpretability logic belongs in GrAIl.
- This is a fork: keep diffs vs. upstream reviewable and avoid gratuitous divergence.
- `CHANGELOG.md` is maintained: under `[unreleased]`, one line per PR, newest first, made of the PR title verbatim followed by a link to the PR (`<title> [#N](https://github.com/jfreboud/GrAIdient/pull/N)\`; every line ends with `\` except the last).
