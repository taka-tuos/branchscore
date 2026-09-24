# Phase 2 - Single-backend baseline

## Goal

Produce a correct, observable sequential baseline: Vision, Prefill, and option
scoring execute on one backend. Do not introduce parallelism.

## Prerequisites

Phase 1 research/design is complete enough to identify Gemma 4 tensors,
tokenization, vision flow, cache behavior, and the initial scoring rule.

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-1-research-design.md`
- Relevant completed notes under `docs/research/`

## Stages

### Stage 2.1 - Project foundation

#### Step 2.1.1
Create the independent CMake project with source, include, tests, tools,
third-party, and docs locations as needed.

#### Step 2.1.2
Integrate ggml as a dependency.

#### Step 2.1.3
Implement backend selection/initialization for CPU, CUDA, and Vulkan where
available.

### Stage 2.2 - GGUF/model loading

#### Step 2.2.1
Load Gemma 4 E2B/E4B GGUF.

#### Step 2.2.2
Locate required tensors by name.

#### Step 2.2.3
Use quantized tensors directly in ggml graphs.

#### Step 2.2.4
Absorb E2B/E4B metadata differences without scattered hard-coding.

### Stage 2.3 - Tokenizer

#### Step 2.3.1
Make the Gemma 4 tokenizer available.

#### Step 2.3.2
Tokenize state and options.

#### Step 2.3.3
Support debug dumps of option IDs and lengths.

### Stage 2.4 - Vision encoder

#### Step 2.4.1
Implement image loading/preprocessing.

#### Step 2.4.2
Build and run the vision graph on the same backend.

#### Step 2.4.3
Obtain visual tokens/embeddings and convert them for LLM Prefill.

### Stage 2.5 - Prefill

#### Step 2.5.1
Build Prefill from text plus visual tokens.

#### Step 2.5.2
Maintain reusable KV-cache-equivalent state.

#### Step 2.5.3
Prefill once and reuse state per option; measure Prefill time.

### Stage 2.6 - Option scoring

#### Step 2.6.1
Implement single-token scoring.

#### Step 2.6.2
Implement multi-token scoring.

#### Step 2.6.3
Accumulate log-probabilities; provide sum and mean normalization.

#### Step 2.6.4
Compute option probabilities with softmax.

### Stage 2.7 - CLI PoC

#### Step 2.7.1
Accept model, optional image, state, question, repeated options, and backend;
print scores, probabilities, selection, and Vision/Prefill/score/total times.

## Deliverables

A standalone single-backend CLI capable of the first practical milestone.

## Completion Criteria

An E2B or E4B GGUF loads; image state, Prefill, and multi-token scoring work;
probabilities and timings are printed; all work is on one backend.

## Notes / Findings

The detailed dated implementation, benchmark, and model/backend findings are
in [the Phase 2 execution record](../records/phase-2-single-backend.md).
Phase 2 established the independent ggml project, single-backend Gemma 4
loader/tokenizer, Vision and Prefill graphs, request cache, continuation
scoring, and the first CLI. The continuation contract and source-owning
template path were subsequently superseded by Phase 2+ and Phase 3+.
