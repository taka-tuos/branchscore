# Phase 4 - Component/backend separation

> Planning hold (2026-09-20): [Phase 3+](phase-3-plus-categorical-readout.md)
> is complete and removed option-continuation workers from the normal decision
> path. The Phase 4–7 separation/worker/scheduler plans still require a revised
> handoff based on categorical readout measurements before implementation; the
> stages below are the previous plan.

## Goal

Make Vision, Prefill, and Logit independently placeable on backends. This is
an architecture phase, not yet a parallelization phase.

## Prerequisites

Phase 3 has characterized baseline costs and Phase 1 ownership design is
validated against the implementation.

## Read First

- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-1-research-design.md`
- `docs/phases/phase-3-evaluation-debug.md`
- Backend findings in `docs/research/llama-ggml.md`

## Stages

### Stage 4.1 - Backend contexts

#### Step 4.1.1
Remove direct global-backend assumptions. Give Vision, Prefill, and Logit
contexts independent handles, allocators, compute/temp buffers, and sync.

### Stage 4.2 - Vision separation

#### Step 4.2.1
Place Vision on its own backend; transfer its output to Prefill and measure the
copy.

### Stage 4.3 - Prefill separation

#### Step 4.3.1
Place Prefill independently and define the scoring state precisely. Decide from
evidence whether scoring receives the KV cache, a subset, hidden state, or
another model-specific reusable representation.

### Stage 4.4 - Logit separation

#### Step 4.4.1
Place OptionScorer independently, implement Prefill-to-Logit transfer, and
retain sequential scoring on a single Logit backend.

### Stage 4.5 - Transfer and synchronization

#### Step 4.5.1
Track `VisionDone`, `PrefillReady`, `LogitReady`, and `LogitDone`. Synchronous
operation is acceptable initially; use events/async copies only when useful.

## Deliverables

Configurable component backends with explicit state transfer and timing.

## Completion Criteria

Vision=A, Prefill=B, and Logit=C can be selected independently; layer splitting
remains unsupported.

## Notes / Findings

_Pending separation work._
