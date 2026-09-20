# Phase 5 - Parallel decision/request workers

> Planning handoff (2026-09-20): categorical readout evaluates every displayed
> option in one decision. If parallelism is needed, the unit of work is a full
> decision or request, not an option continuation.

## Goal

Evaluate independent questions or requests concurrently when Phase 4
measurements justify it. Preserve the invariant that one decision contains all
of its displayed options and performs one categorical readout.

## Prerequisites

- Phase 4 has recorded the useful component/backend boundaries, or explicitly
  deferred separation while retaining the single-backend context.
- A measured request-level latency or throughput need justifies concurrency.
- Request/result ownership and the Phase 3+ categorical schema are stable.

## Read First

- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`
- `docs/phases/phase-4-backend-separation.md`

## Stages

### Stage 5.1 - Decision worker model

#### Step 5.1.1

Define a worker that owns a complete model/backend/engine context and processes
whole decision requests. A worker may remain single-backend; do not create a
candidate worker merely to distribute answer labels.

### Stage 5.2 - Request jobs

#### Step 5.2.1

Define a `DecisionRequest` or equivalent job containing state, question, and
the complete ordered option list. The job returns all option results,
selection, and timing for one categorical readout. There is no
`OptionScoreJob` or candidate-level continuation unit.

### Stage 5.3 - Basic scheduler

#### Step 5.3.1

If concurrency is justified, compare a small first-available or round-robin
policy for full decision requests. Preserve request IDs and result order, and
make queue ownership and failure handling explicit.

### Stage 5.4 - Request-level batching

#### Step 5.4.1

Only if measurements justify it, compare batching of independent requests or
questions. Do not call a list of options a worker batch: all options of one
question already belong to one categorical decision.

### Stage 5.5 - State and cache ownership

#### Step 5.5.1

Define request-scoped Vision/Prefill/readout state and any cache transfer at
worker boundaries. Share or copy state only when measurements and lifetime
rules justify it; do not fan one candidate state out to option workers.

## Deliverables

- A measured decision/request worker model, if concurrency is needed.
- Request-level job and result ownership contracts.
- A small scheduler and optional request-level batching experiment with
  reproducible measurements.

## Completion Criteria

- Independent full decisions can run concurrently without mixing options,
  state, IDs, or result order.
- Each decision performs one categorical readout over its displayed options.
- Queue, worker, state lifetime, synchronization, and failure behavior are
  documented and tested to the extent of the implemented feature.
- No completion criterion requires candidate workers, `OptionScoreJob`,
  option batches, or candidate KV fan-out.

## Notes / Findings

_Pending request-level worker work._
