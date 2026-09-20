# Phase 7 - Scheduler and batch tuning

> Planning handoff (2026-09-20): tune only request-level mechanisms that were
> actually introduced and measured in Phases 4–6.

## Goal

Tune useful throughput and latency on available hardware without turning the
hobby PoC into a production scheduling system or optimizing an unimplemented
candidate-scoring path.

## Prerequisites

- Phase 6 request-pipeline behavior is stable, if a pipeline was introduced.
- Stage, queue, copy/synchronization, and categorical readout measurements are
  available for the target workload.
- The selected worker/backend configuration is explicit.

## Read First

- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`
- `docs/phases/phase-4-backend-separation.md`
- `docs/phases/phase-5-logit-workers.md`
- `docs/phases/phase-6-pipeline-scheduler.md`

## Stages

### Stage 7.1 - Request timing collection

#### Step 7.1.1

Record Vision, Prefill, categorical readout, copy/synchronization, queue wait,
and end-to-end request latency. Keep shared readout timing intact; do not
invent per-option timing by dividing a common operation.

### Stage 7.2 - Worker/resource policy

#### Step 7.2.1

If multiple request workers or backends exist, compare simple assignment
policies and weights using measured completion time. Do not add weights for
resources that the implementation does not have.

### Stage 7.3 - Queue policy

#### Step 7.3.1

Compare FIFO, first-available, shortest queue, or estimated completion time
only when those queues exist. Record latency/throughput trade-offs and keep
the policy small and reproducible.

### Stage 7.4 - Request batch policy

#### Step 7.4.1

If request-level batching is implemented, tune request batch size, wait time,
and queue limits. Measure latency versus throughput. There is no required
`max option batch`: the options of one question are already read together.

### Stage 7.5 - Cache policy

#### Step 7.5.1

Consider Vision-result or state-prefix caches only if those caches were
introduced and their ownership is defined. Keep eviction and invalidation
simple, and measure cache hit cost against recomputation.

## Deliverables

- Reproducible timing data and selected request-level scheduler parameters.
- Hardware/workload comparisons for mechanisms that actually exist.
- A record of tuning decisions, trade-offs, and deferred optimizations.

## Completion Criteria

- The tuned request-level path is correct, measurable, and reproducible.
- Parameter choices are supported by target-workload measurements.
- No tuning requirement refers to an unimplemented Logit worker, candidate
  state fan-out, or option batch.

## Notes / Findings

_Pending request-level tuning work._
