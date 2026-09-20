# Phase 6 - Request pipeline scheduler

> Planning handoff (2026-09-20): the pipeline is organized around complete
> requests and the actual categorical path, not a mandatory Logit queue.

## Goal

Overlap independent requests safely across the stages that actually exist:
Vision, Prefill, categorical readout, and result assembly. Keep ownership and
synchronization explicit without introducing a pipeline for an unimplemented
candidate-scoring mechanism.

## Prerequisites

- Phase 5 request-level workers, if needed, are stable and their measurements
  show that a pipeline can improve the target workload.
- Phase 4 component boundaries and state-transfer rules are known.
- The Phase 3+ result, timing, and categorical readout contracts remain intact.

## Read First

- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`
- `docs/phases/phase-4-backend-separation.md`
- `docs/phases/phase-5-logit-workers.md`

## Stages

### Stage 6.1 - Request pipeline model

#### Step 6.1.1

Model the actual flow as `Request → Vision → Prefill → CategoricalReadout →
Result`. Add queues only at measured ownership boundaries and retain a direct
sequential path for comparison.

### Stage 6.2 - Request state machine

#### Step 6.2.1

Represent at least `Queued`, `VisionRunning`, `VisionDone`,
`PrefillRunning`, `PrefillDone`, `CategoricalReadoutRunning`,
`CategoricalReadoutDone`, `Completed`, and `Failed`. State transitions must
identify the owning request and the resource that signals completion.

### Stage 6.3 - Async execution

#### Step 6.3.1

Use threads, backend events, asynchronous copies, or graph execution only when
they improve a measured request workload. Define synchronization and shutdown
behavior before enabling overlap.

### Stage 6.4 - Independent-request overlap

#### Step 6.4.1

Allow one request to enter Vision while another reaches a later stage when
resources and ownership permit. Do not split one question's options into
separate pipeline jobs; its readout remains one operation.

## Deliverables

- A request-level pipeline model and state-transition record.
- Explicit queue, resource, state lifetime, and synchronization contracts for
  any implemented overlap.
- Measurements comparing the pipeline with the sequential request path.

## Completion Criteria

- Independent requests can traverse the implemented stages without state or
  result mixing.
- Every request performs its categorical readout once and returns all options
  in input order.
- Failure, cancellation, queue ownership, and synchronization behavior are
  defined for the implemented scope.
- No completion criterion requires an independent Logit queue, candidate state
  fan-out, or candidate continuation worker.

## Notes / Findings

_Pending request-pipeline work._
