# Phase 5 - Multiple Logit workers

## Goal

Evaluate options or decision queries concurrently by distributing scoring work
to multiple workers/backends.

## Prerequisites

Phase 4 supports a separately placed, sequential Logit component and explicit
prefill-state transfer.

## Read First

- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-4-backend-separation.md`

## Stages

### Stage 5.1 - Worker model

#### Step 5.1.1
Define a LogitWorker with backend, model state, work queue, input/output
buffers, and status.

### Stage 5.2 - Jobs

#### Step 5.2.1
Define DecisionJob (`state_id`, `question_id`, options) and independently
dispatchable OptionScoreJob units.

### Stage 5.3 - Basic scheduler

#### Step 5.3.1
Start with round-robin or first-available worker scheduling.

### Stage 5.4 - Batched scoring

#### Step 5.4.1
Compare many small option graphs with option batches where backend/model data
supports it.

### Stage 5.5 - Shared-state fan-out

#### Step 5.5.1
Distribute Prefill state to workers, measure copy cost, and consider persistent
worker-local cache only when measurements justify it.

## Deliverables

Multiple workers, job distribution, result aggregation, and a single/multiple
worker switch.

## Completion Criteria

Options can score concurrently across multiple backends/workers and aggregate
into one result.

## Notes / Findings

_Pending worker implementation._
