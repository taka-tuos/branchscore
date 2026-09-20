# Phase 7 - Scheduler tuning

## Goal

Tune the pipeline for useful throughput/latency on available hardware without
turning the hobby PoC into a production scheduling system.

## Prerequisites

Phase 6 pipeline behavior is stable and its stage measurements are available.

## Read First

- `docs/development-rules.md`
- `docs/phases/phase-6-pipeline-scheduler.md`
- Relevant Phase 3 performance findings

## Stages

### Stage 7.1 - Timing collection

#### Step 7.1.1
Maintain moving averages for Vision, Prefill, Logit, copy latency, and queue
wait.

### Stage 7.2 - Worker weights

#### Step 7.2.1
Where justified, apply simple backend weights (for example CUDA0=3, CUDA1=2,
CPU=1).

### Stage 7.3 - Queue policy

#### Step 7.3.1
Compare FIFO, shortest queue, least estimated completion time, backend affinity,
and state-cache affinity.

### Stage 7.4 - Batch policy

#### Step 7.4.1
Tune max option batch, max wait time, and max queued jobs; document latency vs.
batching trade-offs.

### Stage 7.5 - Cache policy

#### Step 7.5.1
Consider Vision-result, Prefill-state, and worker-local caches. Keep eviction
policy simple.

## Deliverables

Recorded scheduler parameters and hardware-specific comparisons.

## Completion Criteria

Pipeline requests and multiple workers run successfully, speed differences are
accounted for, and parameter choices can be compared reproducibly.

## Notes / Findings

_Pending tuning._
