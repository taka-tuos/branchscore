# Phase 6 - Pipeline scheduler

## Goal

Treat Vision, Prefill, and Logit as pipeline stages so multiple requests can
overlap safely.

## Prerequisites

Phase 5 supports multiple Logit workers and has measured state fan-out costs.

## Read First

- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-4-backend-separation.md`
- `docs/phases/phase-5-logit-workers.md`

## Stages

### Stage 6.1 - Pipeline model

#### Step 6.1.1
Implement the logical flow Request → Vision queue → Prefill queue → Logit queue
→ Result.

### Stage 6.2 - Request state machine

#### Step 6.2.1
Represent at least Queued, VisionRunning, VisionDone, PrefillRunning,
PrefillDone, LogitRunning, Completed, and Failed.

### Stage 6.3 - Async execution

#### Step 6.3.1
Use backend async graph compute, events, and async copies where worthwhile;
threads plus synchronization are an acceptable initial implementation.

### Stage 6.4 - Overlap

#### Step 6.4.1
Allow request B's Vision to overlap request A's later stages when resources
allow, and fan one shared state out to multiple Logit workers.

## Deliverables

A multi-request pipeline with observable stage transitions and safe overlap.

## Completion Criteria

Multiple requests can traverse independent queues and use concurrent Logit
workers without violating state ownership.

## Notes / Findings

_Pending scheduler implementation._
