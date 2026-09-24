# Phase 3 - Evaluation, debugging, and observation

> Phase 3+ completed (2026-09-20): the continuation results below remain
> historical evidence. New evaluations use the SemIf-style categorical
> answer-slot readout and schema 2; pending old continuation evaluations do not
> block the migration.

## Goal

Use the single-backend baseline to understand correctness, behavior, and
bottlenecks before backend separation.

## Prerequisites

Phase 2 baseline works with option scoring, and Phase 2+ has stabilized the
request/result, fixed Gemma 4 prompt, and request-timing contracts.

## Read First

- `docs/requirements.md`
- `docs/development-rules.md`
- `docs/phases/phase-2-single-backend.md`
- `docs/phases/phase-2-plus-contract-hardening.md`

## Stages

### Stage 3.1 - Basic behavior

#### Step 3.1.1
Try text-only decisions.

#### Step 3.1.2
Try image-plus-decision inputs.

#### Step 3.1.3
Compare single- and multi-token options.

#### Step 3.1.4
Permute option order and observe changes.

### Stage 3.2 - Quantization comparison

#### Step 3.2.1
Compare practical Q4/Q5/Q6/Q8 and, if available, F16/BF16 variants for
selection, distribution, logit gap, runtime, and memory. Precision benchmarking
is not required.

### Stage 3.3 - Performance observation

#### Step 3.3.1
Measure image preprocessing, Vision, Prefill, option scoring, normalization,
backend synchronization, backend copies, and total time. Use the resulting
Vision/Prefill/Logit ratio to inform later design.

### Stage 3.4 - Debug support

#### Step 3.4.1
Add only useful dumps: token IDs/lengths, tensor shape/type/placement, graph
node count, intermediate timings, logits, raw and normalized scores, and
probabilities.

### Stage 3.5 - Small fixtures

#### Step 3.5.1
Create a small manual/regression fixture set, such as text/simple,
text/ambiguous, vision/object, vision/state, vision/order_bias, and
vision/multi_token. Avoid a large test framework.

### Stage 3.6 - Benchmark harness and frozen evaluations

#### Step 3.6.1
Add a small `branchscore-bench` JSONL runner that loads the model and backend
once, performs an explicit warmup, evaluates requests sequentially through
`Gemma4DecisionEngine`, and writes row-level machine-readable results. Include
request and option IDs, ordered probabilities and scores, selected option,
prompt identity, token counts, stage timings, and enough model/backend metadata
to identify the run. Refuse silent truncation and accidental output overwrite.

Keep JSON request/result encoding reusable by a possible later local server,
but do not add HTTP, a service framework, concurrency, or a general serving
abstraction in this phase.

#### Step 3.6.2
Report aggregate warm-run measurements including total wall time,
decisions/second, and request-latency p50/p95, while retaining the existing
Vision/Prefill/option-scoring/normalization breakdown. Keep model loading,
warmup, and result-file writes outside the measured request interval and record
the timing boundary in the output.

#### Step 3.6.3
Rebuild and evaluate SemIf's frozen 102-row TypeSafe-selected subset when the
four required, hash-verified upstream `typesafe-*-cases.js` snapshots are
available. This is a SemIf-selected subset of TypeSafe's public evaluation,
not an official standalone 102-row TypeSafe benchmark. Record equal-case modal
agreement and equal-case total-variation distance using the frozen 20-case
grouping.

Treat the result as a benchmark of Gemma 4 continuation scoring, not a direct
reproduction of SemIf's single-token letter-logit readout. Preserve the current
`sum_logprob` selection contract rather than changing the production scorer to
match the benchmark.

#### Step 3.6.4
Before running all 102 rows, identify a longest-token candidate and run an E2B
and, where practical, E4B smoke test. Record token count, peak/backend memory
observations, Prefill time, and any context or allocation failure. Do not add
truncation to make a row pass.

#### Step 3.6.5
Run the committed SemIf `shape777.jsonl` fixture as a warm-loaded, fresh,
sequential baseline and report all 777 row-level decisions plus aggregate
throughput and latency. This establishes the current one-request baseline only.
Do not label it as SemIf's serial-prefix or parallel-shared mode.

Cross-request state-prefix reuse, parallel suffix branches, and an
autoregressive generation comparison are separate follow-up work. A minimal
sequential `branchscore-server` may later reuse the JSON codec and resident
engine boundary, but it is not a Phase 3 benchmark deliverable.

## Deliverables

Behavior notes, timing observations, focused debug outputs, small fixtures, a
warm-loaded sequential benchmark runner, row-level benchmark evidence, and
aggregate TypeSafe-selected and shape777 reports where their inputs are
available.

## Completion Criteria

The baseline is stable; scoring behavior and major bottlenecks are understood;
the benchmark runner produces reproducible row-level and aggregate results;
long-context feasibility is recorded; and backend separation is justified by
measurements.

## Notes / Findings

The detailed dated evaluation and benchmark record is in [the Phase 3
continuation record](../records/phase-3-continuation-baseline.md). Phase 3's
continuation measurements are historical: they established the warm-loaded
runner, long-context feasibility, and backend observations, but the production
readout was migrated to Phase 3+ categorical answer-slot logits. The [direction
review](../research/direction-review-2026-09-20.md) remains the interpretation
source for the old measurements and their limitations.
