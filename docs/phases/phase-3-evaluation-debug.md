# Phase 3 - Evaluation, debugging, and observation

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

- 2026-09-20: Planned the Phase 3 benchmark path around a small warm-loaded
  JSONL runner before any server work. The first systems result will be fresh
  sequential shape777 execution; cross-request state-prefix reuse, parallel
  suffix execution, and HTTP serving remain out of scope until baseline
  measurements justify them.
- 2026-09-20: The intended external quality fixture is SemIf's frozen
  102-row/20-case selection reconstructed from TypeSafe public records. Its
  source snapshots are third-party inputs and are not present locally. The
  benchmark will retain branchscore's full-description continuation scoring,
  so results must not be described as reproducing SemIf's single-token
  letter-logit scorer.
- 2026-09-20: Stage 3.1 text smoke coverage now includes simple, ambiguous,
  single-token, multi-token, and permuted-order rows in
  fixtures/phase3-text.jsonl. E2B and E4B both preserve option order and
  prompt identity; the E2B selections for the four rows were
  yes, wait, yes, yes, while E4B selected yes, unknown, no, yes. The
  single-token E2B row strongly favored yes (0.866 versus 0.134), whereas
  the longer Keep it running/Stop it descriptions favored yes at 0.99986.
  These are option-set-relative continuation scores, not calibrated
  confidence.
- 2026-09-20: A 640x488 existing JPEG vision smoke request completed on E2B
  CPU with image preprocessing 12.90 ms, Vision 3,779.04 ms, Prefill
  4,799.78 ms, option scoring 425.14 ms, and request total 9,017.58 ms. A
  previously recorded 1942x809 PNG request took 59,437.41 ms in Vision and
  20,117.91 ms in Prefill, showing that image resolution dominates this CPU
  path.
- 2026-09-20: Added request-scoped backend copy/synchronization timings and
  graph node counts. The counters cover backend tensor set/get/copy operations
  and explicit synchronization calls; model loading remains outside the
  request interval. In the E2B text fixture, mean Prefill copy/synchronization
  was 0.323/0.00040 ms and option copy/synchronization was 0.472/0.00078 ms;
  graph counts were 1,736 for Prefill and 6,090 per option on average. The
  measured CPU synchronization cost is negligible, but the counters remain
  backend-specific observations rather than portable performance claims.
- 2026-09-20: The warm-loaded JSONL runner produced row-level and aggregate
  output for the four-row E2B and E4B text fixture. E2B warm-run aggregate
  throughput was 0.683 decisions/s with request p50/p95 1,366.30/1,817.30
  ms; E4B was 0.358 decisions/s with p50/p95 2,664.97/3,417.36 ms. These
  figures exclude model loading, warmup, and JSONL writes as recorded in the
  output timing boundary.
- 2026-09-20: The longest state candidate found in the committed SemIf
  shape777.jsonl input had 1,793 prefix tokens and option lengths 11/12.
  E2B Q4 CPU completed it without truncation or allocation failure in
  58,464.40 ms (Prefill 55,763.13 ms); E4B Q4 CPU completed it in
  107,071.28 ms (Prefill 101,141.87 ms). The full 777-row sequential run
  was not started because this CPU cost makes it a multi-hour workload; the
  runner is ready for a deliberately provisioned run.
- 2026-09-20: Only the local E2B/E4B Q4_K_M text assets are available.
  Q5/Q6/Q8 and F16/BF16 comparison data is therefore pending rather than
  inferred from the E2B/E4B model-size comparison. The frozen TypeSafe
  snapshots required for the 102-row report are also not present locally.
- 2026-09-20: The same four-row E2B Q4 fixture on the RTX 2060 SUPER CUDA
  backend completed at 16.76 decisions/s with request p50/p95
  54.02/97.27 ms and mean Prefill 25.63 ms. The ambiguous row changed from
  CPU E2B wait=0.599/unknown=0.399, selected wait, to CUDA
  wait=0.470/unknown=0.527, selected unknown. This is a real backend
  sensitivity observation for the current quantized baseline; CPU and CUDA
  distributions must not be treated as numerically identical until a later
  parity study. CUDA focused/model-backed CTest also passed 10/10.
