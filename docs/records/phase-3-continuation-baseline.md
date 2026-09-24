# Phase 3 - Continuation baseline: execution record

This is the dated evaluation and benchmark record extracted from
[`phases/phase-3-evaluation-debug.md`](../phases/phase-3-evaluation-debug.md).
It describes the old continuation scorer and must not be compared directly
with Phase 3+'s categorical readout. The saved benchmark artifacts are
historical evidence, not a current production contract.

## Notes / Findings

- 2026-09-20: Reviewed the initial design against current branchscore,
  SemIf source, and Jev's public documentation. The continuation scorer is an
  intentional Phase 1 choice, not SemIf's categorical letter readout; current
  cache reuse is across options, not questions. Rechecked the saved shape777
  artifacts: Prefill accounts for roughly 69-70% of mean request time and
  E2B/E4B agree on 320/777 decisions, without semantic gold labels. The review
  was subsequently revised to reflect the user's decision to migrate directly
  to SemIf-style categorical readout in Phase 3+. Comparative scoring research
  and permanent retention of the old scorer are not required. Cross-question
  prefix reuse remains a follow-up candidate after migration, not a migration
  prerequisite. Existing measurements describe the old continuation path.
  See [`direction-review-2026-09-20.md`](../research/direction-review-2026-09-20.md)
  for evidence and limitations.
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
  `fixtures/phase3-text.jsonl`. E2B and E4B both preserve option order and
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
  throughput was 0.651 decisions/s with request p50/p95 1,489.63/1,912.19
  ms; E4B was 0.331 decisions/s with request p50/p95 2,917.58/3,767.78 ms.
  These corrected figures exclude model loading, warmup, and JSONL writes as
  recorded in the output timing boundary.
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
  backend completed at 17.61 decisions/s with request p50/p95
  49.76/97.43 ms. The ambiguous row changed from
  CPU E2B wait=0.599/unknown=0.399, selected wait, to CUDA
  wait=0.470/unknown=0.527, selected unknown. This is a real backend
  sensitivity observation for the current quantized baseline; CPU and CUDA
  distributions must not be treated as numerically identical until a later
  parity study. CUDA focused/model-backed CTest also passed 10/10.
- 2026-09-20: The committed shape777 fixture has SHA-256
  `8dcf414b12fc2684e3c4ca5f3ebfd3f525f5346fec4a9bc67eb65138101f55f1`.
  Its longest row passed CUDA smoke on the RTX 2060 SUPER for both E2B and E4B:
  E2B took 920.01 ms (Prefill 640.14 ms), and E4B took 1,475.22 ms
  (Prefill 1,044.51 ms), with no allocation or context failure. The full
  fresh sequential 777-row run then completed for both models. E2B took
  711,256.10 ms (11.85 minutes, 1.0924 decisions/s, request p50/p95
  918.31/930.47 ms, mean Prefill 628.70 ms, mean option scoring 259.03 ms),
  while E4B took 1,161,806.31 ms (19.36 minutes, 0.6688 decisions/s, request
  p50/p95 1,499.00/1,512.87 ms, mean Prefill 1,052.30 ms, mean option scoring
  414.99 ms). Both artifacts contain 777 decision rows plus run and aggregate
  rows and are retained locally under ignored `out/shape777-*-cuda.jsonl`.
  The E2B selections were `no` 573 / `yes` 204 and E4B selections were
  `no` 162 / `yes` 615; they agreed on 320/777 rows (41.18%). Because this
  fixture has no semantic gold labels, these counts are descriptive model
  sensitivity observations, not accuracy measurements.
- 2026-09-20: The Phase 3+ migration supersedes the continuation production
  path. The new E2B/E4B text smoke, E2B image smoke, schema-2 JSONL output,
  and focused readout tests are recorded in
  [`phase-3-plus-categorical-readout.md`](phase-3-plus-categorical-readout.md);
  the historical measurements above must not be compared as if they used the
  same readout contract.
