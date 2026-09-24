# Phase 4 - Backend boundaries: measurement record

This is the dated measurement and ownership record extracted from
[`phases/phase-4-backend-separation.md`](../phases/phase-4-backend-separation.md).
The measurements support the current single-backend handoff; they are local
observations rather than portable performance claims.

## Notes / Findings

- 2026-09-20: Stage 4.1 CPU baseline was measured with the current
  categorical path using the warm-loaded `branchscore-bench` runner
  (`--warmup 1`, four rows from `fixtures/phase3-text.jsonl`). The measured
  interval is one `Gemma4DecisionEngine::evaluate` call; model loading,
  warmup, and JSONL writes are excluded. On E2B Q4_K_M, mean request time was
  2,417.62 ms, p50/p95 was 2,304.81/2,724.10 ms, and throughput was 0.4136
  decisions/s. Prefill accounted for 2,415.31 ms on average (1,736 graph
  nodes); its backend copy and synchronization counters were 0.26997 ms and
  0.00169 ms. Categorical readout was 0.03278 ms (2 graph nodes), with
  0.000254 ms backend copy and 0.0000218 ms synchronization. Normalization
  averaged 0.000948 ms. CUDA initialization reported no CUDA-capable device,
  so this current categorical baseline is CPU-only in this environment.
- 2026-09-20: The same four-row E4B Q4_K_M CPU run measured 4,761.02 ms mean
  request time, p50/p95 4,521.60/5,344.14 ms, and 0.2100 decisions/s.
  Prefill averaged 4,758.72 ms (2,163 graph nodes), with 0.22477 ms copy and
  0.000210 ms synchronization. Readout averaged 0.03304 ms (2 graph nodes),
  with 0.000246 ms copy and 0.0000188 ms synchronization. These values are
  backend-specific observations, not portable performance claims.
- 2026-09-20: A current E2B CPU image request using the local 42x64 PNG
  `third_party/ggml/examples/yolo/data/labels/72_5.png` measured 3.58 ms image
  preprocessing, 1,942.66 ms Vision (877 graph nodes), 1.02045 ms Vision
  backend copy, and 0.000136 ms Vision synchronization. Prefill was
  4,338.34 ms (1,740 graph nodes, 0.24420 ms copy, 0.000239 ms
  synchronization), readout was 0.02911 ms (2 nodes), and request total was
  6,285.99 ms. The small source image is accepted and resized by the model's
  preprocessing contract; this is a boundary/lifetime measurement, not an
  image-resolution performance sweep.
- 2026-09-20: An E2B CPU request with 16 displayed options and a 226-token
  prompt measured 5,865.72 ms Prefill and 5,885.94 ms total. Readout remained
  a single 2-node gather at 0.04467 ms (0.000856 ms copy, 0.000089 ms
  synchronization). Candidate descriptions therefore increase Prefill input
  cost, while categorical readout does not introduce one forward pass per
  option.
- 2026-09-20: The current single-backend ownership and allocation boundaries
  are recorded as follows. `BackendContext` owns the selected backend, buffer
  type, graph allocators, and synchronization for the engine lifetime;
  `ModelLoader` owns backend-resident text/vision weights for that same
  lifetime. `VisionEncoder` owns temporary Vision graph tensors for one call
  and returns request-scoped backend-resident `VisualTokens`, retained until
  Prefill consumes them. `PrefillEngine` uses temporary graph tensors for one
  execution and returns `PrefillState`, whose request-scoped `StateCache` and
  final-position logits remain backend-resident through readout. The
  categorical gather uses a temporary two-node graph and copies only the
  requested 2--16 scalars to host; host softmax and result assembly finish
  before request state is released. No cross-backend copy exists in this
  baseline. These boundaries agree with the placement/lifetime contract in
  `docs/architecture.md`.
- 2026-09-20: Stage 4.1 currently supports deferring backend separation.
  Prefill dominates text requests, and Vision plus Prefill dominate image
  requests; measured copy/synchronization costs are negligible on CPU, while
  the readout is already sub-millisecond. A Vision boundary remains a
  measurement candidate, but no split is justified by the single-backend
  baseline alone.
- 2026-09-20: The initial sandbox run hid `/dev/nvidia*`, causing CUDA
  initialization to report no device. A privileged read-only check found one
  RTX 2060 SUPER (compute capability 7.5, 7.8 GiB VRAM), and the current
  categorical benchmark was rerun on CUDA with the same four-row fixture and
  one warmup. E2B measured 40.71 ms mean request time, p50/p95
  39.24/44.76 ms, and 24.54 decisions/s; Prefill averaged 39.18 ms and
  readout 0.0491 ms (0.0179 ms copy, 0.000485 ms synchronization). E4B
  measured 66.56 ms mean request time, p50/p95 66.85/76.20 ms, and 15.01
  decisions/s; Prefill averaged 64.86 ms and readout 0.0526 ms (0.0186 ms
  copy, 0.000481 ms synchronization). The GPU run confirms the same boundary
  conclusion: Prefill dominates and the readout is not a useful independent
  backend candidate on this path.
- 2026-09-20: The same E2B image request used for the CPU boundary check was
  rerun on CUDA. Image preprocessing was 3.52 ms, Vision 88.94 ms (877 graph
  nodes, 0.53481 ms copy, 0.001016 ms synchronization), Prefill 88.45 ms
  (1,740 nodes, 0.12726 ms copy, 0.00111 ms synchronization), readout
  0.0908 ms (0.02227 ms copy, 0.000480 ms synchronization), and request total
  182.38 ms. The measured Vision-to-Prefill transfer is currently an
  in-backend tensor lifetime, not a cross-backend copy.
