# 2026-10-07 CUDA validation artifacts

These are sequential measurement harnesses, not production API changes.
See [interpretation and limits](../phase-4-plus-option-scale-2/gpu-validation.md).
`manifest.json` records model hashes, revisions and CMake flags.
`report.json` retains candidate logits, centered deltas, probability deltas,
selection/correctness transitions and allocation/sample measurements.
The per-run `*-gpu.json` files record exact commands, exit codes and device-wide
VRAM samples; `*.log`/`*.log.gz` retain stdout/stderr (reference logs are losslessly gzip-compressed). CUDA runs require GPU device access.

## Reproduction

This evidence predates production adoption. Its ordinary/F32 control requires
an isolated checkout at `f08cadf7c210cc1a80224e2943db0da35833c3ec` and copies
of these artifacts; the current CUDA core now uses F16/Flash. Do not rerun
these controls against the new default and label them ordinary attention.
See the [production adoption evidence](../option-scale-2-runtime-512-2026-10-07/README.md)
for the current implementation.

From the branchscore repository root, with the existing Q4 GGUF/mmproj paths:

```sh
cmake -S . -B build-cuda -DBRANCHSCORE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda -j 24
mkdir -p /tmp/branchscore-llama-reference-19e28a2
git -C ../llama.cpp archive 19e28a27702117d8f2eb16b825b9a308111f67d9 \
  | tar -x -C /tmp/branchscore-llama-reference-19e28a2
cmake -S /tmp/branchscore-llama-reference-19e28a2 \
  -B /tmp/branchscore-llama-reference-build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_COMMON=OFF
cmake --build /tmp/branchscore-llama-reference-build --target llama -j 16
python3 docs/records/option-scale-2-gpu-2026-10-07/build-harnesses.py
```

`run-existing.py` executes the configured E2B CUDA CTest, three E4B tests, then
three fixture sets for each model. Set the `BRANCHSCORE_TEST_MODEL`,
`BRANCHSCORE_TEST_MMPROJ`, and `BRANCHSCORE_TEST_BACKEND=cuda` CMake values as
recorded in the manifest before CTest. Configure paths for the local checkout.
The recorded CTest run passed all 11 tests without skips.

`make-inputs.py` regenerates 15 deterministic probe prompts and their label
files. Five controls use the current production prompt bytes. The text scale
cases use 16/384/512 displayed options, a unique blue item, and reversal; the
image scale cases use the original board-base image and options, unrelated
named-zone additions to 384/512, and reversal. The latter are E4B-only.
The temporary extended renderer uses uppercase **label**, while the production
renderer retains uppercase **letter** and A-P. All supplied labels are checked
for one normal token, unique ID, piece and the actual extended prompt boundary.
These are small sensitivity controls, not representative quality guarantees.

Run scripts sequentially (one model/backend/request at a time):

```sh
python3 docs/records/option-scale-2-gpu-2026-10-07/run-existing.py
python3 docs/records/option-scale-2-gpu-2026-10-07/make-inputs.py
python3 docs/records/option-scale-2-gpu-2026-10-07/run-probes.py
python3 docs/records/option-scale-2-gpu-2026-10-07/run-probes.py --image-scale
python3 docs/records/option-scale-2-gpu-2026-10-07/run-references.py
python3 docs/records/option-scale-2-gpu-2026-10-07/report.py
```

**Use a fresh artifact destination for a rerun.** The production benchmark
refuses existing output JSONL; the measurement scripts otherwise replace their
logs. The recorded commands used equivalent temporary orchestration scripts.
`run-probes.py --image-scale` reproduces the separate E4B image-scale run.
`report.py` requires the temporary token/embedding/reference files at
`/tmp/branchscore-gpu-probes`; durable candidate/reference scores are also in
`report.json`. The embedding hashes are retained, not the large binary files.

The ordinary probe directly uses the current core. The Flash executable and
benchmark link the preserved `prefill-cd-flash.cpp` and
`state-cache-cd-f16.cpp` measurement objects before the core, overriding just
Prefill and StateCache. No production default is changed. Their F16 cache bytes,
256-position capacity padding and Flash operation support checks distinguish
this path from the ordinary build. `prefill-probe.cpp` reports stage free VRAM,
exact cache/peak graph allocation bytes and final candidate scores. Stage free
memory after Prefill includes any retained CUDA pool but misses freed transient
graph allocations; neither it nor the allocator bytes is an exact VRAM peak.

Both references use the CPU record's pinned llama.cpp revision with GPU layers,
GPU device selection and an explicit `.*` tensor buffer override to CUDA, F32/ordinary or F16/Flash, full SWA and a 512-token
microbatch limit. Text decode is explicitly looped in chunks of at most 512.
Image reference decodes before/visual/after separately as in the CPU harness;
its decode shapes therefore differ from the branchscore mixed microbatch.
Image pairs consume the probe's exact CUDA-generated F32 embeddings.
Reference logs retain actual layer/KV placement, kernel path and buffer sizes.
The discarded first reference pilot kept 2,208 MiB of E4B embedding weights on
CPU despite reporting 43/43 layers offloaded; its logs/samples are isolated in
`reference-placement-pilot/` and excluded from all reported comparisons.
The rerun has only a CUDA model buffer (5,091.51 MiB), with all KV on CUDA.
This reference allocation includes its tensor duplication/layout and is not
the branchscore weight allocation. Small host input/output staging buffers
still appear in the reference scheduler logs.
The project's pinned ggml and reference ggml differ; this comparison does not
prove identical kernels. No high-precision CUDA A/B model pair was available.

`nvidia-smi` is sampled every **at least** 100 ms (plus command overhead).
The maximum is device-wide and a lower bound on the actual temporal peak.
It includes CUDA context and retained allocator pools; it is not derived from
CPU RSS. Each run is a single observation, with one warmup for fixture benches
and no warmup for the standalone probes. No final GPU adoption threshold was
inferred from the CPU criteria or chosen after observing these results.

Reference stdout/stderr destructor messages can interrupt a buffered score line.
The report removes only the explicit `~llama_context` destructor lines before
parsing, then verifies every candidate index and token ID in order. Log compression
was verified byte-for-byte; uncompressed hashes are in the manifest.

## Image placement follow-up

`run-image-placement.py` compares image-first and image-last full Prefill for
the 27 E4B vision fixtures and four 384/512 image scale inputs, on both paths.
All 124 runs are in `image-placement/`; they do not change the original report.
The image-first small prompts match production bytes and previous scores.
`summarize-image-placement.py` checks embedding identity and records family,
candidate-order/evidence/layout sensitivity and attention-path differences.
`check-image-last-reference.py` diagnoses five cases selected after observing
the results, using their exact tokens/embeddings in both reference paths.
See [results and limits](../phase-4-plus-option-scale-2/image-placement-gpu.md).
There is no cross-request cache reuse in this measurement. These scripts retain
completed files; use a separate destination for an independent rerun.
