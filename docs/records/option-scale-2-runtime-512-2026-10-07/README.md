# 2026-10-07 production 512-option validation

[Adoption and limits](../phase-4-plus-option-scale-2/runtime-512-adoption.md).
This record exercises the production engine and HTTP/CLI entry points.
It does not override Prefill with the earlier measurement-only implementation.

`manifest.json` captures the initial resource-run source/binary/model hashes
on base revision `f08cadf7c210cc1a80224e2943db0da35833c3ec` with working-tree changes.
`tokenizer-measured.cpp` preserves the tokenizer source from that run. The final
implementation additionally checks overlapping special tokens through the
appended label; its live HTTP/core score agreement and final tokenizer/CTest
checks are recorded separately. Final CLI diagnostic output and focused-test
changes also postdate the resource run. `final-manifest.json` identifies the
final sources and binaries; neither snapshot pretends that the base revision
alone contains the new implementation. `report.py` recomputes the summary
from the saved candidate scores and GPU samples.

## Recorded inputs and checks

- `inputs.jsonl`: 17 sequential cases, including 17/255/256/512 labels,
  reversed order, board images, 20-character synthetic part numbers, exact
  16,384/16,385-position admission boundaries, repetitions, and recovery.
- `results.jsonl`: every label/token ID/raw score/probability and allocation,
  stage timing and post-request free GPU bytes. `results.jsonl.effective-inputs.jsonl`
  contains the actual state text produced by the boundary-fill directives.
- `white-hd.png` and `white-fhd.png`: deterministic RGB resource controls,
  1280×720 and 1920×1080. The expected part number comes from text state.
  These controls test resource sizing and mapping, not package OCR accuracy.
- `quality-e4b.jsonl` and `vision-e4b.jsonl`: production outputs for the
  existing 24/27 fixtures. Compared to the prior F16/Flash measurement outputs.
- `*-gpu.json`: device-wide nvidia-smi samples at intervals of at least 100ms,
  plus process exit status and commands. A sample maximum is a lower bound on
  the instantaneous peak, not a precise peak or an allocator-only estimate.
- `http-checks.json`, `http-*.json`, `http-server.log`: live 17/512/FHD requests,
  ordering and score comparisons, health limits, 513-option/position/aggregate
  rejection, recovery, and UI delivery. `served-ui.js` passed `node --check`;
  interactive browser operation was not tested.
- `final-*.log`: final CUDA CTest, CPU host/tokenizer checks, model checks,
  tokenizer boundary checks, and real CLI 512-option output.

The primary GPU is RTX 2060 SUPER / CUDA architecture 75. E4B uses the hashes
in the manifests. The configured CUDA CTest uses E2B Q4_K_M, plus separate E4B
tokenizer/engine/Prefill runs. CPU retains ordinary/F32 Prefill and is checked
separately. Large 512 resource results refer to E4B CUDA only.

## Reproduction

Build the changed implementation with the pinned ggml submodule. Model paths
must match local files; none are included here. For example, from the repo root:

```sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBRANCHSCORE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DBRANCHSCORE_TEST_MODEL="$PWD/../models/e2b/gemma-4-E2B-it-Q4_K_M.gguf" \
  -DBRANCHSCORE_TEST_MMPROJ="$PWD/../models/e2b/mmproj-F16.gguf" \
  -DBRANCHSCORE_TEST_BACKEND=cuda
cmake --build build-cuda -j 24
c++ -std=c++17 -O2 -Iinclude -Ithird_party/ggml/include \
  docs/records/option-scale-2-runtime-512-2026-10-07/runtime-probe.cpp \
  -Lbuild-cuda -Lbuild-cuda/third_party/ggml/src \
  -lbranchscore_core -lggml -lggml-base \
  -Wl,-rpath,"$PWD/build-cuda:$PWD/build-cuda/third_party/ggml/src" \
  -o /tmp/branchscore-runtime-512-probe
```

**Use a fresh artifact directory for a rerun.** The probe overwrites its output,
the runner retains existing fixture JSONL, and the HTTP checker overwrites its
logs. Do not run them over the recorded evidence or mix results from different
builds. Copy `inputs.jsonl`, both PNG controls, `runtime-probe.cpp`,
`run-validation.py` and `check-http.py` into a fresh sibling directory under
`docs/records/`; running those copied scripts preserves their repository-root
discovery while writing new output there. Change input image paths if relocating
the checkout. The existing fixture images are relative to the repository root.

Run the copied `run-validation.py` with GPU device access, then its
`check-http.py` (requires Python 3 and Node.js). The latter uses an ephemeral
loopback port and shuts down the temporary server. Build the probe against the
same core binary used by these runs; do not reuse an executable built against
the earlier `DecisionResult` ABI.

The old F32/Flash-control record uses the original base implementation.
To reproduce that control, use an isolated checkout at its recorded base
revision with copies of those measurement artifacts, rather than comparing
two executables that both now take production CUDA Flash.
