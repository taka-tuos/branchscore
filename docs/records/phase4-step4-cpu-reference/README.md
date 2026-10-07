# Step 4 CPU reference controls (2026-10-01)

These are measurement harnesses, not branchscore runtime components. They do
not generate or consume answer tokens. Run only one model/request at a time.
The upstream source is `/home/ubuntu/llama.cpp` at
`19e28a27702117d8f2eb16b825b9a308111f67d9`; use the matching upstream headers.

Build a separate reference library with Release, native CPU, OpenMP and CPU
REPACK enabled; CUDA, BLAS and LLAMAFILE disabled. This run used
`/tmp/step4-llama-matched-make`. The original upstream build remains intact.
Reference ggml comes from the upstream tree and differs from branchscore's
pinned ggml; matching flags do not imply identical kernels.

```sh
cmake -S /home/ubuntu/llama.cpp -B /tmp/step4-llama-matched-make \
  -DCMAKE_BUILD_TYPE=Release -DGGML_LLAMAFILE=OFF -DGGML_NATIVE=ON \
  -DGGML_CPU_REPACK=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build /tmp/step4-llama-matched-make --target llama -j 2
```

Compile `image-buffer.cpp`, `text-buffer.cpp` and `text-trace.cpp` with C++17,
upstream `include`, `ggml/include` (and `tools/mtmd` for the image harness),
linking `llama`, `ggml` and `ggml-base` from that build's `bin`, with its rpath.
For original-build controls, link the same source to the original `build/bin`.

- `image-buffer.cpp MODEL MMPROJ BEFORE_IDS AFTER_IDS ANSWER_IDS EMBEDDINGS EXTRA`:
  exact whitespace-separated IDs and headerless F32 image embeddings. MMPROJ
  is provenance only, not executed. Text/image/text are three decode calls,
  each capped at 512 positions. EXTRA is `0` or `1` for `use_extra_bufts`.
- `text-buffer.cpp MODEL IDS LOGITS ALL EXTRA`: exact text IDs; ALL selects
  final-position-only (`0`) or every-position (`1`) output. Reads the last
  output with `llama_get_logits_ith(ctx, -1)` in either case.
- `text-trace.cpp`: same arguments, with selected intermediate tensor dumps to
  `/tmp/step4-trace-ref`. Create that directory first. The second trace run
  changed only this output directory to `/tmp/step4-trace-ref-plain`.
- `vision-embeddings.cpp MODEL MMPROJ LIST`: link against branchscore's built
  core and ggml. LIST contains image/output path pairs; encode each distinct
  image once, save headerless projected F32 values and print shape. This run
  reproduced the five previously saved embeddings byte for byte.

The fixed A/B reference path additionally sets `GGML_CPU_TILED_MM=0` and
`use_extra_bufts=false`; it has F32 K/V, ordinary attention, full SWA,
`n_batch=n_ubatch=512`, four threads. `GGML_NO_IQ_PANEL=1` was also set for the
kernel controls; branchscore's old IQ panel type list does not include this
GGUF's Q4_K/Q6_K matrices. No production default was changed.

`prefill-trace.patch` is diagnostic instrumentation for a temporary copy of
`src/prefill_engine.cpp`: preserve selected graph outputs and dump them after
compute to `/tmp/step4-trace-branch`. Compile that copy and place its object
before `libbranchscore_core.a` when linking the existing bench object. The
normal project source/binary remains unchanged. Raw tensor dumps can be large
and are temporary; the record saves their shapes, hashes and comparison
metrics. Trace results are separately checked against uninstrumented scores.

`prefill-key-padding-control.patch` applies on top of the temporary trace copy.
It allocates zeroed F32 K/V capacity rounded to 32 positions (with one reference
context slack position), pads each attention key axis to the reference's
`min(capacity, max(256, ceil(used/256)*256))`, and retains the original logical
positions and causal masks. This is a diagnostic shape control, not an adopted
cache change. Its dumps go to `/tmp/step4-trace-branch-padded`.

## C/D CPU screening (2026-10-02)

`image-cd-flash.cpp` is the reference image harness with F16 K/V and forced
Flash Attention. Link it to the same matched reference build as above.
`prefill-cd-flash.cpp` and `state-cache-cd-f16.cpp` are measurement copies of
current project source, with F16 caches/masks, Flash requesting F32 accumulators,
zeroed 256-position physical padding and per-node backend support checks.
They preserve 512-token all-layer microbatches and full-length SWA. Compile
these two objects before `libbranchscore_core.a` when linking `branch-cd.cpp`
with the project's ggml static libraries, OpenMP, dl, m and pthread.
No trace tensors are retained, and no production source/default is changed.
The CPU vector Flash path still accumulates the weighted V sum in F16 when
V is F16, even with this precision request. Query counts below 64 select that
path; tiled Flash uses an F32 V accumulator. C's initial 38-token decode and
D's 11-token tail therefore differ from their larger query batches. See the
[C/D review](../phase-4-plus-option-scale-2/cd-cpu-evaluation.md#cd-cpu-review-2026-10-02).

`branch-cd.cpp` uses exact saved before/after/answer IDs, encodes the fixture
image to obtain the normal VisualTokens allocation, then replaces its tensor
with the same saved F32 embeddings used by the reference. The timed Prefill
excludes that allocation/encoding; process memory would include it. The final
vocabulary logits are downloaded only by this research harness. No answer is
generated. CPU thread count is four, including the reference's documented
`GGML_DEFAULT_N_THREADS=4` default.

`run-cd.py` records stdout/stderr per case under `/tmp/step4-cd-20261002`, runs
all C then all D sequentially, and stops on failure. Binary paths are
`/tmp/step4-image-cd-flash` and `/tmp/step4-branch-cd-flash`.
`report-cd.py` writes durable candidate scores, probabilities, pairwise deltas,
selection/correctness transitions, headers and source/log hashes to
`../phase-4-plus-option-scale-2-cd-e4b-q4-cpu-2026-10-02.jsonl`.
Its manifest input `/tmp/step4-cd-manifest.json` saves actual CMake flag values;
the emitted JSONL preserves that manifest after temporary files are removed.
Missing C/D pairs are excluded and the denominator is explicit.
The predeclared CPU screening criteria are in
[cd-cpu-evaluation.md](../phase-4-plus-option-scale-2/cd-cpu-evaluation.md).
