# Phase 2 - Single-backend baseline: execution record

This is the dated implementation and verification record extracted from
[`phases/phase-2-single-backend.md`](../phases/phase-2-single-backend.md).
Phase 2 is complete and its continuation scorer was superseded by Phase 2+
and Phase 3+. The entries below are retained as historical evidence only.

## Notes / Findings

- 2026-09-19: Began Stage 2.1 in the independent `branchscore/` repository.
  Added `ggml-org/ggml` as `third_party/ggml` submodule at revision
  `456172ec733a135778adcd32d00e576a58232e45` (ggml 0.24.0), rather than
  depending on or wrapping llama.cpp at runtime.
- 2026-09-19: Completed the initial Stage 2.1 project layout and CMake
  integration. CPU is the default; CUDA and Vulkan are opt-in build options.
  The initial `BackendContext` enumerates ggml devices, selects `auto`, an
  exact device name, or a CPU/CUDA/Vulkan backend family, owns exactly one
  backend and its default buffer type, and provides explicit synchronization.
- 2026-09-19: CPU configure/build, device enumeration, initialization, and the
  focused backend test pass locally. CUDA 13.3 and Vulkan builds pass. On the
  local RTX 2060 SUPER host, both CUDA and Vulkan enumerate, initialize, and
  synchronize successfully. NCCL is disabled because multi-GPU execution is
  outside project scope.
- 2026-09-19: Completed the initial Stage 2.2 loader. It parses and validates
  Gemma 4 metadata, normalizes scalar/array integer differences, locates
  required global/per-layer/vision tensors by name, preserves GGUF tensor
  types, streams weights directly into one selected backend buffer, and keeps
  E2B/E4B dimensions in `TextModelConfig`/`VisionModelConfig`. The mmproj load
  selects only `v.*` and `mm.input_projection.weight`, excluding audio.
- 2026-09-19: Actual E2B loads pass on CPU, CUDA, and Vulkan. E2B contains 601
  text tensors (2.88 GiB) plus 659 selected vision tensors (0.34 GiB). E4B
  loads on CUDA with the same code and contains 720 text tensors (4.62 GiB)
  plus 659 selected vision tensors (0.34 GiB). The matching projector widths
  are validated as 1536 for E2B and 2560 for E4B.
- 2026-09-19: Completed Stage 2.3 with an independent Gemma 4 tokenizer loaded
  from GGUF metadata. It implements raw-UTF-8 BPE, `▁` whitespace escaping,
  newline-run handling, byte fallback, BOS and special-token parsing, exact
  prefix-boundary validation, and observable option IDs/pieces. It does not
  link or call llama.cpp.
- 2026-09-19: Compared token IDs against `llama-tokenize` from the inspected
  llama.cpp revision for English, Japanese, emoji/accent/punctuation, repeated
  newlines, Gemma 4 image/turn/EOS tokens, and special-token-disabled input;
  results match. Focused model-backed tests pass with both local E2B and E4B
  GGUFs, including rejection when appending an option retokenizes the prefix.
- 2026-09-19: Completed Step 2.4.1 image loading and preprocessing. Encoded
  images are decoded through the `stb_image` copy already present in the pinned
  ggml submodule, converted to RGB, aspect-ratio resized with the same
  Pillow-compatible bicubic rules as the inspected Gemma 4V reference, aligned
  to the patch/pooling grid, normalized from GGUF mean/std metadata, and laid
  out as planar floats ready for a ggml input tensor.
- 2026-09-19: Gemma 4V uses a pooling factor of 3 and limits resized images to
  70--1120 post-pooling visual tokens. A 1x1 image becomes 432x432 / 81 visual
  tokens with the local E2B mmproj, matching the inspected llama.cpp
  preprocessing output. A real PNG decode produced a 432x384 / 72-token image;
  focused CPU tests cover upscaling, downscaling, normalization, layout, and
  token counts.
- 2026-09-19: Completed Steps 2.4.2 and 2.4.3 with a direct ggml Gemma 4V
  graph: patch convolution, learned 2D positions and RoPE, 16 RMSNorm
  transformer blocks, clippable attention/GEGLU projections, 3x3 average
  pooling, and final projection to the E2B/E4B LLM width. The result is exposed
  as token-major float32 embeddings ready for Prefill; a binary debug dump is
  available for parity checks.
- 2026-09-19: The minimum 432x432 case produces 81x1536 finite E2B embeddings.
  CPU output was compared elementwise with the inspected llama.cpp reference
  for a white image: 124,416 values had mean absolute error 2.52e-4 and maximum
  absolute error 3.75e-3. The graph runs without CPU fallback on the same
  selected backend as the weights: about 2.0 s on CPU, 133 ms on CUDA, and
  9.62 s on Vulkan on the local RTX 2060 SUPER.
- 2026-09-19: Before starting Prefill, corrected visual-token ownership to
  match the architecture contract: the temporary vision graph copies its
  output directly into a persistent tensor on the selected backend. Normal
  execution no longer round-trips the full embedding through host memory;
  host download is explicit and used only by the parity/debug dump.
- 2026-09-19: Began Stage 2.5 by adding backend-resident `StateCache` storage
  sized to the concrete request, with per-layer full/SWA K/V widths, Gemma 4's
  late-layer reuse mapping (last stored full or SWA source), an immutable
  frozen-prefix boundary, and a rewindable sequential branch cursor. The
  E2B/E4B RoPE and RMSNorm metadata needed by the text graph is now retained
  in `TextModelConfig`.
- 2026-09-19: Implemented the Stage 2.5 Prefill graph for token IDs plus an
  optional backend-resident visual-token span. Text embeddings receive Gemma
  4's input scale while visual embeddings remain raw; padding-token per-layer
  embeddings, per-layer input projection/injection, alternating SWA/full
  attention, shared late-layer KV, gated FFN, layer output scale, tied LM head,
  and final logit softcap follow the inspected reference.
- 2026-09-19: KV copies must be explicitly expanded into the ggml graph before
  attention reads the persistent cache views. Merely feeding a `ggml_cpy`
  result into a later view did not preserve the required execution ordering.
  Gemma 4 text RoPE uses NeoX ordering; full-attention layers additionally use
  the global `rope_freqs.weight` factors when present.
- 2026-09-19: CPU E2B Prefill passes for a two-token text prefix and for a
  mixed 83-position prefix containing 81 real visual embeddings. The top
  logits for `BOS + Hello` match the inspected llama.cpp reference candidate
  set and are numerically close (small differences are expected from its F16
  KV cache and CPU weight repacking versus this baseline's F32 cache). The same
  text and multimodal test passes on the RTX 2060 SUPER CUDA backend in about
  2.4 s total, versus about 6.6 s on CPU.
- 2026-09-19: Added sequential option-tail scoring. Each option rewinds the
  branch cursor, restores prefix logits into a separate working tensor, scores
  the first token from the Prefill result, and runs one-token continuation
  graphs only for preceding tokens. Full-vocabulary `ggml_soft_max_ext` plus
  `log` computes each target log-probability on the selected backend; host
  code accumulates and reports sum/mean values and per-token values.
- 2026-09-19: CPU E2B tests verify a two-token option, target-logprob parity
  against a host calculation, deterministic repeated branch scoring, and
  prefix cursor isolation. Multiple-option probability softmax and final
  decision aggregation remain outside the model graph.
- 2026-09-20: Completed Step 2.6.4 with a host-side stable softmax over
  `sum_logprob`. The summary preserves input order, selects the first exact
  maximum, records ties, and returns relative probabilities explicitly as
  conditional option probabilities. A focused test covers large negative
  scores, normalization to one, and first-max tie behavior.
- 2026-09-20: Completed Stage 2.7 with a `ChatTemplate` source boundary. The
  default source is now required from GGUF `tokenizer.chat_template`; the CLI
  accepts `--chat-template-file` as a source override and reports its origin
  alongside the rendered prefix IDs. The Phase 2 renderer intentionally covers
  only Gemma 4's plain system/user path, one `<|image|>` marker, and the model
  generation prompt. It validates the Jinja source shape but does not add a
  general Jinja engine or tool/media branches; arbitrary template semantics are
  deferred behind the same boundary rather than silently approximated.
- 2026-09-20: The Stage 2.7 CLI now accepts `--state`, `--question`, and
  repeated `--option ID=DESCRIPTION`, validates 2--16 unique options, and
  reports rendered Prefix IDs, option IDs, sum/mean log-probabilities,
  conditional softmax probabilities, selection, and Vision/Prefill/score/total
  timings. The focused CTest suite passes 7/7. An E4B Q4 CPU run succeeded for
  text-only scoring, an explicit file-sourced copy of the GGUF template, and a
  77-visual-token image request; the latter also wrote the existing binary
  vision debug dump. The two implementation commits are `ccf2377` (source
  boundary) and `e04f84f` (CLI). The README usage and current required CLI
  arguments were synchronized in `39b3877`.
- 2026-09-20: Phase 2+ supersedes the Stage 2.7 source-owning `ChatTemplate`
  path with the fixed `gemma4-fixed-v1` renderer and moves request
  orchestration into `Gemma4DecisionEngine`. The Stage 2.7 source-boundary
  details above remain historical evidence only.
