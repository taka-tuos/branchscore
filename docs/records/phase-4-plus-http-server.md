# Phase 4+ - Sequential HTTP server: execution record

This is the dated implementation, portability, and verification record
extracted from
[`phases/phase-4-plus-http-server.md`](../phases/phase-4-plus-http-server.md).
The current HTTP contract remains in the phase document; this file keeps the
longer chronology and environment-specific evidence out of that contract.

## Notes / Findings

- 2026-09-24: Investigated HTTP dependency portability at branchscore
  `04f25fa`. Before the vendoring change, the default server build required
  both pkg-config and exactly `libllhttp=9.3.1`; there was no CMake-package,
  source-build, or bundled fallback. Consequently an Ubuntu 24.04
  installation without the development package could not configure even the
  CLI targets unless the server was disabled. A different installed llhttp
  version also failed the exact check. The core does not use llhttp: only
  `tools/branchscore_server.cpp` and its executable link do. Socket handling
  uses POSIX APIs available on Linux; replacing the HTTP implementation was
  not necessary to solve this dependency issue.
- 2026-09-24: Inspected the upstream generated-source release
  [llhttp release/v9.3.1](https://github.com/nodejs/llhttp/tree/release/v9.3.1).
  It contains `src/{llhttp,http,api}.c`, `include/llhttp.h`, and an MIT
  license; consuming these generated files needs no Node.js/npm. Its CMake
  supports `LLHTTP_BUILD_SHARED_LIBS=OFF` / `LLHTTP_BUILD_STATIC_LIBS=ON`,
  but requires CMake 3.25, above this project's advertised 3.20 minimum. Use
  the generated `release/v9.3.1` archive, not the generator-source `v9.3.1`
  tag. The downloaded codeload archive SHA-256 was
  `c14a93f287d3dbd6580d08af968294f8bcc61e1e1e3c34301549d00f3cf09365`.
- 2026-09-24: Implemented the vendored-source follow-up. Generated
  `third_party/llhttp/generated/{src/llhttp.c,include/llhttp.h}` and the
  native helper C sources are built as a private static
  `branchscore::llhttp` target; the previous pkg-config discovery and exact
  system-version requirement were removed. The v9.3.1 TypeScript grammar,
  generator, npm lockfile, native headers, and license remain under
  `third_party/llhttp/upstream`, and the opt-in
  `branchscore-llhttp-regenerate` target refreshes the generated files. Normal
  builds therefore work without `libllhttp-dev`, pkg-config, Node.js, or
  network access. Regeneration still needs Node.js/npm and the locked package
  downloads. A small local CMake target is used instead of upstream llhttp's
  CMake because the latter requires CMake 3.25 while branchscore supports 3.20.
- 2026-09-24: The vendored generated C and header hashes match the official
  `release/v9.3.1` archive. A build with an empty pkg-config search path
  produced `branchscore-server`; the regeneration target also completed after
  allowing npm registry access and left the generated files byte-identical.
- 2026-09-24: Verification on the available Arch Linux host: hiding system
  pkg-config directories reproduced the configure failure at CMake line 20.
  Built upstream generated llhttp 9.3.1 statically into a temporary prefix,
  then configured the unchanged project with `PKG_CONFIG_LIBDIR` pointing only
  to that prefix's `lib/pkgconfig`. The CPU `branchscore-server` target built
  successfully. This was the pre-vendoring workaround; it is retained as
  diagnostic evidence. The committed vendored target removes the need for
  this user-prefix and pkg-config setup. Ubuntu 24.04 execution and
  model-backed HTTP inference were not tested here.
- 2026-09-23: The sequential HTTP server, adapter, and in-memory image path
  are implemented. `branchscore-server` originally used a separately
  installed llhttp 9.3.1 package, pinned by an exact pkg-config version check;
  the dependency was later vendored as generated C with its generator inputs
  kept under `third_party/llhttp`. Set `BRANCHSCORE_BUILD_SERVER=OFF` for a
  CLI-only build if the server target is not wanted. The server defaults to
  `127.0.0.1:8080`; the advertised model ID is `branchscore-local`. For a
  non-loopback bind it requires `BRANCHSCORE_BEARER_TOKEN` before backend or
  model initialization.
- 2026-09-23: Focused verification passed the adapter, decision-contract, and
  image-preprocessor CTests (3/3). Live E2B CPU HTTP checks covered health,
  text and PNG Choice requests, sequential two-question dispatch, malformed
  JSON (400), unsupported Score (422), unsupported content type (415),
  malformed encoded image (422), oversized body/image (413), failure
  recovery, request-ID echo/logging, and bearer authentication. A
  `localhost.` alias resolving to loopback exercised the non-loopback
  authentication branch without exposing a LAN listener; a missing-token
  startup check with `0.0.0.0` and nonexistent model paths confirmed
  credentials are rejected before model loading.
- 2026-09-23: The HTTP text request and direct core CLI request produced the
  same prompt identity and exact answer-slot logits. The base64 PNG HTTP path
  and path-based core CLI request also produced the same prompt identity and
  exact logits. On live E2B CPU requests, measured HTTP handling time was
  2,373.38 ms for a 2,373.12 ms engine evaluation, 4,942.17 ms for two
  evaluations totaling 4,941.97 ms, and 6,613.51 ms for a 6,613.32 ms image
  evaluation. Access logs measured wall time through response writing.
- 2026-09-23: The initial Choice envelope adapter is implemented in the
  transport-facing `branchscore_systemone` target, outside
  `branchscore_core`. It accepts 1--16 Choice questions and 2--16 string/null
  criteria, traverses question and criteria keys in the JSON codec's lexical
  order, preserves criteria keys as option IDs, and renders non-null options
  as `key: description` (null becomes the key alone). Structured object/array
  state is serialized with the existing deterministic JSON codec. The adapter
  projects choice/probabilities/confidence/usage and namespaced per-question
  prompt, scoring, timing, and optional raw-logit metadata without including
  prompt text, token IDs, or Vision debug values. The user selected 16 as the
  per-request question limit.
- 2026-09-23: The adapter now accepts the documented single-image extension
  as in-memory base64 JPEG/PNG bytes shared across its sequential
  `DecisionRequest`s. `ImagePreprocessor::inspect_encoded` reads dimensions
  from encoded bytes before pixel allocation, while `load_encoded` uses the
  same preprocessing path as path-based images. The engine rejects requests
  that supply both a filesystem path and in-memory bytes; CLI/JSONL path input
  remains unchanged.
- 2026-09-23: `Gemma4DecisionEngine::evaluate` already owns all per-request
  Vision/Prefill/readout state and returns `DecisionResult`; model and backend
  live outside it. `branchscore-bench` demonstrates repeated sequential calls
  with one warm-loaded engine. Its JSONL rows contain several benchmark-only
  fields and prompt token IDs, so the HTTP result needs a smaller explicit
  projection. The original engine boundary remains the only decision entry
  point for both the CLI and HTTP server.
- 2026-09-23: The current Phase 4 CUDA four-row baseline measured mean E2B/E4B
  text requests at 40.71/66.56 ms and one E2B image request at 182.38 ms.
  Prefill dominates text requests, and Vision plus Prefill dominate the image
  request. These are local sequential observations, sufficient to plan an
  HTTP entry point without first splitting the backend.
- 2026-09-23: TypeSafe's public HTTP API uses `POST /v1/systemone` with
  `state`/`model`/question-ID-keyed `questions`, and Choice answers return
  `choice` plus a `probabilities` map. The current engine can map criteria to
  semantic option IDs and scores. TypeSafe additionally requires `confidence`
  for Choice, whose equivalent this engine does not provide; its other
  primitives, calibrated model behavior, and parallel multi-question
  execution cannot be promised by an HTTP shape adapter. See the public API
  and Choice pages: https://docs.typesafe.ai/api and
  https://docs.typesafe.ai/primitives/choice .
- 2026-09-23: For HTTP shape compatibility, the user chose a temporary
  `confidence: 1.0` on every Choice answer. This is a constant placeholder,
  independent of logits/probabilities. Expose that fact in `branchscore`
  metadata and documentation; do not change the engine's probability contract
  or label the value calibrated confidence.
- 2026-09-24: Added a self-contained `GET /ui` page to the server. It provides
  the bearer token field, health check, Choice state/question/options form,
  optional PNG/JPEG input, and result probability/logit/timing display. The
  token remains in page memory and is sent only as an Authorization header.
  The exact `/ui` route is public for browser navigation; all existing API
  routes retain their non-loopback bearer check. The response adds no external
  assets and uses restrictive cache/security headers. A CPU E2B server bound to
  `0.0.0.0` returned 200 for `/ui` without a token, 401 for `/healthz`
  without or with a wrong token, 200 for `/healthz` with the configured token,
  and 401 for `/ui?x=1` without a token.
