# Phase 4+ - Sequential HTTP decision server (draft)

## Position

Phase 4's CPU/CUDA measurements show useful single-request latency without a
backend split. Add a small HTTP entry point around the existing categorical
engine before request workers or a scheduler. This is a planning draft, not an
implementation or a claim that Phase 4's optional backend stages are complete.

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md` (current contract)
- `docs/phases/phase-4-backend-separation.md` (measurements and ownership)

## Proposed first contract

- New `branchscore-server` executable; `--model`, `--mmproj`, `--backend`,
  `--host`, and `--port` are startup options. Load model/tokenizer once before
  listening. Default bind address: `127.0.0.1`; LAN binding is explicit.
- For a non-loopback bind, require a bearer token loaded from an environment
  variable at startup; reject missing credentials before loading the model.
  Do not put the token in command-line arguments or logs. The server itself
  speaks HTTP; use TLS termination at a reverse proxy when the network is not
  fully trusted. No public-internet deployment target in this phase.
- `POST /v1/systemone` is the primary HTTP entry point. It follows the
  TypeSafe HTTP envelope: `state`, `model`, and a `questions` map keyed by
  caller-defined question IDs. Accept only `type: "choice"` with string
  `instructions` and 2--16 `criteria` entries whose values are strings or
  null. Evaluate multiple Choice questions sequentially against the same
  state. Set an explicit small question-count limit; no shared state cache or
  parallel execution is implied.
- Optional `request_id` is a branchscore extension for tracing and response
  echo; it does not enter any model prompt.
- `model` accepts one advertised local model ID selected at server startup;
  reject other IDs. Never accept `jev-latest` as an alias for Gemma 4. A
  structured JSON `state` can be serialized as text for the existing engine,
  but its model semantics are not claimed equivalent to Jev's structured
  state. Preserve each question ID only for response mapping; it is not in
  the prompt.
- Map each `criteria` key to an option ID. Compose the key and its description
  into model-visible option text, since TypeSafe shows both to its model.
  For null descriptions, show the key alone. The current JSON codec stores
  object keys in lexical order, so option order is deterministic by key; this
  differs from a caller's original JSON insertion order and must be tested.
- Optional branchscore extension for one image:
  `image: {"media_type":"image/png|image/jpeg",
  "data_base64":"..."}`. Decode in memory through the existing image
  preprocessing path. Do not expose server filesystem paths in the HTTP API.
  Set body, decoded-image, and image-dimension limits before decoding. TypeSafe
  Jev currently supports text only; image requests are a local extension.
- Success follows the TypeSafe outer shape: `model`, `answers` keyed by the
  supplied question IDs, and `usage`. Each answer has `type: "choice"`,
  `choice` (selected criteria key), `probabilities` keyed by every criteria
  key, and `confidence: 1.0` as a fixed compatibility placeholder.
  `usage.input_tokens` sums rendered text prompt token counts across
  sequential questions (and explicitly excludes visual tokens until those are
  accounted for); `usage.output_tokens` is 0 because no answer token is
  generated. Add a clearly namespaced `branchscore` object for `schema_version:
  2`, `confidence_kind: "constant_placeholder"`, scoring/prompt identities,
  timings, and raw logits when diagnostics are desired. HTTP output omits
  rendered token IDs, full prompt, and Vision debug values. Relative
  probabilities remain uncalibrated and conditional on the supplied option
  set. Clients must not use this fixed `confidence` for threshold decisions;
  it does not express certainty or reproduce TypeSafe's computation.
- `GET /healthz` reports readiness after startup. No generation, OpenAI API
  emulation, model upload, or generic model registry.
- One process owns one model/backend/engine. Handle one decision at a time.
  Bound HTTP request size and connection wait time; close each response
  connection. A later phase may add bounded request admission after measuring
  concurrent callers. Keep `Gemma4DecisionEngine::evaluate` unchanged as the
  decision boundary and avoid sharing its backend state concurrently.
- Return a small JSON error object with a stable error code and message.
  Invalid JSON syntax -> 400; missing/invalid token -> 401; valid JSON with
  unsupported question type or invalid fields -> 422; unsupported media type
  -> 415; too-large body/image -> 413; unexpected engine failure -> 500. Do
  not expose model paths or internal stack details to clients. Server
  logs retain request ID, status, and wall time without logging state or image
  data by default.

Example request (text only):

```json
{"state":"The service is healthy.","model":"branchscore-local","questions":{"action":{"type":"choice","instructions":"Which action should be taken?","criteria":{"keep":"Keep it running","stop":"Stop it"}}}}
```

Example response shape (illustrative probabilities and token count):

```json
{"model":"branchscore-local","answers":{"action":{"type":"choice","choice":"keep","probabilities":{"keep":0.9,"stop":0.1},"confidence":1.0}},"usage":{"input_tokens":86,"output_tokens":0},"branchscore":{"schema_version":2,"confidence_kind":"constant_placeholder"}}
```

## Implementation slices

1. Add a narrow TypeSafe-envelope adapter that converts supported Choice
   questions to existing `DecisionRequest` values and projects
   `DecisionResult` into `answers`. Keep benchmark run/aggregate JSONL fields
   in the runner. Verify option IDs and deterministic ordering.
2. Add in-memory JPEG/PNG decoding to `ImagePreprocessor` and an image-bytes
   input path to `DecisionRequest`; preserve the CLI/JSONL path behavior.
3. Add the HTTP executable with bounded parsing, sequential dispatch, startup
   readiness, authentication, error mapping, and shutdown. Choose and pin a
   small HTTP parser implementation during this slice; keep HTTP code outside
   `branchscore_core`.
4. Verify Choice text/image results against the core engine, question-ID and
   option-ID mapping, unsupported Score/Noul rejection, malformed and oversized
   requests, repeated requests after failure, LAN authentication, and
   sequential latency. Record end-to-end HTTP overhead separately from
   `evaluate` timing.

## Direction

- The first version supports LAN callers, with an explicit non-loopback bind
  and bearer token. TLS termination is a deployment concern for networks that
  need encryption.
- Compatibility targets TypeSafe's URL and HTTP JSON envelope for the Choice
  subset, not its SDK or its model outputs. TypeSafe also supports Score,
  Noul, up to 255 Choice options, and parallel mixed questions; those are
  outside this engine's current contract. Requests can contain multiple
  supported Choice questions, but they run sequentially.
- Keep the first version synchronous. The first concurrency change, if needed,
  should be chosen from observed queue delay and memory use, not inferred from
  single-request compute speed.

## Notes / Findings

- 2026-09-23: `Gemma4DecisionEngine::evaluate` already owns all per-request
  Vision/Prefill/readout state and returns `DecisionResult`; model and backend
  live outside it. `branchscore-bench` demonstrates repeated sequential calls
  with one warm-loaded engine. Its JSONL rows contain several benchmark-only
  fields and prompt token IDs, so the HTTP result needs a smaller explicit
  projection. `ImagePreprocessor` currently decodes from a path, while the HTTP
  boundary needs in-memory bytes for a usable client image contract.
- 2026-09-23: The current Phase 4 CUDA four-row baseline measured mean
  E2B/E4B text requests at 40.71/66.56 ms and one E2B image request at
  182.38 ms. Prefill dominates text requests, and Vision plus Prefill dominate
  the image request. These are local sequential observations, sufficient to
  plan an HTTP entry point without first splitting the backend.
- 2026-09-23: TypeSafe's public HTTP API uses `POST /v1/systemone` with
  `state`/`model`/question-ID-keyed `questions`, and Choice answers return
  `choice` plus a `probabilities` map. The current engine can map criteria to
  semantic option IDs and scores. TypeSafe additionally requires `confidence`
  for Choice, whose equivalent this engine does not provide; its other
  primitives, calibrated model behavior, and parallel multi-question execution
  cannot be promised by an HTTP shape adapter. See the public API and Choice
  pages: https://docs.typesafe.ai/api and
  https://docs.typesafe.ai/primitives/choice .
- 2026-09-23: For HTTP shape compatibility, the user chose a temporary
  `confidence: 1.0` on every Choice answer. This is a constant placeholder,
  independent of logits/probabilities. Expose that fact in `branchscore`
  metadata and documentation; do not change the engine's probability contract
  or label the value calibrated confidence.
