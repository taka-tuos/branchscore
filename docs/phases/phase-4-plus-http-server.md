# Phase 4+ - Sequential HTTP decision server

## Position

Phase 4's CPU/CUDA measurements show useful single-request latency without a
backend split. This phase adds a small HTTP entry point around the existing
categorical engine before request workers or a scheduler. It does not claim
that Phase 4's optional backend stages are complete or justified.

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md` (current contract)
- `docs/phases/phase-4-backend-separation.md` (measurements and ownership)

## Implemented contract

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
  state. Accept 1–16 questions; no shared state cache or parallel execution
  is implied.
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
  differs from a caller's original JSON insertion order.
- Optional branchscore extension for one image:
  `image: {"media_type":"image/png|image/jpeg",
  "data_base64":"..."}`. Decode in memory through the existing image
  preprocessing path. Do not expose server filesystem paths in the HTTP API.
  Set body, decoded-image, and image-dimension limits before decoding. The
  server caps request bodies at 16 MiB, image dimensions at 8,192 pixels per
  side, and decoded source images at 8 megapixels. TypeSafe Jev currently
  supports text only; image requests are a local extension.
- Success follows the TypeSafe outer shape: `model`, `answers` keyed by the
  supplied question IDs, and `usage`. Each answer has `type: "choice"`,
  `choice` (selected criteria key), `probabilities` keyed by every criteria
  key, and `confidence: 1.0` as a fixed compatibility placeholder.
  `usage.input_tokens` sums rendered text prompt token counts across
  sequential questions (and explicitly excludes visual tokens until those are
  accounted for); `usage.output_tokens` is 0 because no answer token is
  generated. Add a clearly namespaced `branchscore` object for `schema_version:
  2`, `confidence_kind: "constant_placeholder"`, scoring/prompt identities,
  timings, and per-option raw logits. HTTP output omits
  rendered token IDs, full prompt, and Vision debug values. Relative
  probabilities remain uncalibrated and conditional on the supplied option
  set. Clients must not use this fixed `confidence` for threshold decisions;
  it does not express certainty or reproduce TypeSafe's computation.
- `GET /healthz` reports readiness after startup. No generation, OpenAI API
  emulation, model upload, or generic model registry.
- One process owns one model/backend/engine. Handle one decision at a time.
  The server uses a listen backlog of 8, a 10-second request read timeout, a
  10-second response write timeout, 16 KiB/64-field header limits, and a 16 MiB
  body limit. Close each response connection. A later phase may add bounded
  request admission after measuring concurrent callers. Keep
  `Gemma4DecisionEngine::evaluate` unchanged as the decision boundary and avoid
  sharing its backend state concurrently.
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

1. [x] Add a narrow TypeSafe-envelope adapter that converts supported Choice
   questions to existing `DecisionRequest` values and projects
   `DecisionResult` into `answers`. Keep benchmark run/aggregate JSONL fields
   in the runner. Verify option IDs and deterministic ordering.
2. [x] Add in-memory JPEG/PNG decoding to `ImagePreprocessor` and an image-bytes
   input path to `DecisionRequest`; preserve the CLI/JSONL path behavior.
3. [x] Add the HTTP executable with bounded parsing, sequential dispatch, startup
   readiness, authentication, error mapping, and shutdown. Choose and pin a
   small HTTP parser implementation during this slice; keep HTTP code outside
   `branchscore_core`.
4. [x] Verify Choice text/image results against the core engine, question-ID and
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

The detailed dated implementation, portability, and HTTP verification record
is in [the Phase 4+ execution record](../records/phase-4-plus-http-server.md).
The current server contract and implementation slices remain above; the record
contains the vendoring investigation, live request checks, and environment-
specific timing evidence.
