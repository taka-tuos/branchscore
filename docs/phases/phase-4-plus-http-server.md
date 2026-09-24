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
- `GET /ui` serves a self-contained browser page for the Choice workflow. It
  is the only exact route that is public on a non-loopback bind, so a browser
  can load the token field; `/healthz` and `/v1/systemone` still require the
  configured bearer token. The page does not persist or log the token.
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

## Follow-up design: small browser UI

This is a design for a later, small implementation slice of the Phase 4+ HTTP
server. It does not change the Choice API or the sequential engine contract.

### Entry point and ownership

- Serve one self-contained HTML page at `GET /ui` from `branchscore-server`.
  Include its CSS and JavaScript in the page; do not add a frontend build,
  package manager, third-party assets, filesystem document root, or proxy.
  The page uses relative, same-origin `/healthz` and `/v1/systemone` URLs.
- `GET /ui` is available without a bearer token so a browser can navigate to
  it. Serve no model paths, token, environment values, or request data in it.
  Keep authentication on `/healthz` and `/v1/systemone`, including when the
  server binds to `0.0.0.0`. Other paths retain the current 404 behavior.
  Match the method and exact path when making the authentication exception;
  query strings and similar prefixes do not inherit it.
- Return `Content-Type: text/html; charset=utf-8`, `Cache-Control: no-store`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and a
  restrictive Content Security Policy. The policy should allow only the
  inline code/style needed by this fixed page, same-origin connections, and
  no framing or external resources. Keep dynamic text out of the HTML source.

### First-screen workflow

1. Show a password-style **Bearer token** field and a **Connect** button.
   The user enters only the token value, without the `Bearer ` prefix. Keep it
   in page memory only; never place it in a URL, cookie, local/session storage,
   HTML attribute, or log. Clearing the field/reloading the page forgets it.
   On a loopback bind, an empty token is valid. `Connect` calls `GET /healthz`
   with `Authorization: Bearer <token>` when nonempty, then displays readiness
   and the returned `model` ID. A 401 leaves the form editable with an
   authentication error; the UI must not retry automatically.
2. Offer one Choice question: text `state`, `instructions`, and 2–16 editable
   option rows with `key` and optional `description`. The first version
   generates a fixed question ID such as `decision`; users may add/remove
   option rows. Require unique nonempty keys, nonempty state/instructions,
   and the supported option count before sending. Explain that keys are
   sorted lexically by the server and that a blank description is sent as
   `null`, making only the key visible to the model.
3. Allow one optional PNG/JPEG file. Convert it in browser memory to
   `{media_type, data_base64}` in the existing top-level `image` field;
   display its filename and a remove action. Check type and the serialized
   request size against the 16 MiB body limit before fetch. Leave image
   dimensions and decoded-pixel validation to the server.
4. Build the existing envelope with the model ID from `/healthz`, `state`,
   `questions.decision`, and optional `image`; submit it to
   `POST /v1/systemone` as JSON with the same Bearer header. Disable repeated
   submission while a request is active. Do not invent model selection,
   generation controls, or a second inference endpoint.
5. Show the selected key prominently, then every key's relative probability
   and raw logit, input-token count, HTTP handling time, and the question's
   stage timings. Label probabilities as relative to these options and
   uncalibrated. Do not present `confidence: 1.0` as a confidence meter: it is
   a compatibility placeholder. Show the server's error code/message and
   HTTP status on failure, preserving the form for correction. Render all
   user/server strings with `textContent`, never `innerHTML`.

### Implementation and verification boundary

- Keep the HTML asset and response helper local to the server executable;
  reuse the current HTTP parser, socket limits, JSON API, and model lifetime.
  A focused helper for the exact unauthenticated UI route is sufficient.
- Verify loopback with an empty token, `0.0.0.0` with correct/missing/wrong
  tokens, and direct API calls without a token. Check that `/ui` works by
  browser navigation but `/healthz` and inference still return 401 when
  required; check route prefixes/query strings, security/cache headers,
  malformed form values, PNG/JPEG, 413/422 errors, and displayed choice/logit
  mapping. No broad frontend test framework is needed.
- The token crosses the network in the Authorization header. For a LAN bind
  over plain HTTP, use only a trusted network; put a TLS-terminating reverse
  proxy in front when transport encryption is needed. The UI does not add
  transport security.

## Notes / Findings

The detailed dated implementation, portability, and HTTP verification record
is in [the Phase 4+ execution record](../records/phase-4-plus-http-server.md).
The current server contract and implementation slices remain above; the record
contains the vendoring investigation, live request checks, and environment-
specific timing evidence.

2026-09-24 UI design finding: the existing server checks Bearer authorization
in `on_headers_complete` before route dispatch. A browser navigation cannot
attach a header from an input field, so `GET /ui` needs one exact, public
exception there. The existing `/healthz` response provides the model ID needed
by the form, and the Choice response already includes probabilities, raw
logits, and timings; no new model or inference API is required.
