# Architecture

> Transition completed (2026-09-20): the implemented path is the Phase 3+
> categorical readout. The continuation baseline below is retained only where
> it explains historical state/cache behavior; it is not a production mode.

## Current categorical execution model

```mermaid
flowchart LR
    I[Text + optional image] --> V[Vision Encoder]
    V --> P[Prefill displayed-options prompt]
    P --> S[Final-position logits]
    S --> G[Gather A-P answer slots]
    G --> N[Stable softmax + first max]
    N --> R[Decision result]
```

Run this fully sequentially using one model, one selected backend, one request,
and 2-16 options. Vision and Prefill run once, followed by one small categorical
readout. Layer split, tensor parallelism, workers, and the ggml multi-backend
scheduler are not part of this baseline.

## Categorical decision contract

The rendered prompt contains the optional image marker, state, question, and
ordered option descriptions. The model-visible option IDs are not included;
input order assigns answer labels A through P. The versioned prompt is:

```text
system: Apply the supplied criterion to the supplied evidence. Choose exactly
        one listed option. Respond with only its uppercase letter, with no
        explanation or reasoning.
user:   [optional image]
        State:
        {state}

        Question:
        {question}

        Options:
        [{"description":"...","letter":"A"}, ...]
model:  <answer slot follows>
```

`Gemma4PromptRenderer` renders this with Gemma 4's native turn and special
tokens as `gemma4-categorical-v1`. The full rendered prompt and token IDs are
observable and the prompt identity hashes the actual prompt. Each A-P label is
checked as a standalone normal token, round-trips to the same one-character
piece, has a distinct token ID, and preserves the full prompt token boundary
when appended. A failure rejects the request; there is no multi-token
fallback.

`--chat-template-file` remains a reserved no-op: its exact filename is retained
in metadata, never opened or validated, and never changes the effective prompt.
GGUF `tokenizer.chat_template` is optional diagnostic metadata and is never
used by the built-in renderer.

For answer token IDs `a[0..n-1]`, the categorical readout is:

```text
raw_score[i]             = final_position_logits[a[i]]
relative_probability[i]  = exp(raw_score[i] - max(raw_score)) /
                           sum_j exp(raw_score[j] - max(raw_score))
selected_index           = first argmax(raw_score)
```

Only the requested 2-16 logits are copied to the host. No answer token is
sampled or consumed, and no EOS or continuation forward is performed.
Relative probabilities are conditional on the supplied option set and are not
calibrated confidence.

## Minimum components

These are narrow current-phase boundaries, not framework extension points.

### `BackendContext`

- Enumerates available ggml devices and initializes exactly one requested
  backend.
- Owns the backend handle, selected buffer type, graph allocator/work buffers,
  and synchronization used for measurement and host reads.
- Allocates model, persistent state, and temporary graph buffers on that same
  backend. It does not create a backend scheduler.

### `ModelLoader`

- Reads the text GGUF and matching mmproj GGUF, validates `gemma4`/`gemma4v`,
  and builds a single `ModelBundle` from metadata and named tensors.
- Localizes E2B/E4B differences in model metadata: widths, layer counts, FFN
  sizes, KV layout, attention pattern, and vision projection width.
- Allocates weights in backend buffers and keeps quantized text weights in
  their stored ggml types. Audio tensors in the mmproj are ignored.
- Owns GGUF mappings/host metadata and the backend weight buffers for the
  lifetime of the engine.

### `VisionEncoder`

- Decodes JPEG/PNG on the host, applies the Gemma 4 dynamic-size preprocessing,
  and creates an `EncodedImage` including the patch grid and positions.
- Builds and executes the Gemma 4 Vision graph on `BackendContext` and returns
  projected `VisualTokens` whose width equals the text embedding width.
- Measures host preprocessing, required input copy, graph compute, and
  synchronization so Vision time has an explicit boundary.

### `PrefillEngine`

- Splices projected visual embeddings at the image placeholder and validates
  the context limit for the already-rendered/tokenized displayed-options prompt.
- Builds and executes the causal Gemma 4 graph, populates a `StateCache`, and
  copies the state-final logits needed for the answer-slot readout into
  request-scoped backend storage outside the temporary graph buffer.
- Requests only the final Prefill output row. It returns a `PrefillState` after
  synchronization at the Prefill timing boundary.

### `StateCache`

- Owns backend-resident K/V storage and the logical position/sequence metadata
  for one request.
- Marks `[0, prefix_length)` immutable after Prefill. The categorical path
  allocates no continuation tail: cache capacity equals the displayed prompt.
- The cache is request-scoped and is not rewound or copied between candidate
  options. The final-position logits are read once after Prefill.
- Encodes Gemma 4's mixed sliding/full attention and shared-KV-layer layout
  behind this boundary; callers do not assume one uniform K/V tensor per layer.

### `CategoricalReadout`

- Receives the Prefill final-position logits and the validated A-P answer token
  IDs in input order.
- Uses one ggml gather graph to transfer only the requested 2-16 logits to the
  host. It rejects out-of-range or non-finite values.
- Applies a temperature-1 stable host softmax, selects the first maximum, and
  records gather/copy/synchronization timing separately from normalization.

### `Gemma4PromptRenderer`

- Renders the displayed-options `gemma4-categorical-v1`
  system/user/image/generation prompt and records `PromptFormatInfo`.
- Expresses reasoning behavior as `PromptPolicy`; the categorical path supports only
  direct-answer/reasoning-disabled behavior and does not invent a think-block
  terminator.
- Retains a requested template filename as metadata without opening or
  interpreting it.

### `Gemma4DecisionEngine`

- Validates one request, orchestrates Vision -> Prefill -> categorical readout
  in order,
  and owns request-scoped intermediate values. Model loading is outside this
  boundary.
- Calls the fixed renderer and validates every answer label with the tokenizer,
  and passes the split prefix/image spans to the existing Gemma-specific
  execution components.
- Gathers and normalizes 2-16 `answer_slot_logit` values. This small result
  aggregation is intentionally host-side; final vocabulary projection remains
  in the Prefill graph.
- Preserves input order in results, selects the first maximum on an exact tie,
  and reports the tie, scoring basis, relative-probability semantics, and
  `terminator_scored=false` explicitly.
- Produces `DecisionResult` and `TimingInfo`; it does not retain request state
  after the result is returned.

## Intermediate data contracts

| Type | Required contents |
|---|---|
| `EncodedImage` | Host RGB samples after exact resize/padding/normalization preparation; width, height, patch-grid dimensions, x/y positions |
| `VisualTokens` | Backend tensor `[text_width, visual_token_count]`, element type, backend identity, image-token placement metadata |
| `AnswerToken` | Answer label, input index, semantic option ID, one normal token ID, standalone/round-trip/boundary validation |
| `PrefillState` | Rendered-prompt hash/version, prompt token count, image-token count, next absolute position, `StateCache`, persistent backend final-position logits |
| `OptionScore` | Option ID/index, answer label/token ID, `raw_score`, relative probability |
| `DecisionRequest` | State, question, 2-16 ordered semantic options, at most one image path or shared in-memory encoded image, prompt policy, and optional reserved template filename |
| `PromptFormatInfo` | Renderer ID/version, model family, effective source, reasoning policy, requested override/applied flag, GGUF-template diagnostic flags |
| `DecisionResult` | Schema 2, ordered option scores, relative probabilities, selected ID/index, exact-tie flag, `answer_slot_logit` basis, readout identity, `terminator_scored=false`, prompt identity/metadata, timings |
| `VisionDebugInfo` | Optional host copy of projected visual tokens, populated only when explicitly requested for a debug dump |
| `TimingInfo` | Prompt rendering, tokenization, image preprocessing, Vision, Prefill, shared readout/copy/synchronization, normalization, enclosing request total |

Host input validation additionally requires 2-16 options, unique nonempty IDs,
nonempty descriptions, a nonempty question, and a text state. The first
milestone accepts at most one image.

## Placement, copies, ownership, and lifetime

| Category | Location | Owner | Lifetime | Required copies |
|---|---|---|---|---|
| Request strings and image path/bytes | Host | `Gemma4DecisionEngine`; encoded bytes may be shared across questions in one envelope | One request | None until token/image inputs are prepared |
| Token IDs, positions, option metadata | Host | `Gemma4DecisionEngine` / `CategoricalReadout` | One request | Host -> backend graph-input tensors per Prefill/readout |
| Decoded/resized image samples | Host | `EncodedImage` | Through Vision input upload | Host -> selected backend once |
| Text and vision weights | Selected backend; GGUF may remain host-mapped as loading source | `ModelLoader` | Engine lifetime | Load/upload once; no per-request backend copy |
| Vision activations | Selected backend temporary graph buffers | `VisionEncoder` via `BackendContext` | One Vision call | None across backends |
| Projected visual tokens | Selected backend | Request-scoped `VisualTokens` | Until Prefill consumes them | No backend copy; do not round-trip through host |
| Prefill activations | Selected backend temporary graph buffers | `PrefillEngine` via `BackendContext` | One graph execution | None across backends |
| Prefix K/V | Selected backend persistent request buffer | `StateCache` | One Prefill/readout request | No candidate branch copy or tail |
| Prefill-final vocabulary logits | Selected backend persistent request buffer | `PrefillState` | Through categorical readout completion | Backend -> host only for requested answer logits/debug output |
| Gathered answer logits | Selected backend readout output | `CategoricalReadout` | One readout graph | Backend -> host only for 2-16 scalars |
| Option scores and probabilities | Host | `Gemma4DecisionEngine` | Result lifetime | One scalar result per requested value |
| Optional Vision debug values | Host | `Gemma4DecisionEngine` / CLI | Result and dump lifetime | Backend -> host only when explicitly requested; file output is after request completion |

All backend work is synchronized before its timing interval ends and before a
host read. Temporary graph tensors must not outlive their graph buffer. A
`PrefillState` cannot outlive its `ModelBundle` or `BackendContext`.

## Confirmed Phase 3+ sequence

1. Validate the request, render the displayed-options prompt, and validate all
   A-P answer labels before model work.
2. If an image exists, decode/preprocess it and run Vision on the selected
   backend, retaining projected tokens there.
3. Run Prefill once from the complete text plus optional visual embeddings.
   Freeze the request cache and retain the final output logits in request-scoped
   backend storage.
4. Gather the A-P answer logits in input order, reject non-finite values, and
   copy only those scalars to the host.
5. Compute stable softmax over answer-slot logits, select the first maximum,
   and emit the ordered semantic result plus timings and contract metadata.

## Later direction

Phase 4 may assign Vision, Prefill, and readout to separate backends only if
measurements justify the copies. The next likely optimization is state-only
prefix reuse across questions on one backend; it requires a separate contract
for token boundaries, image placement, and cache ownership. Request-level
parallelism remains a later measurement-driven decision. These future
constraints do not authorize concurrency, cross-backend copies, or scheduler
use in the current sequential path.
