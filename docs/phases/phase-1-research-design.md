# Phase 1 - Research and minimal design

## Goal

Establish a defensible minimal design before implementation by researching
OpenJev scoring and llama.cpp/ggml Gemma 4 multimodal internals.

## Prerequisites

Phase 0 documentation structure is complete.

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- The relevant file in `docs/research/`

## Stages

### Stage 1.1 - OpenJev scoring research

#### Step 1.1.1

Inspect current OpenJev state/prompt construction, option tokenization,
single/multi-token scoring, shared state/prefix behavior, serial and batch
paths, accumulated log-probabilities, normalization, softmax/confidence,
option-order sensitivity, and timing. Record source revisions and evidence in
`docs/research/openjev.md`.

#### Step 1.1.2

Choose the first scoring behavior: accumulated continuation log-probability,
with mean score when needed. Defer PMI unless findings make it necessary.

### Stage 1.2 - llama.cpp and ggml research

#### Step 1.2.1

Inspect Gemma 4 E2B/E4B model tensors and the embedding, transformer,
attention, RoPE, final norm, output projection, and logits path.

#### Step 1.2.2

Inspect image preprocessing, vision tensor layout, patch embedding, Vision
Transformer, projector, visual-token output, and its connection to the LLM.

#### Step 1.2.3

Compare Prefill and decode graphs: KV cache, positions, sequence handling,
batches, and output selection.

#### Step 1.2.4

Inspect ggml backend enumeration/init, buffer types, tensor/graph allocation,
compute, async/sync, copy, and events.

#### Step 1.2.5

Inspect backend scheduler graph splitting, placement, buffer ownership,
cross-backend copy, and synchronization. Do not commit to using it yet.

### Stage 1.3 - Basic design

#### Step 1.3.1

Define minimum component interfaces: ModelLoader, VisionEncoder,
PrefillEngine, StateCache, OptionScorer, BackendContext, DecisionEngine.

#### Step 1.3.2

Define intermediate data types listed in `docs/architecture.md`.

#### Step 1.3.3

Specify backend/host placement, copy requirements, owner, and lifetime for
each tensor category.

#### Step 1.3.4

Confirm the Phase 2 sequential flow: Vision → Prefill → each option →
normalization → softmax.

## Deliverables

- Evidence-backed OpenJev and llama.cpp/ggml research notes.
- A small component/data/ownership design sufficient for Phase 2.
- A stated initial scoring rule and sequential execution flow.

## Completion Criteria

The team can explain option scoring, Gemma 4 Vision/Prefill/logits handling,
the required ggml-only components, state ownership, and Phase 2 execution.

## Notes / Findings

- 2026-09-19: Inspected local OpenJev/SemIf revision
  `ca3ba65f142967030ecb453346e94d6f476a69df`. Its preferred direct path scores
  one-token answer letters at the final prompt position; it does not accumulate
  multi-token option continuation likelihoods. Prefix reuse applies across
  decisions sharing exact state. See `docs/research/openjev.md`.
- 2026-09-19: Inspected local llama.cpp/ggml revision
  `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f` and all four local E2B/E4B GGUF
  files. Recorded the Gemma 4 text graph, dynamic vision path, cache/output
  behavior, backend lifecycle, and scheduler split/copy behavior in
  `docs/research/llama-ggml.md`.
- Stage 1.1 and Stage 1.2 research are complete for the Phase 2 design. The
  required initial scorer remains accumulated causal continuation
  log-probability; OpenJev's letter-logit path is evidence, not a compatibility
  target.
- 2026-09-19: Fixed the initial scoring contract: the common prefix contains
  state/image and question but no candidate list; an option's exact description
  is the continuation; EOS/turn terminators are excluded; `sum_logprob` drives
  selection and softmax; `mean_logprob` is returned to expose length effects.
- 2026-09-19: Completed the Stage 1.3 minimum interfaces, intermediate data
  contracts, tensor placement/copy/ownership/lifetime table, and Phase 2
  sequential flow in `docs/architecture.md`. The single-backend state cache
  keeps prefix rows immutable and rewinds one reusable tail between options,
  avoiding a full prefix-cache copy without weakening branch isolation.
- Phase 1 completion criteria are met. Phase 2 can begin with the independent
  project foundation and single-backend baseline.
