# Phase 2+ - Contract hardening

## Goal

Turn the Phase 2 proof of concept into a stable, observable Gemma 4 decision
engine boundary before evaluation work begins. Keep the current Gemma 4 prompt
fixed, move request orchestration out of the CLI, make result and timing
semantics explicit, and leave a narrow path for later LLaDA-V and custom prompt
support without generalizing model execution prematurely.

## Prerequisites

The Phase 2 single-backend baseline loads Gemma 4 E2B/E4B, handles optional
images, reuses Prefill state across sequential option branches, and reports
sum/mean continuation log-probabilities and relative option probabilities.

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-2-single-backend.md`
- Relevant completed notes under `docs/research/`

## Scope decisions

- The effective Phase 2+ prompt renderer is a built-in, versioned Gemma 4
  renderer. It does not interpret GGUF Jinja or a user-supplied template.
- `--chat-template-file` remains accepted as a reserved request. Phase 2+
  neither opens nor validates the file, does not apply it, and reports the
  requested filename with `applied=false`.
- Reasoning behavior is expressed as a semantic prompt policy. The built-in
  renderer uses direct-answer / reasoning-disabled behavior, but must not add
  an unverified model-specific `</think>` sequence.
- Shared request/result contracts may be reused by a later LLaDA-V engine.
  Prefill, KV-cache, causal continuation, and other Gemma-specific execution
  details are not generalized ahead of that implementation.
- Model loading is outside request timing. CLI formatting, debug-dump file I/O,
  and other presentation work are also outside request timing.

## Stages

### Stage 2+.1 - Decision contracts

#### Step 2+.1.1
Define `DecisionOption` and `DecisionRequest` with state, question, at most one
optional image, 2--16 ordered options, and prompt policy. Keep semantic option
IDs separate from input positions.

#### Step 2+.1.2
Define `DecisionResult` with ordered option scores, selected ID/index, exact-tie
status, scoring basis, relative-probability semantics,
`terminator_scored=false`, effective prompt metadata, rendered-prompt identity,
and timings.

#### Step 2+.1.3
Define `TimingInfo` with prompt rendering, tokenization, image preprocessing,
Vision, Prefill, per-option scoring, score total, normalization, and request
total. Document the inclusive/exclusive timing boundaries.

#### Step 2+.1.4
Keep the contracts independent of Gemma 4 cache and graph types so a later
LLaDA-V implementation can return the same decision-level result without
pretending that its inference process is Prefill/continuation based.

### Stage 2+.2 - Versioned Gemma 4 prompt rendering

#### Step 2+.2.1
Replace the ambiguous source-owning `ChatTemplate` behavior with a concrete
`Gemma4PromptRenderer` that emits the currently validated fixed system/user,
optional-image, and model-generation prefix.

#### Step 2+.2.2
Add `PromptPolicy`, initially supporting reasoning-disabled direct-answer
behavior, and `PromptFormatInfo` containing the renderer ID, model family,
effective source, reasoning policy, requested override, and whether that
override was applied.

#### Step 2+.2.3
Version the effective renderer as `gemma4-fixed-v1`. Preserve the Phase 2
rendered prefix and token IDs; do not add a think-block closing sequence until
the exact Gemma 4 convention is verified and represented by a new renderer
version.

#### Step 2+.2.4
Make `--chat-template-file FILE` a transparent no-op reservation. Do not open,
check, or parse `FILE`; retain its exact name in prompt metadata, set
`requested_template_applied=false`, and print that it is ignored in Phase 2+.

#### Step 2+.2.5
Treat GGUF `tokenizer.chat_template` as optional diagnostic metadata rather
than the effective Phase 2+ prompt source. When its presence is reported, also
report that it was not used.

### Stage 2+.3 - Gemma 4 decision engine

#### Step 2+.3.1
Add `Gemma4DecisionEngine` as the request-scoped orchestration boundary. It
validates the request, renders and tokenizes the common prefix, boundary-
tokenizes every option, runs optional Vision, runs Prefill once, scores options
sequentially, normalizes results, and returns `DecisionResult`.

#### Step 2+.3.2
Move decision validation and orchestration out of `tools/branchscore.cpp`. The
CLI should parse arguments, construct one request, call the engine once, and
format the returned result.

#### Step 2+.3.3
Keep the engine explicitly Gemma 4-specific. Do not introduce an abstract
inference-engine hierarchy, model registry, or a common cache interface before
a second model family provides concrete requirements.

#### Step 2+.3.4
Preserve the Phase 2 scoring contract: exact option-description continuation,
full-vocabulary log-normalization, sum-based selection and softmax, mean score
for observability, first maximum on exact ties, and no EOS/turn terminator.

### Stage 2+.4 - Timing and result observability

#### Step 2+.4.1
Measure `request_total_ms` from request validation through completed result
normalization. Exclude model loading, CLI output, and optional dump-file I/O.

#### Step 2+.4.2
Report coarse synchronized Phase 2+ intervals for image preprocessing, Vision,
Prefill, and each option. Ensure every backend interval ends after required
synchronization and every host read is covered by its owning interval.

#### Step 2+.4.3
Report `score_total_ms` and normalization separately. Do not compute total time
by summing selected stage values; use an enclosing wall-clock measurement so
prompt construction, tokenization, validation, and orchestration overhead are
included.

#### Step 2+.4.4
Label probabilities as relative to the supplied option set and explicitly
print `scoring_basis=sum_logprob` and `terminator_scored=false`. Do not label
relative probabilities as confidence.

#### Step 2+.4.5
Defer detailed Vision/Prefill upload, compute, persistent-copy, and sync
breakdowns to Phase 3 unless they can be added without scattering timing logic
through graph construction.

### Stage 2+.5 - Focused verification and documentation

#### Step 2+.5.1
Add exact text-only and image prompt-rendering tests for
`gemma4-fixed-v1`, including the effective renderer metadata and reasoning
policy.

#### Step 2+.5.2
Verify that a nonexistent `--chat-template-file` path is accepted, is not
opened, leaves rendered text and token IDs unchanged, and is reported with
`applied=false`.

#### Step 2+.5.3
Add focused engine tests for request validation, ordered results, first-max tie
behavior, probability labeling, scoring metadata, and nonnegative timing
fields. Reuse the existing model-backed test rather than adding a broad test
framework.

#### Step 2+.5.4
Run the existing E2B CPU model-backed suite and one practical CLI request.
Confirm that prefix token IDs, option token IDs, scores, probabilities, and
selection remain unchanged apart from the intentional output-contract changes.

#### Step 2+.5.5
Update CLI help, README usage, architecture contracts, and this phase's
Notes / Findings with the implemented renderer version, timing boundaries,
test evidence, and any discovered model-specific facts.

## Suggested implementation locations

- `include/branchscore/decision.hpp`
- `include/branchscore/gemma4_prompt_renderer.hpp`
- `include/branchscore/gemma4_decision_engine.hpp`
- `src/gemma4_prompt_renderer.cpp`
- `src/gemma4_decision_engine.cpp`
- `tools/branchscore.cpp`
- `tests/gemma4_prompt_renderer_test.cpp`
- `tests/gemma4_decision_engine_test.cpp`

These names are guidance, not a requirement to create additional abstraction.
Keep the implementation smaller if existing files provide a clearer boundary.

## Deliverables

- A library-level Gemma 4 decision entry point used by the CLI.
- Explicit request, result, prompt-format, and timing contracts.
- A versioned fixed Gemma 4 renderer with reasoning-disabled intent.
- Transparent no-op handling of the reserved `--chat-template-file` argument.
- Correct request-total timing and unambiguous scoring/probability output.
- Focused regression coverage and synchronized documentation.

## Completion Criteria

The CLI and library use one decision path; the effective renderer is reported
as `gemma4-fixed-v1`; a missing template-file path is safely retained but not
applied; existing Gemma 4 prompt tokens and scoring results remain stable;
relative probability and terminator semantics are explicit; request total is
an enclosing wall-clock interval; and the E2B model-backed tests pass.

## Deferred work

- Executing GGUF or file-sourced Jinja templates.
- A custom-template sandbox, template checksum, or `custom-jinja-v1` renderer.
- Adding verified model-specific think-block opening or closing sequences.
- LLaDA-V loading, prompt rendering, image handling, or scoring.
- A cross-model inference-engine abstraction.
- Fine-grained copy/compute/synchronization instrumentation beyond what Phase 3
  measurements justify.

## Notes / Findings

The detailed dated implementation and verification notes are in [the Phase 2+
execution record](../records/phase-2-plus-contract-hardening.md). Phase 2+
established the request/result contracts, fixed `gemma4-fixed-v1` renderer,
no-op template override, and `Gemma4DecisionEngine`; its continuation scoring
fields are retained only in that historical record because Phase 3+ changed
the readout contract.
