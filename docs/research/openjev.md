# OpenJev research

This document is the canonical record for Phase 1 / Stage 1.1 research.

## Sources inspected

| Upstream | Revision | Date | Relevant files |
|---|---|---|---|
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) (the local `openjev/` reference tree) | `ca3ba65f142967030ecb453346e94d6f476a69df` | 2026-09-19 | `src/semif_phase1/{core,direct,serial,shared}.py`, `docs/{METHOD,RESULTS}.md`, `results/phase1-summary.json` |

The revision above is the exact local checkout inspected on 2026-09-19. No live
Jev endpoint was used. The upstream repository describes itself as an open
interface-pattern reproduction and baseline, not an implementation of Jev's
undisclosed internals.

## State and prompt construction

- `validate_row()` accepts a nonempty string, JSON object, or JSON array as
  `state`; a nonempty question; and 2-16 options with unique IDs and string
  descriptions.
- `direct_messages()` renders a fixed system instruction and one JSON user
  payload containing `evidence`, `criterion`, and the displayed options. The
  model is instructed to return exactly one uppercase letter. IDs are not
  exposed to the model; options are assigned `A` through `P` by input order.
- The tokenizer's native chat template renders the two messages with a
  generation prompt and thinking disabled. Input truncation is forbidden.
  The exact rendered prompt is SHA-256 recorded in each result.
- Serial/shared reuse extracts a deterministic token prefix ending inside the
  serialized `{"evidence": ...}` object. One token is deliberately removed
  because later JSON punctuation may merge with the boundary token. Every
  full prompt must start with the extracted token sequence.

## Option tokenization and scoring

- The direct path does **not** score option descriptions as continuations.
  Descriptions occur in the prompt; the scored alternatives are the answer
  letters `A`-`P` at the next position.
- Every answer letter must tokenize to one unique token, round-trip exactly,
  and remain that same token when appended to the rendered prompt. A model or
  prompt failing any condition is rejected. Consequently this implementation
  has no multi-token option-score accumulation path.
- One model forward produces the final-position full-vocabulary logits. The
  implementation selects only the declared answer-token logits and applies a
  numerically stable softmax across them. It does not first apply a
  full-vocabulary log-softmax, but the common log-normalizer would cancel in
  this single-position comparison.
- The returned numbers are explicitly conditional option scores, not
  calibrated confidence. No sum, mean-length normalization, or PMI correction
  is used by the direct categorical path.
- A separate reranker path constructs one yes/no proposition per option,
  scores `logit(yes) - logit(no)`, then softmaxes those values. It is a control,
  not the preferred general-decision path.

## Shared state, serial, and batch behavior

- Fresh direct mode evaluates one complete prompt without a cache.
- Serial-prefix mode prefills a state-derived prefix once. For every decision
  it deep-copies the native KV cache, evaluates the remaining prompt suffix,
  and reads the final-position answer-letter logits. A cache hit requires exact
  Python equality of `state`.
- Shared mode requires all rows to have exactly the same state. It prefills
  once, duplicates the cache for the row count, pads the suffixes, supplies
  explicit positions and masks, and requests logits only at each suffix's last
  real token. This batches multiple **decisions**, not multiple candidate
  continuations of one decision.
- Both reuse modes preserve independent branches. A candidate or decision must
  never leave tokens in the state used by the next branch.

## Order effects, timings, and limitations

- Reversing displayed option order changed 10 of 36 direct-path argmaxes in
  the published perturbation run. Therefore semantic IDs must remain separate
  from display positions and order sensitivity must be testable.
- On the owned 777-decision repeated-state fixture, fresh, serial-prefix, and
  parallel-suffix modes measured 2.33, 10.75, and 20.03 decisions/s on one RTX
  3090. Reuse changed 5 and 6 argmaxes versus fresh BF16 scoring, so cache-path
  equivalence must be verified rather than assumed.
- Direct results record input-token count, forward time, total time, prompt
  hash, model revision, and selected option logits. Serial/shared paths add
  prefill, cache-copy/replication, and suffix timing.
- The timing scope includes prompt construction/tokenization, transfers,
  forward passes, and CPU readout after a warm load; it excludes model loading
  and result-file writes.

## Implications for this project

OpenJev supplies useful evidence for immutable shared-state branches, explicit
option ordering, conditional softmax labeling, and stage timings. Its actual
categorical readout is not the scoring rule required here: this project will
score every option's tokenized description as a causal continuation and
accumulate token log-probabilities. Exact OpenJev internal or API reproduction
is not a goal.
