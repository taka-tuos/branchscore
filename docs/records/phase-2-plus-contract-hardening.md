# Phase 2+ - Contract hardening: execution record

This is the dated implementation and verification record extracted from
[`phases/phase-2-plus-contract-hardening.md`](../phases/phase-2-plus-contract-hardening.md).
The continuation contract recorded here was later superseded by Phase 3+;
the entries remain useful for tracing the request and renderer boundaries.

## Notes / Findings

- 2026-09-20: Implemented `DecisionOption`, `DecisionRequest`, `OptionScore`,
  `DecisionResult`, `PromptFormatInfo`, `TimingInfo`, and optional
  `VisionDebugInfo` in `include/branchscore/decision.hpp`. The request owns
  state/question, 2--16 ordered semantic options, an optional image path, the
  direct-answer prompt policy, and an optional reserved template filename.
  Results preserve input order and expose selected ID and input index,
  exact-tie status, `sum_logprob`, relative-probability semantics,
  `terminator_scored=false`, prompt identity/metadata, and stage timings.
- 2026-09-20: Replaced the source-owning `ChatTemplate` with the concrete
  `Gemma4PromptRenderer`. The effective renderer is `gemma4-fixed-v1`, emits
  the Phase 2 text/image prefix unchanged, records a stable SHA-256 identity,
  and supports only direct-answer/reasoning-disabled behavior. GGUF
  `tokenizer.chat_template` is now optional diagnostic metadata.
- 2026-09-20: `--chat-template-file` is now a transparent no-op. The engine
  retains the exact requested filename, never opens it, reports
  `requested_template_applied=false`, and produces identical prefix token IDs,
  prompt identity, and E2B scores for a nonexistent path versus no override.
- 2026-09-20: Added `Gemma4DecisionEngine` as the single request-scoped path
  used by the CLI. It validates the request, renders/tokenizes the common
  prefix, boundary-tokenizes options, runs image preprocessing/Vision once,
  Prefill once, sequential option scoring, and host normalization. It measures
  prompt rendering, tokenization, image preprocessing, Vision, Prefill, each
  option, score total, normalization, and an enclosing request total. Model
  loading and CLI output are outside request timing. An explicitly requested
  Vision debug host read is covered by Vision timing; dump-file writing occurs
  after the result is complete.
- 2026-09-20: Focused renderer and contract tests pass. The E2B CPU
  model-backed engine test verifies request/result metadata, finite
  nonnegative timings, normalized relative probabilities, and no-op template
  equivalence. The complete CTest suite passes 9/9, including the existing
  Prefill/option model-backed test. A practical CPU CLI request reports
  `gemma4-fixed-v1`, `scoring_basis=sum_logprob`, and
  `terminator_scored=false`.
- 2026-09-20: Prepared the Phase 2+ tree for public source hosting. Project
  documentation and `AGENTS.md` now live in the repository; the README labels
  the work as an experimental, largely AI-assisted hobby project rather than
  a stable library; model assets and tested environments are explicit; the
  project is MIT licensed; and a model-free CPU build/test workflow covers the
  focused tests available without separately downloaded weights.
