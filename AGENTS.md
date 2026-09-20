# OpenJev-like GGML Decision Engine

This is a hobby PoC for categorical candidate decisions with Gemma 4 E2B/E4B
GGUF models. The current runtime displays option descriptions and reads a
single-token categorical answer-slot logit. Normal autoregressive generation
is not the primary feature.

## Non-negotiable constraints

- Build an independent C/C++ project using **ggml directly**.  Do not add the
  implementation to llama.cpp tools/examples or make a llama.cpp runtime API
  wrapper.
- Treat llama.cpp as a reference implementation to investigate, not as the
  project's runtime architecture.
- Layer split and tensor parallelism are out of scope.
- Work sequential-first: one model, one backend, one request, and N options
  before component separation or parallel workers.
- Do not add a large framework, speculative abstraction, or broad test suite.
- Do not silently extend unclear requirements: consult the canonical document
  below (and record a finding when appropriate).

## Start every task this way

1. Identify the requested Phase / Stage / Step in `docs/phases/`.
2. Read that phase's **Read First** list, then only the documents needed for
   the assigned step.
3. Inspect the affected source and any relevant upstream reference.
4. Make a small, verifiable change. Update only documentation affected by it.
5. Record discovered facts in the phase document's **Notes / Findings**.

Do not begin by reading `docs/archive/`; it preserves historical context only.

## Document routing

| Work | Read first |
|---|---|
| Document map and canonical sources | `docs/README.md` |
| Project requirements and scope | `docs/requirements.md` |
| Components, data flow, and ownership | `docs/architecture.md` |
| Shared implementation rules | `docs/development-rules.md` |
| OpenJev scoring research | `docs/research/openjev.md` |
| llama.cpp / ggml research | `docs/research/llama-ggml.md` |
| Current task | matching `docs/phases/phase-*.md` |

Phase 1, Phase 2, Phase 2+, and the Phase 3+ migration are complete. Phase 3
has a measured continuation baseline retained as historical evidence. The
categorical contract and current implementation are recorded in
`docs/phases/phase-3-plus-categorical-readout.md`. Do not implement the old
Phase 4–7 option-worker plan before its Phase 3+ handoff revision.
