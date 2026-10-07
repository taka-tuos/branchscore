# Execution records

This directory contains dated implementation, benchmark, review, and
portability records that would otherwise make the phase plans difficult to
scan. The phase documents remain the canonical place for scope, current
contracts, stages, completion criteria, and a short findings summary.

Records are historical evidence, not additional requirements. A later phase
or contract can supersede an observation; read the linked phase document
first when deciding what is currently in scope.

## Reading and adding records

Use the [phase index](../phases/README.md) to find the current plan, then follow
its record routing. Longer histories are split by Step or investigation topic
into a phase-specific directory; the original phase-named Markdown is the
index and retains links for its old sections.

Keep each record focused on one measurement or related investigation. When a
new topic starts or a history becomes long, add a separate record and update
the phase/index links. Keep the conditions, dates, raw-data links and findings
together. Phase Notes / Findings should summarize the result and remaining
work rather than repeat the execution history.

## Record indexes

| Record | Source phase/document |
|---|---|
| [Phase 2 single-backend](phase-2-single-backend.md) | `phases/phase-2-single-backend.md` |
| [Phase 2+ contract hardening](phase-2-plus-contract-hardening.md) | `phases/phase-2-plus-contract-hardening.md` |
| [Phase 3 continuation baseline](phase-3-continuation-baseline.md) | `phases/phase-3-evaluation-debug.md` |
| [Phase 3+ categorical readout](phase-3-plus-categorical-readout.md) | `phases/phase-3-plus-categorical-readout.md` |
| [Phase 3+ implementation review](phase-3-plus-implementation-review.md) | `research/direction-review-2026-09-20.md` |
| [Phase 4 backend measurements](phase-4-backend-measurements.md) | `phases/phase-4-backend-separation.md` |
| [Phase 4+ HTTP server](phase-4-plus-http-server.md) | `phases/phase-4-plus-http-server.md` |
| [Phase 4+ option-scale investigation (discontinued)](phase-4-plus-option-scale.md) | `phases/phase-4-plus-option-scale.md` (discontinued 2026-09-30) |
| [Phase 4+ option-scale-2 topic records](phase-4-plus-option-scale-2.md) | `phases/phase-4-plus-option-scale-2.md` |
| [Current Step 4 CPU baseline and padding review](phase-4-plus-option-scale-2/cpu-kernel-padding-baseline.md) | `phases/phase-4-plus-option-scale-2.md` / Step 4 |

## Raw measurements

JSON/JSONL/TSV and the [CPU reference harness](phase4-step4-cpu-reference/README.md)
remain at their existing paths. Topic records link to the measurements they
discuss; these files do not define current scope or completion status.

| Data | Source phase/document |
|---|---|
| [Phase 4+ option-scale-2 Unsloth provenance evidence](phase-4-plus-option-scale-2-provenance-2026-09-30.json) | `phases/phase-4-plus-option-scale-2.md` / Step 4 |
| [Phase 4+ option-scale-2 E4B Q4 quality screening](phase-4-plus-option-scale-2-quality-e4b-q4-2026-09-30.jsonl) | `phases/phase-4-plus-option-scale-2.md` / Step 4 |
| [Phase 4+ provisional 512-label candidates](phase-4-plus-option-label-candidates.tsv) | `phases/phase-4-plus-option-scale.md` (carried into option-scale-2) |
| [Phase 4+ option-scale initial raw scores](phase-4-plus-option-scale-scores-2026-09-29.jsonl) | `phases/phase-4-plus-option-scale.md` |
| [Phase 4+ CPU option-scale resource sweep](phase-4-plus-option-scale-resource-cpu-2026-09-29.jsonl) | `phases/phase-4-plus-option-scale.md` |
| [Phase 4+ F16 mask CPU comparison](phase-4-plus-option-scale-f16-mask-cpu-2026-09-29.jsonl) | `phases/phase-4-plus-option-scale.md` |
| [Phase 4+ F16 K/V CPU comparison](phase-4-plus-option-scale-f16-kv-cpu-2026-09-29.jsonl) | `phases/phase-4-plus-option-scale.md` |
| [Phase 4+ Flash Attention CPU probe](phase-4-plus-option-scale-flash-cpu-2026-09-29.jsonl) | `phases/phase-4-plus-option-scale.md` |
| [Phase 4+ query-chunk attention CPU probe](phase-4-plus-option-scale-attention-chunk-cpu-2026-09-29.jsonl) | `phases/phase-4-plus-option-scale.md` |

The original unpartitioned proposal remains under `docs/archive/`; it is not
an execution record and is intentionally not linked from this index.
