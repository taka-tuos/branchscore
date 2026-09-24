# Phase 4 - Measurement-led component/backend boundaries

> Planning handoff (2026-09-20): Phase 3+ categorical readout is complete.
> This phase does not restore `OptionScorer` or require an independent Logit
> worker. Separation is selected only when measurements show a useful boundary.

## Goal

Measure and, when justified, define useful ownership boundaries for Vision,
Prefill, and categorical readout. Keep the one-model, one-request,
single-backend path as the baseline while making transfers, lifetimes, and
synchronization explicit.

## Prerequisites

- Phase 3+ categorical readout is complete and its current timing/schema
  contract is understood.
- Phase 3 evaluation findings and the current backend implementation provide a
  reproducible baseline.
- A proposed split has a concrete measurement question; component separation
  is not a deliverable by itself.

## Read First

- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`
- `docs/phases/phase-3-evaluation-debug.md`
- Backend findings in `docs/research/llama-ggml.md`

## Stages

### Stage 4.1 - Baseline ownership and measurements

#### Step 4.1.1

Record the current Vision, Prefill, categorical readout, copy/sync, and
allocation boundaries on one backend. Measure stage cost and lifetime without
introducing a component split.

### Stage 4.2 - Vision boundary

#### Step 4.2.1

If measurements justify it, prototype a Vision-to-Prefill boundary. Define
which representation is transferred, who owns it, when it is synchronized,
and the copy cost. Defer the split when the transfer is not useful.

### Stage 4.3 - Optional categorical readout placement

#### Step 4.3.1

Assess whether the small A–P gather/readout benefits from a separate placement.
An independent readout backend is optional; compare transfer and synchronization
costs against the current backend-resident logits path before adding one.

### Stage 4.4 - Request/backend contexts

#### Step 4.4.1

If a boundary is selected, define request-scoped context handles, allocators,
temporary buffers, and synchronization. Preserve sequential ownership and keep
layer splitting and tensor parallelism out of scope.

### Stage 4.5 - Transfer and handoff decision

#### Step 4.5.1

Document the selected boundaries, explicit transfers, and synchronization
contract, or record why the single-backend design remains preferable. Hand off
only measured, useful boundaries to later phases.

## Deliverables

- A baseline measurement record for Vision, Prefill, categorical readout,
  copies, synchronization, and ownership.
- An explicit transfer/lifetime contract for every implemented boundary.
- A decision to implement, defer, or reject each proposed separation.

## Completion Criteria

- Stage costs and ownership are measured on the current sequential path.
- Any implemented split has explicit state transfer and synchronization, and
  its cost is recorded.
- Deferred work is recorded with a reason and a measurement condition.
- No completion criterion requires `OptionScorer`, candidate continuation
  workers, or a mandatory independent Logit backend.

## Notes / Findings

The detailed dated CPU/CUDA measurements and ownership observations are in [the
Phase 4 measurement record](../records/phase-4-backend-measurements.md). The
current conclusion is unchanged: Prefill dominates text requests, Vision and
Prefill dominate image requests, and the small categorical readout does not
justify an independent backend in the measured baseline.
