# Phase 3 - Evaluation, debugging, and observation

## Goal

Use the single-backend baseline to understand correctness, behavior, and
bottlenecks before backend separation.

## Prerequisites

Phase 2 baseline works with option scoring, and Phase 2+ has stabilized the
request/result, fixed Gemma 4 prompt, and request-timing contracts.

## Read First

- `docs/requirements.md`
- `docs/development-rules.md`
- `docs/phases/phase-2-single-backend.md`
- `docs/phases/phase-2-plus-contract-hardening.md`

## Stages

### Stage 3.1 - Basic behavior

#### Step 3.1.1
Try text-only decisions.

#### Step 3.1.2
Try image-plus-decision inputs.

#### Step 3.1.3
Compare single- and multi-token options.

#### Step 3.1.4
Permute option order and observe changes.

### Stage 3.2 - Quantization comparison

#### Step 3.2.1
Compare practical Q4/Q5/Q6/Q8 and, if available, F16/BF16 variants for
selection, distribution, logit gap, runtime, and memory. Precision benchmarking
is not required.

### Stage 3.3 - Performance observation

#### Step 3.3.1
Measure image preprocessing, Vision, Prefill, option scoring, normalization,
backend synchronization, backend copies, and total time. Use the resulting
Vision/Prefill/Logit ratio to inform later design.

### Stage 3.4 - Debug support

#### Step 3.4.1
Add only useful dumps: token IDs/lengths, tensor shape/type/placement, graph
node count, intermediate timings, logits, raw and normalized scores, and
probabilities.

### Stage 3.5 - Small fixtures

#### Step 3.5.1
Create a small manual/regression fixture set, such as text/simple,
text/ambiguous, vision/object, vision/state, vision/order_bias, and
vision/multi_token. Avoid a large test framework.

## Deliverables

Behavior notes, timing observations, focused debug outputs, and small fixtures.

## Completion Criteria

The baseline is stable; scoring behavior and major bottlenecks are understood;
backend separation is justified by measurements.

## Notes / Findings

_Pending evaluation._
