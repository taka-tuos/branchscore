# Requirements and scope

> Transition completed (2026-09-20): the current runtime uses SemIf-style
> displayed options and categorical answer-slot logits. The old continuation
> results remain historical evidence and are not a second production mode.

## Goal

Build a hobby, GGML-based Jev-like multimodal decision engine for Gemma 4 E2B
and E4B. It must evaluate displayed candidate options semantically from a
text/image state, rather than serve general-purpose autoregressive generation.
E2B is the development/debug model; E4B is the evaluation target. Keep
model-specific differences localized.

## Required decision flow

1. Render the text/image state, question, and all ordered option descriptions
   in one fixed Gemma 4 prompt.
2. Encode images with a vision encoder when present.
3. Prefill the complete displayed-options prompt once.
4. Gather the logits of the single-token A–P answer labels at the next model
   position.
5. Apply a temperature-1 stable softmax across the supplied labels and return
   the selected semantic option plus timing information.

The production scoring target is `answer_slot_logit`. Option descriptions are
judgment context in the prompt, not continuations to score. The model never
consumes a sampled answer token, and EOS/turn-ending tokens are not scored.
The returned probabilities are conditional on the supplied option set and are
not calibrated confidence. PMI, calibration, and sampling remain out of scope.

## Implementation boundaries

- Use ggml directly in an independent C/C++ project.
- Initially run all work sequentially on one selected backend (CPU, CUDA, or
  Vulkan). Backend portability follows via ggml abstractions.
- Never make transformer layer split, tensor parallelism, distributed
  inference, or multi-node operation a project goal.
- llama.cpp and OpenJev are upstream references, not dependencies to wrap as
  the decision runtime.

## First practical milestone

Input: JPEG/PNG image, text state, question, and 2–16 options. Model: Gemma 4
E2B Q4/Q5 GGUF. Output: answer-slot raw logits, relative probabilities,
semantic selection, prompt/readout identity, and Vision/Prefill/readout
timings. The same code is checked with E4B. Probabilities are relative to the
supplied option set and must not be labeled calibrated confidence.

## Non-goals

Chat completion, long-form generation, llama.cpp or OpenAI API compatibility,
training/fine-tuning, production fault tolerance, large benchmark suites, and
exact Jev API/internal reproduction are out of scope.
