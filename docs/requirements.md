# Requirements and scope

## Goal

Build a hobby, GGML-based Jev-like multimodal decision engine for Gemma 4 E2B
and E4B. It must evaluate candidate options semantically from a pre-filled
text/image state, rather than serve general-purpose autoregressive generation.
E2B is the development/debug model; E4B is the evaluation target. Keep
model-specific differences localized.

## Required decision flow

1. Build text and optional image state.
2. Encode images with a vision encoder when present.
3. Prefill the state once.
4. Score each candidate as a continuation.
5. Return logits or log-probabilities, normalized scores, softmax
   probabilities, the selected option, and timing information.

The initial scoring target is accumulated continuation log-probability:
`sum(log P(token_i | state, option_<i))`. Mean log-probability is also
required as a reported length-normalized value. Selection and softmax use the
sum score. The common prefix contains the text/image state and question, but
not the candidate list. Each option description is scored as the exact
continuation; EOS and turn-ending tokens are excluded. PMI and calibration are
later experiments.

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
E2B Q4/Q5 GGUF. Output: raw and normalized option scores, probabilities,
selection, and Vision/Prefill/Logit timings. The same code should later be
checked with E4B. Probabilities are relative to the supplied option set and
must not be labeled calibrated confidence.

## Non-goals

Chat completion, long-form generation, llama.cpp or OpenAI API compatibility,
training/fine-tuning, production fault tolerance, large benchmark suites, and
exact Jev API/internal reproduction are out of scope.
