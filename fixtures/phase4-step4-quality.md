# Step 4: focused judgment fixtures

[`phase4-step4-quality.jsonl`](phase4-step4-quality.jsonl) contains 8 independent
scenarios and 24 request rows. Each scenario has a base order, a reversed order,
and an order with four unrelated options added. The 16-option long scenario
uses a five-position rotation instead of adding options beyond the current
production limit. Original options retain their semantic IDs and relative order
when distractors are inserted.

| Scenario | Judgment | Expected semantic ID |
|---|---|---|
| `service-policy` | Apply an explicit service action rule | `keep_running` |
| `report-policy` | Apply two publishing prerequisites and an explicit fallback | `wait_for_review` |
| `approval-latest` | Interpret withdrawal, negation, and the latest event | `withdrawn` |
| `close-scores` | Reject the invalid candidate and compare close valid scores | `birch` |
| `japanese-priority` | Apply priority, then an arrival-time tie breaker in Japanese | `shiro` |
| `long-route-16` | Retrieve one route assignment among 48 entries and 16 teams | `juniper` |
| `image-red` | Identify the supplied image's dominant color | `red` |
| `image-blue` | Identify the supplied image's dominant color | `blue` |

All rows contain `expected_selected_id` and `expected_reason`, fixed from the
stated rule or image pixels before inference. `fixture_group`, `variant`, and
`verification_role` are evaluation metadata. The current benchmark reader
ignores these fields and the expected-answer fields; only state, question,
option descriptions, and an optional image enter the production renderer.
Compare the returned semantic ID with the expected ID outside the runner.

The red and blue images are deterministic 96×96 RGB PNGs with every pixel
equal to `(255, 0, 0)` or `(0, 0, 255)`. Their requests have identical text and
option order; the image provides the distinguishing evidence. Image paths are
relative to this JSONL file, as required by `branchscore-bench`.

This is a small sanity and sensitivity set, not a representative accuracy
benchmark. Count the 8 scenarios separately from the 24 correlated variants.
Record failures and order sensitivity rather than asserting that the model
must always pass. The original `phase4-step4-cpu.jsonl` remains a numerical
control; historical results must retain their original inputs.

`close-scores` has a unique rule-based answer and close evidence values; its
`margin_screen` role does **not** claim a small answer-logit margin. Measure the
margin in the A/B baseline and designate near-tie cases from those observations,
before evaluating optimized C/D. The first E4B Q4 base screen had a margin of
10.970 for this case, so it did not supply a numerical near-tie.

The production renderer yields 943 text tokens for the base long scenario,
crossing both the 512-token microbatch boundary and the sliding-window width.
Other base text scenarios have 135–176 tokens. Image prompts have 120 rendered
tokens including the placeholder; Prefill positions must be measured after
visual-token splicing.

For weight A/B comparisons, render and validate each row once per tokenizer,
then use identical token IDs and answer-ID order on both reference paths.
For images, reuse the same visual embeddings to separate the text-weight
comparison from an end-to-end encoder comparison. CPU and CUDA results remain
separate. The provenance follow-up and dated screen results are recorded in
[`option-scale-2 records`](../docs/records/phase-4-plus-option-scale-2.md).

For structured counting, dashboard, and table evidence, use the separate
[`vision fixture set`](phase4-step4-vision/README.md). It adds image-only evidence
changes and layout invariance controls without changing this set or its results.
