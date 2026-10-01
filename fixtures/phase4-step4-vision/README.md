# Step 4: structured visual evidence

[`../phase4-step4-vision.jsonl`](../phase4-step4-vision.jsonl) supplies 9 RGB PNG
images and 27 categorical requests. Each image has a base candidate order,
reversed candidates, and two unrelated options added. Candidate counts are
4–8, within the current A–P contract. Original semantic IDs and descriptions
are preserved.

| Family | Rule | Base answer | Evidence change / new answer | Relayout answer |
|---|---|---|---|---|
| Board | Count only red squares within each printed region | `east` (4) | Move one red square from EAST to NORTH → `north` (4) | `east` |
| Status | FAILED and OPEN, then highest severity | `lyra` | LYRA response OPEN → ACK → `vega` | `lyra` |
| Table | READY and quality ≥80, then lowest cost | `birch` | BIRCH quality 84 → 78 → `aster` | `birch` |

At the same candidate order, all three visual variants have identical state,
question, and option descriptions. The changed image supplies the only new
evidence. Relayout moves named cards or table rows and changes decorative
colors while preserving the evidence and answer. Compare semantic IDs, since
candidate reversal changes answer labels.

Expected IDs and reasons are fixed from the source facts before inference.
The production benchmark reader ignores the evaluation metadata and does not
place expected answers or source facts in the prompt. Image paths resolve
relative to the input JSONL. `contact-sheet.png` is a preview only; feed the
individual images through the JSONL.

## Regeneration and verification

From the repository root:

```sh
python3 fixtures/phase4-step4-vision/generate.py
```

Rendering uses ImageMagick (`convert`/`montage` or `magick`) and DejaVu Sans,
Sans Bold, and Sans Mono. Python uses only its standard library to orchestrate
rendering and derive answers. [`manifest.json`](manifest.json) contains source
facts, relationships, dimensions, tool version, and SHA-256 hashes. Repeated
generation matched all image and fixture hashes with ImageMagick 6.9.12-98;
different font or ImageMagick versions can change rasterization and hashes.

Each evidence image is 960×720. The production decoder and preprocessor
accepted all images with the local E2B/E4B mmproj configurations: unchanged
960×720 prepared size, 2,700 patches, and 300 visual tokens. File and encoded-byte
inputs produced identical finite normalized tensors. The production renderer
and answer boundary checks passed all 27 rows for E2B/E4B BF16/Q4; their token
and answer ID sequences matched. Pixel differences for changed images were
confined to the moved object or edited cell. These checks validate inputs;
model inference and accuracy measurement for this set remain pending.

## Evaluation scope

Treat this as 3 task families, 6 distinct evidence scenarios, and 3 layout
derivatives. The 27 requests are correlated transformations, not 27 independent
accuracy samples. Report base accuracy, whether evidence changes cause the
expected selection changes, and whether relayout/reversal/distractors preserve
the expected semantic selection. Record top-two logit margins separately:
visual complexity does not establish numerical near-ties.

For Step 4 weight A/B, hold token IDs, candidate order, and visual embeddings
fixed between BF16 and Q4. Measure encoder differences separately in an
end-to-end comparison. CPU and CUDA results remain separate. The original
numeric controls and simpler quality set retain their historical inputs.
Validation details are in the
[`option-scale-2 records`](../../docs/records/phase-4-plus-option-scale-2.md#step-4-structured-vision-fixtures).

![Preview: rows are board/status/table; columns are base/changed/relayout](contact-sheet.png)
