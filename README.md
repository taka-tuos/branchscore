# branchscore

`branchscore` is an independent C++/ggml decision-scoring engine for Gemma 4
E2B/E4B GGUF models. Phase 2 begins with a sequential baseline: one model, one
backend, one request, and multiple option continuations.

## Configure and build

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

CPU is enabled by default. Build an optional GPU backend with
`-DBRANCHSCORE_CUDA=ON` or `-DBRANCHSCORE_VULKAN=ON`.

The CLI exposes the backend foundation and the sequential Phase 2+ decision
path:

```sh
./build/branchscore --list-backends
./build/branchscore --backend cpu
./build/branchscore --backend cuda
./build/branchscore --backend vulkan
```

Selectors are case-insensitive and accept either a ggml device name or backend
family. `auto` chooses the first GPU/integrated GPU, falling back to the first
available device.

Stage 2.2 model loading and Stage 2.7 option scoring can be exercised with a
matching text GGUF and mmproj:

```sh
./build/branchscore --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --state "The service is healthy." \
  --question "Which action should be taken?" \
  --option yes="Keep it running" \
  --option no="Stop it"
```

The request accepts 2--16 unique `--option ID=DESCRIPTION` values. The
library-level `Gemma4DecisionEngine` renders the fixed, versioned
`gemma4-fixed-v1` system/state/question prefix, tokenizes and prints the
prefix, then reports sum/mean continuation log-probabilities, relative
softmax probabilities, the selected option, explicit scoring metadata, and
prompt/option/stage timings. Relative probabilities are conditional on the
supplied option set, not calibrated confidence. Use `--image FILE` to include
one image in the request.

`--chat-template-file FILE` is accepted as a reserved request. Phase 2+ never
opens or validates `FILE`; it prints the exact filename with `applied=false`.
The optional GGUF `tokenizer.chat_template` field is reported as diagnostic
metadata and is not used by the effective renderer.

Weights retain their GGUF tensor types when uploaded to the selected backend.
Only Gemma 4 vision and projector tensors are loaded from mmproj; bundled audio
tensors are deliberately excluded.

Pass `--image FILE` with the model arguments to decode and preprocess a PNG,
JPEG, BMP, TGA, GIF, PSD, HDR, PIC, or PNM image. The image keeps its aspect
ratio, is aligned to the Gemma 4 patch/pooling grid, and is bicubic-resized to
the model's supported visual-token range.
Use `--vision-dump FILE` with `--image` to request the projected embeddings.
The engine includes that explicit host debug read in the Vision interval; the
CLI writes the two int32 dimensions (`tokens`, `width`) followed by row-major
float32 data after the decision result is complete.

Tokenizer IDs and option-boundary behavior can be inspected without loading
model weights:

```sh
./build/branchscore-tokenize --model /path/to/model.gguf --text "Hello world"
./build/branchscore-tokenize --model /path/to/model.gguf \
  --prefix $'<|turn>model\n' --option "candidate text"
```

To enable the model-backed tokenizer CTest, configure with
`-DBRANCHSCORE_TEST_MODEL=/path/to/gemma-4-E2B-it-Q4_K_M.gguf`.
