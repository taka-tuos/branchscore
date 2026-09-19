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

The current CLI exposes the Stage 2.1 backend foundation:

```sh
./build/branchscore --list-backends
./build/branchscore --backend cpu
./build/branchscore --backend cuda
./build/branchscore --backend vulkan
```

Selectors are case-insensitive and accept either a ggml device name or backend
family. `auto` chooses the first GPU/integrated GPU, falling back to the first
available device.

Stage 2.2 model loading can be exercised with a matching text GGUF and mmproj:

```sh
./build/branchscore --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf
```

Weights retain their GGUF tensor types when uploaded to the selected backend.
Only Gemma 4 vision and projector tensors are loaded from mmproj; bundled audio
tensors are deliberately excluded.

Tokenizer IDs and option-boundary behavior can be inspected without loading
model weights:

```sh
./build/branchscore-tokenize --model /path/to/model.gguf --text "Hello world"
./build/branchscore-tokenize --model /path/to/model.gguf \
  --prefix $'<|turn>model\n' --option "candidate text"
```

To enable the model-backed tokenizer CTest, configure with
`-DBRANCHSCORE_TEST_MODEL=/path/to/gemma-4-E2B-it-Q4_K_M.gguf`.
