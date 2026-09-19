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

