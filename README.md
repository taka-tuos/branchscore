# branchscore

`branchscore` is a standalone C++/ggml proof of concept for categorical
decisions with Gemma 4 E2B/E4B GGUF models. It displays all candidate
descriptions in one prompt, reads the single-token A-P answer logits, and
reports relative option probabilities. Autoregressive chat generation is not
its primary purpose.

This is a hobby and experimental project developed largely through
AI-assisted ("vibe coding") workflows. The implementation has been checked
against pinned upstream sources and exercised with focused tests, but it is
not production-ready, security-hardened, or covered by API, compatibility, or
support guarantees.

The core decision call remains the sequential Phase 3+ path: one model, one
backend, one decision, and 2–16 ordered displayed options. The Phase 4+ server
can accept up to 16 Choice questions in one envelope and evaluates them one at
a time. See [`docs/README.md`](docs/README.md) for the design, research record,
phase history, and roadmap.

## What it currently does

- Loads matching Gemma 4 E2B/E4B text and multimodal-projector GGUF files.
- Selects one ggml CPU, CUDA, or Vulkan backend.
- Preprocesses an optional image and executes the Gemma 4 vision graph.
- Renders the fixed, versioned `gemma4-categorical-v1` decision prompt.
- Prefills the complete displayed-options prompt once.
- Gathers only the A-P answer-slot logits with one small ggml graph.
- Reports raw answer-slot logits, probabilities relative to the supplied
  option set, the selected option, and shared readout timings.
- Provides a warm-loaded, sequential branchscore-bench JSONL runner with
  row-level scores and aggregate latency/throughput measurements.

Run a small sequential benchmark. Each non-empty input line is one decision
object with required id, state, question, and
options: [{"id": ..., "description": ...}] fields. Extra fields are ignored
so existing project fixtures can be reused. An optional image path is resolved
relative to the input JSONL file.

    ./build/branchscore-bench --backend cpu \
      --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
      --mmproj /path/to/mmproj-F16.gguf \
      --input fixtures/phase3-text.jsonl \
      --output /tmp/branchscore-e2b.jsonl

The default warmup evaluates the first request once and discards its result;
use --warmup COUNT to change it. The output contains one run metadata row, one
decision row per input row, and one aggregate row. It uses schema version 2 and
includes ordered answer labels and token IDs, raw logits, prompt/readout
identity, stage timings, p50/p95 request latency, and decisions/second. The
output path must not already exist; model loading, warmup, and result-file
writes are outside the measured request interval. Context overflow is an error
and is never silently truncated.

The `branchscore_core` CMake target is an internal implementation boundary,
not a stable or installable library API.

## Models used for development

Model files are not included in this repository. Phase 2/2+ development and
model-backed tests used the following Unsloth quantizations:

- [`unsloth/gemma-4-E2B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
  - `gemma-4-E2B-it-Q4_K_M.gguf`
  - `mmproj-F16.gguf`
- [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)
  - `gemma-4-E4B-it-Q4_K_M.gguf`
  - `mmproj-F16.gguf`

Those model assets are separately licensed under Apache-2.0; the repository's
MIT license covers this project's own code and documentation. Consult the
model cards before downloading or using the weights.

## Configure and build

Requirements are a C++17 compiler, CMake 3.20 or newer, and the tools needed
by the selected ggml backend. Ninja is used in the examples but is not a
project requirement. The default build includes `branchscore-server` and uses
the checked-in generated llhttp 9.3.1 C source, so it does not require
pkg-config or `libllhttp-dev`. Set `-DBRANCHSCORE_BUILD_SERVER=OFF` for a
CLI-only build.

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

The parser's TypeScript grammar and pinned npm lockfile are retained under
[`third_party/llhttp/upstream`](third_party/llhttp/upstream). After changing
those generator inputs, regenerate the checked-in C source with:

```sh
cmake --build build --target branchscore-llhttp-regenerate
```

Regeneration requires Node.js/npm and network access for the locked npm
packages; normal CMake builds do not.

CPU is enabled by default. Enable one optional GPU backend with
`-DBRANCHSCORE_CUDA=ON` or `-DBRANCHSCORE_VULKAN=ON`. NCCL and multi-GPU
execution are deliberately disabled.

Tests that require model weights return CTest's skip code when model paths are
not configured. Run the complete E2B model-backed CPU suite with:

```sh
cmake -S . -B build-model -G Ninja \
  -DBRANCHSCORE_TEST_MODEL=/path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  -DBRANCHSCORE_TEST_MMPROJ=/path/to/mmproj-F16.gguf \
  -DBRANCHSCORE_TEST_BACKEND=cpu
cmake --build build-model
ctest --test-dir build-model --output-on-failure
```

## CLI usage

List the ggml devices visible to the process:

```sh
./build/branchscore --list-backends
```

Score a text-only decision:

```sh
./build/branchscore --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --state "The service is healthy." \
  --question "Which action should be taken?" \
  --option yes="Keep it running" \
  --option no="Stop it"
```

Selectors are case-insensitive and accept a ggml device name or backend
family. `auto` chooses the first GPU/integrated GPU and otherwise the first
available device. Add `--image FILE` for one image.

`--chat-template-file FILE` is currently a transparent reserved no-op. The
path is retained in result metadata but is not opened or applied. The GGUF
`tokenizer.chat_template` value is likewise diagnostic metadata; the fixed
`gemma4-categorical-v1` renderer remains the effective prompt source.

Use `--vision-dump FILE` with `--image` to request projected embeddings. The
target file is created or truncated after inference completes and contains two
int32 dimensions (`tokens`, `width`) followed by row-major float32 values.

## Sequential HTTP server

`branchscore-server` loads one model/backend before it starts listening. It
serves `GET /healthz` and the Choice subset of `POST /v1/systemone`, evaluating
up to 16 questions sequentially. Each Choice supports 2–16 string or null
criteria. The request's `model` must be `branchscore-local`; other model IDs,
Score, and Noul questions are rejected. Question and criteria maps use lexical
key order. A string criterion is shown to the model as `key: description`,
while a null criterion shows only its key.

The optional image extension accepts base64 PNG or JPEG bytes. Request bodies
are limited to 16 MiB; source images are limited to 8,192 pixels per side and
8 megapixels total before pixel decoding. Requests run one at a time, and the
server closes each response connection. It logs only request ID, status, and
wall time by default.

The default bind is loopback on port 8080. A non-loopback bind requires
`BRANCHSCORE_BEARER_TOKEN` in the environment before model loading; the token
is sent as `Authorization: Bearer ...`, never as a command-line argument.
Use a TLS-terminating reverse proxy on untrusted networks.

```sh
./build/branchscore-server \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --backend cuda --host 127.0.0.1 --port 8080
```

Choice responses include the TypeSafe-compatible fixed `confidence: 1.0`
placeholder, identified by `branchscore.confidence_kind`. It is not calibrated
confidence and must not be used for threshold decisions. `usage.input_tokens`
counts rendered text prompt tokens and excludes visual tokens;
`usage.output_tokens` is zero. Per-question timings and prompt/readout
identities are in `branchscore.questions`; raw answer-slot logits are included
there for diagnostics. The separate `branchscore.http_handling_ms` and server
`wall_ms` log make HTTP request time visible alongside `evaluate` timings.

Tokenizer IDs and answer-label boundary behavior can be inspected without loading
model weights:

```sh
./build/branchscore-tokenize --model /path/to/model.gguf --text "Hello world"
./build/branchscore-tokenize --model /path/to/model.gguf \
  --prefix $'<|turn>model\n' --answer-label A
```

## Tested development environment

These are reference environments, not a portability guarantee. On
2026-09-20, the Phase 2/2+ work was built and exercised with:

- Arch Linux (rolling)
- GCC 16.2.1
- CMake 4.4.3
- Ninja 1.13.2
- ggml 0.24.0 at `456172ec733a135778adcd32d00e576a58232e45`
- CUDA Toolkit 13.3 (`nvcc` 13.3.73)
- NVIDIA GeForce RTX 2060 SUPER for the recorded CUDA and Vulkan runs
- the Unsloth E2B/E4B Q4_K_M text models and matching F16 projectors listed
  above

The complete local E2B CPU suite passes 9/9 tests. CPU, CUDA, and Vulkan
backend initialization and model execution have also been exercised locally;
the phase notes contain the exact observations and indicative timings.

## Important limitations

- Inputs, model files, and images are assumed to be trusted local files. The
  parsers and decoders have not been fuzzed or hardened for hostile input.
- Probabilities are relative only to the supplied option set and are not
  calibrated confidence values.
- Selection uses the requested A–P answer-slot logits and a temperature-1
  softmax across the supplied options.
- EOS and turn-ending tokens are not scored.
- There is no stable API, package installation contract, production fault
  tolerance, or general chat-completion interface.

## License and acknowledgements

The project is licensed under the [MIT License](LICENSE). Third-party runtime,
research, and model sources are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
