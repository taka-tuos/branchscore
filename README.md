# branchscore

[日本語](README.ja.md) · English

`branchscore` is a standalone C++/ggml decision engine for Gemma 4 E2B/E4B
GGUF models. Give it a text state, an optional image, a question, and candidate
options; it returns the selected option, raw scores, relative probabilities,
and timings.

The goal is a small local runtime for choosing among supplied options. One
practical use is matching a part package shown by a camera to a bill of materials
(BOM). It is a hobby proof of concept developed largely with AI assistance,
with no stable API or production support guarantees.

## Build and try it

Run these commands from the `branchscore` repository root. Requirements:
a C++17 compiler, CMake 3.20+, and the development tools for your selected ggml
backend. The examples use Ninja; another CMake generator is also fine.

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

CPU is enabled by default. Add `-DBRANCHSCORE_CUDA=ON` or
`-DBRANCHSCORE_VULKAN=ON` to the configure command to build a GPU backend.
The server is built by default using vendored llhttp C sources; a normal build
needs no Node.js, npm, pkg-config, or system llhttp package. For a CLI-only build,
set `-DBRANCHSCORE_BUILD_SERVER=OFF`.

Model weights are not included. Development uses the Q4_K_M text GGUF and
matching `mmproj-F16.gguf` from the Unsloth Gemma 4 E2B/E4B repositories listed
in [THIRD_PARTY.md](THIRD_PARTY.md#model-assets). Supply both files, including
for text-only requests.

```sh
./build/branchscore --list-backends
./build/branchscore --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --state "The service is healthy." \
  --question "Which action should be taken?" \
  --option keep="Keep it running" \
  --option stop="Stop it"
```

Add `--image FILE` for one PNG or JPEG image. Backend selectors accept a device
name or family, case-insensitively. `--backend auto` chooses the first GPU or
integrated GPU, falling back to the first available device.

Tests needing weights are skipped unless model paths are configured. To run
them, configure with `-DBRANCHSCORE_TEST_MODEL=/path/to/text.gguf`,
`-DBRANCHSCORE_TEST_MMPROJ=/path/to/mmproj.gguf`, and
`-DBRANCHSCORE_TEST_BACKEND=cpu` (or your selected backend), then rebuild and
run CTest.

## Browser UI and HTTP API

Start a server that keeps one model loaded:

```sh
./build/branchscore-server --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --host 127.0.0.1 --port 8080
```

Open [the browser UI](http://127.0.0.1:8080/ui), press **Connect**, and enter the
state, question, and options. The bearer-token field can be empty for this
loopback bind. For a BOM, bulk-paste one part number per line; an optional
description goes after a tab. Blank descriptions are sent as `null` so that
the model sees the part number alone.

The endpoints are `GET /ui`, `GET /healthz`, and `POST /v1/systemone`.
A minimal text request is:

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data '{"state":"The service is healthy.","model":"branchscore-local","questions":{"action":{"type":"choice","instructions":"Which action should be taken?","criteria":{"keep":"Keep it running","stop":"Stop it"}}}}'
```

The API implements the TypeSafe Choice envelope: 1–16 questions per request,
with 2–512 criteria per question, evaluated sequentially. Only the model ID
`branchscore-local` is accepted. String criteria appear as `key: description`;
null criteria appear as the key alone. Questions and criteria run in lexical
key order, as determined by the current JSON codec. CLI/JSONL options retain
input order. Images and more than 255 criteria are branchscore extensions;
Score and Noul are unsupported.

For a LAN bind, set `BRANCHSCORE_BEARER_TOKEN` in the environment before startup
and choose a non-loopback `--host`. `/healthz` and `/v1/systemone` then require
`Authorization: Bearer ...`; `/ui` stays public so the browser can load the
form. The UI keeps the token in memory. The server speaks plain HTTP; use TLS
termination on untrusted networks.

Responses contain the selected criterion key, relative probabilities, and
`branchscore` diagnostics with raw logits, actual option order, prompt/readout
identities, and timings. `confidence: 1.0` is a fixed compatibility placeholder,
not a measured certainty. `usage.input_tokens` counts rendered text tokens,
excluding expanded visual tokens; `usage.output_tokens` is zero.
`branchscore.prefill_positions` reports the expanded position count across
questions. See the [HTTP contract](docs/phases/phase-4-plus-http-server.md)
for the image envelope, authentication, errors, and response details.

## How scoring works

1. Render the optional image, state, question, and **all** option descriptions
   in one fixed Gemma 4 prompt.
2. Run Vision when an image is present, then Prefill the complete prompt.
3. Read each option's answer-label logit at the same next-token position.
4. Apply a temperature-1 softmax over those logits and select the first maximum.

For 2–16 options, `gemma4-categorical-v1` assigns A–P labels. For 17–512,
`gemma4-categorical-512-v1` uses a fixed set of two-letter labels. Each label
must be a single normal token at the answer boundary; validation failure
rejects the request. The number of printed letters is not the token count.

Option descriptions provide judgment context. Their continuation likelihoods
are not scored, and no answer token is sampled or consumed. The final
vocabulary projection runs on the backend; a small ggml gather transfers only
the requested answer logits to the host. Probabilities are relative to the
supplied option set, **not calibrated confidence**. They can change when options,
order, or wording change.

Prefill runs in 512-token microbatches within one logical prompt evaluation.
CUDA uses F16 K/V and masks with Flash Attention and F32 accumulation;
CPU/Vulkan use F32 K/V with ordinary Prefill attention. This is the current
implementation, not a requirement that every future backend use those types.
See [architecture](docs/architecture.md) for prompt bytes, scoring formulas,
component ownership, and execution details.

## Scope and deliberate design choices

- Use ggml directly in an independent C++ project. llama.cpp and SemIf are
  implementation/research references, not wrapped runtimes.
- Focus on categorical selection. General chat completion, long-form
  generation, training, calibration, and exact Jev output reproduction are
  outside the project scope.
- Keep the current baseline sequential: one model, one selected backend, one
  decision at a time. Request workers and scheduling are later work only if
  measurements justify them; transformer layer splitting and tensor
  parallelism are out of scope.
- Use a versioned built-in categorical prompt with direct-answer instructions.
  GGUF chat templates are diagnostic metadata. `--chat-template-file` is a
  reserved no-op: it records the path without reading or applying the file.
- Reject input exceeding the supported budgets without truncating it.

These choices and the current input contract are defined in
[requirements](docs/requirements.md). The internal `branchscore_core` target
is not a stable or installable library API.

## Current limits and remaining work

These are the limits of this implementation and its validation, rather than
permanent model or project restrictions.

| Input | Current limit |
|---|---|
| Options per decision | 2–512 |
| Images per decision | At most one PNG/JPEG |
| Expanded Prefill positions per decision | 16,384, also within the model context |
| HTTP questions / combined Prefill positions | 16 / 32,768 |
| HTTP request body | 16 MiB |
| HTTP source image | 8,192 pixels per side and 8 megapixels total |

Expanded positions include visual tokens and all prompt text. Supporting 512
options does not mean arbitrarily long descriptions will fit, nor does the
model's context metadata guarantee enough memory. HTTP preflights all questions
before inference: position overflow returns 413, while 513+ options returns
422. `/healthz` advertises the decision and position limits.

The prompt currently places the image first. K/V cache belongs to one decision
and is discarded afterward; repeated images/questions do not reuse a prefix.
Image placement and cache reuse are deferred investigations. The server handles
one request at a time and closes the connection after each response.

Large-input resource validation primarily covers E4B Q4_K_M on CUDA with an
RTX 2060 SUPER (8 GiB). The same 512-option resource workload has not been
validated on CPU/Vulkan. On that GPU, 512 synthetic 20-digit part numbers with
a blank FHD image took about 11.5 seconds; the largest observed VRAM sample
across resource checks was 6,352 MiB. These are resource measurements, not
package OCR accuracy results or universal hardware requirements.

Known quality differences include an E2B close-score regression with CUDA
F16/Flash and option-order sensitivity in E4B large-option image fixtures.
High-precision CUDA comparisons and real BOM/image evaluation remain follow-up
work. See the [512 adoption record](docs/records/phase-4-plus-option-scale-2/runtime-512-adoption.md)
for dated results and what was actually verified.

The project has no security-hardening, stable API, or long-term compatibility
guarantees. Local model/input parsers and image decoders have not been fuzzed
for hostile input.

## Benchmark and diagnostics

`branchscore-bench` keeps the model loaded and evaluates JSONL rows sequentially.
Each nonempty line requires `id`, `state`, `question`, and
`options: [{"id": "...", "description": "..."}]`. Extra fields are ignored;
an optional `image` path is relative to the input file.

```sh
./build/branchscore-bench --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --input fixtures/phase3-text.jsonl \
  --output /tmp/branchscore-e2b.jsonl
```

The output must not already exist. Default warmup runs the first request once;
change it with `--warmup COUNT`. Schema 2 output includes a run row, a decision
row per input, and an aggregate row with p50/p95 latency and decisions/second.
Model loading, warmup, and file writes are outside the request measurements.

`branchscore-tokenize` inspects token IDs and answer boundaries without loading
model weights:

```sh
./build/branchscore-tokenize --model /path/to/model.gguf --text "Hello world"
./build/branchscore-tokenize --model /path/to/model.gguf \
  --prefix $'<|turn>model\n' --answer-label A
```

For projected image embeddings, add `--vision-dump FILE` and `--image FILE`
to the decision CLI. The dump is written after inference, creating or
truncating the file: two int32 dimensions (`tokens`, `width`), then row-major
float32 values. HTTP responses omit full prompts, token IDs, and Vision dumps.

## Documentation and license

Start with the [documentation map](docs/README.md). Current plans are in the
[phase index](docs/phases/README.md); dated experiments and development
environments are in [records](docs/records/README.md). For HTTP parser
regeneration, see [llhttp notes](third_party/llhttp/README.md).

Project code and documentation use the [MIT License](LICENSE). Dependencies,
research references, and separately licensed model assets are documented in
[THIRD_PARTY.md](THIRD_PARTY.md).
