# Third-party software, references, and model assets

## Runtime dependency

- [ggml](https://github.com/ggml-org/ggml), pinned as the
  `third_party/ggml` submodule at
  `456172ec733a135778adcd32d00e576a58232e45` (ggml 0.24.0), is licensed under
  the MIT License. Its license is retained in the submodule. The project also
  uses the `stb_image` copy exposed by that pinned source tree.

## Research references

The following repositories were inspected as implementation references. They
are not linked as runtime dependencies and are not vendored by this repository.

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at
  `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f`, MIT License.
- [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) at
  `ca3ba65f142967030ecb453346e94d6f476a69df`, MIT License.

The detailed research record and inspected source locations are in
`docs/research/`.

## Model assets

No model weights or projectors are distributed with this repository. Local
development used the Q4_K_M text GGUF and F16 multimodal projector files from:

- [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
- [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)

The model cards and the
[official Gemma 4 license](https://ai.google.dev/gemma/apache_2) identify those
assets as Apache-2.0. Their license is separate from the MIT license for this
repository; users are responsible for reviewing the applicable model terms.
