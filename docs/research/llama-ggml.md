# llama.cpp and ggml research

This document is the canonical record for Phase 1 / Stage 1.2 research.

## Sources inspected

| Upstream | Revision | Date | Relevant files |
|---|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (local `llama.cpp/` reference tree) | `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f` | 2026-09-19 | `src/models/gemma4.cpp`, `src/llama-{context,batch}.cpp`, `tools/mtmd/{clip,mtmd}.cpp`, `tools/mtmd/models/gemma4v.cpp` |
| ggml in the same checkout | same revision | 2026-09-19 | `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp` |
| Local E2B/E4B GGUF and mmproj files | file metadata shown below | inspected 2026-09-19 | `models/{e2b,e4b}/*.gguf` |

llama.cpp is evidence only. The Phase 2 runtime remains an independent project
using ggml directly.

## Local model evidence

The four local files are GGUF v3. Relevant metadata and shapes were read from
their headers rather than inferred from model names.

| Property | E2B | E4B |
|---|---:|---:|
| Reported model size | 4.6B | 7.5B |
| Text blocks | 35 | 42 |
| Text embedding width | 1536 | 2560 |
| Vocabulary | 262144 | 262144 |
| Attention heads / KV heads | 8 / 1 | 8 / 2 |
| Full-attention K/V head width | 512 / 512 | 512 / 512 |
| Sliding-attention K/V head width | 256 / 256 | 256 / 256 |
| Sliding window | 512 | 512 |
| Layers with shared/reused KV | 20 | 18 |
| Per-layer input width | 256 | 256 |
| Context metadata | 131072 | 131072 |
| Final logit softcap | 30 | 30 |

E2B has varying FFN widths (6144 then 12288 in the inspected first/last
layers); E4B reports 10240. Both carry a `per_layer_token_embd` tensor whose
first dimension is `256 * block_count`, plus projection and per-layer gate
tensors. No separate output tensor was present in the inspected tensor list,
so llama.cpp's optional-output rule ties the LM head to the token embedding.

Both mmproj files identify `gemma4v`, contain a 16-block, width-768 Vision
Transformer with 12 heads and FFN width 3072, and use 16x16 RGB patch weights.
The projector tensor is `[768, text_width]`, producing width 1536 for E2B and
2560 for E4B. The files also contain audio tensors; audio is outside the first
milestone and must not leak into the initial interfaces.

## Text graph

- Token input is gathered from `token_embd` and scaled by `sqrt(n_embd)`.
  Raw multimodal embeddings bypass this scaling.
- A second per-layer input stream is gathered from
  `per_layer_token_embd`, projected together with the main embedding, then
  gated into every transformer block. For raw multimodal embeddings the
  reference currently uses token 0's per-layer embedding.
- Gemma 4 mixes sliding-window and full-attention layers. Q and K receive RMS
  normalization and RoPE; V receives RMS normalization. Some later layers
  reuse earlier KV rather than writing distinct K/V tensors, as directed by
  model metadata.
- Each block performs post-attention normalization and residual addition,
  gated GELU FFN work, post-FFN normalization and residual addition, then the
  per-layer embedding injection and optional layer-output scale.
- The final hidden state receives RMS normalization, the tied/explicit output
  projection, and `30 * tanh(logit / 30)` softcapping. These softcapped logits
  are the values from which option-token log-probabilities must be computed.

## Vision preprocessing and graph

- The Gemma 4 vision path is dynamically sized, uses bicubic resize, limits
  the projected image-token count to 70-1120, and uses `<|image>` / `<image|>`
  boundary tokens in the multimodal prompt path. Exact resize/padding behavior
  should follow the reference for Phase 2 compatibility.
- The graph maps input samples from `[0, 1]` to `[-1, 1]`, applies a stride-16
  2D patch convolution without bias, then adds learned x and y position-table
  rows.
- Each ViT layer uses two-dimensional NEOX-order RoPE with theta 100, RMS
  normalization, attention, and a gated FFN. Gemma 4 additionally RMS-normalizes
  V in attention.
- A 3x3 stride-3 average pool reduces both patch-grid dimensions. The result is
  scaled by `sqrt(768)`, optionally standardized, RMS-normalized, and projected
  from vision width 768 to the text width.
- The projected tensor is inserted at the image placeholder as raw LLM input
  embeddings. E2B/E4B remain causal even for these image chunks; larger Gemma 4
  variants may select non-causal handling and must not determine E2B/E4B logic.

### 2026-09-24 follow-up: large-image CUDA memory

At the pinned revision, `clip_graph_gemma4v::build()` calls the shared ViT graph.
Its attention builder uses `ggml_flash_attn_ext` when Vision Flash Attention is
enabled, casting K/V to F16 and requesting F32 accumulation. Otherwise it
materializes `K*Q`, softmax, and `V*attention` tensors. Vision Flash Attention
defaults to auto; warmup probes backend support and disables it with a memory
warning if unsupported. The Gemma 4 Vision warmup uses 256 output tokens to
avoid a maximum-size warmup allocation. This does not cap real images.

Gemma 4 Vision's default image range is 70--1120 pooled output tokens, with
3x3 pooling after ViT. Therefore each output token represents nine input
patches before attention. A 1920x1080 input aligns to 1920x1104, giving 8280
ViT patches and 920 output tokens. With 12 heads, an F32 full attention score
tensor at that size is about 3.06 GiB by itself. `--image-max-tokens` can lower
the preprocessing ceiling for dynamic-resolution models; Flash Attention is
the upstream way to avoid materializing this large score tensor. These are
source-based size estimates, not measured CUDA peak memory.
The separate text-model KV cache defaults to F16 for both K and V in
`llama_context_default_params()`; its type is not the Vision attention fix.

The Phase 4 implementation builds the Vision graph with the actual Q/K/V
shapes, checks each `GGML_OP_FLASH_ATTN_EXT` node with the selected backend's
`ggml_backend_supports_op()` before graph allocation, and rebuilds the existing
materialized attention path when support is false. It keeps all tensors on the
selected backend and exposes `vision_attention_path` as `flash`, `standard`, or
`not_used` in timing diagnostics. This is an implementation finding; it does
not claim that every ggml backend supports Flash Attention.

## Prefill, continuation decode, and outputs

- Prefill and continuation use the same model graph family. The concrete graph
  differs through the micro-batch: token IDs or raw embeddings, token count,
  absolute positions, sequence IDs, attention mask/cache view, and requested
  output rows.
- Prefill writes K/V for the whole state and only needs the final state's
  logits. An option branch starts from the same immutable prefix cache and
  advances positions causally through its tokens.
- To score option tokens `o[0..m-1]`, the state-final logits score `o[0]` and
  the logits after option token `o[i-1]` score `o[i]`. There is no reason to
  compute or copy logits after the final option token unless a terminator is
  deliberately included in the scoring contract.
- llama.cpp marks requested output positions in the batch and uses an
  `inp_out_ids` gather before final output work. Phase 2 should likewise build
  only the required output rows, rather than copy `[vocab, all_tokens]` to the
  host.
- Branch isolation is a cache ownership problem: each option needs the same
  prefix snapshot plus private writable continuation rows. The sequential
  baseline may restore/copy a snapshot per option; sharing writable cache rows
  is invalid.

## ggml backend lifecycle

- Backends are discoverable through registry/device enumeration and can be
  selected by name or type. Device initialization returns a backend; its
  default buffer type determines graph/weight allocation unless a host buffer
  type is explicitly required.
- Tensor metadata lives in a no-allocation `ggml_context`. Data storage is
  owned by backend buffers. Graph allocation can use the graph allocator for
  one backend; tensor set/get calls cross the host boundary.
- `ggml_backend_graph_compute_async()` queues work. The synchronous wrapper
  calls it then synchronizes. Async tensor copies use a backend-specific path
  when available; otherwise ggml synchronizes both endpoints and performs a
  blocking copy. Events are optional device capabilities, not universally
  available primitives.
- For the Phase 2 one-backend path, explicit synchronization is required at
  timing boundaries and before host reads. No cross-backend tensor copies are
  needed after initial host-to-backend inputs and weights.

## Backend scheduler findings

- The scheduler assigns nodes using existing buffer placement, weight
  placement, supported operations, priority, and optional op offload. It then
  groups adjacent nodes into backend-specific graph splits.
- When a split consumes an incompatible buffer from another backend, the
  scheduler allocates a layout-compatible destination tensor and substitutes
  it as the split input. It tries async backend copy; otherwise it synchronizes
  and performs a blocking copy. Events order reusable copies in parallel mode.
- `alloc_graph`, `graph_compute[_async]`, `synchronize`, and `reset` have
  distinct lifecycle roles. Scheduler-owned copies and allocation state remain
  valid only within that lifecycle.
- The scheduler solves a later multi-backend graph-placement problem. Phase 2
  has exactly one selected backend, so a direct single-backend graph allocator
  is smaller and makes ownership/copies easier to verify. Do not adopt the
  scheduler until Phase 4 measurements show a need.
