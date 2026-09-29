# Phase 4+ - Larger Choice option sets: investigation record

This is the dated tokenizer, prompt-size, memory, and reference-source evidence
from [the Phase 4+ larger-option plan](../phases/phase-4-plus-option-scale.md).
The phase document remains the source for current scope, implementation steps,
and adoption gates. The provisional label mapping is in
[the 512-label candidate table](phase-4-plus-option-label-candidates.tsv).

## Notes / Findings

- 2026-09-29: 現行の上限は `src/gemma4_decision_engine.cpp`、
  `src/gemma4_prompt_renderer.cpp`、`src/tokenizer.cpp`、
  `src/systemone_adapter.cpp`、`tools/branchscore_server.cpp` に分散する。
  ラベルは A–P で、HTTP は criteria key を辞書順に option ID へ写す。
- 2026-09-29: ローカルの Gemma 4 語彙参照
  `/home/ubuntu/llama.cpp/models/ggml-vocab-gemma-4.gguf` を
  `/home/ubuntu/llama.cpp/build/bin/llama-tokenize` で読み取り専用確認。
  `--no-bos` で `000`、`001`、`015`、`099`、`100`、`255`、`511` は
  各 3 token。短い英語説明を持つ試作 prompt の token 数は
  16/64/128/256/512 件で 329/1,193/2,345/4,649/9,257。
  この試作 prompt は本番 renderer と完全一致せず、推論時間・判断品質は未測定。
- 2026-09-29: 9,257 token の二つの F32 正方形 mask の値は
  `2 × 9,257² × 4 = 685,536,392 bytes`（約 654 MiB）。これは
  `src/prefill_engine.cpp` の入力 mask だけの計算であり、host 側にも同じ大きさの
  二つの vector を作る。peak 使用量そのものではない。
  手元の checkout は `third_party/ggml` submodule とビルド出力がないため、
  branchscore 自体の 512 件 model-backed 実測は未実施。
- 2026-09-29: F16 Prefill 案を追加調査。`src/state_cache.cpp` の K/V と
  `src/prefill_engine.cpp` の二つの mask は F32。手元の llama.cpp 参照 checkout
  `b36798957` の ggml では `ggml_soft_max_ext` の mask は F16/F32 を許容し、
  `ggml_mul_mat` の出力型は常に F32。9,257 token、8 attention heads の
  score テンソル 1 個は `9,257² × 8 × 4 = 2,742,145,568 bytes`
  （約 2.55 GiB）。これも peak 使用量そのものではない。project pinned ggml
  `456172ec733a135778adcd32d00e576a58232e45` は checkout に存在せず、
  CPU/CUDA の実 graph 対応と F16 数値差は未確認。
- 2026-09-29: 外部研究でも選択肢の位置とラベル token による selection bias が
  報告されている（[Zheng et al.](https://arxiv.org/abs/2309.03882)、
  [Pezeshkpour and Hruschka](https://arxiv.org/abs/2308.11483)）。
  Gemma 4 の 512 件での劣化幅はこの研究から推定せず、
  [phase 計画の focused verification](../phases/phase-4-plus-option-scale.md)
  で測る。
- 2026-09-29: ローカルの E4B Q4_K_M GGUF の token table は語彙参照 GGUF と
  SHA-256 が同じ `e5dd1a3d42323fe4c865b6554f5fa697dc93fb2dfa67579f7111a5bea9f4f638`。
  通常 token の英大文字 2 文字は 653 個あり、同一の実 GGUF を使った
  llama.cpp 参照 tokenizer で 653 個すべてが単独 1 token、piece 一致、
  試作 Gemma 回答位置で境界維持。そこから
  [保存した 512 個](phase-4-plus-option-label-candidates.tsv) を選び、branchscore の
  `GemmaTokenizer::tokenize`/`piece`/`is_special_token` でも現行
  `Gemma4PromptRenderer` が作る回答位置に対し 512/512 を確認した。
  project tokenizer 検証には、submodule 不在のため別 checkout の ggml を
  一時的にリンクした。これは語彙・token 境界の確認であり、モデルが 512 件を
  正しく選べることや現在の A–P と同程度の偏りを持つことは示さない。
- 2026-09-29: 追加されたローカルの E2B Q4_K_M GGUF も調査した。E2B/E4B の
  `tokenizer.ggml.tokens`、`tokenizer.ggml.token_type`、
  `tokenizer.ggml.merges` の内容 hash はそれぞれ一致した。
  E2B の context metadata は 131,072 token。llama.cpp 参照 tokenizer で
  英大文字二文字 653/653 が単独 1 token・回答位置の境界維持、E4B の 653 件
  ID 表と byte 一致した。branchscore の tokenizer と現行 renderer の回答位置
  でも保存済み 512/512 の ID・piece・境界を確認。512 件を表示した
  model-backed 判断品質・latency・peak memory は未測定。
