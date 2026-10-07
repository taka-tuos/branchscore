# Phase 4+ option-scale: ラベル形式・初期判断品質

[記録索引](../phase-4-plus-option-scale.md) · [phase計画](../../phases/phase-4-plus-option-scale.md)

2026-09-29のtokenizer照合、2例の品質screening、暫定512ラベル候補。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

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
  [後続計画の focused verification](../../phases/phase-4-plus-option-scale-2.md)
  で測る。
- 2026-09-29: ローカルの E4B Q4_K_M GGUF の token table は語彙参照 GGUF と
  SHA-256 が同じ `e5dd1a3d42323fe4c865b6554f5fa697dc93fb2dfa67579f7111a5bea9f4f638`。
  通常 token の英大文字 2 文字は 653 個あり、同一の実 GGUF を使った
  llama.cpp 参照 tokenizer で 653 個すべてが単独 1 token、piece 一致、
  試作 Gemma 回答位置で境界維持。そこから
  [保存した 512 個](../phase-4-plus-option-label-candidates.tsv) を選び、branchscore の
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
- 2026-09-29: project checkout に pinned ggml `456172ec733a135778adcd32d00e576a58232e45`
  が存在することを確認し、CPU build を構成して `branchscore-tokenize` をビルドした。
  一時的な読み取り専用 probe は project の `Gemma4PromptRenderer` と
  `GemmaTokenizer` を使い、現行 renderer が作る 16-option prompt の回答位置に
  保存済み 512 ラベルを追加して検証した。E2B/E4B とも 512/512 が単一 normal
  token、piece 一致、個別 ID 一意、prompt 境界維持となり、TSV の token ID も
  512/512 一致した。これは形式検証であり、512 option prompt の判断品質を示さない。
- 2026-09-29: この環境には E2B/E4B の text GGUF はあるが、対応する
  `mmproj` GGUF が `/home/ubuntu` 配下に見つからなかった。現行 `ModelLoader` と
  `branchscore-bench` は text model と `mmproj` の組を要求するため、Step 1.2 の
  model-backed label/順序比較はその時点では未実施だった。続けてユーザーが
  `/home/ubuntu/llama.cpp/models` に対応 mmproj を配置したため、以下の測定を行った。
- 2026-09-29: E2B/E4B Q4_K_M text GGUF と、それぞれの
  `e2b-mmproj-F16.gguf` / `e4b-mmproj-F16.gguf` を使用。単一 CPU backend、逐次実行。
  Quadro P620 は VRAM 2 GiB のため使用していない。2つの手作り text fixture
  (`healthy_service`, `incomplete_report`) で、production renderer の A–P 16件と
  逆順、probe renderer の A–P 16件 control、保存済み二文字ラベル16件と逆順、
  さらに16件の無関係選択肢を加えた32件、48件を加えた64件を各モデルで測定した。
  Probe は現行 `ModelLoader`、`PrefillEngine`、`gather_categorical_logits` を使用。
  拡張 prompt は system 指示の `letter` を一般化した `label` に置き換えた試験用
  renderer であり、production renderer/readout/schema の変更ではない。
- 2026-09-29: 健康状態 fixture は全条件で両モデルとも期待した `keep_running` を選択。
  未完成レポート fixture は E2B が期待 ID `wait_for_review` を一度も選ばず、元順では
  `assign_owner`、逆順では `complete_facts` に winner が変わった。E4B は全条件で
  `wait_for_review` を選び、順序反転でも維持した。32/64件への無関係候補追加は、
  同じ順序の16件二文字ラベル条件から top-1 を変えなかった。手作りの2例だけなので
  品質結論ではなく、E2B の順序依存とモデル差を示す初期観察である。
- 2026-09-29: prompt token 数は健康状態 fixture が16/32/64件で278/467/831、
  レポート fixture が297/486/850。1回ずつの CPU Prefill は E2B で16件5.88–6.21秒、
  32件10.15–10.43秒、64件19.23–19.36秒。E4B はそれぞれ11.35–12.11秒、
  19.44–20.10秒、36.08–36.56秒。warmup/反復なしであり、分位 latency や
  peak memory の測定ではない。全28条件の各 option `raw_score`、relative probability、
  token ID と選択結果は
  [初回 score sweep JSONL](../phase-4-plus-option-scale-scores-2026-09-29.jsonl) に保存。

## 暫定 512 ラベル候補

[512 件の二文字ラベルと token ID](../phase-4-plus-option-label-candidates.tsv)
を調査用に保存した。`index` は 0 始まりの入力位置候補で、caller の
semantic option ID ではない。英大文字 2 文字で統一し、先頭文字と末尾文字が
それぞれ A–R で 20 回、S–Z で 19 回現れるように選んだ。その条件の下で
token ID の合計が小さい集合を選び、先頭文字ごとに巡回して並べた。
token ID は BPE 語彙上の値であり、モデルの回答 logit や判断品質の代用ではない。
この表は Phase 3+ の A–P ラベルや 2–16 件の本番契約を変更しない。
現行の `tokenize_answer_label` は A–P 以外を拒否し、system 指示も 1 文字の
回答を求めるため、本番で使うには新しい renderer/validator の契約が必要。
