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
  [初回 score sweep JSONL](phase-4-plus-option-scale-scores-2026-09-29.jsonl) に保存。

## CPU resource sweep

- 2026-09-29: 追加された対応 mmproj と pinned project ggml を使い、CPU backend、逐次、
  15 GiB RAM の同一マシンで E2B/E4B の text/image を16/64/128/256/512件ずつ実行した。
  GPU はユーザー指示に従い使用していない。モデルは各 mode ごとにロードし、count は
  昇順に一度ずつ実行した。20条件すべて成功。prefill を含む全 option の raw scores は
  各 benchmark 条件ではなく選択 ID のみ出力する probe を使ったため、これは
  resource/latency sweep であり、512件の判断品質比較ではない。
- 固定 workload は各候補の説明が `Candidate 0001 is a possible response.` と同じ長さの
  一文、question が `Which response best matches the request?`。画像には checkout 内の
  `third_party/ggml/examples/yolo/data/labels/72_5.png`（42×64 grayscale）を使った。
  画像条件では visual token が77。従ってこれは制御された短い説明の測定であり、実運用の
  説明長分布を代表しない。
- 下表の `peak RSS` は `getrusage(RUSAGE_SELF).ru_maxrss` を probe process 内で読んだ
  累積 high-water mark。各 count は別 process ではなく昇順の同一 process なので、値は
  model load 後からその行までの cumulative peak であり、MiB に換算した。model load 後の
  peak は E2B 3,443.7 MiB、E4B 5,229.0 MiB。JSONL の
  `tokenization_and_boundary_validation_ms` は prompt の tokenization に加え、全候補ごとの
  standalone label と `prompt + label` 境界再検証を含むため、純粋な prompt tokenization
  時間ではない。各条件1回、warmup/反復なし。

| Model | Mode | Options | Prompt tokens | Visual tokens | Prefill ms | Readout ms | Peak RSS MiB, cumulative |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B | text | 16 | 367 | 0 | 7,752 | 0.025 | 3,530 |
| E2B | text | 64 | 1,231 | 0 | 29,609 | 0.023 | 3,778 |
| E2B | text | 128 | 2,383 | 0 | 66,268 | 0.034 | 4,150 |
| E2B | text | 256 | 4,687 | 0 | 181,555 | 0.046 | 5,136 |
| E2B | text | 512 | 9,295 | 0 | 567,964 | 0.060 | 8,754 |
| E2B | image | 16 | 369 | 77 | 9,809 | 0.017 | 3,549 |
| E2B | image | 64 | 1,233 | 77 | 32,445 | 0.023 | 3,814 |
| E2B | image | 128 | 2,385 | 77 | 70,188 | 0.030 | 4,188 |
| E2B | image | 256 | 4,689 | 77 | 184,950 | 0.039 | 5,207 |
| E2B | image | 512 | 9,297 | 77 | 590,807 | 0.053 | 8,858 |
| E4B | text | 16 | 367 | 0 | 15,307 | 0.026 | 5,356 |
| E4B | text | 64 | 1,231 | 0 | 55,362 | 0.025 | 5,676 |
| E4B | text | 128 | 2,383 | 0 | 118,331 | 0.030 | 6,171 |
| E4B | text | 256 | 4,687 | 0 | 292,988 | 0.040 | 7,409 |
| E4B | text | 512 | 9,295 | 0 | 843,069 | 0.505 | 10,898 |
| E4B | image | 16 | 369 | 77 | 18,770 | 0.018 | 5,384 |
| E4B | image | 64 | 1,233 | 77 | 59,515 | 0.022 | 5,721 |
| E4B | image | 128 | 2,385 | 77 | 123,636 | 0.030 | 6,212 |
| E4B | image | 256 | 4,689 | 77 | 301,672 | 0.040 | 7,449 |
| E4B | image | 512 | 9,297 | 77 | 854,980 | 0.063 | 11,602 |

Mask host build/upload at 512件 were 481/321 ms (E2B text), 488/324 ms (E2B image),
481/319 ms (E4B text), and 490/324 ms (E4B image). Image preprocessing was about 3 ms,
vision encoding about 1.83–1.86 s per condition. Readout graph size was 2 nodes throughout.
The two F32 masks at 9,295 text positions occupy 659.2 MiB per copy of the pair; host vectors
and backend tensors coexist during prefill. This storage estimate does not include the much
larger graph activations/attention scores or model weights.

During periodic live `/proc/<pid>/status` samples, E2B 512 runs showed 0 process `VmSwap`;
E4B text 512 reached about 0.68 GiB process `VmSwap`, while E4B image samples showed 0. These
swap observations were not continuously profiled and are not fields in the resource JSONL.
All four 512件 CPU runs completed, but took 9.47/9.85 minutes for E2B text/image and
14.05/14.25 minutes for E4B text/image. This confirms executability on this machine under the
fixed workload only; it does not establish an acceptable request budget or public support limit.
Full raw rows are in
[CPU resource JSONL](phase-4-plus-option-scale-resource-cpu-2026-09-29.jsonl).

## F16 mask probe

- 2026-09-29: pinned ggml `ggml_soft_max_ext` accepts contiguous F16 or F32 masks, and the
  pinned CPU softmax implementation reads an F16 mask by converting each element to F32.
  A temporary build changed only the two mask tensors and host mask vectors to F16; both kept
  exact `0` and `-∞` values. No tracked runtime source was changed.
- E2B/E4B text prompts at128/256件 were compared on CPU, once each. All 768 candidate logits
  compared across four conditions were bit-identical (max and mean absolute raw-score difference
  `0`); all winners matched. F16 cut backend mask upload roughly in half. Converting/building
  host masks took longer, and Prefill time stayed within about 2% of F32.

| Model | Options | Prefill F32/F16 ms | Host mask build F32/F16 ms | Backend mask upload F32/F16 ms | Peak RSS F32/F16 MiB |
|---|---:|---:|---:|---:|---:|
| E2B | 128 | 67,090 / 67,261 | 32.2 / 40.2 | 21.6 / 10.6 | 4,145 / 4,091 |
| E2B | 256 | 178,944 / 179,857 | 123.0 / 186.9 | 81.1 / 41.9 | 5,131 / 4,963 |
| E4B | 128 | 121,238 / 119,053 | 31.9 / 40.2 | 21.4 / 10.8 | 4,789† / 6,104 |
| E4B | 256 | 292,467 / 294,308 | 122.0 / 187.2 | 80.6 / 42.2 | 7,395 / 7,228 |

† E4B F32 128件 process was swapped during the run (`VmSwap` sampled at about1.46 GiB),
so its RSS is not directly comparable with the F16 run. At256件, F16 peak RSS was about168 MiB
lower in both models. These are one-shot process measurements; F16 mask remains a probe result,
not a production change. Candidate raw scores are in
[F16 mask comparison JSONL](phase-4-plus-option-scale-f16-mask-cpu-2026-09-29.jsonl).

## F16 K/V cache probe

- 2026-09-29: a temporary `StateCache` variant stored K and V in F16, while Prefill graph output
  remained F32 and was copied into the cache. The pinned ggml CPU graph successfully wrote and
  read the F16 cache for E2B/E4B text prompts at128/256件. Masks stayed F32 to isolate cache type.
  No production source or cache contract was changed.
- All four winners matched the F32 baseline in this single controlled fixture. Raw logits changed;
  therefore the result probabilities are not numerically interchangeable with F32.

| Model | Options | Max / mean absolute raw-logit diff | Winner F32/F16 KV | Prefill F32/F16 KV ms | Peak RSS F32/F16 KV MiB |
|---|---:|---:|---|---:|---:|
| E2B | 128 | 2.382 / 1.004 | option_0014 / option_0014 | 66,268 / 66,527 | 4,150 / 4,132 |
| E2B | 256 | 4.965 / 3.021 | option_0014 / option_0014 | 181,555 / 164,905 | 5,136 / 5,284 |
| E4B | 128 | 1.059 / 0.286 | option_0013 / option_0013 | 118,331 / 117,118 | 6,171 / 6,043 |
| E4B | 256 | 1.160 / 0.263 | option_0013 / option_0013 | 292,988 / 271,596 | 7,409 / 7,331 |

The E2B 256件 row is a clear counterexample to assuming F16 cache must lower observed peak RSS;
single-run process residency and system swap conditions varied. Prefill was faster in the two
256件 F16 runs, but one run per condition does not establish a stable speedup. F16 K/V needs
more quality/numerical study before adoption. All option scores are in
[F16 K/V comparison JSONL](phase-4-plus-option-scale-f16-kv-cpu-2026-09-29.jsonl).

## Flash Attention CPU probe

- 2026-09-29: pinned ggmlの`ggml_flash_attn_ext`を使い、CPU・text・F32 Q/K/V・F16 maskで
  E2B/E4Bの16/64/128件を実行した。maskはFlash APIの要求に合わせてF16にした。
  128件は同じF16 mask baselineと比べて一回のPrefillが短縮したが、raw scoreは一致せず、
  E4Bでwinnerが変わった。数値差の理由はFlash kernelの演算順/精度差を含むため、このprobeだけでは
  同じscoring contractとして扱えない。Flashは採用していない。

| Model | Options | Baseline / Flash Prefill ms | Speed change | Max / mean absolute raw-score diff | Winner baseline / Flash | Flash peak RSS MiB |
|---|---:|---:|---:|---:|---|---:|
| E2B | 128 | 67,261 / 52,221 | -22.4% | 1.713 / 0.705 | option_0014 / option_0014 | 4,065 |
| E4B | 128 | 119,053 / 99,678 | -16.3% | 0.879 / 0.254 | option_0013 / option_0012 | 6,013 |

E4Bのbaseline top-two差は0.0255、Flash側は0.259。FlashはE2Bでwinnerを保ったがraw scoreを
変え、E4Bではwinnerも変えた。16/64件も実行できたが、数値差は128件の同一条件を基準に記録した。
各条件一回でwarmupなし。raw scoresは[Flash CPU JSONL](phase-4-plus-option-scale-flash-cpu-2026-09-29.jsonl)。

## F32 query-chunk attention CPU probe

- 2026-09-29: baselineのF32 `mul_mat` → `soft_max_ext` → `mul_mat`を維持し、query軸だけを
  128 token単位のviewに分けて出力をconcatする一時probeを作った。全attention scoreを一度に作る経路と
  同じ候補順・固定短文を使用。E2B 16件でraw scoresがbit-identical、E2B/E4B 128件でも全128 logitsが
  bit-identicalだった。512件ではbaselineのraw-score vectorがresource probeにないためwinnerのみ照合した。
- 現行PrefillEngineのgraph/context容量8,192ではE2B 256件のgraph組立て中にcontext pool不足でassertした
  （3,345,216 bytes必要、3,344,848 bytes利用可能）。256/512件測定は一時probeの容量を65,536にして成功した。
  したがってchunk案を実装するならgraph容量の扱いも変更対象になる。tracked production sourceは変更していない。

| Model | Options | Baseline / chunk Prefill ms | Change | Baseline / chunk peak RSS MiB | Winner baseline / chunk | Score comparison |
|---|---:|---:|---:|---:|---|---|
| E2B | 16 | 7,898 / 7,884 | -0.2% | 3,530 / 3,527 | option_0014 / option_0014 | 16 logits exact |
| E2B | 128 | 66,268 / 67,770 | +2.3% | 4,150 / 4,175 | option_0014 / option_0014 | 128 logits exact |
| E2B | 256 | 181,555 / 182,737 | +0.7% | 5,136 / 5,063 | option_0014 / option_0014 | winner only |
| E2B | 512 | 567,964 / 587,554 | +3.4% | 8,754 / 6,049 | option_0014 / option_0014 | winner only |
| E4B | 128 | 118,331 / 120,141 | +1.5% | 6,171 / 6,176 | option_0013 / option_0013 | 128 logits exact |
| E4B | 512 | 843,069 / 932,396 | +10.6% | 10,898 / 7,411 | option_0013 / option_0013 | winner only |

`peak RSS`は`getrusage`のprocess high-water値。512件のchunk中`/proc`でE2B process swapを約1.6–1.7 GiB、
E4Bを約2.2–3.0 GiB観測した（各一回の断続sample）。baseline側はE2B 0、E4B text約0.68 GiBの観測だった。
したがってchunkのRSS差はresident pagesの低下を示すが、同量が丸ごと解放された意味ではない。
測定回数が少なくCPU速度は両モデルで遅く、swap量も増える。memory/latency budgetの採用値やproduction
変更を決める結果ではない。F32 raw-score rows、条件、graph capacityは
[query-chunk CPU JSONL](phase-4-plus-option-scale-attention-chunk-cpu-2026-09-29.jsonl)。

## llama.cpp Gemma 4 text attention reference

- 2026-09-29: 読み取り専用でローカル llama.cpp checkout
  `aa39d7a3e145a88202793a89462d65e94a5fc25f` の
  `src/models/gemma4.cpp`、`src/llama-graph.cpp`、`src/llama-context.cpp`、
  `common/common.h` を確認した。Gemma 4 text graph は共通の `build_attn` を使い、
  Gemma 4 専用に Flash Attention を禁止していない。
- llama.cpp の context と common CLI の既定 `flash_attn_type` は `AUTO`。
  対応 device では `ggml_flash_attn_ext` を使用し、auto probe で fused op が
  対象 device に配置できない場合は通常の `K×Q → softmax → V×score` 経路へ戻る。
  実行環境でどちらが選ばれたかは runtime log で確認が必要。
- Flash text graph では Q をそのまま渡し、F32 の K/V は F16 に cast してから
  `ggml_flash_attn_ext` を呼び、F32 accumulation を要求する。context の既定
  K/V cache type は双方 F16。prompt 処理の既定 physical microbatch は512 token。
  したがって今回の branchscore の CPU Flash probe（F32 Q/K/V、prompt 全体を
  一つの Prefill graph で処理）は llama.cpp の既定構成との同条件比較ではない。
  これは実装方式の確認であり、llama.cpp の Gemma 4 における実測精度差や
  winner 反転の有無を示さない。
