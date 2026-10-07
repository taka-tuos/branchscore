# Phase 4+ option-scale: CPU資源・F16/Flash/query-chunk probe

[記録索引](../phase-4-plus-option-scale.md) · [phase計画](../../phases/phase-4-plus-option-scale.md)

2026-09-29のCPU資源sweep、F16 mask/KV、Flash、query-only chunkとupstream参照。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

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
[CPU resource JSONL](../phase-4-plus-option-scale-resource-cpu-2026-09-29.jsonl).

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
[F16 mask comparison JSONL](../phase-4-plus-option-scale-f16-mask-cpu-2026-09-29.jsonl).

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
[F16 K/V comparison JSONL](../phase-4-plus-option-scale-f16-kv-cpu-2026-09-29.jsonl).

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
各条件一回でwarmupなし。raw scoresは[Flash CPU JSONL](../phase-4-plus-option-scale-flash-cpu-2026-09-29.jsonl)。

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
[query-chunk CPU JSONL](../phase-4-plus-option-scale-attention-chunk-cpu-2026-09-29.jsonl)。

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
