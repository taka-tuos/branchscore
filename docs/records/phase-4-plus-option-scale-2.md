# Phase 4+ option-scale-2: source investigation and memory estimates

2026-09-30時点のllama.cpp/ggml source調査、GGUF容量集計、保存済みlogitsの再解析。
現在の実装順・採用条件は[option-scale-2計画](../phases/phase-4-plus-option-scale-2.md)を参照。
以下は旧option-scale記録から移管した調査結果で、GPU実測や実装完了を示さない。
基礎となるtokenizer照合とCPU probeは[旧調査記録](phase-4-plus-option-scale.md)に保持する。

## 2026-09-30: GPU 全載せを前提としたメモリ削減案

### 評価条件と今回の根拠

主眼はメモリ削減。本番は E4B を8 GiB以上のVRAMを持つ単一 NVIDIA GPUへ全載せする。
512件を目標とし、384件も比較点にする。256件では用途上の余裕が足りない懸念がある。
検証機のCPU時間・RSS・swapから本番CUDAの採用判断をしない。

今回は既存JSONLの再集計、GGUF headerの読み取り、source/Web調査のみを行った。
本番相当GPUのpeak VRAM、CUDA logits、384件のmodel-backed実行は未測定。
local llama.cpp参照revisionは`aa39d7a3e145a88202793a89462d65e94a5fc25f`、
branchscore pinned ggmlは`456172ec733a135778adcd32d00e576a58232e45`。
確認した追加sourceはllama.cppの`src/llama-kv-cache-iswa.cpp`、
`src/llama-model.cpp`、`src/llama-context.cpp`のmicrobatch loopと、
pinned ggmlの`src/ggml-cuda/fattn.cu`、`fattn-common.cuh`、
`src/ggml-cpu/ggml-cpu.c`、`ops.cpp`。

Webで確認した一次資料:

- [llama.cpp CLIの既定設定](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md):
  Flashは`auto`、physical microbatchは512 token、full-size SWA cacheは既定off。
- [llama.cpp SWA cache](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache-iswa.cpp):
  full/sliding cacheを分け、単一sequenceのsliding容量は概ね`window + microbatch`を256へ丸める。
- [ggml CUDA Flash](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu):
  kernel選択は型・head幅・GQA・stride/paddingに依存する。容量だけで実行可否を判断しない。
- [FlashAttention論文](https://arxiv.org/abs/2205.14135)と
  [著者の実装・検証方針](https://github.com/Dao-AILab/flash-attention#tests):
  Flashはdense attentionと同じ数式をtileで計算する方式で、bit一致は保証しない。
  著者実装の検証も数値許容差を使う。著者のCUDA実装とggml kernelは別実装であり、
  その検証結果をbranchscoreへ転用しない。

### 実際に載せる重み容量の訂正

前の会話で使った約5.56 GiBはtext GGUFとmmproj GGUFの**ファイル全体**の合計だった。
`src/model_loader.cpp`はtextの全tensorと、mmprojの`v.*`および
`mm.input_projection.weight`だけを載せる。ローカルファイルのGGUF headerから
各tensorのshape/typeに応じた値のbytesを集計すると以下になる。

| Selected weights | Bytes | GiB |
|---|---:|---:|
| E4B Q4_K_M text | 4,961,343,656 | 4.621 |
| E4B F16 vision/projector | 370,222,848 | 0.345 |
| 合計 | 5,331,566,504 | 4.965 |

これは実測VRAMではなくtensor値の容量。backend alignment、CUDA context、graph、
KV、workspace/pool等は別。8 GiBなら重み値を差し引いた約3.03 GiBがそれらの候補枠となる。
mmprojの音声tensorを除外している既存実装を計算にも反映する必要がある。

### 第一候補: Prefill全体をtoken microbatchで逐次実行する

既存query-chunk probeは一つのlayerのattention queryだけを分け、concatする一つのgraphだった。
llama.cppの`llama_context::decode`は`process_ubatch`を順に呼び、cacheを引き継ぐ。
この違いが大きい。branchscoreでも例えば256/512 tokenずつ全layerを通すなら、
FFNやGemma 4のper-layer inputもmicrobatch長だけの一時tensorになる。
既存`build_layer`には`query_count`、`cache_start`、`cache_count`があり、
`causal_mask`もabsolute query offsetを受け取るため、狭い実装変更の足場がある。

全候補を含む一つのpromptを、一回の論理的なPrefill内で最後まで読む案。
absolute positionsと全attentionの過去KVを引き継ぎ、最後の位置だけをreadoutする。
optionごとに独立promptを作る案ではない。数学上のcausal attention範囲を保てるが、
matrix shape/kernelの違いによりbit一致するとは限らない。

### Flashとcacheの型・形状

llama.cpp相当の候補は**QはF32、K/V cacheはF16、Flash accumulation/outputはF32**。
pinned CUDA Flashの共通launchコードはQのF32をassertするので、QまでF16へcastする
変更はそのまま接続できない。F32 K/VをFlashへ渡す経路はkernel次第で追加F16領域を
確保する。F16 cacheならその変換領域も避けられる。

Gemma 4 full attentionの512次元headでは、pinned CUDA kernelが`gqa_opt_applies`を要求する。
E4BのQ/KV head比8/2は条件に合うが、key数の256-token padding、各tensor strideの
16-byte alignment、F16 mask等も必要。logical token数とphysical cache容量を分け、
padding位置をmaskで遮断した実shapeで`ggml_backend_supports_op`を確認する必要がある。
CPU probeが任意長で動いたことだけではCUDA対応を証明できない。

F16 K/V試験の差はcache値の丸めだけで説明しない。pinned CPU通常`mul_mat`の
fallbackは左入力の型に応じて右入力をvec-dot型へ変換するため、KがF16ならQ、
VがF16ならsoftmax値もF16へ変換され得る。CPU Flashのdefault/F32 precision指定は
同じF32 accumulator経路へ入る。既存CPU結果からCUDAの誤差幅を推定しない。

### 次の削減: sliding cacheを短く保つ

E4BのGGUF metadataは42 layer、18 shared KV layer。独立cacheは24 layerで、
そのうち20がsliding、4がfull。後半shared layerは既存と同じsource layerを使う。
現在はsliding cacheにもprompt全長を確保している。

microbatch長B=512、sliding window=512なら、llama.cpp参照ではsliding容量は1024 token。
これは学習済みsliding windowを縮める変更ではない。古いsliding KVを破棄する時は、
現在microbatchの全queryと後半shared layerが読み終わるまで必要な値を保持し、
absolute positionと物理slotを対応させる。cacheのimmutable-prefix契約にも変更が必要なため、
まず全長cacheでmicrobatchを成立させ、その後に別sliceで実装する候補。

### 固定短文workloadでの概算

T=prompt positions、K=ceil(T/256)×256、B=512とする。384件のT=6,991は、
既測定workloadの1候補18 tokenからの補間で、tokenizer/modelによる新規実測ではない。
512件のT=9,295は既存CPU測定値。画像・長いstate/説明は含まない。
各行は一つの容量項目であり、総peak VRAMを足し上げた表ではない。

| 容量項目 | 384件 | 512件 |
|---|---:|---:|
| 現行full-prompt F32 attention score 1個 | 1,491.5 MiB | 2,636.6 MiB |
| B=512通常attention score 1個（K padding込みの上限） | 112 MiB | 148 MiB |
| 現行二つのF32 mask、backend側だけ | 372.9 MiB | 659.2 MiB |
| 二つの全長key F16 microbatch mask、backend側だけ | 14 MiB | 18.5 MiB |
| full/sliding別key容量のF16 microbatch mask | 8 MiB | 10.25 MiB |
| 現行F32、24 layer全長KV | 764.6 MiB | 1,016.6 MiB |
| F16、24 layer全長KV（padding除外） | 382.3 MiB | 508.3 MiB |
| F16、fullはK・slidingは1024 tokenのKV | 152 MiB | 188 MiB |
| per-layer input F32値1個、prompt全長 | 286.7 MiB | 381.2 MiB |
| per-layer input F32値1個、B=512 | 21 MiB | 21 MiB |
| FFN width10240のF32値1個、prompt全長 | 273.1 MiB | 363.1 MiB |
| FFN width10240のF32値1個、B=512 | 20 MiB | 20 MiB |

式はscore=`8×queries×keys×4`、mask pair=`2×queries×keys×element_bytes`。
KV全長F32は`T×4×(20×(512+512)+4×(1024+1024))`、
F16 short-SWAは`2×(20×(512+512)×1024+4×(1024+1024)×K)`。
FAは上表の全attention scoreをmaterializeしないが、outputやkernel workspaceは必要。
既存ggml allocatorはsoftmaxのin-place再利用にも対応しているため、scoreとsoftmax値の
二つが常に別々にpeakへ加算されるとは仮定しない。

### 数値差の読み直しと推奨する検証順

保存済みJSONLを再集計すると、E2B F16 KV 256件の最大raw差4.965には平均shift
+3.021が含まれる。shiftを引いた最大差は2.023、候補softmaxの最大絶対差は0.174。
E4B CPU Flash 128件は平均shift−0.022、最大raw差0.879、確率の最大絶対差0.062。
共通shiftだけならsoftmax/argmaxは変わらないが、残る相対差もある。
これらは同じ固定fixtureの再解析であり、意味判断の正解率低下の測定ではない。

ユーザー方針に従い、llama.cppと同程度の実装・数値挙動を採用基準にする候補とする。
従来F32全長graphへのbit一致をFA/F16採用の必須条件にせず、同じGGUF・prompt tokens・
cache型・backend・FA設定でreferenceと照合する。backend/precisionによる正常な差と
mask・position・shared-KV等の実装差を区別する。

1. 全長F32 KVのまま、Prefill全体をtoken microbatchで逐次実行する最小probe。
   中間chunkはvocabulary出力を省き、最後のchunkの最終位置だけreadoutする。
   短い既存fixtureと、chunk/window境界をまたぐtext/image入力を照合する。
2. F16 K/V + F16 mask + text FA + CUDA paddingを追加し、llama.cpp相当条件で照合する。
   CUDA FAが非対応でもmicrobatchの通常attention経路を比較候補にできる。
3. 必要ならSWA cache短縮を別sliceで試す。まず全長cacheから短縮cacheへの差を確認する。
4. 本番相当GPUで384/512件の重み・cache・graph buffer容量、model load後/各stage後の
   空きVRAMと実行中peak、selected ID・候補logits/相対差を記録する。
   Vision graphはencode内で解放されるがCUDA workspace/poolの保有分も含めて測る。
   その後に説明長/画像を含めたrequest token budgetを決める。

QのF16化、KVの4/8-bit化、重みの追加量子化、候補説明の短縮は、上記で足りない場合の
追加候補にする。prompt書式の簡略化もtoken数を減らせる可能性があるが、renderer/判断品質の
再検証が必要。候補を小グループへ分けてwinnerを集める案は別の採点契約になる。

この構造なら512件を8 GiBで検証する根拠はある。384件だけに下げることを先に決める
段階ではない。8 GiBでの成立はGGUF・実際の説明長・画像・GPU/kernel・workspace次第で、
上表はそれを保証しない。上記のsource調査を記録した時点の変更は文書のみ。

## 2026-09-30: Step 1 Prefill microbatch implementation and CPU/reference checks

`PrefillEngine`は最大512 tokenごとに全layerを実行し、request内の全長F32 KVへ追記する。
absolute positionとchunk終端までのcausal maskを渡し、visual spanはbackend tensor viewで
各chunkへ切り出す。中間chunkはvocabulary projectionを省略し、graph contextとallocatorは
同期後にchunk単位で解放する。`PrefillState`からchunk数、上限、cache buffer bytes、
peak temporary graph bytesを観測できる。HTTP/runtime契約には公開していない。

513-token text fixtureと631-position image fixture（visual span `[470,551)`、512境界を横断）を
E2B/E4B CPUで実行。どちらもtext 513 tokenは2 graph、image fixtureは2 graphで完了した。
position 511/512のlayer-0 K値が異なり、A/B readoutがfinal-position logitsと一致した。
513-token textのcache/peak graph bufferはE2B 18,911,232/120,068,096 bytes、
E4B 58,834,944/126,883,840 bytes。これらはCPU buffer値でありVRAM推定に使わない。

長文16候補の画像fixtureは1x1白画像、prompt placeholder込み1,345 token、展開後1,425位置
(text前37、visual 81、text後1,307)。Branchscore CPUの3 graphで処理した。llama.cpp revision
`19e28a27702117d8f2eb16b825b9a308111f67d9`のtext modelとmtmd image encoderをCPUへ固定し、
F32 K/V、Flash Attention disabled、full-size SWA cache、最大physical batch 512で比較した。
同一のBranchscore-rendered token IDsからimage placeholderを除き、その位置へmtmdで作った
image embeddingsを入力した。mtmdが通常tokenizeで挿入する`<|image>`/`<image|>` wrapperは
比較streamへ加えず、Branchscoreと同じtext-token/visual-embedding streamに揃えた。

| Model | Max abs raw diff | Mean common shift | Max centered abs diff | Max relative-probability diff | Top-1 |
|---|---:|---:|---:|---:|---|
| E2B | 4.060 | -3.064 | 2.601 | 0.2426 | `option_00`一致 |
| E4B | 0.690 | +0.112 | 0.578 | 0.0793 | `option_00`一致 |

差はbranchscore minus llama.cpp。E2B vision outputsは両側81×1,536、max abs diff 0.00267、
mean abs diff 0.000248、cosine similarity 0.99999992。E2Bだけ、Branchscore側vision outputを
そのままllama.cpp text stackに入力したcontrolも実施し、max raw diff 2.368、mean shift -1.745、
centered max 0.623、relative-probability max差0.0455、top-1一致だった。encoder出力の差と
Prefill graph batch shape/kernel差は異なる要因として保持する。これらは各1画像/各1長文promptで、
相対差許容値、拡張候補数、意味判断品質を決めない。

Recorded GPU-reference trials used explicit CPU devices. One earlier auto-device trial selected Vulkan
on the 2 GiB Quadro P620; that trial is excluded from all recorded numeric comparisons and the text
comparison was rerun CPU-only. Current `branchscore --list-backends` reports CPU only, so Step 2's
CUDA attention/model check was not started. The phase's Step 4 explicitly requires an E4B all-loaded
single NVIDIA GPU with at least 8 GiB VRAM; it was not attempted.
