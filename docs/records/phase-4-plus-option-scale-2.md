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

2026-09-30: Step 1の実装では最大512 tokenごとに全layerを逐次実行し、全長F32 KVへ
追記する。absolute position、query chunk幅×chunk終端までのkey幅を持つcausal mask、
image spanのbackend viewを使い、中間chunkのvocabulary projectionを省略した。
chunkごとにgraph allocatorとcontextを解放する。PrefillStateからchunk上限、graph数、
cache buffer bytes、最大一時graph buffer bytesを参照できる。`cmake --build build -j2` は成功。

同日、E2B/E4BのCPU Prefill testを実行し、両方成功。513-token text caseは2 graph、image span
`[470,551)`を512境界で分割する631-token入力も2 graphで完了し、logitsはfinite。
513-token text caseのE2B KV bufferは18,911,232 bytes、最大一時graph bufferは120,068,096 bytes。
E4Bはそれぞれ58,834,944 bytes、126,883,840 bytes。key layer 0のposition 511/512は異なる値となり、
513-token PrefillでA/B categorical gatherとfinal logitsの選択IDが一致した。image境界入力でも
categorical gatherのA/B raw logitsはfinal-position logitsと一致した。
これはCPU bufferの計測値であり、VRAM見積りには使わない。

16-labelのtext promptを同一のrendered token IDsでllama.cpp (`aa39d7a3e145a88202793a89462d65e94a5fc25f`)
と比較。双方CPUの既定thread設定、F32 K/V、Flash Attention disabled、referenceの最大physical microbatch 512。
全4条件でtop-1 (`option_00`)は一致。raw差はbranchscore minus reference。

| Model | Prompt tokens | Max abs raw diff | Mean common shift | Max centered abs diff | Max relative-probability diff |
|---|---:|---:|---:|---:|---:|
| E2B | 328 | 2.049 | -1.069 | 1.230 | 0.0551 |
| E2B | 737 | 0.781 | -0.233 | 1.014 | 0.00095 |
| E4B | 328 | 0.434 | +0.028 | 0.406 | 0.0374 |
| E4B | 737 | 0.970 | +0.136 | 0.834 | 0.00890 |

比較した品質例は各1件で、差の許容値や拡張件数の意味判断品質を確定しない。

同日、1x1白画像と同じ16候補の長文promptをllama.cppのmtmd image encoderでも照合した。
branchscoreのrenderer token列はplaceholder込み1,345 tokenで、placeholderを81 visual
embeddingへ置き換えたPrefill位置数は1,425（画像前37、後1,307）。最大512位置の3 graphで処理し、
graph node数はE2B 5,199、E4B 6,480。両方のtop-1は`option_00`（A）で一致した。

llama.cpp revision `19e28a27702117d8f2eb16b825b9a308111f67d9`のmodelとmtmdをCPUへ固定し、
F32 K/V、Flash Attention disabled、full-size SWA cache、最大physical batch 512で照合した。
同じprompt token IDsから画像placeholderだけを除き、mtmdで作ったimage chunkのembeddingを
その位置に入力した。標準mtmd tokenizerが加える`<|image>`/`<image|>`境界tokenは含めていない。
差はbranchscore minus llama.cpp。

| Model | Prompt positions | Max abs raw diff | Mean common shift | Max centered abs diff | Max relative-probability diff | Top-1 |
|---|---:|---:|---:|---:|---:|---|
| E2B | 1,425 | 4.060 | -3.064 | 2.601 | 0.2426 | 一致 |
| E4B | 1,425 | 0.690 | +0.112 | 0.578 | 0.0793 | 一致 |

E2B vision encoderの出力自体は双方81×1,536、max absolute diff 0.00267、mean absolute
diff 0.000248、cosine similarity 0.99999992だった。E2Bのvision embeddingだけをbranchscoreと
同一にしてllama.cpp text stackへ入力した場合はtop-1一致、max raw diff 2.368、共通shift -1.745、
centered max diff 0.623、相対確率max差0.0455となった。通常encoder同士の差がlogitに現れるため、
これは単一画像fixtureの比較であり、許容誤差や意味判断品質の確定には使わない。

Step 1の位置・chunk・KV・最終位置readoutをCPUで確認し、差分を記録したためStep 1を完了とする。
これは拡張候補数の品質承認ではない。`branchscore --list-backends`は現buildでCPUのみを表示し、
llama.cpp比較もCPU固定で実施した。CUDA attention pathの実行確認を含むStep 2と、
8 GiB以上GPUでのStep 4（現行Step 5）には未着手。

Recorded GPU-reference trials used explicit CPU devices. One earlier auto-device trial selected Vulkan
on the 2 GiB Quadro P620; that trial is excluded from all recorded numeric comparisons and the text
comparison was rerun CPU-only. Current `branchscore --list-backends` reports CPU only, so Step 2's
CUDA attention/model check was not started. The production GPU measurement step (now Step 5) explicitly requires an E4B all-loaded
single NVIDIA GPU with at least 8 GiB VRAM; it was not attempted.

<a id="precision-baseline-review"></a>

## 2026-09-30: 量子化差を基準にする採用方針の検討

Step 1のCPU/reference結果をレビューした後、ユーザーは、既にQ4_K_Mを使う前提なら、
高精度重みからQ4_K_Mへの回答差よりruntime最適化の追加差が十分小さいことを
実用上の採用基準にできるのではないか、と提案した。この方向で測定計画を追加することに合意。
現在の契約・測定手順は[phaseのStep 4](../phases/phase-4-plus-option-scale-2.md#step-4-precision-baseline)に定める。

重み量子化で既に受け入れている差を物差しにするのは合理的。ただし、差は候補ごとに異なり、
僅差のtop-1は小さな追加差でも反転し得る。共通logit offsetはsoftmax/argmaxを変えないため、
最大raw差だけで精度低下を判断しない。選択変更と正解→誤答/誤答→正解を分けて見る。
量子化差が大きいことをmask・position・shared KV等の実装不具合の許容理由にしない。

比較は、高精度llama.cpp基準(A)、Q4_K_Mの同じ基準(B)、Q4_K_Mのllama.cpp最適化(C)、
同じQ4_K_M/最適化条件のbranchscore(D)へ分ける。A→Bが量子化、B→Cが最適化、
C→Dが実装、B→Dが最終的な合成差。画像では同じvisual embeddingを使うcontrolで
vision差を分離する。モデル規模/bit数や他モデルのPPLから、Gemma 4の候補logits差の
数値を直接推定できる根拠は得ていない。普遍的な許容比率もこのレビューでは決めていない。

既存Step 1比較はすべてQ4_K_M/CPUで、同一モデルの高精度→Q4_K_M比較ではない。
E4B textの相対確率max差0.0374/0.00890、画像E4B 0.0793、画像E2B 0.2426、
同一vision embeddingのE2B control 0.0455は、量子化差より小さいとまだ判断できない。
拡張ラベルの形式照合と旧16/32/64件の2例のscreeningも、384/512件の品質を承認しない。

計画追加時点でE2B/E4BのBF16 GGUFとQ4_K_M、対応F16 mmprojがlocalにあることを確認した。
後続のCPU A/BでGGUF metadata、tokenizer一致、file hashを確認し、A→Bの初期比較を実施した。
ただし元checkpoint revisionとimatrix実体・再現設定は未確認のため、同一checkpoint由来の確定と
量子化だけの因果分離は未完了。詳細は[Step 4 CPU A/B記録](#2026-09-30-step-4-cpu-precision-baseline)を参照。

検討時に確認した一次資料（2026-09-30）:

- [llama.cpp quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md):
  高精度GGUFからの量子化、imatrix/個別tensor設定、PPL/KLによる差の評価。
- [llama.cpp perplexity](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md):
  FP16 logitsとのKL、確率差の分布、top-1一致率を測る。今回の候補限定readoutへは
  同じ指標の考え方を用い、全語彙/PPLの数値をそのまま許容差へ転用しない。
- [Lee et al., 2024](https://arxiv.org/abs/2409.11055):
  量子化方式・モデル規模・bit幅・taskごとに性能が異なることを評価した研究。
  Gemma 4/Q4_K_Mの今回のfixtureを直接評価した資料ではない。

同日、既存変更のコミット時に`cmake --build build -j2`と`git diff --check`を実行し成功。
`ctest --test-dir build -R '^prefill$' --output-on-failure`もE4B Q4_K_M/対応F16 mmproj、
CPU backendの設定で成功（1/1、58.66秒）。これは既存Prefill境界testの再確認であり、
高精度比較やCUDA検証の実施を意味しない。

<a id="2026-09-30-step-4-cpu-precision-baseline"></a>

## 2026-09-30: Step 4 CPU A/B baseline (partial)

CPUで実行できるA/B（高精度重み vs Q4_K_M、どちらも同じllama.cpp基準経路）を測定した。GPU/CUDA経路は使用していない。以下のA→B差はlocalファイル間の観測差であり、元checkpoint revisionの一致を証明できていないため、量子化だけの因果差と確定しない。

### GGUF inventory, provenance, tokenizer

| Model | Variant | File bytes | GGUF `general.file_type` | SHA-256 |
|---|---|---:|---:|---|
| E2B | BF16 | 9,311,305,568 | 32 | `1eafd61d010ce8ca09db38f370aadd64c6d792db269c365ad0d9ea2709701890` |
| E2B | Q4_K_M | 3,106,738,272 | 15 | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` |
| E4B | BF16 | 15,053,097,856 | 32 | `38e0dba6818d18e2b41062c7b3b9083dcfddc44767078c415ff7e13661911ba5` |
| E4B | Q4_K_M | 4,977,169,568 | 15 | `519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d` |

4ファイルはすべてGGUF v3。各サイズ内でtensor数はE2B=601、E4B=720。各ペアの`general.name`/`general.base_model.0.name`と`general.base_model.0.repo_url`は一致し、公式base URLはそれぞれ`google/gemma-4-E2B-it`、`google/gemma-4-E4B-it`。4ファイルとも`general.repo_url=https://huggingface.co/unsloth`、`general.quantized_by=Unsloth`、`general.quantization_version=2`を持つ。BF16はfile_type 32、Q4_K_Mは15。

GGUF内に元checkpointのcommit/revisionはなく、同じ公式repo URLだけではpairの重み由来一致を立証できない。2026-09-30の再確認で、Q4_K_MのGGUF本体と保存済みloader logの双方に以下のimatrix metadataを確認した。初期記録の「GGUF metadataに記録されていない」を訂正する。

| Model | `quantize.imatrix.file` | `quantize.imatrix.dataset` | `entries_count` | `chunks_count` |
|---|---|---|---:|---:|
| E2B | `gemma-4-E2B-it-GGUF/imatrix_unsloth.gguf` | `unsloth_calibration_gemma-4-E2B-it.txt` | 275 | 141 |
| E4B | `gemma-4-E4B-it-GGUF/imatrix_unsloth.gguf` | `unsloth_calibration_gemma-4-E4B-it.txt` | 342 | 141 |

これはimatrixを用いた量子化のメタデータ上の申告を確認したもの。imatrix実体のhash、calibrationデータの内容、量子化commandとtensor別設定は未確認であり、再現性やBF16との同一checkpoint由来を立証しない。Step 4.1は引き続き未完了。

tokenizer metadataはE2B/E4BおよびBF16/Q4_K_M全4ファイルでbyte内容一致。canonical JSON内容のSHA-256は`tokenizer.ggml.tokens`（262,144件）=`0c42e767893bb4c7361a016da8c9b89e52ebc86b58b366dfef5d75a0b7bdfd02`、`token_type`（262,144件）=`987bc200faf7bd20738013daab9cda6a005b2a2842d4f8be9449458663e0bdf9`、`merges`（514,906件）=`b2feb0918e7eca60e2fbb5701149d00735e653f6d69a78d72b5b9e33c5983a12`。BOS=2、EOS=106、`add_bos_token=true`、`add_space_prefix=false`。固定105-token ambiguity promptはbranchscore/llama.cpp tokenizer双方で全4 GGUFのID列一致を確認した。

### Reference conditions and fixed inputs

Referenceはlocal llama.cpp `19e28a27702117d8f2eb16b825b9a308111f67d9`（build 11276）。一時C API runnerはmodel/deviceをCPUに固定し、`n_ctx=token_count+1`、logical batch=prompt長、physical `n_ubatch≤512`、F32 K/V、Flash Attention disabled、`swa_full=true`でpromptを一度prefillし、最後の位置のanswer-slot logitsだけを取得した。高精度版/Q4版は一度に一つずつロード。生成は行わない。

- `short_16`/`long_16`: 固定入力は[`fixtures/phase4-step4-cpu.jsonl`](../../fixtures/phase4-step4-cpu.jsonl)。16 optionすべての説明が`Candidate NN is a possible response.`というsynthetic control。`short_16`は328 token、`long_16`は同じ健康文を反復した737 tokenで512-token microbatch境界を越える。どちらも正解optionは定義していない。
- `text-ambiguous`: 既存[`fixtures/phase3-text.jsonl`](../../fixtures/phase3-text.jsonl)の3-option caseをproduction rendererで描画。105 token。A=`publish`、B=`wait`、C=`unknown`。fixtureに`expected_selected_id`はなく、結果を誤答と分類しない。
- `e2b_image_white_1x1`: Step 1の`image-long-16`入力を再利用し、1×1白画像[`fixtures/phase4-step4-white.ppm`](../../fixtures/phase4-step4-white.ppm)を使う。stateは`Evidence item NNN confirms an active change in category alpha.`を000–071の72文、空白区切りで並べた4,535 byteの文。questionは`Which option best matches the evidence?`、16候補は`Choose category NN for the evidence case.`（00–15）。初期記録の「short_16と同じtext/options」を訂正する。同じE2B mmproj F16（SHA-256 `140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa`）で得た81×1,536のBranchscore visual embeddingsをBF16/Q4の両方へ同一に入力した。37 text + 81 visual + 1,307 text = 1,425 Prefill positions。これはencoderを分離したcontrolで、end-to-end結果ではない。short_16との違いにはstate/question/候補説明も含まれ、両者の差を画像追加だけの効果とは扱わない。

固定text input JSONLのSHA-256は`76b81c681df8118bb7c4c58acbbba2d219882d779a9abb101bb2ffaba3867a48`。short/long rendered ID列のSHA-256はそれぞれ`74613d145086d02d32a8941a6ae86e75e8a6c33e11f897c22c7b2ecf727f064b` / `f8e07a42d719c980779751b0c7075b5a3bae19faf7eb36c0c703dfa6a41da4cd`。imageのbefore 37 IDs / after 1,307 IDs / answer IDs / shared embeddingのSHA-256は`f286a8a6501ea057f309c3a571d2745d0ce55caaf1e5d621827b36b23525280d` / `c55ee33684fd8ed5bc4532ffffe2531ca054282cc5c57917b394b57379c94065` / `7999ff192aff01d1e1ac3cd155094abb1750d7e0fde5107e68b13e779cc1a575` / `9211262ac1609fce74d2a8b12c68310dfbfd4e0b23172614c372d326ad71042b`。ambiguity rendered prompt SHA-256=`96cb860eef16bc452359296bbd5709ce7dd36584917651c2411c4ef080f40877`、105-ID列SHA-256=`8b12bc56d649555955796835560270cbcd5e248fcfc7c6cb4572571256e6e52d`。

### A→B summary

`delta = Q4_K_M − BF16`。`mean shift`は候補raw差の共通平均、`centered max`はその平均を候補ごとに差し引いた最大絶対差。probabilityは同じ候補集合内の温度1 softmax。

| Model | Fixture | Tokens / positions | Top-1 BF16 → Q4 | Top-two margin BF16 / Q4 | Max abs Δraw | Mean shift | Max abs Δcentered | Max abs Δp |
|---|---|---:|---|---:|---:|---:|---:|---:|
| E2B | short_16 | 328 | option_00 → option_00 | 4.399330 / 2.951118 | 3.586758 | 2.001117 | 2.551005 | 0.133042 |
| E2B | long_16 | 737 | option_00 → option_00 | 10.734138 / 8.446993 | 6.003855 | 4.432599 | 3.506151 | 0.000444 |
| E4B | short_16 | 328 | option_00 → option_00 | 2.337423 / 2.366575 | 1.834423 | 0.163774 | 1.670649 | 0.069161 |
| E4B | long_16 | 737 | option_00 → option_00 | 4.075382 / 3.689526 | 2.673573 | -0.476135 | 2.197438 | 0.014787 |
| E2B | e2b_image_white_1x1 | 1,425 | option_00 → option_00 | 5.105050 / 1.001248 | 4.137471 | -1.718526 | 4.863133 | 0.485147 |
| E2B | text-ambiguous | 105 | unknown → unknown | 15.253202 / 13.802769 | 1.183726 | 0.691512 | 0.971324 | 0.000000924 |
| E4B | text-ambiguous | 105 | unknown → unknown | 4.631159 / 2.657742 | 1.931570 | 0.759584 | 1.171986 | 0.056566 |

top-1変更は0/7 paired comparisons（E2B=4、E4B=3）。意図的に作った僅差fixtureはない。この小標本中の最小marginはimage E2B/Q4の1.001だが、synthetic option上の観測であり、意味的なnear-tie品質証明ではない。正解→誤答/誤答→正解率は、16-option synthetic inputsに正解labelがなく、ambiguous fixtureも期待IDを持たないため算出していない。少数のtop-1一致を一般的な品質保証にしない。

### All candidate logits and conditional softmax values

表中`centered Δ`=`(Q4−BF16)−mean shift`、`Δp`=`p(Q4)−p(BF16)`。logitは全候補のraw値。

#### short_16 — E2B

| Option ID | BF16 logit | Q4_K_M logit | Centered Δ | p(BF16) | p(Q4) | Δp |
|---|---:|---:|---:|---:|---:|---:|---:|
| option_00 | 10.3355217 | 9.78563404 | -2.55100507 | 0.97961493 | 0.846573372 | -0.133041558 |
| option_01 | 5.93619156 | 6.83451605 | -1.10279292 | 0.0120351246 | 0.0442598914 | 0.0322247668 |
| option_02 | 3.91034389 | 5.49185562 | -0.419605674 | 0.00158721634 | 0.0115584695 | 0.00997125312 |
| option_03 | 4.51607323 | 6.61364603 | 0.0964553931 | 0.0029087141 | 0.0354885058 | 0.0325797917 |
| option_04 | 4.3880043 | 6.8106904 | 0.42156869 | 0.00255906555 | 0.0432178341 | 0.0406587686 |
| option_05 | 1.31047451 | 3.42203236 | 0.110440431 | 0.000117903638 | 0.0014587723 | 0.00134086866 |
| option_06 | 1.48942709 | 2.922261 | -0.5682835 | 0.000141008509 | 0.000884992448 | 0.000743983939 |
| option_07 | 2.39648223 | 4.97284079 | 0.575241146 | 0.000349280518 | 0.00687852074 | 0.00652924022 |
| option_08 | 0.0337550528 | 0.0877203867 | -1.94715208 | 3.28893271e-05 | 5.19893297e-05 | 1.91000026e-05 |
| option_09 | -1.3347317 | 0.284872085 | -0.381513627 | 8.37006331e-06 | 6.33193011e-05 | 5.49492378e-05 |
| option_10 | -0.831087291 | 1.35512483 | 0.185094712 | 1.38502856e-05 | 0.000184646455 | 0.00017079617 |
| option_11 | 0.561732233 | 4.13672829 | 1.57387864 | 5.5763826e-05 | 0.0029810963 | 0.00292533248 |
| option_12 | -1.2369945 | 2.34976363 | 1.58564073 | 9.22944246e-06 | 0.000499237405 | 0.000490007963 |
| option_13 | 2.84561896 | 4.67208672 | -0.174649658 | 0.000547308216 | 0.0050918924 | 0.00454458418 |
| option_14 | -1.20085347 | 1.87120295 | 1.070939 | 9.569105e-06 | 0.000309364768 | 0.000299795663 |
| option_15 | -1.17938638 | 2.34747481 | 1.52574378 | 9.77674659e-06 | 0.000498096048 | 0.000488319301 |

#### short_16 — E4B

| Option ID | BF16 logit | Q4_K_M logit | Centered Δ | p(BF16) | p(Q4) | Δp |
|---|---:|---:|---:|---:|---:|---:|---:|
| option_00 | 28.8699112 | 28.5935135 | -0.440171838 | 0.825430796 | 0.756269363 | -0.0691614323 |
| option_01 | 26.5324879 | 25.9891224 | -0.707139611 | 0.0797169394 | 0.0559247208 | -0.0237922185 |
| option_02 | 24.4740639 | 24.1283741 | -0.509463906 | 0.0101762656 | 0.00869943625 | -0.00147682934 |
| option_03 | 22.664465 | 21.4432755 | -1.38496363 | 0.00166605604 | 0.000593427385 | -0.00107262866 |
| option_04 | 24.2453079 | 24.7854271 | 0.376345038 | 0.00809544984 | 0.0167820727 | 0.00868662291 |
| option_05 | 22.6076946 | 23.4097195 | 0.638250709 | 0.00157410815 | 0.00424017097 | 0.00266606282 |
| option_06 | 24.6756725 | 26.2269382 | 1.38749158 | 0.0124493295 | 0.0709391433 | 0.0584898138 |
| option_07 | 24.2898216 | 23.351429 | -1.10216677 | 0.00846394907 | 0.004000075 | -0.00446387407 |
| option_08 | 24.728857 | 24.3343945 | -0.558236718 | 0.0131293643 | 0.0106896778 | -0.00243968657 |
| option_09 | 24.2041416 | 23.6085281 | -0.759387612 | 0.00776895645 | 0.00517279035 | -0.0025961661 |
| option_10 | 23.6033039 | 24.121397 | 0.354318976 | 0.00426012346 | 0.00863895083 | 0.00437882736 |
| option_11 | 24.9603252 | 25.2077026 | 0.0836032629 | 0.0165488973 | 0.0255998569 | 0.00905095955 |
| option_12 | 24.0293522 | 24.6383724 | 0.4452461 | 0.00652307846 | 0.0144870703 | 0.00796399181 |
| option_13 | 22.016531 | 21.5411148 | -0.639190316 | 0.000871556333 | 0.000654423183 | -0.00021713315 |
| option_14 | 23.0841599 | 24.3927498 | 1.1448158 | 0.00253489973 | 0.0113320377 | 0.00879713799 |
| option_15 | 21.9185753 | 23.7529984 | 1.67064893 | 0.000790230598 | 0.00597678308 | 0.00518655248 |

#### long_16 — E2B

| Option ID | BF16 logit | Q4_K_M logit | Centered Δ | p(BF16) | p(Q4) | Δp |
|---|---:|---:|---:|---:|---:|---:|---:|
| option_00 | 5.41801691 | 6.34446478 | -3.50615114 | 0.999972516 | 0.999528948 | -0.000443567679 |
| option_01 | -5.31612158 | -2.10252786 | -1.21900529 | 2.17876781e-05 | 0.000214443598 | 0.00019265592 |
| option_02 | -7.95094824 | -3.32228637 | 0.196062863 | 1.5628607e-06 | 6.33255099e-05 | 6.17626492e-05 |
| option_03 | -7.68914318 | -2.96439815 | 0.292146027 | 2.03058308e-06 | 9.057484e-05 | 8.85442569e-05 |
| option_04 | -8.36877632 | -3.45684552 | 0.479331791 | 1.02910536e-06 | 5.53529023e-05 | 5.4323797e-05 |
| option_05 | -10.5742607 | -5.386374 | 0.755287707 | 1.13404461e-07 | 8.03816336e-06 | 7.9247589e-06 |
| option_06 | -12.2143879 | -7.85007477 | -0.0682858825 | 2.19954046e-08 | 6.84203321e-07 | 6.62207917e-07 |
| option_07 | -10.8642635 | -6.2690053 | 0.162659228 | 8.48561871e-08 | 3.32533133e-06 | 3.24047514e-06 |
| option_08 | -11.551671 | -7.57147264 | -0.452400625 | 4.26723177e-08 | 9.04025213e-07 | 8.61352895e-07 |
| option_09 | -13.0734615 | -9.17110729 | -0.530244768 | 9.31624741e-09 | 1.82586239e-07 | 1.73269991e-07 |
| option_10 | -13.3070297 | -8.34896088 | 0.52546984 | 7.37570667e-09 | 4.15452804e-07 | 4.08077097e-07 |
| option_11 | -12.7214756 | -6.71762085 | 1.57125574 | 1.32466652e-08 | 2.12326238e-06 | 2.11001572e-06 |
| option_12 | -13.2377453 | -8.23555279 | 0.569593489 | 7.90484732e-09 | 4.65344088e-07 | 4.57439241e-07 |
| option_13 | -8.67874527 | -4.06326628 | 0.182879984 | 7.54817631e-07 | 3.018389e-05 | 2.94290724e-05 |
| option_14 | -13.2590485 | -8.05537796 | 0.771071494 | 7.73823e-09 | 5.57215431e-07 | 5.49477201e-07 |
| option_15 | -12.9162359 | -8.21330738 | 0.270329535 | 1.0902428e-08 | 4.75811855e-07 | 4.64909427e-07 |

#### long_16 — E4B

| Option ID | BF16 logit | Q4_K_M logit | Centered Δ | p(BF16) | p(Q4) | Δp |
|---|---:|---:|---:|---:|---:|---:|---:|
| option_00 | 28.6527634 | 28.3986645 | 0.222035766 | 0.946366791 | 0.931580205 | -0.0147865858 |
| option_01 | 24.5773811 | 24.0261593 | -0.0750871897 | 0.0160747222 | 0.011756157 | -0.00431856524 |
| option_02 | 22.6664772 | 21.3919773 | -0.798365235 | 0.00237820038 | 0.000843829453 | -0.00153437093 |
| option_03 | 20.4141521 | 19.0409641 | -0.897053361 | 0.000250078353 | 8.0393813e-05 | -0.00016968454 |
| option_04 | 21.9915409 | 21.2717018 | -0.243704438 | 0.00121095314 | 0.000748203431 | -0.000462749705 |
| option_05 | 21.2833042 | 21.5011425 | 0.693972945 | 0.000596408807 | 0.000941162296 | 0.000344753489 |
| option_06 | 22.1848793 | 23.8206654 | 2.11192071 | 0.00146924132 | 0.00957239251 | 0.00810315119 |
| option_07 | 22.6569157 | 20.8952942 | -1.28548682 | 0.00235556949 | 0.00051350886 | -0.00184206063 |
| option_08 | 21.7107105 | 20.6968842 | -0.537691712 | 0.000914459024 | 0.00042109449 | -0.000493364535 |
| option_09 | 23.2723866 | 22.5609303 | -0.235321641 | 0.00435904679 | 0.00271596702 | -0.00164307977 |
| option_10 | 24.2573624 | 24.7091389 | 0.927911162 | 0.011672425 | 0.0232744615 | 0.0116020365 |
| option_11 | 24.1194363 | 24.3403664 | 0.697064757 | 0.0101685856 | 0.0160962154 | 0.0059276299 |
| option_12 | 22.3235416 | 21.7171726 | -0.130234361 | 0.00168777062 | 0.00116811395 | -0.00051965667 |
| option_13 | 20.7637291 | 18.0901566 | -2.19743788 | 0.000354727976 | 3.1066487e-05 | -0.000323661489 |
| option_14 | 19.6487198 | 19.9912338 | 0.818648696 | 0.00011631964 | 0.00020793113 | 9.16114898e-05 |
| option_15 | 18.0992012 | 18.5518951 | 0.928828597 | 2.47004963e-05 | 4.92972469e-05 | 2.45967506e-05 |

#### e2b_image_white_1x1 — E2B

| Option ID | BF16 logit | Q4_K_M logit | Centered Δ | p(BF16) | p(Q4) | Δp |
|---|---:|---:|---:|---:|---:|---:|---:|
| option_00 | 4.82079887 | 0.683327377 | -2.41894576 | 0.981196206 | 0.496049228 | -0.485146978 |
| option_01 | -0.424392998 | -0.561253548 | 1.58166518 | 0.00517366108 | 0.142892735 | 0.137719074 |
| option_02 | -1.11077881 | -1.57338846 | 1.25591608 | 0.00260438032 | 0.0519332549 | 0.0493288746 |
| option_03 | -0.284250945 | -0.887993753 | 1.11478292 | 0.00595197222 | 0.103064401 | 0.0971124284 |
| option_04 | -1.9019891 | -3.0120399 | 0.608474931 | 0.00118055477 | 0.0123210341 | 0.0111404794 |
| option_05 | -1.96428645 | -3.49104857 | 0.191763611 | 0.00110925334 | 0.00763161299 | 0.00652235965 |
| option_06 | -2.67565942 | -6.2376771 | -1.84349195 | 0.000544609724 | 0.00048952032 | -5.50894035e-05 |
| option_07 | -3.45213318 | -6.13123703 | -0.960578119 | 0.000250534259 | 0.000544498966 | 0.000293964707 |
| option_08 | -5.30783415 | -7.52357864 | -0.497218759 | 3.91693559e-05 | 0.000135304045 | 9.61346893e-05 |
| option_09 | -4.39202023 | -8.03255844 | -1.92201248 | 9.78767099e-05 | 8.1332414e-05 | -1.6544296e-05 |
| option_10 | -5.54538822 | -7.8870554 | -0.623141449 | 3.08871622e-05 | 9.40707984e-05 | 6.31836362e-05 |
| option_11 | -6.14891291 | -7.0606513 | 0.806787341 | 1.68915914e-05 | 0.000214959959 | 0.000198068368 |
| option_12 | -6.1485157 | -8.26318932 | -0.396147889 | 1.68983022e-05 | 6.45805804e-05 | 4.76822782e-05 |
| option_13 | -1.66669738 | -4.81017351 | -1.4249504 | 0.00149372977 | 0.00204045798 | 0.00054672821 |
| option_14 | -5.15951633 | -7.21407843 | -0.336036369 | 4.54318102e-05 | 0.000184384757 | 0.000138952946 |
| option_15 | -3.46252847 | -0.317921102 | 4.8631331 | 0.000247943373 | 0.182258625 | 0.182010681 |

#### text-ambiguous — E2B

| Semantic option | Slot | BF16 logit | Q4_K_M logit | Centered Δ | p(BF16) | p(Q4) | Δp |
|---|:---:|---:|---:|---:|---:|---:|---:|
| publish | A | -2.84037185 | -1.65664577 | 0.492214322 | 4.48524516e-08 | 1.93817163e-07 | 1.48964712e-07 |
| wait | B | -1.17368615 | -0.00306510949 | 0.479109287 | 2.37475379e-07 | 1.01282215e-06 | 7.75346773e-07 |
| unknown | C | 14.0795155 | 13.7997036 | -0.971323609 | 0.999999718 | 0.999998793 | -9.24311484e-07 |

#### text-ambiguous — E4B

| Semantic option | Slot | BF16 logit | Q4_K_M logit | Centered Δ | p(BF16) | p(Q4) | Δp |
|---|:---:|---:|---:|---:|---:|---:|---:|
| publish | A | 23.3461876 | 23.735218 | -0.37055397 | 0.00192298002 | 0.0027893917 | 0.000866411683 |
| wait | B | 24.9572868 | 26.8888569 | 1.17198563 | 0.0096308869 | 0.0653307209 | 0.055699834 |
| unknown | C | 29.5884457 | 29.5465984 | -0.801431656 | 0.988446133 | 0.931879887 | -0.0565662457 |

### Interpretation / remaining work

A→Bのlogit差はfixtureとmodelで大きく異なる。共通offset除去後の最大差もprobability差も揃わず、bit幅や別fixtureの結果から許容値を推定できない。image E2BではQ4のtop-two marginが小さくなった一方、top-1は同じだった。これらのCPU値をCUDA許容差やruntime最適化(B→C/C→D)へ転用しない。

元checkpoint revision/imatrix実体・再現設定の追跡、意味正解と意図的near-tieを持つ固定fixtureの追加、全ケースのA/B測定、order reversal/distractor、Step 5の384/512件比較は残る。Step 2/3のCUDA最適化経路とreference C/Dは実行しておらず、GPUは使用していない。利用条件に合うGPUがないためCUDA比較・Step 5には入らずここで止めた。A/Bの意味品質基準もまだ設定しない。

<a id="step-4-interim-review"></a>

## 2026-09-30: Step 4途中結果のまとめと考察

旧records、Step 1、Step 4の全候補表と保存済みreference runner/logを照合した。
今回の作業は既存結果の再解析と文書訂正で、新規model inferenceやCUDA検証は行っていない。
7比較すべてについてraw logitsから温度1 softmax、centered差、top-two marginを再計算し、
掲載値との一致を表示丸めの範囲で確認した。

### この比較が必要になった経緯

1. Phase 3+で候補説明continuationから単一answer-slot readoutへ移行した。
   現在の精度評価対象は、候補を含む一つのpromptの最終位置におけるラベル分布とsemantic ID選択。
2. 旧option-scaleで暫定512二文字ラベルの形式を確認したが、意味判断は16/32/64件の2例のみ。
   E2Bの未完成レポートは順序反転でwinnerが変わり、E4Bは期待IDを維持した。
   別のCPU Flash probeではE4B/128件のmargin 0.0255のwinnerが反転した。
   そのfixtureに正解labelはなく、反転を誤答増加と確定する結果ではない。
3. 旧query-only chunk/RSS試験では全layerの一時tensorと本番VRAMの課題が残ったため、
   option-scale-2ではE4B/単一GPU全載せに向けたメモリ削減へ優先順位を変更した。
   Step 1は全layer/512-token microbatchを実装・CPU確認済み。
   F16 KV/FAのStep 2、必要時のSWA短縮のStep 3は未着手。
4. Step 1でtop-1が一致しても相対logit/確率差が残ったため、ユーザー提案の
   「BF16→Q4差を最適化の追加差の物差しにする」をStep 4へ追加した。
   今回は先に実行できるCPU A/Bを測った段階で、CUDA最適化の採用評価には達していない。

### A/Bから現時点で読み取れること

top-1変更0/7は、E2Bの4比較とE4Bの3比較で同じwinnerが残ったという結果。
全7比較に評価用の正解IDがなく、synthetic 16候補は意味的な正解を設計していない。
従って誤答率の分母は得られておらず、「誤答0/7」ではない。
全5個の16候補比較が先頭の`option_00`を選ぶが、候補内容が正解を定義しないため、
この一致から意味理解の成立も先頭位置の偏りの大きさも確定できない。
`text-ambiguous`も名称に反して観測marginはE2B/Q4で13.803、E4B/Q4で2.658あり、
意図的near-tieの代わりにはならない。

共通offsetだけでは説明できない差がある。E2B/longでは最大raw差6.004のうち平均shiftは
+4.433だが、centered最大差も3.506ある。それでもwinnerの確率は0.999973→0.999529で、
最大確率差は0.000444にとどまる。softmaxがwinnerへ集中している条件では、
小さな確率差だけでlogit差が小さいと判断できない。

E4B/shortではmarginが2.337→2.367とほぼ同じでも、winner確率は0.8254→0.7563へ変化し、
2位は`option_01`→`option_06`へ移る。E4B/longでも2位は`option_01`→`option_10`。
marginは各条件の2位との差なので、同じ対抗候補とのgapを比較しているとは限らない。

最大の分布変化はE2B/shared-image control。winner確率は0.9812→0.4960、
marginは5.105→1.001、2位は`option_03`→`option_15`となる。
同じvisual embeddingsを入力しているので、A/B間のencoder出力差が原因ではない。
ただしcheckpoint由来が未確定で、text/画像配置を固定した一例のため、
量子化だけの因果差や一般的な画像判断品質の低下とは断定しない。

### Step 1の差を、この物差しでどう見るか

以下は同名・同token列のfixtureで、Step 1記録のbranchscore−Q4 reference差と
Step 4のQ4−BF16差を並べたもの。imageは双方shared-embedding controlを使う。
Step 1のtext reference revisionは`aa39d7a3...`、Step 4は`19e28a27...`であり、
厳密な同一revisionのA/B/C/D比較ではない。image referenceは双方`19e28a27...`だが、
referenceはtext/image区間ごとのdecode、branchscoreは全位置を512ずつ切るのでchunk構成も異なる。
これはCPUでの差の規模を見直す材料で、C/Dの測定やCUDA採用基準にはしない。

| Model / fixture | A/B centered最大差 | Step 1 centered最大差 | A/B最大確率差 | Step 1最大確率差 |
|---|---:|---:|---:|---:|
| E2B / short_16 | 2.551 | 1.230 | 0.133042 | 0.0551 |
| E2B / long_16 | 3.506 | 1.014 | 0.000444 | 0.00095 |
| E4B / short_16 | 1.671 | 0.406 | 0.069161 | 0.0374 |
| E4B / long_16 | 2.197 | 0.834 | 0.014787 | 0.00890 |
| E2B / shared-image | 4.863 | 0.623 | 0.485147 | 0.0455 |

この5条件のcentered最大差ではStep 1の差がA/B差より小さい。
ただしE4Bの最大確率差はA/B差の約54%/60%で、桁違いに小さいとは言えない。
E2B/longでは確率差の大小が逆になる。これにより、
「top-1一致、または一つの差の比率だけで十分小さいと認定する」方法には無理がある。
E2B/shared-imageの数値は比較を進める材料だが、E4B画像のA/Bは未測定。
E2Bで得た大きい量子化差をE4Bの画像実装差0.0793の許容根拠にはできない。
別々のencoderを動かしたE2B画像の差0.2426も、この0.0455のcontrolとは分ける。

### 採用判断へ進むための残り

量子化差を物差しにする方針は維持できる。ただし現在の結論は
「入力別のA/B観測値を得た」であり、「最適化差を許容できる」はまだ未判定。
数値差と意味判断の両方を揃えるため、次は以下を小さく順に確認する。

1. 同一checkpoint revisionと量子化設定を追跡する。由来を立証できなければ、
   固定した高精度GGUFから自前量子化するcontrolも候補。ただし現在のQ4ファイルとは別artifactになり、
   productionで使うQ4との比較も必要になる。今回この量子化は実行していない。
2. 既存の正解付き`text-explicit-yes-no`、旧screeningの健康状態/未完成レポートを足場に、
   意味正解・僅差・順序反転・無関係候補追加を持つ少数fixtureを固定し、E2B/E4BのA/Bを測る。
   BF16のwinnerも正解とは仮定せず、正解→誤答/誤答→正解を別々に数える。
   E4B/shared-image A/Bも追加し、同じembeddingのcontrolとend-to-endを区別する。
3. 採用するbackendでA/Bの基準を得て、C/Dの採用評価前に絶対差・相対差と追加誤答の
   許容範囲をrecordsへ明示する。現在の7比較から普遍的な許容比率は設定しない。
4. Step 2/3の対応経路ができたら同じrevision・入力・型・FA・padding・chunk条件で
   B→C、C→D、B→Dを照合する。mask/position/shared KV等の不具合は数値許容とは別に修正する。
5. Step 5のE4B/384・512件と実際の画像・説明長で、同じ比較とpeak VRAMを測る。
   形式上の512ラベル、16候補のA/B、CPU RSSのどれも公開件数の保証にはならない。

GPUがなくても由来確認と正解付きCPU A/Bの補強は進められる。
一方、CUDA経路の最終採用、8 GiB以上の単一GPU全載せ、384/512件品質はGPU実測が必要。
Step 4は未完了のまま、productionの2–16件/A–P契約と現行F32通常attentionを維持する。
