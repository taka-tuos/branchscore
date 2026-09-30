# Phase 4+ option-scale-2 - GPU 全載せに向けた Prefill メモリ削減と候補拡大

## Status / Position

2026-09-30 計画作成。Step 1の実装とCPU/reference focused verificationを完了。
Step 2は未着手。CUDA pathのmodel-backed照合に必要なGPU実行環境がなく、Step 4の
E4B全載せ計測にはユーザー指定の8 GiB以上GPUが必要なため、ここで保留する。
[旧 option-scale](phase-4-plus-option-scale.md) は調査途中で打ち切り、
未完了の資源・品質・公開契約の判断を本計画へ引き継ぐ。
旧CPU probeの記録は採用済みruntimeではない。

優先順位はメモリ削減。本番はE4Bを8 GiB以上のVRAMを持つ単一NVIDIA GPUへ全載せする。
512件を目標とし、384件を比較点にする。256件では用途上の余裕が足りない懸念がある。
検証機のCPU時間・RSS・swapを本番の採用基準にしない。
現在のproduction契約は2–16件、A–Pの単一token answer-slot readout。
上限公開は本計画の資源・ラベル・判断品質の検証後に行う。

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`（現行採点契約）
- `docs/records/phase-4-plus-option-scale-2.md`（今回のsource調査・容量推算）
- 必要な項目だけ `docs/records/phase-4-plus-option-scale.md`（旧tokenizer/CPU probe）
- 入口変更時は `docs/phases/phase-4-plus-http-server.md`
- 計測境界は `docs/phases/phase-4-backend-separation.md` と
  `docs/records/phase-4-backend-measurements.md`
- 必要時のみ `docs/research/llama-ggml.md`

## Goal / Constraints

- ggml直接利用の独立C/C++ runtimeを維持し、llama.cppは実装照合のreferenceとして使う。
- `1 model / 1 backend / 1 request / N options` の逐次実行。
  Layer split、tensor parallel、option worker、backend schedulerは導入しない。
- 全候補を含む一つのpromptを最後まで読む。token分割は一回の論理的なPrefillの内部処理。
  最終位置のlabel logitsを一度readoutし、回答tokenの生成・消費は行わない。
- 数値検証は同じGGUF・prompt tokens・backend・cache型・FA設定のllama.cppと照合する。
  元のF32全長graphへのbit一致をFA/F16採用の必須条件にしない。
  mask、absolute position、画像配置、shared KVの実装差は修正対象として区別する。
- 件数上限と、state・説明・画像を含むrequest token/資源予算を別に定める。
  context metadataの131,072 tokenをそのまま実行可能上限にしない。

## Steps

### Step 1 - Prefill全体のtoken microbatch化（全長F32 KV）

1. `PrefillEngine`の全layer計算を、例えば256/512 tokenずつ逐次実行する小さな変更にする。
   最初は全長F32 K/Vと通常attentionを維持し、型・FA・SWA短縮を混ぜない。
2. 現在の`build_layer`の`query_count`、`cache_start`、`cache_count`と
   `causal_mask`のabsolute query offsetを使い、各chunkのKVをrequest cacheへ追記する。
   positionsはprompt全体の絶対位置。full/sliding maskはchunkのquery数×必要なkey数で作る。
3. text/imageのsplicingと画像位置のper-layer token IDの扱いを保ち、
   FFNとper-layer inputもchunk長だけのgraph tensorにする。
   中間chunkのvocabulary projectionは省き、最後のchunkの最終位置だけlogitsを保存する。
4. 一つのchunkの一時graphを次へ再利用または解放してから進み、
   全chunk完了後にprefixをfreezeする。失敗したrequestのcacheを次へ持ち越さない。
5. 既存の短い16件入力と、chunk/sliding-window境界をまたぐ少数のtext/image入力で
   E2B/E4Bの最終位置logits・selected ID・KV位置を確認する。
   shape/kernel変更に伴う差はraw差と相対差で記録し、bit一致だけで判定しない。

**完了条件:** 一つのpromptを最後まで処理でき、chunk境界の位置・mask・画像・shared KVが
一致する。cacheと一時graphのbytes、chunk長、実行pathを記録する。
旧query-only chunkの実行結果だけでこのStepの完了としない。

### Step 2 - F16 KV + F16 mask + text Flash Attention

1. Step 1の通常attentionをcontrolにし、K/V cacheをF16にする。
   maskはhost側もF16で直接作り、`0`/`-∞`を保つ。QはF32、accumulation/outputはF32。
2. pinned ggmlのCUDA kernelに合わせ、logical token数とphysical cache容量を分ける。
   full attentionの512次元headに必要な256-token key paddingとstride alignmentを満たし、
   padding位置をmaskで遮断する。未使用値をattentionへ混入させない。
3. 実際のQ/K/V/mask shapeで選択backendの`ggml_backend_supports_op`を確認する。
   非対応時は同じbackendのmicrobatch通常attentionを比較候補とする。
   本番全載せ条件の検証では意図しないCPU tensor配置を許容しない。
4. 同条件のllama.cppと少数の入力で照合し、attention path、cache/mask型、
   chunk長、padding、raw logit差、共通offsetを除いた相対差、selected IDを記録する。
   誤差許容はreferenceとの観測に基づいて明示する。
   旧CPU Flash/F16 KV probeの差をCUDAの精度差として扱わない。

**完了条件:** 型・padding・backend対応とfinal-position readoutが確認でき、
referenceとの数値差を説明できる。FA採用・通常経路継続の根拠とbuffer容量を残す。

### Step 3 - 必要ならSWA cache短縮

1. Step 2の全長cacheのメモリ量を基に、追加の余裕が必要か判断する。
   見送る場合は判断根拠をrecordsへ記録してStep 4へ進む。
2. full attentionは必要なprompt全長を保ち、slidingは概ね`window + microbatch`にする。
   学習済みwindow幅を変えず、absolute positionと物理slotを対応させる。
3. 各chunkの全queryとshared layerが読むKVを保持してから古いsliding KVを破棄する。
   Gemma 4の既存shared-KV source mappingを維持する。
   `StateCache`のimmutable-prefixと容量・位置の契約を、この処理に合わせて明文化する。
4. Step 2と同じcache型・attention pathの全長cacheをcontrolにし、
   chunk/window境界とshared layerを含む入力でlogits・selected ID・cache容量を確認する。

**完了条件:** short-SWAを採用または見送りと判断し、数値・位置・メモリの根拠を残す。

### Step 4 - 本番相当GPUで384/512件の資源・判断品質を測る

1. E4B Q4_K_Mと対応mmprojを単一NVIDIA GPUへ全載せする。
   GPU型番、VRAM総量/利用可能量、GGUF、project/ggml revision、path/型/chunk長を記録する。
2. 調査用の拡張label rendererで384/512件を比較し、textと実際に使う画像・説明長の
   workloadを含める。16件controlと旧短文workloadは照合用に使う。
3. weight/cache/graph buffer bytes、model load後/各stage後の空きVRAM、
   実行中peak VRAM、prompt/visual token数、失敗段階を分けて記録する。
   CUDA workspace/poolの保有分も含め、CPU cumulative RSSからVRAMを推定しない。
4. 少数の意味判断fixtureで正解ID、順序反転、無関係候補追加を確認する。
   全候補raw logits、selected ID、referenceとの差を保存する。
   相対確率は条件付き・未校正のまま扱う。
5. latencyも記録するがメモリ削減を主評価にする。
   単発値と反復結果を区別し、公開上限の判断に必要な実運用の処理時間も示す。

**完了条件:** 全載せとpeak VRAMが実測で成立する範囲、判断品質、失敗条件を示せる。
384件だけに下げる判断はこの測定を基に行う。8 GiBでの成立を概算だけで保証しない。

### Step 5 - ラベル契約とrequest budgetの確定

1. 旧調査の暫定512二文字labelを引き継ぎ、採用する拡張rendererの実際の回答位置で
   E2B/E4Bの単一normal token、piece、ID一意性、prompt境界を再確認する。
   旧16件rendererでの形式照合を拡張promptの意味判断品質の証明にしない。
2. label集合・順序・無関係候補に対するStep 4の品質結果から公開件数を決める。
   label方式が成立しなければ公開拡大を止め、別の採点方式を自動導入しない。
3. 既存16件A–P入力のprompt契約を保ち、拡張label用renderer IDと
   scoring/readout identity、execution metadataの扱いを確定する。
   単一回答位置logit、温度1の候補softmax、semantic ID対応を明文化する。
4. 説明長・state・visual tokensを含む単問token予算と、HTTPの複数問を合算した
   request予算を決める。件数・context・資源超過を区別し、モデル実行前の拒否条件を定める。

**完了条件:** 採用件数、renderer/label/scoring契約、モデル/backend別の資源予算が確定する。

### Step 6 - core / adapter / CLI / HTTP / UIへの反映と採用判断

1. Step 5の決定後に`Gemma4DecisionEngine`、renderer/tokenizer/readout、
   `SystemOne` adapter、CLI/JSONL、HTTP/UIの順で小さく変更する。
   件数定数を共有し、HTTPのkey辞書順とCLI/JSONL入力順をmetadataに残す。
2. 255件を超えるChoiceはbranchscore固有拡張として示す。
   UIは説明の貼り付け/一括入力を検討し、エラーは現行HTTP契約へ対応付ける。
3. 既存16件入力、拡張label/semantic ID対応、同点順序、確率和、context/資源拒否、
   実行失敗後の次requestをfocused verificationする。
4. 本番相当GPUの資源・品質が成立した最大件数を公開する。
   512件が成立しなければ測定に基づく上限と妨げた条件を記録する。

## Deliverables / Completion Criteria

- token microbatch Prefill、選択したattention/cache経路と明確なrequest内寿命。
- E4B GPU全載せの384/512件計測記録と、同条件referenceとの数値・判断品質の照合。
- 検証済み公開件数、単問/requestのtoken・資源予算、label/renderer契約。
- 現行16件入力の回帰確認、予算拒否と失敗後の次requestの確認。
- 未採用のSWA短縮や512件未達は理由を記録する。

## Notes / Findings

2026-09-30: 旧option-scaleはStep 1/2の調査途中で打ち切り、メモリ削減を優先する本計画へ移行。
旧tokenizer照合・初期品質screening・CPU resource/F16/FA/query-chunk probeは
[旧調査記録](../records/phase-4-plus-option-scale.md)に保持する。
重み容量の訂正、llama.cppのmicrobatch/SWA、CUDA FAの型・padding条件、
容量式と数値差の再解析は[option-scale-2調査記録](../records/phase-4-plus-option-scale-2.md)に移管した。
512件を目標、384件を比較点とする。GPU peak VRAM・拡張label品質・request budgetは未確定。

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
8 GiB以上GPUでのStep 4には未着手。
