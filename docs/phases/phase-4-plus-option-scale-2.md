# Phase 4+ option-scale-2 - GPU 全載せに向けた Prefill メモリ削減と候補拡大

## Status / Position

2026-09-30 計画作成。Step 1の実装とCPU/reference focused verificationを完了。
Step 2以降は未着手。Step 4の高精度比較計画を追加済み。
CUDA pathのmodel-backed照合に必要なGPU実行環境がなく、Step 5の
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
- runtime最適化の追加差は、Step 4で同じモデルの高精度重み→Q4_K_Mの差を物差しにし、
  追加誤答と同条件referenceとの実装照合を併せて採用判断する。
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
   referenceとの実装差を記録し、実用上の誤差許容はStep 4の量子化差・追加誤答と併せて判断する。
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

<a id="step-4-precision-baseline"></a>

### Step 4 - 高精度重みとの比較で量子化差・最適化差・実装差を測る

1. E2B/E4Bそれぞれで同一checkpoint由来のF16または元のBF16とQ4_K_Mを用意し、
   GGUF hash、元checkpoint、量子化設定/imatrixの確認状況、tokenizer一致、reference revisionを記録する。
   未確認の由来・設定は未確認と明示する。モデル規模やbit数だけから回答差を推定しない。
2. 少数の代表fixtureを固定し、同じrendered prompt token IDs、label集合/順序、backend、
   microbatch上限で下表を逐次実行する。基準経路はF32 KV・通常attention・全長SWAとし、
   最適化経路はStep 2/3で選んだcache型・FA設定・SWA容量・paddingに揃える。
   画像入力は同じmmprojと前処理を用い、数値差の切り分けには同じvisual embeddingsを入力する。
   encoderを別々に動かすend-to-end結果は、重み量子化差とは分けて保存する。

   | 条件 | Text weights | Runtime / path |
   |---|---|---|
   | A | F16 / 元のBF16 | llama.cppの基準経路 |
   | B | Q4_K_M | Aと同じllama.cpp基準経路 |
   | C | Bと同じQ4_K_M | llama.cppの対応する最適化経路 |
   | D | Bと同じQ4_K_M | branchscoreの対応する最適化経路 |

   A→Bを重み量子化差、B→Cをruntime最適化差、C→Dを実装差として記録する。
   B→Dの合成差も測り、個別差が小さいことだけで最終結果を判断しない。
   referenceで対応経路を再現できなければ、再現できた条件と未分離の要因を明示する。
3. 既存16件control、chunk/window境界、画像、僅差の上位候補を含める。
   Step 5の384/512件fixtureにも同じ比較を適用し、順序反転・無関係候補追加を揃える。
   全候補raw logits、共通offsetを除いた差、候補softmax差、上位2候補のlogit間隔、
   selected ID変更の件数/率、正解→誤答と誤答→正解の件数/率を保存する。
   件数の分母とfixture別の結果を残し、少数例の一致を一般的な誤答率の保証にしない。
4. A→Bの観測を物差しにし、B→C/B→Dの追加差が十分小さいか、
   追加の誤答が実用上許容できるかを併せて判定する。
   「十分小さい」の数値基準と追加誤答の許容範囲は、A/Bの基準測定後、
   C/Dの結果を採用評価する前にrecordsへ明示する。Step 2/3の経路選択はこの評価まで暫定とする。
   共通offsetを含む最大raw差だけで判定しない。量子化差がゼロに近い場合は
   比率で無理に比較せず、絶対差・選択一致・誤答の変化を示す。
   mask・position・padding・画像配置・shared KVの不具合は量子化差が大きくても修正する。
5. 高精度重みとQ4_K_Mは一つずつロードし、同時常駐を要求しない。
   高精度版のGPU実行には8 GiBを超える容量が必要になり得るため、比較用環境を別途確認する。
   CPUで先にA/Bを測れる場合もCUDAと別集計にし、CPU量子化差をCUDAの許容差へ直接転用しない。
   高精度版の不足や比較条件の不一致は未完了事項として残し、Step 2/3/5の実装・資源調査は進められる。
   公開件数とruntimeの最終採用は、このStepとStep 5の結果を揃えて判断する。

**完了条件:** 同一モデル・入力・backendの量子化差を基準に、最適化の追加差と追加誤答、
referenceとの実装差を説明できる。採用基準・判定・比較できなかった範囲をrecordsへ残す。
他モデルのPPLやbit数だけの推定を、今回のanswer-slot精度の確定値にしない。

### Step 5 - 本番相当GPUで384/512件の資源・判断品質を測る

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

### Step 6 - ラベル契約とrequest budgetの確定

1. 旧調査の暫定512二文字labelを引き継ぎ、採用する拡張rendererの実際の回答位置で
   E2B/E4Bの単一normal token、piece、ID一意性、prompt境界を再確認する。
   旧16件rendererでの形式照合を拡張promptの意味判断品質の証明にしない。
2. label集合・順序・無関係候補に対するStep 5の品質結果から公開件数を決める。
   label方式が成立しなければ公開拡大を止め、別の採点方式を自動導入しない。
3. 既存16件A–P入力のprompt契約を保ち、拡張label用renderer IDと
   scoring/readout identity、execution metadataの扱いを確定する。
   単一回答位置logit、温度1の候補softmax、semantic ID対応を明文化する。
4. 説明長・state・visual tokensを含む単問token予算と、HTTPの複数問を合算した
   request予算を決める。件数・context・資源超過を区別し、モデル実行前の拒否条件を定める。

**完了条件:** 採用件数、renderer/label/scoring契約、モデル/backend別の資源予算が確定する。

### Step 7 - core / adapter / CLI / HTTP / UIへの反映と採用判断

1. Step 6の決定後に`Gemma4DecisionEngine`、renderer/tokenizer/readout、
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
- 高精度重み/Q4_K_Mの量子化差、最適化の追加差・追加誤答、採用基準の測定記録。
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

2026-09-30: Step 1を完了。最大512 tokenの全layer microbatch、全長F32 KV、
absolute position、画像spanのchunk分割、最終位置readoutを実装した。
E2B/E4BのCPU境界testとllama.cppのtext/image照合を実施し、今回のfixtureではtop-1が一致。
相対logits/probability差は残り、拡張候補数の品質とCUDAの許容差は未確定。
容量値・比較条件・全数値表・vision差の切り分けは
[Step 1実行記録](../records/phase-4-plus-option-scale-2.md#2026-09-30-step-1-prefill-microbatch-implementation-and-cpureference-checks)へ移管した。

2026-09-30: 重み量子化差をruntime最適化の追加差の物差しにする方針をユーザーと合意。
追加誤答と同条件referenceとの実装照合を併せて評価するStep 4を追加した。
従来Step 4–6はStep 5–7へ繰り下げた。判断の背景と未測定事項は
[採用基準の検討記録](../records/phase-4-plus-option-scale-2.md#precision-baseline-review)に保持する。
