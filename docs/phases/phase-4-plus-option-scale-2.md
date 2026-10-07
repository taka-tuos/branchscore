# Phase 4+ option-scale-2 - GPU 全載せに向けた Prefill メモリ削減と候補拡大

## Status / Position

2026-09-30計画作成。旧[option-scale](phase-4-plus-option-scale.md)の未完了判断を引き継ぐ。
2026-10-07: 利用者の「まず512件で使える版を出す」方針に従い、初期512対応を実装・検証。
Step 1のmicrobatchに加え、Step 2のCUDA F16 KV/mask＋Flashを採用した。
Step 3は最大VRAMサンプル6,352 MiBで16,384 positionsまで成立したためSWA短縮を見送る。
Step 6/7の512件label契約、core/CLI/JSONL/HTTP/UI、予算と拒否を実装した。
E2B CUDA CTest 11/11、E4B focused checks 3/3、既存quality 24/24・vision 14/27。
Step 5の初期資源確認は合成20文字品番512件＋HD/FHD画像と逐次反復で完了。
FHD/14,854 positionsは約11.5秒、16,384 positionsは約13.1秒。
白画像は容量controlであり、実パッケージのOCR品質検証ではない。
現在の契約は2–512件。2–16件のprompt/A–Pは維持し、17–512件は固定二文字label。
単問16,384 expanded positions、HTTP合計32,768 positionsを実行前に確認し、切り捨てない。
[採用実装・検証記録](../records/phase-4-plus-option-scale-2/runtime-512-adoption.md)を参照。

Step 4の高精度CUDA A/BとCPU残差研究は未完了。CPUの事前screening未達判定は保持し、
CPU/VulkanのPrefillは従来F32のままとする。CUDA E2Bには既存僅差fixtureの追加誤答1件がある。
E4B画像384/512の順序依存も残る。実BOM/画像での品質と運用時間は利用後の追加確認とし、
初期公開の開始条件にはしない。利用者の成功例報告は本測定とは区別する。
画像配置と固定prefix cacheは別の調査へ回し、現行画像先頭・request所有cacheで提供する。
[実用方針](../records/phase-4-plus-option-scale-2/practical-512-adoption.md)と
[画像配置比較](../records/phase-4-plus-option-scale-2/image-placement-gpu.md)を参照。

優先順位はメモリ削減。E4Bを8 GiB以上のVRAMの単一NVIDIA GPUへ全載せする。
384件を比較点とし、512件を初期公開上限とする。件数とtoken予算を分け、
任意の説明・backend/GPUでの資源成立までは保証しない。

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`（現行採点契約）
- [option-scale-2記録索引](../records/phase-4-plus-option-scale-2.md)（下表から担当Stepの記録だけ読む）
- 必要な項目だけ `docs/records/phase-4-plus-option-scale.md`（旧tokenizer/CPU probe）
- 入口変更時は `docs/phases/phase-4-plus-http-server.md`
- 計測境界は `docs/phases/phase-4-backend-separation.md` と
  `docs/records/phase-4-backend-measurements.md`
- 必要時のみ `docs/research/llama-ggml.md`

## Record routing

| 作業・確認 | 必要な記録 |
|---|---|
| Step 1–3の容量・Prefill分割 | [メモリ・Prefill](../records/phase-4-plus-option-scale-2/memory-prefill.md) |
| Step 4のC/D CPU測定・事前基準・判定 | [C/D CPU評価](../records/phase-4-plus-option-scale-2/cd-cpu-evaluation.md) |
| Step 4の現在のCPU baseline・残る確認 | [CPU kernel/key padding](../records/phase-4-plus-option-scale-2/cpu-kernel-padding-baseline.md) |
| Step 4の初期A/BとGGUF由来 | [初期baseline](../records/phase-4-plus-option-scale-2/precision-baseline.md)、[由来・quality fixture](../records/phase-4-plus-option-scale-2/precision-review-provenance.md) |
| 画像fixtureと境界修正の経緯 | [vision screening](../records/phase-4-plus-option-scale-2/vision-screening.md)、[reference/control](../records/phase-4-plus-option-scale-2/vision-reference-controls.md)、[境界・buffer](../records/phase-4-plus-option-scale-2/image-boundary-buffer.md) |
| CUDA回帰、F16/Flash probe、384/512候補の容量・品質 | [GPU検証](../records/phase-4-plus-option-scale-2/gpu-validation.md)、[再現資料](../records/option-scale-2-gpu-2026-10-07/README.md) |
| 512候補の早期対応・BOM用途・採用条件 | [実用採用方針](../records/phase-4-plus-option-scale-2/practical-512-adoption.md) |
| 初期512実装・公開契約・HD/FHD資源とHTTP回帰 | [512採用記録](../records/phase-4-plus-option-scale-2/runtime-512-adoption.md)、[再現資料](../records/option-scale-2-runtime-512-2026-10-07/README.md) |
| 固定prefix再利用に向けた画像とtextの配置比較 | [画像先頭/末尾CUDA比較](../records/phase-4-plus-option-scale-2/image-placement-gpu.md) |
| 以前のF16/Flash/query-only chunk実験 | [旧CPU probe](../records/phase-4-plus-option-scale/cpu-memory-probes.md) |

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
  初期512対応は実用採用方針を優先し、高精度CUDA不足だけで公開を止めない。
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
   C/Dの結果を研究評価する前にrecordsへ明示する。初期CUDA採用は下記の実用方針を優先する。
   共通offsetを含む最大raw差だけで判定しない。量子化差がゼロに近い場合は
   比率で無理に比較せず、絶対差・選択一致・誤答の変化を示す。
   mask・position・padding・画像配置・shared KVの不具合は量子化差が大きくても修正する。
5. 高精度重みとQ4_K_Mは一つずつロードし、同時常駐を要求しない。
   高精度版のGPU実行には8 GiBを超える容量が必要になり得るため、比較用環境を別途確認する。
   CPUで先にA/Bを測れる場合もCUDAと別集計にし、CPU量子化差をCUDAの許容差へ直接転用しない。
   高精度版の不足や比較条件の不一致は未完了事項として残し、Step 2/3/5の実装・資源調査は進められる。
   高精度比較を含むStep 4全体の完了は継続課題とする。初期512対応の公開と経路採用は、
   実用採用方針に従い、既存品質回帰・BOM相当の合成品番/HD・FHD資源・逐次request確認を基に判断する。
   実BOM/画像の品質評価は利用後の追加確認へ回し、初期公開の必須条件にしない。
   高精度CUDA A/B未実施は明記し、Step 6/7の一律の開始阻害条件にしない。

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
- 比較可能な高精度重み/Q4_K_Mの量子化差、最適化の追加差・追加誤答、実用採用判断の記録。
  高精度CUDA A/Bは環境が得られた際の追加検証とし、初期512対応の必須成果物としない。
- 検証済み公開件数、単問/requestのtoken・資源予算、label/renderer契約。
- 現行16件入力の回帰確認、予算拒否と失敗後の次requestの確認。
- 未採用のSWA短縮や512件未達は理由を記録する。

## Notes / Findings

- 2026-09-30: Step 1を完了。512-token全layer microbatch、全長F32 KV、absolute position、
  画像span分割と最終位置readoutをCPU/referenceで確認。[実行記録](../records/phase-4-plus-option-scale-2/memory-prefill.md)。
- 2026-09-30〜2026-10-02: Step 4の初期A/B、GGUF由来、quality/vision fixture、画像境界修正、
  CPU buffer/kernel/key軸の診断を実施。[最新baseline・再レビュー](../records/phase-4-plus-option-scale-2/cpu-kernel-padding-baseline.md)。
  BF16/Q4各13/27は正誤変化各1件の相殺で、画像判断の品質達成を示さない。
  診断6条件の一致を全条件・長文・CUDAへ一般化しない。
- 2026-10-02時点の次作業: 未採用C/Dのkernel・decode shape差を切り分け、padding採用候補の全27条件・
  chunk/window境界と時間/メモリ差を確認する。E2B修正後基準・CUDA品質・384/512件・VRAMは未測定。
  Step 4と公開件数の採用判定は未完了。
- 2026-10-02: recordsをテーマ別に分割し、phaseは計画・現在の進捗・読む記録の案内に整理。
  過去のFindings全文は[履歴](../records/phase-4-plus-option-scale-2/plan-findings-history.md)へ移管。
  測定本文・rawデータを保持し、移動後の参照リンクを更新した。
  元の24節・Findingsの本文保存、計画のGoal/Steps/完了条件不変、raw hash、相対リンク・旧anchorを検証。

- 2026-10-02: Step 4のC/D CPU測定に着手。F16 KV・host F16 mask・Flash Attention・
  全長SWA・512-token上限の測定専用copyを作成し、結果を見る前に
  [採用screening基準](../records/phase-4-plus-option-scale-2/cd-cpu-evaluation.md)を固定。
  E4B修正後27画像条件へ同じ保存済みvisual embeddingsを入力する。通常runtimeは未変更。
  E2B・純text/16件・長文/window境界・CUDAと公開件数の判定は別途残る。

- 2026-10-02: E4B修正後27画像条件のCPU C/Dを完了。B/C/Dの正解は13/14/13。
  B→Cは誤答→正解1件、C→Dはその正解→誤答1件、B→Dは選択変更0件。
  事前数値screeningはB→C 12/27、C→D 0/27、B→D 11/27で未達のため未採用。
  [結果・限界](../records/phase-4-plus-option-scale-2/cd-cpu-evaluation.md)。523 positionsの
  2 chunkは成功。通常runtimeは未変更。ggml/kernel・decode shape差の切り分け、
  E2B・純text/16件・window境界・CUDA・384/512候補は残り、Step 4全体は未完了。

- 2026-10-02: [C/D再レビュー](../records/phase-4-plus-option-scale-2/cd-cpu-evaluation.md#cd-cpu-review-2026-10-02)で
  生出力・入力/embedding・source/binary/事前基準hashと27条件の数値・正誤集計を独立照合。
  B→Dの選択一致27/27は確認できたが、C→Dの数値基準は0/27で未達。
  両ggmlのCPU Flashは64未満のqueryでvector経路を使い、F16 Vでは加重和をF16で蓄積する。
  F32指定が全経路の内部F32 accumulationを保証するとは扱わない。
  Cの冒頭38 queryとDの混在chunk、Dの11-query tailでこの経路差が生じる。
  少数例でdecode shapeとFlash kernel/内部精度を揃えるcontrolを次の切り分けとし、原因の確定は留保する。

- 2026-10-07: GPU検証への引き継ぎ準備として、画像境界修正・CPU測定資料・記録整理を分けてcommit。
  既存CPU buildでrenderer/tokenizer/E4B engineの対象をbuildし、focused CTest 3/3を確認。
  CUDAの品質・384/512候補・peak VRAMとruntime採用判定は未完了のまま引き継ぐ。

- 2026-10-07: マージ後CUDA検証を実施。[GPU記録](../records/phase-4-plus-option-scale-2/gpu-validation.md)。
  E2BのCUDA構成CTest 11/11（skipなし）、E4B tokenizer/engine/Prefill 3/3。
  現行qualityはE2B 23/24・E4B 24/24、structured visionは6/27・13/27。
  測定専用52 probeと52同条件referenceは実行成功、選択一致47/52。
  E4B短文512候補（9,808 positions）の最大VRAMサンプルは現行7,060 MiB、F16/Flash 5,976 MiB。
  384/512画像も容量は成立したが、正順north（誤答）/逆順east（正解）の順序依存を観測。
  F16/FlashでE2B close-scoresが追加誤答となり、高精度CUDA A/Bと数値差の説明も未完了。
  runtime/default・公開2–16/A–Pは変更せず、Step 2/4/5の最終採用とStep 6/7を留保する。
  referenceの初回pilotはembedding CPU配置を検出して集計から除外し、全tensor CUDA overrideで再測定。

- 2026-10-07: 利用者は200件超のBOM照合に512候補の早期対応が必要で、超厳密な判断精度を
  要求しないと明示。HD～FHD Webカメラで部品パッケージを映す連続requestが主用途。
  [実用採用方針](../records/phase-4-plus-option-scale-2/practical-512-adoption.md)へ反映し、
  E4B CUDA F16/Flashを第一候補、高精度CUDA A/Bを初期公開の必須条件から外した。
  実装不具合の修正と実BOM/画像の資源・判断品質、逐次request確認は維持する。
  既存CPU screeningの未達は過去の事前基準による結果として保持。
  llama.cppの通常FA AUTO/F16 KVと、logit取得がattentionを再計算しないことをsourceで確認。
  runtimeはまだ変更していない。候補descriptionは枝番末尾まで含む品番のみで20文字程度。
  合成20文字品番512件のtokenizer実測は11,085–13,909 text tokens（画像placeholder込み）。
  次はこの長さにHD/FHD visual tokensを足した資源・時間確認を行う。要求total latencyは未確定。

- 2026-10-07: 利用者は現状、10文字超・BOM内の途中挿入文字列・末尾1文字違いを識別でき、
  480pセンサをFHDへ拡大したカメラでも品質は十分以上と報告。
  [実用採用方針](../records/phase-4-plus-option-scale-2/practical-512-adoption.md)に観測の出所と
  説明用仮例を記録。品質確認はこの既存成功例のFA/512拡張後の維持を中心とする。
  実画像・BOM・使用model/backendの取得は未了で、本GPU probeの実測とは混同しない。

- 2026-10-07: 連続画像requestのcache検討を
  [実用採用方針](../records/phase-4-plus-option-scale-2/practical-512-adoption.md)へ記録。
  現行の画像→BOM順ではBOM KVも画像に依存する。固定判定条件/BOM→今回の画像とする
  拡張rendererと単一GPU prefix再利用を候補とし、順序変更の品質とcacheの正しさを別に確認する。
  現行cacheはrequest所有でappend/reset契約がなく、runtimeは未変更。
  固定KVのVRAMと新画像のVision/tail計算は残るため、FAの容量確認とは別の逐次検証単位とする。

- 2026-10-07: 利用者の提案で
  [画像先頭/末尾CUDA比較](../records/phase-4-plus-option-scale-2/image-placement-gpu.md)を実施。
  E4B既存27条件＋384/512候補4条件×配置2種×非FA/FAの124実行は全成功。
  27条件の正解は非FA 13→14、FA 14→14だが、選択変更15/17条件、追加誤答5/6条件。
  board悪化・status改善があり、画像末尾の384/512 probeは両経路とも4/4誤答。
  元配置の54比較は全candidate raw差0、全9画像のembedding hashも全run一致。
  出力観測後の同token/embedding末尾reference診断10実行は全成功、選択9/10一致。
  512逆順の無関係候補zone506への誤答もreferenceで再現。単純な画像末尾配置の採用は留保。
  固定候補/BOM→画像→判定条件・質問を次の比較候補として記録し、未測定と明示。
  production renderer/cache再利用/runtimeは変更していない。

- 2026-10-07: 利用者の早期提供方針で[初期512対応](../records/phase-4-plus-option-scale-2/runtime-512-adoption.md)を採用。
  CUDA F16/Flash・512固定二文字label・legacy prompt維持、単問16,384/HTTP合計32,768位置。
  core/CLI/bench/adapter/server/UIの対応、全問preflight、予算拒否と回復を確認。
  既存E4B quality/visionの全候補raw値は先行FA測定と一致。資源probe 16成功＋1期待拒否、
  FHD＋20文字品番512件は約11.5秒、上限16,384位置は約13.1秒、最大VRAMサンプル6,352 MiB。
  大入力・拒否・反復後の空きVRAMは安定。SWA短縮は不要として見送り。
  白画像による資源測定と実パッケージ品質を区別し、実BOM品質・高精度CUDA比較・cache/配置変更は追加調査。

- 2026-10-07: READMEを現行512契約に合わせて再整理し、日本語版 `README.ja.md` を追加。
  目的・利用手順・採点方式・意図した設計方針・現行実装/検証の制限を分離した。
  旧A–P限定の採点説明と固定の旧test件数を除き、開発環境/測定履歴はrecordsへ案内。
  逐次実行は現段階の方針、画像先頭/request所有cacheは現行実装、
  CPU/Vulkanの512資源成立は未検証として区別。runtimeと公開契約は変更していない。
