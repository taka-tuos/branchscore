# Phase 4+ option-scale-2: 初期精度レビュー・GGUF由来・quality fixture

[記録索引](../phase-4-plus-option-scale-2.md) · [phase計画](../../phases/phase-4-plus-option-scale-2.md)

初期A/Bの考察、Unsloth公開履歴の照合、24条件の正解付きquality screening。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

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

<a id="step-4-provenance-and-quality-fixtures"></a>

## 2026-09-30: Unsloth公開履歴の照合と正解付きfixtureの補強

ユーザーから、全GGUFはHugging FaceのUnsloth配布品であり、ローカルファイルの時刻と
repoのcommit履歴を照合できること、既存fixtureに固執せず必要なデータを作ってよいことを確認した。
公開ファイルhash・履歴・GGUFヘッダを照合し、正解付きの小さい判断fixtureを追加した。
runtime、最適化経路、公開件数は変更していない。

### 由来の確認はどこまで進んだか

ローカル時刻は`stat`のmtime/birthを読み取った。これは取得・コピー時刻を反映し得るため、
モデルの作成・変換日時とは扱わない。既存Step 4 inventoryのSHA-256と、Hugging Face APIの
公開LFS `oid`（ファイルSHA-256）を照合した。このfollow-upで全ファイルの再hashは行っていない。
APIの取得結果、size、mtime、revision、hash照合、下記のtensor確認は
[由来の根拠JSON](../phase-4-plus-option-scale-2-provenance-2026-09-30.json)に保存した。

| Local GGUF | Local mtime（JST） | 公開hashと一致する最終変更commit |
|---|---|---|
| E2B BF16 | 2026-09-30 15:38:01 | `0314792d7f1f7e229411f620751375812bb9faf2` / 2026-07-17 |
| E2B Q4_K_M | 2026-09-29 12:20:28 | 同じ`0314792d...` / 2026-07-17 |
| E4B BF16 | 2026-09-30 15:06:37 | `bfc15c382204943c3a8fff0c750b94ae2364d7a3` / 2026-07-17 |
| E4B Q4_K_M | 2026-06-18 10:31:05 | `653803f092503c04a65164346f3208a36e707693` / 2026-05-04 |

E2Bの両ファイルは[7月更新版の公開ファイル](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/0314792d7f1f7e229411f620751375812bb9faf2)に一致。
E4Bは[BF16が7月版](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/bfc15c382204943c3a8fff0c750b94ae2364d7a3)、
[Q4が5月版](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/653803f092503c04a65164346f3208a36e707693)に一致した。
現在公開中のE4B Q4は4,977,171,584 bytes、SHA-256
`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`で、
ローカル5月版より2,016 bytes大きい。現在の`main`と無条件に同一視しない。

[E4Bの7月commit](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/commit/bfc15c382204943c3a8fff0c750b94ae2364d7a3)の題名は
chat template更新。題名だけで判断せず、固定revisionのQ4の先頭20 MiBのみをHTTP Rangeで取得し、
ローカル5月版のヘッダと直接比較した。全metadata値の差は`tokenizer.chat_template`のみ。
720個のtensor記述（名前・shape・type・相対offset）はすべて一致し、data開始位置だけが
15,824,736→15,826,752へ2,016 bytesずれる。data領域の容量は双方4,961,344,832 bytes。
**全tensor payloadの新旧一致は検証していない**ため、ヘッダ比較を重み値の完全一致証明にはしない。

ローカルBF16/Q4間も、tensor名とshapeはE2B/E4Bそれぞれ全数一致。
同じ非量子化type（F32/F16/BF16）、1個8 MiB以下のtensorを直接byte比較し、
E2Bは283個/1,169,548 bytes、E4Bは339個/2,263,208 bytesで不一致0だった。
これは共通の変換元を支持する追加根拠だが、量子化された大きい行列の元checkpoint revisionを特定しない。

この結果により、Unsloth公開品との対応とE4Bのファイル世代差は説明できた。
同じモデル系統の量子化比較として調査を継続する実用上の根拠は強くなった。
元Google checkpointの厳密なrevision、imatrix実体と全量子化recipeは引き続き未確定と明示するが、
それだけで正解付きCPU A/Bの補強を止める必要はない。
現行branchscore rendererはGGUFのchat templateを使わず、referenceにも同じ固定token列を渡すので、
確認されたchat template metadata差自体は既存A/Bの入力差にはならない。

### 既存データの弱点と追加したデータ

旧synthetic inputsは数値・chunk境界controlとして保持する。
しかし意味的な正解がなく、白画像も判断に必要な証拠を提供せず、
正誤・順序依存・追加誤答を測る用途には不足していた。問題はその用途との不一致。
既存`text-ambiguous`へ結果を見てから正解を付けることはせず、別の明示ルールfixtureを作った。

新規入力は[quality JSONL](../../../fixtures/phase4-step4-quality.jsonl)、設計と利用方法は
[fixture説明](../../../fixtures/phase4-step4-quality.md)。SHA-256は
`48e43bd6efd77dcc420f765997b6f3a78b866ae92dac47a721769ba06d5bfc21`。
以下の8独立シナリオから24条件を作り、推論前に正解IDと根拠を固定した。

| シナリオ | 内容 | 基本prompt tokens |
|---|---|---:|
| service-policy | checks/incidentから明示ルールで行動選択 | 138 |
| report-policy | 公開の二条件と明示fallback | 135 |
| approval-latest | 時系列・承認撤回・否定の解釈 | 166 |
| close-scores | 無効候補を除外して僅差の数値を比較 | 176 |
| japanese-priority | 日本語の優先順位と受付時刻 | 174 |
| long-route-16 | 48件のregisterから特定routeの担当を参照、16候補 | 943 |
| image-red / image-blue | 同一text/候補で画像だけを変え、色を識別 | 各120（placeholder込み） |

各シナリオに候補逆順と無関係候補4件追加を用意した。16候補の長文は上限を保つため
候補追加の代わりに5位置rotationとした。意味ID、正解、既存候補の説明は保持。
画像は全pixelが赤/青の96×96 RGB PNGを決定的に作り、PNG CRCと全pixelを確認した。
metadataの`expected_selected_id`/`expected_reason`等はbenchmark readerがpromptへ入れない。

全24条件をproduction renderer/tokenizerで確認した。E2B/E4B、BF16/Q4の4 GGUFで
prompt IDs/answer IDsは全数一致し、A–Pの回答境界検証も成功。
`(id, token_ids, answer_ids)`配列のcompact JSON SHA-256は
`8eb0d03ca2389e7ff4d4021b6dc4e7d75d17f685bfb26c72859e558b163c54df`。
943-token長文は512-token microbatchとSWA幅を越える意味参照controlになる。

### E4B Q4での初期screening

current branchscore、CPU、E4B Q4_K_M/対応F16 mmproj、一度に一つのmodel/requestで逐次実行。
textはF32 KV・通常attention・全長SWA・microbatch上限512、vision pathは`flash`。
基本8条件と未測定variant16条件を別runで実行した。これはbranchscoreの現行基準経路のscreeningで、
BF16 A/B、reference C/D、CUDA最適化比較ではない。
全候補raw scores/probabilities、prompt IDs、正解照合、margin、revision、run metadataは
[screen JSONL](../phase-4-plus-option-scale-2-quality-e4b-q4-2026-09-30.jsonl)に保存。

- 正解一致24/24条件、8/8シナリオで全variantが正解。
- 逆順/rotationの選択変更0/8比較、無関係候補追加の選択変更0/7比較。
- 最小top-two marginは日本語baseの3.667572。close-scores baseは10.970278。
- close-scoresの証拠値は近いが、観測logitはnear-tieではなかった。
  既存のmargin 0.0255のCPU Flash反転例のような数値ストレスを、この24条件が再現したとは言わない。

24条件は8シナリオの相関する変形で、独立24標本や一般的な正解率の保証にはしない。
この小セットは正誤と候補順序の評価を始める基準にはなるが、
僅差logitのstress cases、複雑な画像判断、384/512候補、実運用の代表性は未充足。
今後のnear-tie探索で追加fixtureを作る場合は、観測marginによる選定を記録し、
C/Dを評価する前に固定する。正解をQ4/BF16のwinnerへ合わせて書き換えない。

E2Bのmodel inferenceと全4重みの正解付きreference A/Bは、このfollow-upでは未実施。
Step 4は未完了を維持し、次のCPU A/Bに新fixtureを使用できる状態にした。
