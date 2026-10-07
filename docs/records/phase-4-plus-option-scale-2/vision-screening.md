# Phase 4+ option-scale-2: Structured vision fixture・4 GGUF screening

[記録索引](../phase-4-plus-option-scale-2.md) · [phase計画](../../phases/phase-4-plus-option-scale-2.md)

画像fixtureの設計と、境界修正前の4 GGUF/108行CPU screening・再レビュー。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

<a id="step-4-structured-vision-fixtures"></a>
## 2026-10-01: Step 4 structured vision fixtures

ユーザーの指示で、インストール済みImageMagickを使って複雑な画像証拠を作成した。
既存の数値control・24条件quality fixtureは保持し、別の
[vision JSONL](../../../fixtures/phase4-step4-vision.jsonl)と
[画像・再生成方法](../../../fixtures/phase4-step4-vision/README.md)を追加。
正解・根拠は推論前に生成元の事実と明示ルールから決め、
[manifest](../../../fixtures/phase4-step4-vision/manifest.json)に全画像hashと事実を保存した。
画像内にはbase/changedの識別や正解を記載していない。

| 系統 | 基本の証拠と正解 | 証拠変更と正解 | 配置変更 |
|---|---|---|---|
| board | 赤い四角: NORTH 3 / EAST 4 / SOUTH 2 / WEST 1 → `east` | EASTの四角1個をNORTHへ移動: 4/3/2/1 → `north` | 印刷名と物体を保持しcard位置を変更 → `east` |
| status | FAILEDかつOPENのうち最高severity → `lyra` | LYRAのRESPONSEだけOPEN→ACK → `vega` | 行順と装飾色を変更 → `lyra` |
| table | READYかつQUALITY≥80のうち最小COST → `birch` | BIRCHのQUALITYだけ84→78（barも同期）→ `aster` | 行順と装飾色を変更 → `birch` |

9枚の各画像に候補基本順・逆順・無関係候補2件追加を用意し、27 request rowsとした。
4–8候補で現行16候補契約の範囲。各候補順について画像3種間のstate/question/optionsは完全一致。
画像だけで証拠を変えたときの選択感度と、同じ証拠の配置変更に対する不変性を分けて確認できる。
3課題系統、6異なる証拠シナリオ、3配置変形であり、独立27標本とは数えない。
今回の作成ではモデル出力やmarginを見て正解・データを調整していない。

### Fixture作成・入力検証

- ImageMagick 6.9.12-98 Q16、DejaVu Sans/Bold/Monoで全画像960×720のRGB PNGを生成。
  一覧previewは1032×870で、モデル入力には使わない。9画像・preview・JSONLの
  再生成SHA-256一致を確認。同じフォント/ツール環境での再現であり他versionのbyte一致は未確認。
- JSONL SHA-256: `a98647965faa3d53db0d0da08ea0ac2be89ce74e1fed0b174565d99eba02416c`。
  全27行の一意ID、候補数/ID/正解包含、画像path、manifest hashを確認。
  逆順と候補追加でも元のsemantic ID/説明を保持。
- 画像のraw RGB差分を確認。boardは2,738 pixelsで物体の移動元/移動先だけ、
  statusは7,624 pixelsでLYRAのRESPONSE cellだけ、tableは579 pixelsで
  BIRCHのQUALITY数値/barだけが変化。relayoutの生成元事実と正解はbaseと一致。
- production renderer/tokenizerでE2B/E4B BF16/Q4の全27条件を検証。
  prompt IDsとanswer IDsが全4 GGUFで一致し回答境界検証に成功。
  `(id, token_ids, answer_ids)`配列のcompact JSON SHA-256:
  `38d404027731a10bb1790f636681d95f89352417fa75fa1e2eb012eb6e1be339`。
  基本prompt tokensはboard 149 / status 192 / table 162（画像placeholder込み）。
  reader/rendererがexpected ID/reasonやmanifestの事実をpromptへ入れないことも確認。
- production decoder/preprocessorを、local E2B/E4B mmprojのmetadataから得た設定で実行。
  全画像でprepared size 960×720、2,700 patches、300 visual tokens。
  path入力とencoded-byte入力の正規化tensorは完全一致し、全値finite。
  基本promptの画像splice後の計算上のpositionsは448 / 491 / 461。
  この入力準備検証ではencoder/Prefillを実行しておらず、これらを実測latency/accuracyとは扱わない。

fixture準備時のproduction sourceは`4afcb7c449987dfcf5693659c53f900b96f61d08`、
ggmlは`456172ec733a135778adcd32d00e576a58232e45`。runtime source変更なし。
この時点の作成・入力検証ではモデル正解率・marginを測っていない。後続のCPU screenを以下に記録する。
画像が複雑でもnear-tieの保証にはならない。次のA/Bは同じvisual embeddingsで
text重み差を分離し、encoderを含むend-to-end比較は別に行う。

### E4B Q4/CPU semantic screening (2026-10-01)

追加した27 request rowsを、current `branchscore-bench`で一括・逐次評価した。
`--backend cpu`を明示し、buildはCUDA/Vulkanとも無効。実行中のGPU使用量は0 MiBだった。
E4B Q4_K_M text weightsとE4B F16 mmprojを使い、warmupなしの1回実行。
text pathはF32 KV・normal attention・full-length SWA・512 microbatch、vision pathはflash。
これはCPU基準経路のsemantic screenで、BF16/Q4 A/B、llama.cpp reference比較、optimized C/Dではない。

| 系統 | base | changed | relayout | 正解合計 |
|---|---:|---:|---:|---:|
| board | 0/3 | 3/3 | 0/3 | 3/9 |
| status | 2/3 | 0/3 | 3/3 | 5/9 |
| table | 1/3 | 3/3 | 1/3 | 5/9 |
| 合計 | 3/9 | 6/9 | 4/9 | 13/27 |

- 正解は13/27。boardは基本/relayoutの全候補順で`north`を選び、正解`east`を外した。
  statusは証拠変更後が0/3正解。tableは証拠変更後が3/3正解だが、base/relayoutでは
  reversed/distractorsに弱く、semantic selectionが揺れた。
- 候補順3種すべてで予測IDが一致した画像条件は5/9。
  base→changedで予測IDが変わった比較は2/9（board 0/3、status 1/3、table 1/3）。
  base→relayoutで予測IDが保たれた比較は6/9（board 3/3、status 2/3、table 1/3）。
- 最小top-two logit marginは`vision-board-relayout-distractors`の
  `0.0259361267`。このfixtureはmarginを見て作成・選定していない。
- evaluate区間は合計756.228秒、request latencyはmedian 28.217秒 / p95 29.813秒。
  単一CPU runの診断値であり、model load・warmup・出力書き込みを含むlatency benchmarkではない。

27行は3系統・6証拠シナリオと候補順/配置変形からなる相関データで、独立27標本とは数えない。
全候補のraw score/probability、prompt token IDs、margin、個別時間とaggregateは
[CPU結果JSONL](../phase-4-plus-option-scale-2-vision-e4b-q4-cpu-2026-10-01.jsonl)に保存した。
入力SHA-256は`a98647965faa3d53db0d0da08ea0ac2be89ce74e1fed0b174565d99eba02416c`、
E4B Q4 GGUF SHA-256は`519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d`、
mmproj SHA-256は`ddf46c21d7078e95338cfc22306b19b276a29a5ad089023449dd54d4b6170a51`。
実行revisionはbranchscore `36e47a94cdc961dbef0d89bbc126ec0dcb552b35`、
ggml `456172ec733a135778adcd32d00e576a58232e45`。Step 4は未完了。

<a id="vision-multi-model-cpu-comparison-2026-10-01"></a>
### BF16 / E2Bのformat comparison (2026-10-01)

同じvision JSONL 27行を、E4B BF16、E2B BF16、E2B Q4_K_MでもそれぞれCPU逐次実行した。
E4B高精度text GGUFは手元の`gemma-4-E4B-it-BF16.gguf`（BF16）を指す。
各format内で入力JSONL・prompt/answer token IDs・candidate orderを揃え、warmup 0、`--backend cpu`。
4 runともtextはF32 KV・normal attention・full-length SWA・512 microbatch、vision pathはflash。
buildはCUDA/Vulkan無効で、GPU使用量は0 MiB。
全4 GGUFでprompt token IDsとanswer token IDsが全27行で完全一致した。
E4B pairは同一E4B F16 mmproj、E2B pairは同一E2B F16 mmprojを使用。
画像embeddingは各run内で再計算しており、BF16/Q4間で同じtensorを直接受け渡すstrict shared-embedding比較ではない。
そのため以下はCPU end-to-end screeningの比較値とする。

| Text GGUF | board | status | table | base | changed | relayout | 正解合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E4B BF16 | 3/9 | 5/9 | 5/9 | 4/9 | 6/9 | 3/9 | 13/27 |
| E4B Q4_K_M | 3/9 | 5/9 | 5/9 | 3/9 | 6/9 | 4/9 | 13/27 |
| E2B BF16 | 4/9 | 4/9 | 0/9 | 3/9 | 2/9 | 3/9 | 8/27 |
| E2B Q4_K_M | 4/9 | 2/9 | 0/9 | 2/9 | 3/9 | 1/9 | 6/27 |

| Text GGUF | candidate order不変 | base→changedで選択切替 | base→relayoutで選択維持 | 最小top-two margin |
|---|---:|---:|---:|---|
| E4B BF16 | 5/9 | 4/9 | 4/9 | 0.301872 (`vision-table-changed-reversed`) |
| E4B Q4_K_M | 5/9 | 2/9 | 6/9 | 0.025936 (`vision-board-relayout-distractors`) |
| E2B BF16 | 0/9 | 4/9 | 6/9 | 0.157578 (`vision-status-base-reversed`) |
| E2B Q4_K_M | 2/9 | 3/9 | 6/9 | 0.139041 (`vision-table-base-distractors`) |

候補順不変は、同じ画像でbase/reversed/distractorsの3条件が同じsemantic IDを選んだ画像数。
証拠変更の選択切替と配置変更の選択維持は各task family内3候補variantずつ、計9比較。
全27 request rowsは3系統・6証拠シナリオから作った相関変形で、独立標本ではない。

同じmodel family内のBF16対Q4_K_Mで、予測IDが変わったのはE4B 6/27、E2B 8/27。
BF16→Q4の向きで、E4Bは正解→誤答2件、誤答→正解2件（残る2 flipは双方誤答）。
E2Bは正解→誤答3件、誤答→正解1件（残る4 flipは双方誤答）。
各request内で候補raw logitsの平均を引いた後のBF16/Q4差は、平均絶対値がE4B 1.218 / E2B 1.081、
最大絶対値がE4B 4.514 / E2B 5.864。候補softmaxのtotal variation平均はE4B 0.209 / E2B 0.325。
確率は候補集合に条件付けされた値で校正confidenceではない。

この小画像セットではE4B両formatの正解件数は同じ13/27だが誤答fixtureが異なる。
E2B BF16は8/27、Q4は6/27で、どの結果も一般的なaccuracy保証にしない。
E4Bの公開Q4は5月版、BF16は7月版と記録されており、元checkpointの完全一致も立証されていない。
よってpaired差を量子化だけの因果差とは断定しない。E2B/E4Bのmodel間差も別に扱う。
strict shared-embedding A/Bとllama.cpp reference比較、optimized C/D、CUDAは未実施。
Step 4は未完了のままとする。

各出力には全candidate raw score/probability、prompt token IDs、正解照合、margin、timing、aggregateを含む。
[E4B BF16](../phase-4-plus-option-scale-2-vision-e4b-bf16-cpu-2026-10-01.jsonl)、
[E4B Q4_K_M](../phase-4-plus-option-scale-2-vision-e4b-q4-cpu-2026-10-01.jsonl)、
[E2B BF16](../phase-4-plus-option-scale-2-vision-e2b-bf16-cpu-2026-10-01.jsonl)、
[E2B Q4_K_M](../phase-4-plus-option-scale-2-vision-e2b-q4-cpu-2026-10-01.jsonl)。
model SHA-256は上記GGUF inventory、fixture hashは本記録の直前のsection、
mmproj hashはE2B `140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa`、
E4B `ddf46c21d7078e95338cfc22306b19b276a29a5ad089023449dd54d4b6170a51`。
全runでbranchscore `36e47a94cdc961dbef0d89bbc126ec0dcb552b35`、
ggml `456172ec733a135778adcd32d00e576a58232e45`。CPU時間はJSONLに診断値として保存し、評価主眼にはしない。

<a id="vision-cpu-interim-review-2026-10-01"></a>
## 2026-10-01: structured vision CPU結果の再レビュー

既存の4 run / 108 decision rowsを独立に再集計した。新しいモデル推論は行っていない。
全runについてfixture/manifest hash、27 request IDsの対応、候補ID/説明/順序、
正解ID、raw scoreのfinite性、argmaxとselected ID、温度1 softmax、top-two margin、
正解数を照合し、全4 GGUFのprompt/answer token IDs一致も再確認した。
上記の正解数・flip数・centered差・TV平均は再計算と一致。
E2Bのformat比較説明はBF16→Q4の変化方向が逆になっていたため訂正した。

### 同じ正解数でも、同じ判断ではない

| CPU screen | 正解 | BF16→Q4選択変更 | 正解→誤答 | 誤答→正解 | 双方誤答で選択変更 |
|---|---:|---:|---:|---:|---:|
| E4B BF16 / Q4 | 13/27 / 13/27 | 6/27 | 2 | 2 | 2 |
| E2B BF16 / Q4 | 8/27 / 6/27 | 8/27 | 3 | 1 | 4 |

E4Bの13/27一致は、2件の悪化と2件の改善が相殺した結果。
例として`vision-table-base`はBF16の`aster`誤答からQ4の`birch`正解へ変わるが、
同じ画像の`vision-table-base-reversed`はBF16の`birch`正解からQ4の`aster`誤答へ変わる。
従って「BF16なら常に正解」「同率なのでQ4の影響なし」のどちらも支持しない。
これらは現行branchscore end-to-endのformat間差で、reference基準A→Bや量子化だけの因果差ではない。

### 選択が動く／保たれるだけでは証拠理解を証明しない

各familyと候補variantを一組にした9比較について、単なる選択IDの変化/維持に加え、
変更前後の両方が正解だった組を数えた。分母9は相関する比較で独立標本ではない。

| CPU screen | 証拠変更で前後とも正解 | 配置変更で前後とも正解 | 候補逆順で選択維持 | 候補追加で選択維持 |
|---|---:|---:|---:|---:|
| E4B BF16 | 1/9 | 2/9 | 6/9 | 7/9 |
| E4B Q4 | 1/9 | 3/9 | 6/9 | 6/9 |
| E2B BF16 | 0/9 | 2/9 | 5/9 | 1/9 |
| E2B Q4 | 0/9 | 1/9 | 4/9 | 2/9 |

E4Bのboardは両formatともbase/changedの全候補順で`north`を選ぶ。
changedの3/3正解は、baseの誤答を保持したまま正解側が`north`へ変わった結果で、
移動した四角を認識して判断を切り替えた証拠にはならない。
statusのchangedはE4B両formatとも0/3で、ACK済みの`orion`や`lyra`を選ぶ。
tableはE2B両formatとも0/9で、QUALITY 79の不適格`dahlia`を多く選ぶ。
応答状態/品質閾値の読み取りまたはルール適用が弱い可能性があるが、
現在の選択結果だけでは画像認識・論理適用・prompt/ラベル対応・実装差のどこが原因か分離できない。

以前の単色画像を含むquality setのE4B Q4 24/24と、今回の13/27は別の入力集合。
同じfixtureでの時系列悪化を示すものではなく、簡単なcontrolでは見えなかった弱点が表れた。
3系統の小さい人工画像だけなので、実運用の一般的accuracyを13/27と推定しない。

### 数値stressと意味品質は別に読む

E4B Q4の`vision-board-relayout-distractors`で観測marginは0.025936。
上位2候補は`north`と`south`で双方誤答、正解`east`は4位、winnerとの差は3.895346。
fixture自体は推論前に固定しており、今回は観測後にnear-tieと分類した。
今後のC/D比較の数値stress controlとして有用だが、ここでwinnerが入れ替わっても
正解→誤答の追加を直接測れない。正解が上位で競るcaseも併用する。

最大候補softmaxが0.9以上でも誤答だった行はE4B BF16 7件、Q4 4件、
E2B BF16 5件、Q4 8件。0.9は今回の集計用閾値で採用基準ではない。
候補内確率やmarginだけを意味正解のconfidenceとして扱わない。

### 次に切り分ける順序とStep 4の判断

まずE4Bの少数失敗例で、現行prompt/候補順/GGUF/CPU条件を固定したllama.cpp照合を行う。
対象はboard-base、status-changed、table-base/base-reversedと、数値stress用の
board-relayout-distractorsを候補にする。同じvisual embeddingsを渡すtext Prefill照合と、
各encoderを動かすend-to-end照合を分ければ、既知のvision数値差とtext実装差を区別できる。
同じmmprojで再計算したことは不一致の証拠ではないが、今回embedding値の直接照合/hashは保存されていない。

次に、画像の事実を正確な文章にした少数controlで同じルール/候補を評価すると、
画像が必要な部分と文章だけのルール適用を分ける手掛かりになる。
これはまだ未作成・未実行の診断案で、画像fixtureの正解を書き換えるものではない。
referenceでも同様に失敗するなら現行直接回答契約でのモデル/入力の課題を検討し、
referenceだけ正解なら実装経路の差を優先して調べる。どちらも推論方式を自動変更する理由にはしない。

BF16でも複雑な画像判断が安定しないので、bit数だけで解消するという見通しは得られていない。
同時にreference未照合のため、モデル固有の能力不足と断定もしない。
今回の主な前進は、意味誤答・順序/候補追加の感度・自然に出たnear-tieを持つ固定controlが得られたこと。
数値最適化の品質保証には未到達で、strict reference A/B、C/D、CUDA、384/512候補の測定を残す。
Step 4は未完了を維持する。Step 2/3の実装・資源調査は進められるが、最終採用は別途評価する。
