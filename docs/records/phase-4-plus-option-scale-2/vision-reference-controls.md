# Phase 4+ option-scale-2: 画像reference・文章control

[記録索引](../phase-4-plus-option-scale-2.md) · [phase計画](../../phases/phase-4-plus-option-scale-2.md)

同一embedding/別encoder referenceと文章controlによる切り分け。境界修正前。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

## 2026-10-01: E4B Q4 CPU referenceとtext control

考察末尾の次作業をCPUで実施した。対象は`vision-board-base`、
`vision-status-changed`、`vision-table-base`、`vision-table-base-reversed`、
`vision-board-relayout-distractors`の5条件。branchscore revision
`36e47a94cdc961dbef0d89bbc126ec0dcb552b35`、local llama.cpp revision
`19e28a27702117d8f2eb16b825b9a308111f67d9`、E4B Q4_K_MとE4B mmproj F16を使用した。
model/deviceはCPU固定、F32 K/V、Flash Attention disabled、full SWA、
`n_ubatch=512`、生成なし。CUDA/Vulkan/GPUは使っていない。

### 画像入力のreference照合

| 条件 | 正解 | branchscore画像E2E | llama.cpp同一埋め込み・token列を厳密維持 | llama.cpp mtmdへbranchscore埋め込みを注入 | llama.cpp mmproj画像E2E |
|---|---|---|---|---|---|
| board-base | east | north | north | east | east |
| status-changed | vega | orion | orion | orion | orion |
| table-base | birch | birch | birch | aster | aster |
| table-base-reversed | birch | aster | aster | elm | elm |
| board-relayout-distractors | east | north | south | north | south |

5件は相関する診断例で、accuracy推定には使わない。画像E2Eの正解数はbranchscore、
llama.cppとも1/5。top-1が両E2Eで一致したのはstatus-changedの1/5のみ。
同じbranchscore埋め込みを使う厳密token列のllama.cpp text prefillではwinnerが
branchscoreと4/5で一致し、正解は1/5だった。唯一winnerが異なるのは
`board-relayout-distractors`で、branchscoreはnorth、llama.cppはsouthを選んだ。

### E4B BF16/Q4 reference A/B

同じbefore/after token IDs、answer IDs、同一300×2560 branchscore float32 visual embeddingを
固定して、llama.cppのBF16とQ4 text GGUFを比較した。誤った81-token placeholder入力の
初回runは破棄し、embedding byte数から300 tokenを得るrunnerで全行を再実行した。
最終runはすべて`image_tokens=300`をassertしている。

| 条件 | Q4 winner | BF16 winner | Q4 / BF16正解 | Q4 / BF16 top-two margin |
|---|---|---|---|---:|
| board-base | north | north | 誤 / 誤 | 0.256966 / 2.113487 |
| status-changed | orion | orion | 誤 / 誤 | 4.876539 / 2.095234 |
| table-base | birch | aster | 正 / 誤 | 1.323273 / 3.791323 |
| table-base-reversed | aster | birch | 誤 / 正 | 0.425386 / 1.068028 |
| board-relayout-distractors | south | south | 誤 / 誤 | 0.355751 / 1.020647 |

両方1/5正解、winnerは3/5一致。`table-base`はQ4のみ正解、
`table-base-reversed`はBF16のみ正解となり、正解→誤答/誤答→正解が各1件だった。
これは固定入力でのlocal GGUF artifact A/Bである。由来照合で記録したBF16/Q4の
公開元世代差が残るため、結果を量子化だけに帰属しない。また、5行から許容差や
一般accuracyを定めない。候補ごとのraw/centered/probability差はreference/control JSONLに保存した。

llama.cpp mtmdで通常の画像chunkを作ると、branchscoreが置換する単独の`<|image|>`
tokenに加えて、テキスト側へtoken ID `255999`と`258882`が挿入される。
前後の通常テキストIDは全件でbranchscore保存列と一致した。通常mtmd経路で
branchscoreのfloat32画像embeddingへ差し替えたときと、llama.cpp mmprojで画像を
再計算したときのwinnerは4/5で一致し、`board-relayout-distractors`だけnorth/southで
逆転した（どちらも正解eastではない）。従ってmmprojの小さな数値差だけで全体差を
説明できず、画像境界tokenとtext prefill経路も独立に観測対象として残す。

4画像（各300×2560）のbranchscore/mmproj embeddingは平均cosine similarity
`0.999999860`、平均relative RMS difference `0.000529`だった。hashは異なるが近い値で、
この小標本のwinnerではnear-tieの1例だけencoder差し替えがwinnerを変えた。
`board-base`のmmproj winnerはeastで正解だがeast/north marginは`0.011448`。
near-tie stressの`board-relayout-distractors`はmmproj経路margin`0.079681`で、
north/southの誤答が接近した。これらのmarginは意味正解のconfidenceではない。

### 画像事実を文章化したcontrol

5行の[text control fixture](../../../fixtures/phase4-step4-vision-text-controls.jsonl)は、
同じ質問・候補ID/説明/順序に画像内の事実を文章で追記した。winnerはbranchscoreで
5/5正解、llama.cppのtext-only prefillで4/5正解だった。唯一のllama.cpp誤答は
status-changed controlで、期待vegaに対してlyraを選んだ。このためstatusの失敗を
画像抽出だけで説明できない。一方、両方のtext controlが正解したtable-baseでも
llama.cpp画像E2Eとbranchscore埋め込みを受けたllama.cppはasterを選び、画像経路に
関する差が残った。board-base controlは文章なら両者正解だが、branchscore画像入力は
誤答し、llama.cppは正解と次点のmarginが`0.011448`だった。relayout controlも文章なら
両者正解だが、画像では両経路とも誤答が近接した。小さい合成controlなので実画像性能へ一般化しない。

条件、各answer-slot logits、token/embedding hash、embedding統計、control結果は
[reference/control JSONL](../phase-4-plus-option-scale-2-vision-reference-controls-e4b-q4-cpu-2026-10-01.jsonl)
に保存した。branchscore側text-control全出力は
[別のraw JSONL](../phase-4-plus-option-scale-2-vision-text-controls-e4b-q4-cpu-2026-10-01.jsonl)。
通常mtmd経路と完全token列を使った共有embedding prefillは別条件であり、混同しない。
このCPU診断には選択5件のBF16/Q4 shared-embedding reference A/Bを含むが、
全27行のA/Bやoptimized B/C/D、CUDAでの品質・VRAM確認を置き換えない。
Step 4は未完了を維持する。

<a id="vision-reference-control-review-2026-10-01"></a>
## 2026-10-01: reference / text controlの再レビュー

追加した5条件の診断JSONL、5行の文章control、branchscore raw出力、local runnerを再確認した。
新規モデル推論は行っていない。画像/入力hash、元の27行との候補・質問・正解対応、
int32 little-endian token hash、全経路のargmax/正解/margin、BF16/Q4 centered差を再計算。
4異なる画像の保存済みfloat32 embeddingは全て300×2560、3,072,000 bytesで、
branchscore/mmproj双方のhashが記録と一致した。cosine/RMS差も再計算で整合した。
初回81-token runnerの出力を採用していないことは、最終raw headerの300 tokensとpositionsで確認した。

### 今回分離できた要因

| 比較 | 観測 | 解釈できる範囲 |
|---|---|---|
| branchscore vs llama.cpp、同じtoken列/300-token埋め込み | 選択4/5一致、両方1/5正解 | この5例の誤答はbranchscore独自のencoder差だけでは説明できない。数値一致の証明ではない |
| llama.cpp内、同じ埋め込みで境界なし→通常mtmd | 選択4/5変更、正解1/5→1/5 | 画像の入力列/実行経路が判断へ影響。正解数が同じでも同じ条件ではない |
| 通常mtmd、branchscore埋め込み→mmproj埋め込み | 選択4/5一致、残る1件は双方誤答 | encoderの微小差はnear-tieを動かすが、他の選択差の主因ではない |
| 同じ埋め込み/token列でreference BF16/Q4 | 両方1/5正解、選択3/5一致 | 画像encoderを固定してもformat間差と誤答が残る |
| 文章controlのbranchscore vs llama.cpp | 5/5 vs 4/5正解 | 文章なら概ね解けるが、statusで画像を含まない実装間の選択差も残る |

5条件は前回結果を見て診断対象として選んだもので、一般accuracyの評価用holdoutではない。
base/reversedは同じ画像を使い、独立5画像とも数えない。

### 優先して扱うのは画像境界の入力契約差

現行`src/gemma4_decision_engine.cpp`は`<|image|>` markerを除いて前後をsplitし、
`src/prefill_engine.cpp`はその間へraw visual embeddingsだけを挿入する。
一方、local upstream `tools/mtmd/mtmd.cpp`の`PROJECTOR_TYPE_GEMMA4V`は
`<|image>` / `<image|>`を前後に加える。今回のIDは255999 / 258882。
この境界tokenの存在は既存`docs/research/llama-ggml.md`にも記載されていたが、
現行runtimeの画像spliceに反映されていない。入力契約の実装差として扱うべき事項が具体化した。

たとえばboard-baseは同じbranchscore埋め込みでも、境界なしreferenceでは`north`誤答、
通常mtmdでは`east`正解になる。逆にtable-baseは`birch`正解から`aster`誤答になる。
従って境界追加を「正解数を上げるprompt調整」として選別せず、upstreamに合わせた
画像入力の互換性を先に確認する。元のtoken列と修正後の列の測定を区別する必要がある。
wrapper有無の比較はmanual decoderとmtmd helperも異なるので、4/5の選択変更全てを
境界token2個だけの因果効果とまでは確定しない。固定decoderで境界有無を比較するとさらに分離できる。
最終raw headerではmtmdの画像decodeもcausal、mRoPEなしで、E4Bの非causal化は今回の差ではない。

### 同じ入力で残る数値差も採用判断に必要

共通offsetを除いた最大候補logit差と候補softmaxの最大絶対差を再計算した。
左側は厳密な入力列を持つQ4 referenceとbranchscore、右側は同じreference経路のBF16/Q4。
後者の元GGUF世代差は引き続き未分離。画像のdecodeはreferenceが前/画像/後の3回、
branchscoreが最大512位置ずつなので、入力一致でもgraphの分割は一致していない。

| 条件 | reference/branchscore centered最大差 | 最大確率差 | BF16/Q4 centered最大差 | 最大確率差 |
|---|---:|---:|---:|---:|
| board-base | 0.324350 | 0.069530 | 1.649988 | 0.324590 |
| status-changed | 0.670218 | 0.003663 | 3.058331 | 0.101994 |
| table-base | 0.739127 | 0.134343 | 3.923496 | 0.767666 |
| table-base-reversed | 0.306633 | 0.027687 | 2.867343 | 0.611673 |
| board-relayout-distractors | 0.412975 | 0.094069 | 1.945558 | 0.167937 |

今回の5例では両指標ともformat間差より小さい。しかしrelayoutでは誤答winnerが変わり、
別の純text status controlでも選択が変わるので、小さい差をそのまま許容済みとはしない。
status文章controlはbranchscoreの`vega`対`lyra` gapが+0.653730、referenceでは−0.567751。
候補centered最大差0.706434、最大確率差0.290103で、正解/誤答の境界を跨いだ。
これは画像抽出とは独立した実装間の数値差の具体例であり、
referenceの誤答だけからモデルのルール適用能力不足を断定できない。

### 文章controlから言えることと次の確認

branchscoreは文章control5/5なので、この固定ルール・候補で正解を選べることは確認できた。
画像での失敗は、視覚からの事実取得・画像/text接続・それらを含むモデル挙動へ調査対象を絞れる。
ただし盤面の文章controlは各領域の赤い四角数を既に集計しており、
画像で必要な形/色識別と数え上げを省略する。OCRだけの診断や画像認識の完全分離ではない。
status/tableは行の値を転記したcontrolで、画像の配置や色等も含まない。

次は画像境界tokenを保持する小さいsplice修正を最優先の候補とし、
位置・per-layer IDs・prompt identity・visual token数と境界token数を分けた記録を確認する。
まずこの5条件で入力一致を検証し、その後に固定済み27条件の前後比較を行う。
画像を含まないstatusの数値差は別のfocused caseとして同じcache/attention/chunk条件で追う。
encoderの近さとBF16の結果だけで品質問題を閉じず、上記を終えてからモデル/入力難易度の限界を評価する。

今回のA/Bはshared embedding付きreferenceの5条件まで進んだ。全27条件のA/B、
最適化B/C/D、CUDA、384/512候補とVRAM評価は未完了。入力契約差の是正は
量子化差の物差しで許容する事項ではなく、Step 4の最終品質判断に先立って扱う。
このレビューではruntime変更を行わず、Step 4は未完了を維持する。
