# Phase 4+ option-scale-2: 画像境界修正・CPU buffer control

[記録索引](../phase-4-plus-option-scale-2.md) · [phase計画](../../phases/phase-4-plus-option-scale-2.md)

画像境界の契約是正、修正後27条件、CPU bufferと出力行選択のcontrol。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

<a id="vision-boundary-fix-2026-10-01"></a>
## 2026-10-01: 再レビュー対応 — 画像境界tokenの保持

末尾のreference/control再レビューを受け、rendererの画像表現を
`<|image><|image|><image|>`へ修正した。engineは境界がそれぞれ独立tokenとして
中央markerの直前/直後にあることを検証し、中央markerだけをvisual embeddingsへ置換する。
境界は通常textとしてPrefillへ渡るため、token embeddingとper-layer token IDの双方で
255999 / 258882を保持する。visual位置は従来どおりper-layer ID 0を用いる。
Prefillのabsolute position、mask、shared KV、512-position microbatchの実装は既存のまま。

境界をrendered textに明示したことで、保存prompt IDsとSHA-256 identityも実際の入力変更を
反映する。text-only promptは変更されない。renderer IDは契約是正として
`gemma4-categorical-v1`を継続し、画像の旧測定と新測定はprompt hashおよび記録で区別する。
今回の境界2tokenをvisual token数へ加算しない。300 visual tokensの場合、
実処理位置数は新rendered prompt token数−1+300となる。

local llama.cpp revision `19e28a27702117d8f2eb16b825b9a308111f67d9`の
`tools/mtmd/mtmd.cpp`、`tools/mtmd/mtmd-helper.cpp`、`src/models/gemma4.cpp`を確認した。
画像はcausal、mRoPEなし、raw embeddingをtext embedding scaleで再scaleしない。
画像境界の文字列はmtmdのGemma4Vと一致する。

固定27条件すべてで、新token IDsが旧列へ境界2tokenだけを追加したものと一致し、
回答token IDsが変わらないことを確認した。まず選択5条件で修正branchscoreを実行し、
同じ保存300×2560 embedding・同じmanual decoderで境界ありreferenceを実行した。
referenceはF32 K/V、通常attention、full SWA、n_ubatch=512、CPUのみ、生成なし。
manual decoderの境界あり全候補logitは、前回の通常mtmdへ同じembeddingを注入した
全候補logitと5/5で一致した。旧manual decoderと境界ありmanual decoderの比較では
winnerが4/5変化し、helperの違いを固定して境界有無の影響を確認できた。

| 条件 | 境界なしreference | 境界ありreference | 修正branchscore |
|---|---|---|---|
| board-base | north | east | north |
| status-changed | orion | orion | orion |
| table-base | birch | aster | aster |
| table-base-reversed | aster | elm | elm |
| board-relayout-distractors | south | north | north |

修正branchscoreと境界ありreferenceのwinnerは4/5一致。board-baseのbranchscoreは
north/east差0.012167で誤答、referenceはeast/north差0.538445で正解だった。
境界保持を正解数を上げるprompt選別とは扱わず、入力契約の是正として採用する。
CPU上の数値差と画像fixtureの難しさは別途残る。


### 固定27条件の修正前後比較

E4B Q4/CPUの27条件を修正後に再実行した。正解は13/27→12/27、winner変更4/27、
正解→誤答2/27、誤答→正解1/27。同じ9画像を使う相関した条件であり、一般accuracyではない。
先行5条件と27条件内の同じ5行は、prompt IDs・winner・全候補logitsが完全一致した。

| 条件 | 正解ID | 修正前 | 修正後 |
|---|---|---|---|
| status-base-distractors | lyra | orion（誤） | lyra（正） |
| table-base | birch | birch（正） | aster（誤） |
| table-base-reversed | birch | aster（誤） | elm（誤） |
| table-changed-reversed | aster | aster（正） | elm（誤） |

[修正後raw JSONL](../phase-4-plus-option-scale-2-vision-boundary-e4b-q4-cpu-2026-10-01.jsonl)と
[27条件の前後比較・5条件reference](../phase-4-plus-option-scale-2-vision-boundary-comparison-e4b-q4-cpu-2026-10-01.jsonl)
に全候補logits、margin、raw/centered/probability差、prompt/embedding hashを保存した。
位置とper-layer ID hashは`src/prefill_engine.cpp`のspliceから導いた期待値として明記し、
内部tensorの実測dumpとは扱わない。manual referenceの実消費position数はheaderと照合した。
画像encoderは今回変更していない。全27条件のreference A/Bは未実施で、今回はbranchscoreの
入力契約修正前後の比較と、5条件の同一embedding reference照合を行った。

### 純text status control: REPACKと最終layer出力位置の切り分け

同じE4B Q4 GGUF・253 prompt IDs・回答ID・CPUで、F32 K/V、通常attention、full SWA、
`n_batch=n_ubatch=512`、一つのchunkを固定した。画像と生成はない。
llama.cppの`use_extra_bufts`と、最終位置だけ/全位置のlogits出力指定を2×2で比較した。
後者は`gemma4.cpp`の最終layerの行選択を変えるcontrolである。

| llama.cpp weight buffer | 出力位置 | winner | vega−lyra | branchscoreとのcentered最大差 | 最大確率差 |
|---|---|---|---:|---:|---:|
| CPU_Mapped + CPU_REPACK | 最終位置 | lyra（誤） | −0.567751 | 0.706434 | 0.290103 |
| CPU_Mapped + CPU_REPACK | 全位置 | lyra（誤） | −0.567751 | 0.706434 | 0.290103 |
| CPU_Mapped | 最終位置 | vega（正） | +1.300638 | 0.422551 | 0.124403 |
| CPU_Mapped | 全位置 | vega（正） | +1.300638 | 0.422551 | 0.124403 |

REPACK有効時の再測定は前回の全候補logitsを再現した。REPACK無効時にbuffer logから
CPU_REPACKの使用が消えることも確認した。最終/全位置の差は有効時0、無効時最大
0.000001907なので、このstatus例のwinner差を最終layerの行選択では説明できない。
`use_extra_bufts=false`のみでwinnerが変わり、branchscoreと同じvegaになるため、
今回のstatus純text選択差にはCPU weight buffer/kernel経路が影響することを確認できた。
branchscoreは通常CPU bufferであり、比較baselineは画像変更に影響しない保存済みtext runを使用した。

ただし通常CPU buffer同士でもcentered差0.422551が残る。ggml buildではbranchscoreの
`GGML_LLAMAFILE=OFF`に対してlocal llama.cppはONであり、graph形状・kernelの完全一致を
確認したことにはならない。今回の切り分けでruntimeの数値処理は変更せず、差を許容済みともしない。
この一例からモデルのルール適用能力不足やCUDAの誤差を推定しない。
[status focused raw JSONL](../phase-4-plus-option-scale-2-status-text-buffer-control-e4b-q4-cpu-2026-10-01.jsonl)
へ2×2の全候補logits、全語彙logits hash、入力ID、設定と差を保存した。

build、renderer/tokenizer focused CTest 2/2、E2B tokenizer、E4B engine focused CTest 1/1、
JSONL整合性とdiff whitespaceを確認した。モデル測定はCPUで逐次実行し、GPUは使用していない。
Step 4は未完了。次は通常CPU bufferとggml build/kernel条件をさらに揃えた数値照合、
修正後画像契約での全27条件shared-embedding reference A/Bを行う。
最適化B/C/D、CUDA品質・384/512候補・VRAM実測も引き続き未完了。

<a id="vision-boundary-buffer-review-2026-10-01"></a>
## 2026-10-01: 境界修正とCPU buffer切り分けの再レビュー

今回のsource差分、27条件の境界前後raw/比較JSONL、5条件reference、statusの2×2 controlを
再確認した。新規モデル推論は行っていない。変更sourceのSHA-256が測定manifestと一致し、
全27行で旧prompt列への境界2token追加のみ、回答ID不変、token hash・argmax・正解・margin、
raw/centered/確率差の再計算が記録と一致した。位置/per-layer ID hashはsourceから導いた
期待値として再確認し、内部tensorの実測確認とは区別した。
status control runnerの保存source hashも一致し、全位置出力時にも
`llama_get_logits_ith(ctx, -1)`で最後の位置を読み出していた。

### 画像境界の問題は修正され、影響の因果分離も進んだ

rendererが`<|image><|image|><image|>`を出し、engineは左右の境界tokenを検証して
中央markerだけを置換する。境界はtext embeddingとper-layer IDsで保持される。
修正はupstream Gemma4Vと整合し、直接ggml・単一回答位置の採点契約を維持している。
今回は同じmanual decoderと同じ保存embeddingで境界有無を比較でき、選択4/5変化。
境界ありmanual decoderと従来mtmd共有embedding経路の全候補logitも5/5一致したため、
前回残った「helper差を含む」という交絡はこの5条件で解消した。

| E4B Q4 CPU | 修正前 | 修正後 |
|---|---:|---:|
| 全27条件の正解 | 13/27 | 12/27 |
| board正解 | 3/9 | 3/9 |
| status正解 | 5/9 | 6/9 |
| table正解 | 5/9 | 3/9 |
| 同じ画像の3候補variantで選択一致 | 5/9 | 5/9 |
| 証拠変更の前後が双方正解 | 1/9 | 0/9 |

winner変更は4/27、正解→誤答2件、誤答→正解1件、双方誤答での変更1件。
この小さい相関セットの正解数低下は、入力契約の是正を取り消す根拠にはしない。
同時に、この修正だけで意味判断の安定性が改善したとも言えない。
旧BF16/E2B画像screenは境界なし入力の測定で、修正後Q4 12/27と旧BF16 13/27を
同入力の量子化比較として並べてはならない。修正後の契約で基準を取り直す。

### 画像側にも、正誤を分けるreference差が残る

修正branchscoreと境界ありreferenceは4/5選択一致。
唯一の相違はboard-baseで、referenceは`east`正解、branchscoreは`north`誤答。
branchscoreのtop-two marginは0.012167で、今回は正解`east`と誤答`north`が競っている。
以前の両候補が誤答だったrelayoutのnear-tieに加え、追加誤答の評価に使える診断例を得た。
これは観測後の分類で、一般accuracyのholdoutとして選んだものではない。

| 境界あり入力、同一embedding | reference/branchscore centered最大差 | 最大確率差 | 選択一致 |
|---|---:|---:|:---:|
| board-base | 0.420477 | 0.133077 | 不一致（referenceのみ正解） |
| status-changed | 0.912992 | 0.017063 | 一致（双方誤答） |
| table-base | 0.821639 | 0.129944 | 一致（双方誤答） |
| table-base-reversed | 0.822993 | 0.197840 | 一致（双方誤答） |
| board-relayout-distractors | 0.613609 | 0.219578 | 一致（双方誤答） |

入力契約差を除いても数値差は残る。境界なしの旧A/B差との比率だけで許容とはしない。
以下のCPU buffer controlを画像5条件にも適用すると、board-baseの不一致の切り分けを進められる。

### statusの純text選択差はCPU buffer/kernel経路に追跡できた

同じGGUF・token IDs・F32 KV・通常attention・full SWA・一つのchunkで、
`use_extra_bufts`だけを変えるとreferenceは`lyra`誤答→`vega`正解へ変わった。
最終位置のみ/全位置出力ではwinnerが変わらず、候補logit差はREPACK有効時0、
無効時1.907349e−6。最終layerの行選択という仮説はこの一例では支持されなかった。

| CPU reference経路 | vega−lyra | branchscoreとの差（centered最大） |
|---|---:|---:|
| CPU_Mapped + CPU_REPACK | −0.567751 | 0.706434 |
| CPU_Mapped（extra buffers無効） | +1.300638 | 0.422551 |
| branchscore通常CPU buffer | +0.653730 | — |

このcontrolは、前回の純text誤答をモデルのルール適用能力不足の根拠にしないことを支持する。
同じモデルでもCPUのbuffer/kernel選択で正誤が変わる事例である。
ただしREPACKが常に品質を落とす、または一方のkernelが数学的に正しいという判定にはならない。
通常buffer同士でもcentered差0.422551・最大確率差0.124403が残る。

sourceではbranchscoreは選択deviceの通常bufferへ重みを確保し、referenceの
`use_extra_bufts`は追加buffer typeを候補にする。両buildで`GGML_CPU_REPACK=ON`でも、
実際にREPACKを使うかは別で、compile flagだけでは経路を揃えたことにならない。
さらにlocal buildはbranchscore `GGML_LLAMAFILE=OFF`、llama.cpp `ON`。
今回特定できたのはbuffer/kernel経路の影響までで、残る差の原因はまだ未分離。

### 次に優先する検証とStep 4の位置づけ

まずreferenceの通常CPU bufferとggml build条件をbranchscoreへ揃え、純text statusと
境界修正後の画像5条件を照合する。特にboard-baseでREPACK有無の影響を確認する。
同じcache型・backend名だけで同経路と扱わず、使用bufferとbuild flagを記録する。
その後も差が残ればchunk構成とgraph/kernelの差を少数例で追う。

この経路を固定してから、修正後27条件のshared-embedding reference BF16/Q4を取り直し、
最適化の追加差・追加誤答の基準を定める。以前の測定はその入力/経路の歴史的根拠として保持する。
境界の是正とCPU数値差の診断は進んだが、意味判断品質は未解決。
CPU一例のREPACK差をCUDAの許容差へ転用せず、B/C/D・CUDA・384/512候補・VRAMは別途測る。
Step 4は未完了を維持する。この再レビューではruntimeを追加変更していない。
