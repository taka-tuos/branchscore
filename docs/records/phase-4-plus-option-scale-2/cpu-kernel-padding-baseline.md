# Phase 4+ option-scale-2: CPU kernel・key padding・修正後A/B

[記録索引](../phase-4-plus-option-scale-2.md) · [phase計画](../../phases/phase-4-plus-option-scale-2.md)

修正後契約のE4B BF16/Q4 CPU baseline、kernel/key軸診断、2026-10-02再レビュー。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

<a id="cpu-kernel-alignment-2026-10-01"></a>
## 2026-10-01: 追加レビュー対応 — CPU build/buffer/kernel照合と修正後27条件A/B

末尾の境界/buffer再レビューに従い、純text statusと境界修正後の画像5条件を先に調べた。
既存のllama.cpp buildを保持し、同じsource revisionから別のCPU reference buildを作成した。
Release、native CPU、OpenMP、CPU_REPACK compiled ON、BLAS/CUDA OFFを維持し、
`GGML_LLAMAFILE=OFF`へ揃えた。`use_extra_bufts`を独立に切り替え、実際のload logから
CPU_Mappedのみ/CPU_Mapped + CPU_REPACKを区別した。F32 K/V、通常attention、full SWA、
`n_batch=n_ubatch=512`、4 threads、生成なし。branchscore runtimeは追加変更していない。

### build flag以外に、ggml source/kernelも異なっていた

branchscoreのggmlは`456172ec733a135778adcd32d00e576a58232e45`（0.24.0）。
llama.cppは`19e28a27702117d8f2eb16b825b9a308111f67d9`のtree内にggml（0.25.3）を持ち、
独立submoduleの同一revisionではない。tracked upstream sourceにlocal差分はない。
従来manifestの単一`ggml_revision`はproject側のrevisionであり、reference側のsource一致を
意味しない。新manifestは両者の出所とCPU source file hashを分けた。

upstream `ggml-cpu.c` / `tiled/tiled.cpp`を調べると、新版には通常bufferのQ4_K/Q6_Kなどにも
使われるtiled matmulがあり、`GGML_CPU_TILED_MM=0`で通常vec_dot経路へ戻せる。
旧project側のIQ panel type listはIQ型のみであり、このGGUFのQ4_K/Q6_K行列には使われない。
`GGML_NO_IQ_PANEL=1`もcontrol環境に記録したが、この型の差を消すswitchではない。
REPACKのcompile有無、実buffer、LLAMAFILE、tiled経路はそれぞれ別の条件である。
新版ggmlへのproject移行や、production defaultの変更は行っていない。

### 純text status: build/kernel control

| reference経路 | winner | vega−lyra | branchscoreとのcentered最大差 | 最大確率差 |
|---|---|---:|---:|---:|
| LLAMAFILE ON、REPACK ON、tiled既定 | lyra（誤） | −0.567751 | 0.706434 | 0.290103 |
| LLAMAFILE ON、REPACK OFF、tiled既定 | vega（正） | +1.300638 | 0.422551 | 0.124403 |
| LLAMAFILE OFF、REPACK ON、tiled既定 | vega（正） | +2.283419 | 0.910229 | 0.237396 |
| LLAMAFILE OFF、REPACK OFF、tiled既定 | vega（正） | +3.514721 | 1.790469 | 0.352653 |
| LLAMAFILE OFF、REPACK OFF、tiled OFF | vega（正） | +0.755550 | 0.249105 | 0.035813 |

LLAMAFILEだけをOFFへ揃えても数値差は減らなかった。同じ新build・同じ通常bufferのまま
tiledだけをOFFにするとcentered差1.790469→0.249105へ縮小した。
REPACKが常に悪い、または一つの経路が数学的に正しいとの判断はしない。
今回も最終位置のみ/全位置出力の候補logitsはtiled OFFで一致し、最終layerのrow選択では
残る選択/数値差を説明できない。branchscoreは保存済み同入力の純text baselineを使った。

一時的なreference callbackとproject graph output保持によって、同じ253-position純textの
中間tensorを実測した。入力embedding、per-layer projection/input、layer 0のattn_normは
tiled既定/OFFのどちらでもbyte一致。既定tiledでは最初のQ投影に最大6.103516e−5、
relative RMS 1.444544e−7の差があり、tiled OFFではQ投影・Q正規化・RoPE後までbyte一致した。
その先のlayer出力には差が残り、layer 0の最大差は0.018511→0.000073314へ縮小した。
最終layer全位置出力のrelative RMS差は0.040856→0.014062。
入力token/embedding違いに帰する差ではなく、kernel差の影響を中間tensorでも確認できた。

traceはgraph分割やtensor保持を変える診断である。通常の候補scoresを別途照合し、
branchscore traceはbaselineと完全一致、tiled既定reference traceは最大1.907349e−6、
tiled OFFのreference traceは通常runと完全一致だった。最終位置の候補差と全位置tensorの
最大差は別物として記録する。中間tensor全体を保存するproduction機能は追加していない。

### 境界あり画像5条件: board-baseの不一致はbuffer/buildだけでは消えない

LLAMAFILE ON/OFF × REPACK ON/OFFの4経路で、5条件のreference winnerは同じだった。
さらにtiled OFFでも同じwinnerを保持した。board-baseはすべて`east`正解で、
branchscoreの`north`誤答と不一致。残る4条件はすべてbranchscoreと選択一致する。
既存5条件と今回再生成した同じ画像のvisual embeddingsはbyte一致した。

| tiled OFF・通常buffer・LLAMAFILE OFF | reference winner | centered最大差 | 最大確率差 |
|---|---|---:|---:|
| board-base | east（正） | 0.265978 | 0.073517 |
| status-changed | orion（誤） | 0.243405 | 0.002398 |
| table-base | aster（誤） | 0.537682 | 0.215771 |
| table-base-reversed | elm（誤） | 0.646164 | 0.089305 |
| board-relayout-distractors | north（誤） | 0.699308 | 0.165468 |

境界・論理token/位置・per-layer画像IDは揃っているが、referenceはtext/image/textの
3 decode、branchscoreは混在した450–493 positionsの1 chunkである。
純textもreferenceはkey軸のpaddingを持ち、projectは論理cache長そのまま。
同じchunk上限を指定してもgraph shapeが同一になるわけではない。
この比較を、実装差が解消または許容されたとの判定には使わない。

### key軸paddingの診断: statusと画像5条件でreferenceと一致

upstream `llama-kv-cache.cpp::get_n_kv`は、使用位置数を最低256 positions単位に
切り上げ、物理cache容量で切り詰めたkey軸を使う。context/cache容量は32 positionsへ
切り上げられ、padding KVは初期化される。projectの既存F32経路は論理使用長そのまま。
この差を調べるため、production sourceから分けた一時的なPrefill copyで、cache容量を
referenceのcontext slack込みで32単位へ、attention key軸を同じ規則へ揃えた。
paddingはF32のゼロで初期化し、既存causal maskで遮断した。token IDs・論理位置・
visual splice・per-layer IDs・cache型・重みは変えていない。

純text statusは253 logical positions / 256 physical key positionsとして実行した。
新build・通常buffer・tiled OFFのreferenceと、候補logitsがF32値で完全一致した。
追跡した213個の中間tensorもすべてbyte一致（全42 layerの選択したtensorを含む）。
したがって、この条件でtiledを除いた後に残ったcentered差0.249105は、key軸形状を
揃えるcontrolで解消した。最終layerのrow選択ではなく、attentionの物理shapeの差に
数値差を追跡できた。数学的に同じmasked paddingでもCPU reductionの結果は変わり得る。

画像5条件にも同じ一時copyを適用した。全候補logitsは同一embedding referenceと5/5で
F32値として完全一致し、winnerも5/5一致した。board-baseはnorth誤答→east正解へ変化。
残り4条件の誤答winnerはそのままである。referenceのtext/image/text分割とprojectの
混在chunkはなお異なるが、この5条件の最終候補差はbuffer/kernel/key軸を揃えることで消えた。
referenceの9有効桁logit出力をF32へ戻して比較し、文字列丸め差を数値差に数えない。

これは診断用のshape controlで、productionのcache容量や既定kernelを変更したものではない。
修正後27条件のbranchscore 12/27は既存の非padding経路の測定であり、診断5条件の結果で
全27条件の結果を置き換えない。特に523 positions / 2 chunkの条件、長文・window境界、
CUDAやStep 2のF16/FA paddingに一般化しない。診断patchと6条件raw結果は
[key padding control記録](../phase-4-plus-option-scale-2-key-padding-diagnostic-e4b-q4-cpu-2026-10-01.jsonl)へ保存した。


### 修正後27条件: 経路を固定したshared-embedding BF16/Q4 reference

入力契約とreference経路を固定し、全27条件をQ4→BF16の順に逐次実行した。
採用したreference経路は通常buffer、LLAMAFILE OFF、tiled OFF、F32 KV、通常attention、
full SWA、512-position上限、4 threads。これはprojectの通常CPU経路への照合条件として
選んだものであり、正解数による経路選別ではない。
9画像をbranchscoreのE4B mmprojで一度ずつencodeし、BF16/Q4へ同じ300×2560 F32値を渡した。
修正後prompt IDs・回答IDs・rendered promptは両GGUFで全27条件完全一致した。
referenceは全条件3 decode、projectは450–523 positionsを1/2 chunkで処理する。
statusのdistractors 3条件のみ512+11 positionsであり、chunk構成差は分離済みではない。

| 同一画像境界契約・同一embedding | BF16 reference | Q4 reference | branchscore Q4 |
|---|---:|---:|---:|
| 正解 | 13/27 | 13/27 | 12/27 |
| board | 3/9 | 4/9 | 3/9 |
| status | 6/9 | 6/9 | 6/9 |
| table | 4/9 | 3/9 | 3/9 |

BF16→Q4のwinner変更は3/27、正解→誤答1/27、誤答→正解1/27、双方誤答の変更1/27。
board-baseはnorth誤→east正、table-base-reversedはbirch正→elm誤、
status-changed-reversedはorion誤→lyra誤となった。総正解数だけでは差を相殺してしまう。
Q4 reference→branchscoreはwinner変更1/27、正解→誤答1/27、誤答→正解0/27。
その1件はboard-baseのeast正→north誤で、画像5条件で残った不一致が全27条件でも残った。
先行5条件の同一reference経路は全候補logit・headerが再現した。

BF16/Q4のcentered最大差のfixture別範囲は0.754721–5.473963、最大確率差は
0.002502–0.673995。Q4 reference/branchscoreはそれぞれ0.062317–0.918849、
0.000267–0.215771。centered差がA/Bより小さいのは27/27、
確率差が小さいのは25/27。追加誤答が1件残るため、
数値差がA/Bより小さい条件が多いことだけでruntime差の許容/採用とはしない。


これらは同じ9画像を使う相関した27条件。旧境界なしBF16/E2B screeningと新Q4の比較ではなく、
今回の修正後契約・同一embedding・同一reference経路のE4B BF16/Q4比較である。
公開GGUFのcheckpoint/世代差の確認限界は引き続き残り、純粋な量子化だけの因果効果と断定しない。
E2Bの修正後reference基準は未取得。CPU結果をCUDAの許容差へ転用しない。

[build/buffer/kernel controls](../phase-4-plus-option-scale-2-cpu-build-buffer-controls-2026-10-01.jsonl)、
[status中間tensor比較](../phase-4-plus-option-scale-2-status-text-trace-e4b-q4-cpu-2026-10-01.jsonl)、
[27条件shared-embedding reference raw/comparison](../phase-4-plus-option-scale-2-vision-shared-reference-e4b-cpu-2026-10-01.jsonl)、
[測定sourceと再現条件](../phase4-step4-cpu-reference/README.md)へ保存した。
全候補logits、margin、selected ID、正解変化、centered/確率差、入力/embedding/source hashを残した。
CPUのみでモデル推論を逐次実行。全27条件のtoken/回答ID一致、全9 embedding shape、先行5 embeddingのbyte一致、
先行5 reference再現、trace有無の候補差、診断6条件のF32一致を確認した。
新規JSONLのargmax・正解・margin・差の再計算、source hash、diff whitespaceを確認した。
通常runtimeのsourceは今回追加変更しておらず、追加のCTestは実行していない。
Step 4は未完了。残る少数例のchunk/graph差、E2B修正後基準、最適化B/C/D、
CUDA品質・384/512候補・VRAMと採用基準の確定は別途必要である。

<a id="cpu-kernel-padding-review-2026-10-02"></a>
## 2026-10-02: CPU kernel/key paddingと修正後A/Bの再レビュー

前節の追加検証を、生ログ・診断patch・現行Prefillとupstream cacheのsourceから再レビューした。
27条件のargmax、正解、margin、centered/確率差、family集計を再計算し、記録と一致。
修正後branchscore rawとのprompt identity/token IDs/全候補scores、画像hash、harness/patch hashを照合した。
5画像×5経路と純text×5経路の計30比較も再計算した。診断6条件の候補scoresは、別記録の
固定reference経路のscoresをF32へ戻した値と一致し、診断JSONL内の一致flagだけに依存しない。
新規推論・runtime変更・CTestは行っていない。

### 原因の切り分けと確認範囲

build flagを揃えるだけではvendor ggmlのsource/kernel差を除けないこと、tiled matmulを
無効にすると純textの最初のQ投影差が消えること、その後の残差はkey軸形状を揃えると
解消することを確認できた。診断patchはpadding KVをゼロ初期化し、現行causal maskの
`key > absolute_query`で未使用位置を遮断する。論理位置や画像境界を変更していない。
このCPU条件のstatusと画像5条件では、kernel/key軸のcontrolによって残差を説明できる。
3 decode対1 chunkという分割差があっても、画像5条件の最終候補は一致した。
したがって、この5条件について分割差だけを引き続き未解決原因として列挙する必要はない。

statusの213 tensorは記録上すべて同shape・同hash・差ゼロ。現在残るreference実dump
213個（計546,641,920 bytes）のshape/サイズ/hashも記録と一致した。ただしprojectの
`/tmp/step4-trace-branch-padded`は後続画像runで253→463 positionsのdumpへ上書きされている。
今回のレビューではstatus両側の実dumpを再比較できていない。213個のbyte一致は保存済み
比較記録に基づく結論と区別する。次のtraceはrun別出力先にすると再検証しやすい。
またbuild control manifestの`build_flags`は3 buildとも空objectであり、READMEの設定と
compile flagsのhashはあるものの、manifest単体で実際の全flagを復元できない。
次の測定では検証済みcache/compile flagsの値も保存する。

一致を確認したのは純text 253 positionsと画像450–493 positionsの診断条件。
productionへのpadding採用、全27条件のpadding版、523 positions / 2 chunk、長文/window境界、
CUDA・F16/FAの一致を実測したことにはしない。現行production Q4の測定値は12/27のまま。

### 判断品質と量子化の考察

修正後・同一embedding・固定reference経路のBF16/Q4は各13/27。
winner変更3件のうち正解→誤答と誤答→正解が各1件あり、総正解数では相殺される。
Q4の無劣化を示す結果ではない。reference Q4→現行branchscoreの追加誤答はboard-baseの1件で、
診断paddingでこの例の選択差を解消したことと、全27条件で解消したことは区別する。
centered差がA/Bより小さい27/27、確率差25/27も再確認した。
確率差の例外はboard-relayout-distractorsとstatus-base。単一の差閾値だけで採用判定しない。

証拠変更前後を双方正解した組は、BF16 reference 0/9、Q4 reference 1/9、
現行branchscore Q4 0/9。base/relayoutで選択が同じ組はそれぞれ6/9、6/9、7/9だった。
入力9画像から派生した相関のある27条件で、一般的な正答率や統計的な量子化劣化率を
推定できる規模ではない。それでも、reference BF16にも誤答14件と証拠変更への弱さが残るため、
kernel差を解消するだけでこの画像タスクの実用品質が得られるとは考えない。
この限定的なdirect-answer契約・画像入力経路での品質として評価し、モデル一般の能力へ拡張しない。
今回のreferenceもproject生成embeddingを共有するため、残る共通誤答の原因はvision処理・
視認・direct-answer契約・モデル判断を分離していない。同内容の既存text controlsとの照合を
品質の切り分けに使い、reference BF16の誤答だけからvision処理の正しさを認定しない。
公開GGUFのcheckpoint/量子化recipeの同一性に関する既存留保も継続する。

### 次の確認

CPU差の探索は今回の診断条件で一旦区切れる。次はpaddingを採用候補として、修正後全27条件と
512付近の1/2 chunk境界・window境界をfocused比較し、追加誤答・margin/確率差と
cacheゼロ初期化/容量増加の時間・メモリ差を記録する。bit一致を全backendの採用条件にはしない。
その結果と品質基準を固定して最適化B/C/Dの評価へ進む。E2B修正後基準とCUDAの品質・資源評価、
384/512候補への拡張は引き続き別測定が必要で、Step 4完了判定には至らない。
