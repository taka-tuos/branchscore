# Step 4 C/D CPU評価（2026-10-02）

[記録索引](../phase-4-plus-option-scale-2.md) · [phase計画](../../phases/phase-4-plus-option-scale-2.md)

## 結果を見る前に固定する基準

E4B修正後27画像条件のA/Bを物差しにする。CPU限定の暫定採用候補は
F16 K/V、host直接F16 mask、F32 Q/accumulation/output、Flash Attention、全長SWA、
512-token microbatch上限、256位置単位のphysical key padding（ゼロ初期化・causal mask遮断）。
Cはllama.cppの通常buffer・LLAMAFILE OFF・tiled OFF、Dは独立ggml測定copy。
通常runtimeは変更しない。short-SWAは測定対象に含めない。

品質条件はB→CとB→Dの正解→誤答が各0/27。数値条件は各fixtureでcentered最大差が
A→Bの25%以下、最大候補softmax差が `max(0.01, A→B差の25%)` 以下。
A→B centered差が0.04未満なら絶対0.01以下とする。C→Dはcentered差0.01以下、
最大確率差0.001以下、winner一致を要求する。B→C/B→Dのwinner変更はすべて個別記録する。
これは採用評価のための保守的なscreening基準であり、一般的な誤答率の保証ではない。
いずれか未達なら当該CPU経路を採用しない。mask/position/画像配置/shared KVの不具合は
数値基準とは別に修正対象とする。CUDAにはこの閾値を転用しない。

27条件は9画像から派生した相関条件。E2B、純text/16件、長文/window境界、384/512候補、
CUDAは別測定として残す。referenceはtext/image/textの3 decode、Dは混在1/2 chunk。
cache physical capacityとggml revision/kernel差も記録し、C→Dを純粋な同一kernel比較とは呼ばない。

## 照合したsourceと比較の限界

ローカルreferenceの `src/models/gemma4.cpp`、`src/llama-graph.cpp`、
`src/llama-kv-cache.cpp` と、projectのPrefill/StateCache、両treeのCPU `ops.cpp` を確認。
referenceはGemma 4でFlashを許可し、F16 K/V・F32 Q・F32 accumulator指定を使う。
指定とCPU kernel内部の実際の蓄積型の違いは後述の再レビューで確認した。
Dの各graph nodeは `ggml_backend_supports_op` でCPU対応を検証し、非対応なら停止する。
Q/K/Vはpermute `[head_dimension, key/query_positions, heads, 1]`、Flash出力は
`[value_dimension, heads, query_positions, 1]` として通常pathと同じFFN入力へreshapeする。
shared-KV mapping、absolute position、画像位置のper-layer IDは既存sourceを維持。

通常matmulの `GGML_CPU_TILED_MM=0` はFlash kernelのtiling無効化を意味しない。
両CPU sourceのFlashはquery数・head幅・SIMD条件でtiled/vector pathを選ぶ。
新版はx86で `DV % GGML_F32_EPR` の制限を外している。今回の差の因果原因をこの
source差と断定しない。Cの3 decodeとDの混在chunkのshape差も残る。
このC→Dはそれらを含む実装比較であり、同一ggml/kernel条件の純粋な差ではない。

Cのactual logは全27条件 `flash_attn = enabled`、通常CPU weight bufferを確認。
Dでは同じ保存済みF32 embeddingsをtensorへ上書きしてからPrefillを開始する。
初期VisualTokensの確保のためvisionを実行するが、その出力値は採点に使わず、
今回のPrefill時間にも含めない。モデル/GGUF由来の留保はA/B記録から継続する。

## 全27条件の結果と判断

E4B Q4_K_M、CPU、各条件1回、warmupなし。A/Bは修正後の保存済みreferenceを
F32へ戻して使用。C/Dは今回新規推論。正解数はA=13/27、B=13/27、C=14/27、D=13/27。

| 比較 | winner変更 | 正解→誤答 | 誤答→正解 | 事前screening通過 |
|---|---:|---:|---:|---:|
| A→B | 3/27 | 1/27 | 1/27 | 基準 |
| B→C | 1/27 | 0/27 | 1/27 | 12/27 |
| C→D | 1/27 | 1/27 | 0/27 | 0/27 |
| B→D | 0/27 | 0/27 | 0/27 | 11/27 |

| 比較 | centered最大差のfixture別範囲 | 最大確率差のfixture別範囲 |
|---|---:|---:|
| A→B | 0.754721–5.473963 | 0.002502–0.673995 |
| B→C | 0.144696–1.063488 | 0.001327–0.322243 |
| C→D | 0.090570–1.502758 | 0.000292–0.432231 |
| B→D | 0.205404–1.197018 | 0.000274–0.140019 |

唯一の変更は `vision-table-relayout-distractors`（正解birch）。Bはelm誤答、Cはbirch正解、
Dはelm誤答。C→Dの最大確率差0.432231、centered最大差1.391756。
B→Cの改善がDで失われるため、B→Dの選択一致だけでC/D実装差の解消とはしない。
board-baseはB/C/Dいずれもeast正解だが、top-two marginはB=0.280720、C=0.371681、
D=0.029707。選択が一致しても僅差条件の数値余裕は変わる。

事前基準に従い、今回のCPU F16/Flash経路は未採用。B→C/B→Dの追加誤答は0だが、
数値条件を全fixtureで満たさず、C→Dの選択変更と追加誤答も1件ある。
この27条件ではB→Dは全選択一致。画像タスク自体の低い正解数を改善したとの結論にはしない。

Dは450–523 positionsを1/2 chunkで処理。523 positionsのstatus distractors 3条件は
512+11で成功し、cache=44,040,192 bytes、一時graph peak=107,348,832 bytes。
それ以外はcache=29,360,128 bytes。ゼロ初期化を含む単発Prefill時間とgraph bytesはcase別に保存。
同条件の新規B時間・peak RSS・swapは測っていないため、速度改善やresident memory削減の
比較結論は出さない。CUDA workspace/VRAMの証拠にも使わない。

[全候補raw/comparison](../phase-4-plus-option-scale-2-cd-e4b-q4-cpu-2026-10-02.jsonl)に
全logits/softmax、offset/centered/raw/確率差、margin、選択・正誤変化、入力/embedding/model/
source/binary hash、実build flag、C実行設定、D資源bytesを保存。
[測定sourceと再現手順](../phase4-step4-cpu-reference/README.md)も保持。
回答token ID、prompt IDsから中央placeholderを除いたtext/image/text分割、9 embeddingの
hash/shape、全CのFlash/CPU_Mapped実log、全D graphのbackend対応、有限logits、
A/Bの保存済み選択・正誤結果を照合。argmax/softmax/margin/差/集計を再計算した。
通常runtimeを変更していないため新規CTestは実施していない。

Step 4全体は未完了。次はC/Dの残差をkernel・decode shape条件を揃えたcontrolで切り分ける。
純text/16件、window境界、E2B修正後基準とC/D、CUDA、384/512候補は未測定。
Step 2/3のproduction採用・公開件数の変更は行わない。

<a id="cd-cpu-review-2026-10-02"></a>
## 2026-10-02: C/D結果の独立レビューと考察

全27条件を54本のC/D生出力と各stderrから再照合し、argmax、候補softmax、top-two margin、
raw/offset/centered/確率差、事前閾値と通過件数を独立再計算した。A/B値は元の
shared-embedding記録をF32に戻した値と一致。fixture正解・候補順、prompt/answer IDs、
画像境界を保持したbefore/after分割、保存済みembeddingのhash/shapeも一致した。
測定source 6本・binary 2本・両CPU ops sourceのhashを実ファイルへ照合。
JSONLに保存された事前基準本文のhashを確認し、現在の文書冒頭にも同じ本文が残っている。
27件のC実logはFlash有効・通常CPU bufferを示す。新規推論、runtime変更、CTestは行っていない。

### 測定完了と採用判定を分ける

E4B・CPU・この27画像条件のA/B/C/D比較は揃った。B→Dの選択一致27/27と
追加誤答0/27は確認でき、現行通常runtimeの12/27から測定版Dでは13/27になった。
ただしDはF16/Flashだけでなくkey paddingも加えており、この改善をFlash単独の効果とはしない。
通常F32 attentionのpadding版は以前の画像5条件までで、全27条件の比較は依然別途必要である。

事前の数値条件の未達は以下のとおり。二つの未達集合には重複がある。

| 比較 | centered条件未達 | 確率条件未達 | 両数値条件を通過 |
|---|---:|---:|---:|
| B→C | 10/27 | 11/27 | 12/27 |
| C→D | 27/27 | 25/27 | 0/27 |
| B→D | 11/27 | 11/27 | 11/27 |

centered差がA→Bより小さいのは三比較とも27/27。確率差はB→C/C→Dで25/27、
B→Dで26/27。しかし「量子化差より小さい」と事前の「その25%以内」は異なる。
全条件で閾値を満たしておらず、今回の結果を見て基準を緩める根拠にはしない。
C→Dのcentered差は最小でも0.090570で、指定0.01を超える。唯一の選択変更だけに
問題を限定できず、全条件に数値残差がある。

`vision-table-relayout-distractors`はCのbirch正解がDでelm誤答へ戻る。
`vision-board-base`はB/C/Dのeast選択が同じでも、Dのmarginが0.029707まで縮む。
したがってB→Dの選択一致をC/Dの実装照合完了や、僅差条件への余裕の証明にはしない。
一方、この範囲で最適化によるBからの追加誤答が実測されなかったこと自体は肯定的な結果である。

### CPU Flashの内部精度とquery形状

両treeの `ggml-cpu/common.h` はquery tileを64に設定し、`ops.cpp` の
`ggml_compute_forward_flash_attn_ext_f16` はquery数64以上などの条件でtiled経路を選ぶ。
`GGML_CPU_TILED_MM=0` はこのFlash選択を止めない。

重要な追加確認として、vector側の `ggml_compute_forward_flash_attn_ext_f16_one_chunk` は
F16 Vに対して `VKQ16` と `ggml_vec_scale_f16` / `ggml_vec_mad_f16` で加重和を蓄積し、
最後にF32へ戻して出力する。Qもdot用にF16へ変換する。tiled側はQをF32で扱い、
Vの加重和にF32 `VKQ32` を使う。precision指定のDEFAULT/F32は同じCPU dispatchへ進むため、
`GGML_PREC_F32` の指定だけでvector内部までF32になるとは言えない。
この検証の「F32 accumulator」は要求設定を示し、全queryでの実際の内部型の保証ではない。

Cは全条件の冒頭textが38 queryなのでvector経路、続く画像300 queryはtiled対象。
Dは最初の450–512 queryを混在chunkとして処理するため、その冒頭位置もtiled対象になる。
523 positionsの3条件ではDの末尾11 queryがvector経路へ入り、Cの末尾185 queryとは異なる。
今回のE4B head幅はfull 512 / sliding 256。新版で外れたx86のDV整除制限のsource差だけを
原因とせず、まずこの具体的なquery形状・内部精度差をcontrolで切り分ける。
実行時のlayer別kernel traceと揃えたcontrolは未取得であり、最終logit残差の原因確定とはしない。
F32通常attentionで以前一致した画像5条件も、このFlash経路の一致を保証しない。

### 判断品質と資源の解釈

family別正解はboardでA/B/C/Dが3/4/4/4（各9）、statusは全て6/9、tableは4/3/4/3。
証拠変更前後を双方正解した組はA=0/9、B/C/D=1/9。
base/relayoutで選択が同じ組は全て6/9で、Cの総正解1件増を証拠への追従やlayout安定性の
改善とは判断しない。9画像から派生した相関27条件であり、一般的な品質保証には使わない。
BF16でも共通誤答が残る点、shared embeddingsのためvision処理・視認・direct-answer契約・
モデル判断を分離していない点、GGUF由来の留保は従来どおりである。

Dのcacheは512 physical positionsで28 MiB、768で42 MiB。
一時graph bufferは約90.02–102.38 MiB、Prefillは各1回で約18.0–21.0秒。
これらは資源の内訳を確認する値で、Bとの同条件時間・process peakやVRAMの比較ではない。
523 logical positions / 512+11 chunkの成功は確認できたが、523「候補」の成功ではなく、
384/512候補・SWA window境界やGPU全載せの証拠にもならない。

次はboard-base、table-relayout-distractors、523 positionsのstatusなど少数条件で、
測定copyのDをCと同じtext/image/text境界で区切り、kernel選択と中間値を記録する。
F32通常attention＋padding、F16 KV＋通常attention、F16 KV＋Flashを段階的に比較すれば、
padding・cache丸め・Flash内部精度を切り分けやすい。残差が続けばggml revisionも揃える。
vector経路を強制するcontrolは内部F16加重和を使うため、F32 accumulation controlとは呼ばない。
全条件の再測定は原因を切り分けた後に行い、事前基準で再評価する。
E2B、純text/16件、window境界、CUDA、384/512候補は引き続き未完了。
