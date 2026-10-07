# 2026-10-07: 画像先頭/末尾promptのCUDA比較

[phase計画](../../phases/phase-4-plus-option-scale-2.md) ·
[固定prefix案](practical-512-adoption.md#連続画像requestの固定prefix再利用候補) ·
[再現script](../option-scale-2-gpu-2026-10-07/run-image-placement.py)

## 目的と比較条件

利用者の提案に従い、既存サンプルで画像と固定textの順序を入れ替えて測定する。
固定BOM prefix再利用に適したprompt配置の探索であり、cache再利用自体は実装・測定しない。
production renderer/API、公開2–16/A–P、runtimeのattention/cache型は変更しない。

- 対象はE4B Q4_K_M + F16 mmproj、RTX 2060 SUPER、単一CUDA/逐次request。
  GGUF/source/binary hashは[manifest](../option-scale-2-gpu-2026-10-07/image-placement/manifest.json)。
- 既存structured visionの27条件。9画像のbase/changed/relayoutと、候補の基本順/逆順/
  無関係候補追加を含む。同じfixtureの条件を両配置で比較する。
- 既存board-base画像の384/512候補、基本順/逆順の4条件を追加。
  31 request条件 × 画像配置2種 × attention/cache経路2種 = 124実行。
- 画像先頭はsystem→画像→State→Question→Options→回答prefix。
  27条件のprompt bytesは保存済みproduction rendererのidentity hashと完全一致。
- 画像末尾はsystem→State→Question→Options→改行→画像→回答prefix。
  画像spanをuser turn内で移動し、JSON arrayの後に区切りの改行を一つ加える。
  State/Question/候補description/順序/labelとnative画像begin/endは保持する。
  これは空白変更を含むprompt配置比較であり、画像位置だけの純粋な因果差とは断定しない。
  測定専用renderer IDは`research-gemma4-image-last-v1`。
- 非FA controlは現行F32 KV/mask、削減候補はF16 KV/mask + FA、256-position padding。
  両方full SWA/512 microbatch。既存測定用Prefill/StateCache copyを同じprobeから実行する。
- 画像は同じencoderから各runで再計算し、配置間の保存embedding SHA-256一致を確認する。
  回答labelのsingle normal token/ID一意性/piece/実prompt境界はprobe内で検証する。

## 計測と再現

各条件はwarmupなしの1回観測。`prefill_ms`はVision/model loadを含まない。
monitorのprocess時間はload、embedding download/保存、出力を含むため、サービスの
request latencyとは呼ばない。VRAMはnvidia-smiによる100ms以上間隔のdevice-wide sample最大で、
時間的な真のpeakの下限。全Prefillの測定であり、cache warm時の時間を示さない。

```sh
python3 docs/records/option-scale-2-gpu-2026-10-07/run-image-placement.py
python3 docs/records/option-scale-2-gpu-2026-10-07/run-image-placement.py --report-only
python3 docs/records/option-scale-2-gpu-2026-10-07/summarize-image-placement.py
python3 docs/records/option-scale-2-gpu-2026-10-07/check-image-last-reference.py
```

既存CUDA probeのbuildは[再現資料](../option-scale-2-gpu-2026-10-07/README.md)に従う。
scriptは完了済みの結果を保持するので、独立した再測定には出力先を別にする。
prompt/label、splice前後のtoken IDs、全候補logits/確率/semantic ID、embedding hash、
exit code/log/VRAM sampleを[raw資料](../option-scale-2-gpu-2026-10-07/image-placement/)へ保存する。

## 124実行の結果

全124実行が成功し、candidate logits/確率はfinite、確率和とlabel境界検証も通過。
画像先頭の27条件×2経路は以前のGPU benchmarkの全candidate raw logitsと一致
（54/54選択一致、最大raw差0）。全9画像について全run間のembedding hashも一致した。
[配置比較report](../option-scale-2-gpu-2026-10-07/image-placement/report.json)と
[追加集計](../option-scale-2-gpu-2026-10-07/image-placement/summary.json)に全差分を保持する。

| 正解数 | 非FA・画像先頭 | 非FA・画像末尾 | F16/FA・画像先頭 | F16/FA・画像末尾 |
|---|---:|---:|---:|---:|
| board（9条件） | 4 | 2 | 4 | 2 |
| status（9条件） | 6 | 8 | 6 | 8 |
| table（9条件） | 3 | 4 | 4 | 4 |
| 合計（27条件） | 13 | 14 | 14 | 14 |

画像先頭→末尾の選択変更は非FA 15/27、FA 17/27。非FAは正解→誤答5、誤答→正解6、
FAは各6で、合計正解数の近さは同じ入力で判断が保たれることを意味しない。
条件付き候補確率の最大差を各条件で取った中央値は非FA 0.549、FA 0.613。
同じ方式のcentered logit最大差の中央値は5.095/5.121。

対して同一配置で非FA→F16/FAの選択変更は先頭2/27、末尾1/27。
先頭は追加正解1、追加誤答0、末尾の1変更は双方誤答だった。
centered最大差の中央値は先頭0.610、末尾0.597、条件付き確率最大差の中央値は0.039/0.046。
今回のruntime・fixtureでは、削減経路への変更より配置変更の影響が大きかった。
これらの確率は校正confidenceではない。

候補基本順/逆順/追加の3条件すべてで選択が一致した画像は、非FA 3/9→2/9、
FA 4/9→2/9。画像内の証拠変更に応じた選択変化は両経路3/9→5/9だが、
同じ証拠の配置変更で選択が維持された比較は非FA 5/9→4/9、FA 6/9→4/9。
選択変化自体を正解への追従と同一視しない。

### 384/512候補

両経路とも次のsemantic selectionを返した。期待するIDは全条件`east`。

| 候補数・候補順 | 画像先頭 | 画像末尾 |
|---|---|---|
| 384・基本順 | north（誤答） | west（誤答） |
| 384・逆順 | east（正解） | zone378（無関係候補・誤答） |
| 512・基本順 | north（誤答） | west（誤答） |
| 512・逆順 | east（正解） | zone506（無関係候補・誤答） |

画像末尾は4条件すべて誤答で、単純な画像末尾移動による大候補数の採用は支持しない。
これはboardを無関係なZONE説明で拡張したprobeで、実BOMの512件の品質を直接示さない。

| 画像末尾・1回観測 | positions | 非FA Prefill | F16/FA Prefill | 非FA最大VRAM sample | F16/FA最大VRAM sample |
|---|---:|---:|---:|---:|---:|
| 384（基本順/逆順） | 6,911 | 5.36–5.38 s | 約4.11 s | 6,542 MiB | 5,808 MiB |
| 512（基本順/逆順） | 9,087 | 約7.99 s | 約5.77 s | 6,920 MiB | 5,938 MiB |

画像先頭は各1 position短いが、全Prefillの時間・容量はほぼ同程度。
cache再利用による時間短縮はまだ測定していない。

### 同一token/embeddingのreference診断

出力観測後に、board-changed、status-base、table-relayout-reversed、512候補の
基本順/逆順の5条件を選び、画像末尾の非FA/FA各経路を追加照合した。
これは原因調査のsubsetで、独立した品質評価集合ではない。
保存した同じbefore/after/answer token IDsとF32 image embeddingsをreferenceへ渡し、
全model/KV bufferがCUDAにあることをlogで確認。referenceではVisionを再実行しない。
元の[GPU reference条件・限界](gpu-validation.md)を引き継ぎ、両projectのggml revisionや
image前/画像/後を分けるdecode shapeの差は残る。

10実行すべて成功し、semantic selectionは9/10一致した。
非FAのtable-relayout-reversedだけbranchscore `elm` / reference `aster`で、双方誤答。
画像末尾のboard-changed `east`（期待`north`）、512基本順 `west`、逆順 `zone506`
（いずれも期待`east`）はreferenceでも両経路で再現した。
これらの誤答をbranchscore固有の挙動とは扱わない。数値の完全一致や全入力の正しさの
証明ではなく、画像配置の採用判断に使う補助診断とする。
[全scores・差分・provenance](../option-scale-2-gpu-2026-10-07/image-placement/reference-diagnostic.json)。

## 用途上の限界

27条件は相関した既存fixtureで、独立27標本として扱わない。
部品パッケージ実写、末尾1文字違い、BOM側の途中挿入文字列の実運用例とは異なる。
この比較で現状のユーザー成功例を検証済みとは称さない。
結果を踏まえ、採用するprompt配置を選んでから同一promptの全Prefill対prefix再利用を
別の小さな検証単位で比較する。画像末尾がよいという事前結論は置かない。

今回の単純な画像末尾配置は、そのまま採用する根拠が足りない。
次の比較候補は、長い固定候補/BOMをprefixへ置き、画像の後に判定条件・質問を置く配置。
短い判定textを動的tailで再計算しても、長いBOMの再計算を省く目的は保てる。
この配置はまだ測定しておらず、改善やcache再利用の成功を保証しない。
