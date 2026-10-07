# 2026-10-07: マージ後のCUDA検証

[記録索引](../phase-4-plus-option-scale-2.md) · [phase計画](../../phases/phase-4-plus-option-scale-2.md)

## 対象と結論

merge `f08cadf` に含まれる `20daa9b`（全layer token microbatch）、
`f5cf099`（画像境界token）、正解付きfixtureと保存済みCPU測定用sourceを検証した。
現行runtimeのCUDA回帰は通過し、今回の短文・board画像workloadではE4Bの384/512候補を
8 GiB GPU上で実行できた。測定専用F16 KV/Flashは容量を削減できたが、
画像判断の誤答・候補順序依存と同条件referenceとの差が残るため、runtime採用・公開上限は変更しない。
Step 4の高精度CUDA A/Bと最終採用基準、Step 5の実運用workload/反復測定は未完了。

## 環境と再現資料

- NVIDIA GeForce RTX 2060 SUPER、compute capability 7.5、`nvidia-smi`総VRAM 8,192 MiB。
  ドライバ615.71.09、CUDA compiler 13.4.92。CUDAが報告する利用可能な総容量は約7,800 MiB。
- project `f08cadf`、ggml `456172ec733a135778adcd32d00e576a58232e45`。
  referenceはCPU記録と同じllama.cpp `19e28a27702117d8f2eb16b825b9a308111f67d9`。
  referenceのggmlはprojectのggmlと異なる。両方Release、CUDA architecture 75。
- E2B/E4BのQ4_K_Mと対応F16 mmprojだけがローカルにあり、BF16/F16 text GGUFはない。
  GGUF/mmprojのSHA-256、build flags、測定sourceのhashは
  [manifest](../option-scale-2-gpu-2026-10-07/manifest.json)に保存。
- sandbox内ではGPU deviceが見えず、sandbox外で実行した。GPU未検出のnative buildを避けるため
  architectureを75に明示した。GPU計算は一つのmodel/backend/requestずつ逐次実行した。
- [再現手順とharness](../option-scale-2-gpu-2026-10-07/README.md)、
  [全候補・差分・正誤集計](../option-scale-2-gpu-2026-10-07/report.json)。
  per-run JSONはcommand/return code/VRAM samples、logは配置と実行結果を保持する。

VRAMは`nvidia-smi`を少なくとも100 ms間隔（command overheadあり）で読み、device全体の
最大サンプル値を報告する。真の時間方向peakの下限であり、厳密なpeak保証ではない。
CUDA context/pool保持分を含み、CPU RSSからの推定ではない。
probeはmodel load後・Vision後・Prefill後の空き容量、weight/cache/graph bytesも保存する。
Prefill後の空き容量には解放済みgraphの瞬間peakが含まれず、allocator bytesとも区別する。
benchはwarmup 1回、probeはwarmupなし。いずれも単発測定である。

## 現行CUDA回帰と判断fixture

CUDA構成の既存CTestは11/11、skipなし。E4Bでもtokenizer/engine/Prefillの3対象が成功した。
513 positionsの2 chunk、absolute KV位置の違い、画像がchunkをまたぐ631 positionsの2 chunk、
finite logits、final-position gatherとdebug logitsの一致を確認した。
renderer/tokenizer検証には画像境界2tokenの保持が含まれる。

| 現行F32 KV・通常attention | E2B | E4B |
|---|---:|---:|
| quality fixture | 23/24 | 24/24 |
| structured vision fixture | 6/27 | 13/27 |
| image-fact text controls | 2/5 | 5/5 |

qualityのE2B誤答は`japanese-priority-reversed`（expected `shiro`、selected `kiiro`）。
structured visionは3 task familyの相関した変形27条件であり、独立27標本のaccuracy保証ではない。
E4Bの画像誤答には候補逆順・distractor・relayoutによる変化があり、実行正常性とは分けて扱う。
全decisionでschema 2、finite scores/probabilities、確率和、first-max/semantic ID対応、
answer-slot契約、回答token非消費を出力から再検証した。
CPU値とCUDA値の精度差を同一集計に混ぜていない。

## 384/512候補の容量probe

productionの2–16/A–P上限は維持する。測定専用promptは暫定二文字labelを使い、
実際に候補を全件表示した回答位置でsingle normal token、piece、ID一意性、境界を検証した。
拡張用system指示はuppercase **label**、renderer識別は`research-gemma4-two-letter-v1`。
現行rendererのuppercase **letter**とは区別する。

短文は各itemの色を全候補に提示し、唯一blueの`item007`を選ぶfixture。
16/384/512件で元のsemantic IDを維持し、正順・逆順を測った。
画像は実際の960×720 board-base（300 visual tokens）、元の候補・質問に無関係なZONE候補を
追加して384/512件にし、正順・逆順を測った。画像拡張はE4Bのみ。
52 probe（両modelのcontrol/text scale 44、E4B image scale 8）は全てexit 0。

| E4B workload（正順） | positions / chunks | 現行VRAM最大サンプル | F16/Flash最大サンプル | 現行Prefill | F16/Flash Prefill |
|---|---:|---:|---:|---:|---:|
| text 16 | 384 / 1 | 5,444 MiB | 5,414 MiB | 0.247 s | 0.253 s |
| text 384 | 7,376 / 15 | 6,628 MiB | 5,832 MiB | 5.880 s | 4.514 s |
| text 512 | 9,808 / 20 | 7,060 MiB | 5,976 MiB | 9.008 s | 6.424 s |
| image 384 | 6,910 / 14 | 6,550 MiB | 5,808 MiB | 5.305 s | 4.093 s |
| image 512 | 9,086 / 18 | 6,920 MiB | 5,938 MiB | 7.958 s | 5.738 s |

text 512のcacheは現行1,124,859,904 bytes、F16 572,522,496 bytes、
peak graph allocationは707,270,656 → 124,786,688 bytes。
F16 cacheは256-position physical paddingを含むため、単純にちょうど半分ではない。
VRAM最大サンプルの差は1,084 MiB（約1.06 GiB）。SWA短縮なしでこのworkloadは成立した。
Step 3の追加変更は行わず、より長いstate/説明/複数問予算を測って必要性を判断する。

textは両model・両経路の16/384/512正順・逆順すべて`item007`。
しかしE4B image 384/512は両経路とも正順`north`（誤答）、逆順`east`（正解）だった。
元の4件board-baseの現行CUDAは`east`であり、候補追加・label変更で判断が変わる。
この画像4条件の2/4を一般的な画像accuracyとして扱わない。
容量が成立したことだけで512件を公開しない。旧9,295-token CPU短文とは入力も異なる。

## F16/Flashとreferenceの照合

測定用copyは保存済みCPU C/D sourceをそのままリンクし、backendだけCUDAにした。
F16 KV、host F16 mask、Q F32、FlashのF32 precision request、256-position padding、
全長SWA、512 microbatchを使用し、全graph nodeのbackend supportを確認する。
normal runtime/default/sourceは変更していない。

referenceは同一GGUF/token IDs/label順、同一backend/cache型/FA/full-SWA/512上限。
画像はprobeでCUDA生成した同一F32 visual embeddingsを入力する。
通常経路とFlash probeのembedding hashも一致した。
referenceのbefore/visual/after decode形状とbranchscoreの混在chunk形状、key padding、
ggml/kernel revisionは異なり、bit一致や差の原因確定は主張しない。

最初のreference pilotは43/43 layer offload表示でも2,208 MiBのembeddingをCPUへ置いた。
配置ログを確認して停止し、全tensorの`.*` CUDA buffer overrideで再測定した。
pilotは別directoryに隔離し、最終集計に含めない。修正後E4BはCUDA model bufferだけ
5,091.51 MiB、全KVもCUDA。referenceのtensor duplication/layoutと小さなhost stagingは
branchscoreのweight bytesと別である。referenceは画像encoder/mmprojを実行しないため、
そのVRAMをbranchscoreのend-to-end容量と直接比較しない。

最終的な同条件reference比較は52/52実行成功、選択一致47/52。
全候補raw/共通offsetを除いた差/softmax差、B→C/C→D/B→Dをreportへ保存した。
A（高精度CUDA）は未測定。以下の「centered」は候補差分の平均offsetを除いた最大絶対差。

| model / matched path | 選択一致 | max centered logit差 | max候補確率差 |
|---|---:|---:|---:|
| E2B F32/通常 | 10/11 | 3.0063 | 0.7569 |
| E2B F16/Flash | 9/11 | 8.7887 | 0.5251 |
| E4B F32/通常 | 15/15 | 2.0365 | 0.2081 |
| E4B F16/Flash | 13/15 | 2.2466 | 0.4454 |

不一致は画像controlの5比較：E4B Flashのboard（branch `east` / ref `north`）、
table（`aster` / `birch`）、E2B F32/Flashのtable（`dahlia` / `cedar`）、
E2B Flashのstatus（`vega` / `atlas`）。画像embeddingとtoken/answer IDは同一。
この差をencoder差だけで説明しない。通常経路にも差があり、Flashだけを原因と断定しない。
52条件は相関したcontrol/変形であり、この一致率を実運用の保証にしない。

F16/Flash benchmarkではE4B quality 24/24、vision 14/27、
E2B quality 22/24、vision 7/27。現行経路との選択変化はE4B quality 0/24・vision 2/27、
E2B quality 1/24・vision 4/27。
E2B `close-scores`が正解→誤答1件、visionは両modelとも誤答→正解1件・正解→誤答0件。
個別誤答の相殺や同じwinnerだけで数値同等性を認定しない。
E2B qualityのmax候補確率差は0.5541、E4B visionは0.2461。
production benchmark出力には測定用Flash pathの独立metadataがないため、
`*-flash.jsonl`とcommand/source manifestで通常結果から区別する。

GPUの高精度A/Bがないので、CPUの量子化差・screening基準をGPUへ転用せず、
今回の差分は記述統計として保存する。F16/Flashのメモリ削減と最終採用を同一視しない。

## 残る確認

- 同一checkpoint高精度/Q4のCUDA A/Bと、GPUでの事前採用基準。
- 同条件referenceとの差のkernel・decode/key shape/内部精度の切り分け。
- 384/512件のより多様な意味判断と実運用の画像・state・説明長、反復latency/VRAM、失敗境界。
- 件数/単問/request予算・拡張rendererの最終契約。Step 6/7は開始しない。
