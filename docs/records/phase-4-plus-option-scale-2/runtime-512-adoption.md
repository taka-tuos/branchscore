# 2026-10-07: 初期512対応の採用

[phase](../../phases/phase-4-plus-option-scale-2.md) · [実用方針](practical-512-adoption.md) ·
[raw/source/再現資料](../option-scale-2-runtime-512-2026-10-07/README.md)

## 判断と実装範囲

利用者は200件超のBOM照合を必要とし、画像配置/cacheの調査をいったん区切って
現状の512対応を先に使える形にする方針を示した。CUDA F16 KV/mask＋Flash Attentionと
512件の単一回答位置readoutを採用する。既存画像先頭配置、全長SWA、512-token
microbatch、1 model/backend/requestの逐次実行を維持する。
高精度CUDA A/B、実パッケージ品質確認と固定prefix再利用は追加調査とし、初期公開を止めない。
過去のCPU事前screening未達は履歴として保持し、CPU/Vulkan PrefillはF32/通常attentionを保つ。

CUDAのQ、accumulation/outputはF32。physical K/V容量とchunk key数を256位置へ切り上げ、
絶対位置のcausal/sliding maskで未使用paddingを遮断する。F16 cacheはゼロ初期化し、
masked K/Vの未初期化NaNを防ぐ。全graph nodeのbackend対応を確認し、非対応時はエラー。
shared KV mapping、画像境界splicing、最終位置だけのreadoutとrequest所有cacheを保つ。
16,384位置で容量が成立したので、この版ではSWA cache短縮を見送る。

## 公開契約

| 項目 | 契約 |
|---|---|
| 候補数 | 2–512、uniqueで空でないsemantic IDとdescription |
| 2–16件 | `gemma4-categorical-v1`、A–P、既存prompt bytesとidentityを維持 |
| 17–512件 | `gemma4-categorical-512-v1`、固定順序の二文字labels、指示はuppercase label、JSON fieldは`letter` |
| labels | `src/gemma4_answer_labels.hpp`の固定512個。既存TSVと一致、全件single normal token、piece/ID一意性/境界を実行時確認 |
| scoring | `answer_slot_logit` / `gemma4-next-token-categorical-v1`、温度1の候補softmax、最初の最大値を選択、回答tokenは消費しない |
| 単問予算 | model context以内かつ16,384 actual Prefill positions、画像placeholderをvisual tokensに置換した数 |
| HTTP合計予算 | 最大16 questions、合計32,768 positions。全questionをCPU token/画像geometryでpreflightしてから逐次実行 |
| 拒否 | 単問/context/合計超過はHTTP 413の各別code、513件以上は422。切り捨てない |
| 順序 | core/CLI/JSONLは入力順、HTTPはkey辞書順。HTTP diagnosticsに実際の`option_order`と`renderer_id` |
| UI | 512件まで、1行1key・任意TAB descriptionの一括貼り付け。blank descriptionはnull |

255件超Choiceはbranchscore拡張。BOM品番だけならHTTP `criteria`は品番key/null valueとし、
`key: description`による品番の重複を避けられる。`/healthz`にlimitsを公開。
HTTP `usage.input_tokens`は従来のtext/placeholder token数のまま、
`branchscore.prefill_positions`にvisual展開後の合計を追加した。
schema 2を維持し、timingsにPrefill attention path/KV type/positions/chunks/microbatchと
cache/peak graph allocation bytesを追加する。確率は未校正、`confidence: 1.0`は従来のplaceholder。

512ラベルの検査では、最後のspecial `<|turn>model\n`をBPE境界として、そのsuffixと
labelを検査する。元promptから境界に重なるspecial tokenがあればfull promptへ戻す。
最適化前は各labelごとに長いpromptを二度tokenizeする費用が大きかった。
focused testでは512全ラベルを確認し、代表index 0/16/255/256/511を独立したfull prompt
tokenizationとも比較した。これはモデル計算や提示textを変更しない。

## GPU・入口の検証

対象はRTX 2060 SUPER compute 7.5、物理8 GiB（nvidia-smi total 7,800 MiB）、
E4B Q4_K_M＋F16 mmproj。pinned ggml `456172ec733a135778adcd32d00e576a58232e45`。
GGUF、sourceとbinary hashは再現資料のmanifestに保存した。

- CUDA構成のE2B CTest 11/11、E4B tokenizer/engine/Prefill 3/3。
  最終buildでもCUDA CTest 11/11とE4B tokenizer、CLIの512候補実行を確認。
  pure CPU buildとhost/tokenizer CTest 9/9、別実行のE2B engine/Prefillも通過。
- production E4B quality 24/24、vision 14/27。51条件すべてのcandidate raw値が
  先行F16/Flash測定copyと一致し、production移植による追加選択差は0。
- production engine probeは16成功と1期待拒否。17/255/256/512のtext、順序反転、
  384/512画像、20文字品番＋HD/FHD、16,384/16,385境界、反復・拒否後の小入力を含む。
  single-token ID、semantic対応、finite logits/probabilities、確率和と最大値選択を確認。
- live HTTPは17/512件とFHD画像でcoreと選択・順序・raw値（差1e-4未満）が一致。
  513件422、16,385位置413、14,854位置×3質問の合計超過413、次の小requestの200を確認。
  UIの512定数・bulk fieldの配信とJavaScript syntaxを確認。browserでの操作は未検証。

## 資源と待ち時間

以下はwarm-loaded engineのrequest時間。HD/FHDは白いRGB画像の資源controlであり、
品番の正解はstateに記載したtext control。パッケージOCR精度の測定ではない。

| 入力 | expanded positions | Visual tokens | request時間 |
|---|---:|---:|---:|
| 短文512 | 9,808 | 0 | 約6.2秒 |
| 合成20文字英数字512＋HD 1280×720 | 11,509 | 405 | 約7.9秒 |
| 合成20桁数字512＋FHD 1920×1080 | 14,854 | 920 | 約11.5秒 |
| FHDを含む公開予算上限 | 16,384 | 920 | 約13.1秒 |

FHD/14,854の3反復は約11.55/11.56/11.58秒。16,384位置ではcache 939,524,096 bytes、
peak temporary graph allocation 138,418,176 bytes。device-wide nvidia-smiの最大サンプルは
6,352 MiB。100ms以上間隔で取得した観測値なので、正確な瞬間peakの下限である。
大入力後、予算拒否、反復と小入力への復帰後の空きVRAMは2,609,971,200 bytesで安定。
これは今回の有限回数での確認であり、長時間運用や他GPU/backendの保証ではない。

## 残す制約

GPU E4Bでは既存非FAからquality 24/24を維持、vision 13/27→14/27。
E2Bでは僅差quality 1条件が正解→誤答となる[既知の差](gpu-validation.md)がある。
CPU BF16比較の数値や採用基準をCUDAへ直接移したとは扱わない。
E4B画像の384/512は正順north（誤答）、逆順east（正解）の順序依存を維持。
小さいモデルやFAという理由だけで実BOM品質まで保証しない。
利用者の現状成功例の実画像/BOM/model/backendは取得していない。
CPU/Vulkanの大型512資源は未測定。件数・token予算は全backend共通の上限であり、
資源成立を測定した主対象はこのE4B CUDA構成。

[単純な画像末尾移動](image-placement-gpu.md)は多数の選択変更と大型画像controlの悪化を
招いたため採用しない。新配置/cacheは独立した次の課題とする。

資源probe後にtokenizerのoverlap guardを`rendered_prompt + label`まで先読みする形へ
補強した。測定時sourceは`tokenizer-measured.cpp`として初期manifestのhashと一致を確認して
保存。最終guardのbinaryでlive HTTPのcore比較、512 full-boundary tokenizer test、CUDA
回帰を確認した。最終source/binary manifestを別に保存し、初期測定のhashは上書きしない。
