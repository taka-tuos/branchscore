# Phase 4 extension - Gemma 4 Vision Flash Attention

## Status / Position

2026-09-24 実装・focused verification 中。Phase 4 の単一バックエンド計測から
生じた、画像入力時の CUDA メモリ改善に限定する。Phase 4+ の HTTP 契約、
Phase 3+ の categorical readout、1 model / 1 backend / 1 request の逐次実行は
維持する。
llama.cpp は実装の参照元であり、runtime 依存にはしない。

## Goal

Gemma 4 E2B/E4B の Vision self-attention を、選択中の ggml backend が
対応する場合は `ggml_flash_attn_ext` で実行する。Full HD 画像で発生する
Vision graph の巨大な attention score テンソルを避け、同じ前処理・
視覚トークン・判断結果の契約を維持する。効果は CUDA の peak memory と
実際の OOM 再現入力で確認する。

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`
- `docs/phases/phase-4-backend-separation.md` と
  `docs/records/phase-4-backend-measurements.md`
- `docs/research/llama-ggml.md` の large-image CUDA memory finding
- pinned llama.cpp `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f` の
  `tools/mtmd/models/gemma4v.cpp`, `tools/mtmd/clip.cpp` の Vision attention と
  backend support probe

## Current finding / memory hypothesis

当初の `src/vision_encoder.cpp` は全パッチで `ggml_mul_mat(K, Q)`、
softmax、`ggml_mul_mat(V, scores)` を行う。1920x1080 画像は前処理後
1920x1104、Vision の 16x16 パッチ 8,280 個、3x3 pooling 後の視覚
トークン 920 個となる。12 head の F32 score テンソル 1 個だけで概算
3.06 GiB。これにモデル重み、他の graph buffer、Prefill の領域が加わる。
これは実装前のソースからの容量推算だった。実測結果は Notes / Findings に
記録し、推算だけで OOM と断定しない。
KV cache の F16 化だけではこの Vision 側の二乗メモリを解消しない。

## Scope / decisions

- 変更対象は Gemma 4 Vision graph の attention と、必要最小限の backend
  capability 判定・計測。画像前処理、位置埋め込み、RoPE、pooling、projection、
  text Prefill、KV cache、HTTP 入出力は変更しない。
- 選択中の単一 backend で Flash Attention が対応するなら使う。
  対応判定は実際の Q/K/V 型・形状で作る graph の
  `GGML_OP_FLASH_ATTN_EXT` に対して行い、確保前に経路を確定する。
  backend 名だけから対応を推測しない。
- 非対応 backend では既存の通常 attention を維持し、選んだ経路を診断可能に
  する。非対応の大画像を暗黙に縮小したり、CPU/別 backend へ移したりしない。
  必要なら別途、明示的な画像トークン上限を検討する。
- llama.cpp の Gemma 4 Vision 経路に倣い、Q の形状を保ち K/V を F16 に
  cast して `ggml_flash_attn_ext` を呼び、F32 accumulation を指定する。
  K/V の cast と最終出力型をこの ggml revision で確認する。mask は現在の
  Vision graph と同じく無し、scale は既存の `1.0f` とする。
- ggml の graph と backend API のみ使う。独自 CUDA kernel、
  llama.cpp runtime wrapper、backend scheduler、並列 worker は導入しない。

## Steps

### 1. Baseline and support probe

1. OOM を再現する画像・モデル・GPU・backend・コマンド・失敗段階
   （Vision graph allocation / compute / Prefill）を記録する。
   画像の前処理後サイズ、パッチ数、視覚トークン数も記録する。
2. 小・中・Full HD 画像で現行経路の Vision 時間と peak VRAM を採る。
   GPU 使用量はモデルロード後の空き容量と推論中の最大値を分けて記録する。
   CUDA が使えない環境では未測定と明示し、CPU 結果で代用しない。
3. 現在 vendoring している ggml に `ggml_flash_attn_ext` と backend
   support probe があることを確認する。選択 backend の実際の graph op
   対応と必要な tensor 型・layout を小さい検証 graph で確かめる。

### 2. Minimal Vision graph change

1. `VisionEncoder` の attention node 構築を小さく分岐する。
   Flash 経路では既存の Q/K/V norm と 2D RoPE の後に K/V を F16 に cast、
   `ggml_flash_attn_ext` を追加し、従来の `[embedding, patches]` 形状へ戻す。
2. unsupported の場合は既存の通常経路を使う。対応判定から graph allocation
   まで同じ backend を使用し、途中で別 backend へ切り替えない。
3. 選択経路と Vision graph allocation/compute の失敗段階をログまたは
  明示的な診断値で区別できるようにする。通常の判断 result/schema の意味は
  増やさず、経路は既存 timing diagnostics に限定する。

### 3. Focused verification

1. CPU と利用可能な CUDA backend で、小画像の従来経路と Flash 経路の
   投影済み視覚トークンを比較する。最大/平均絶対誤差、相対誤差、非有限値を
   記録し、採用する許容差を結果に併記する。Flash 非対応 backend は
   従来経路で正しく動くことを確認する。
2. 同じ画像・prompt・E2B/E4B で回答ラベル logits と選択 ID を比較する。
   明確な勝者の fixture では選択一致を確認し、僅差の変更は数値差として
   調査する。text-only、画像前処理サイズ、visual token 数、画像位置の
   既存契約を確認する。
3. Full HD の再現入力を実行し、Flash 使用の有無、Vision/Prefill の
   成否、peak VRAM、段階別時間を baseline と比較する。E4B が重みだけで
   容量制約に近い場合、その限界を Vision 改善の成否と分けて報告する。

### 4. Decision and documentation

計測により CUDA OOM の改善と数値差を評価し、既定経路を確定する。
`docs/research/llama-ggml.md` とこの Phase の Notes / Findings に
backend 対応、実測値、失敗条件を記録する。画像上限や text Prefill の
最適化がなお必要なら、別の根拠付き作業として切り出す。

## Completion criteria

- 対応 backend の Gemma 4 Vision が Flash Attention graph を実行し、
  非対応 backend は既存経路で動く。
- 小画像の視覚トークンと回答 logits の数値差を測定し、判断結果と
  許容差を説明できる。
- 再現 Full HD 入力について、OOM の改善有無、peak VRAM、Vision 時間を
  同条件の baseline と比較できる。改善しなかった場合も失敗段階を特定する。
- 独立 ggml、単一 backend、単一 request、categorical readout の契約を
  維持する。

## Notes / Findings

- 2026-09-24: pinned llama.cpp は Gemma 4 Vision を共有 ViT graph で組み、
  Vision Flash Attention を auto probe する。非対応では通常 attention に
  戻り、メモリ増加を警告する。Gemma 4 の warmup は 256 視覚トークンに
  抑えるが、実リクエストの最大 1120 トークンは変えない。
- 2026-09-24: vendored ggml `456172ec733a135778adcd32d00e576a58232e45` に
  `ggml_flash_attn_ext` と `ggml_backend_supports_op` があることを確認した。
  実装は実際の Q/K/V shape の Flash graph を allocation 前に probe し、
  非対応なら同じ選択 backend 上で既存の通常 graph を再構築する。経路は
  timing diagnostics / CLI に `vision_attention_path=flash|standard|not_used`
  として出る。独自 kernel、scheduler、別 backend への移動は追加していない。
- 2026-09-24: CPU baseline は E2B Q4_K_M、42x64 source（前処理後の最小域）、
  512x512 synthetic source、1920x1080 synthetic source で取得した。通常経路の
  Vision はそれぞれ 1998.9 ms、3521.2 ms、104000.7 ms、Flash 経路は
  1457.1 ms、未計測、31276.2 ms だった。CPU では VRAM は測定対象外であり、
  CPU 時間を CUDA peak の代用にはしない。
- 2026-09-24: CUDA baseline は NVIDIA GeForce RTX 2060 SUPER
  (compute capability 7.5, 7800 MiB total) で、測定時の空きは 7800 MiB。
  E2B Full HD の通常経路は allocation/compute とも成功し、Vision 1523.1 ms、
  peak 6708 MiB。Flash は Vision 505.9 ms、peak 3844 MiB で、同じ gray
  fixture の選択 ID は一致した。これは OOM 解消ではなく、約 2864 MiB の
  peak 削減と Vision 時間短縮だった。
- 2026-09-24: E4B Full HD の通常経路は、重みロード後の Vision graph allocation
  で 3392614144-byte CUDA buffer の OOM により失敗した。Flash 経路は peak
  5674 MiB で Vision 509.0 ms、Prefill 527.6 ms、readout まで成功した。
  失敗段階は text Prefill ではなく Vision allocation と特定できる。
- 2026-09-24: 小画像の標準経路との debug dump 比較では、E2B CPU が
  max/mean abs `2.9753e-3 / 3.1344e-4`、E2B CUDA が
  `1.5625e-2 / 1.7409e-3`、E4B CPU が `2.7777e-3 / 2.7444e-4`、
  E4B CUDA が `1.6602e-2 / 1.4800e-3`。全て非有限値はなく、明確な gray
  fixture の selected ID は一致した。これは数値差の記録であり、全入力の
  意味判断同等性や calibrated confidence を主張しない。この focused 比較の
  受け入れ目安は max abs <= `2e-2`、mean abs <= `2e-3`、max relative
  <= `4e-3`、非有限値なし、明確な fixture の selected ID 一致とする。
  これは本番の普遍的な数値契約ではない。
- 2026-09-24: 通常 sandbox では CUDA device が不可視だったため、CUDA 実測は
  昇格した読み取り専用実行で行った。非対応 Flash backend はこの環境で利用
  できず、fallback の実機走行は未測定だが、support probe と通常 graph の
  分岐は CPU/CUDA build で検証した。
