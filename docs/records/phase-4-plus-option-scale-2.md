# Phase 4+ option-scale-2: execution record index

Step 1のメモリ・Prefill調査と、Step 4の精度・画像・CPU reference検証の記録索引。
現在の実装順・採用条件は[option-scale-2計画](../phases/phase-4-plus-option-scale-2.md)を参照。
基礎となるtokenizer照合とCPU probeは[旧調査記録](phase-4-plus-option-scale.md)に保持する。

2026-10-02整理: 長い実行記録を以下のテーマ別Markdownへ分割した。
最新のCPU比較baselineは[CPU kernel/key padding記録](phase-4-plus-option-scale-2/cpu-kernel-padding-baseline.md)。
そのbaselineに対する最適化評価と再レビューは[C/D CPU評価](phase-4-plus-option-scale-2/cd-cpu-evaluation.md)。
初期の画像記録と境界修正後の記録は測定契約を区別して読む。

## テーマ別記録

| 記録 | 読む内容 |
|---|---|
| [メモリ容量・Prefill分割](phase-4-plus-option-scale-2/memory-prefill.md) | Step 1–3のsource調査、容量推算、Step 1実装とCPU/reference照合。 |
| [初期CPU精度baseline・採用方針](phase-4-plus-option-scale-2/precision-baseline.md) | Step 4の採用方針と、正解labelのない初期7 paired comparisons。画像境界修正前の測定。 |
| [初期精度レビュー・GGUF由来・quality fixture](phase-4-plus-option-scale-2/precision-review-provenance.md) | 初期A/Bの考察、Unsloth公開履歴の照合、24条件の正解付きquality screening。 |
| [Structured vision fixture・4 GGUF screening](phase-4-plus-option-scale-2/vision-screening.md) | 画像fixtureの設計と、境界修正前の4 GGUF/108行CPU screening・再レビュー。 |
| [画像reference・文章control](phase-4-plus-option-scale-2/vision-reference-controls.md) | 同一embedding/別encoder referenceと文章controlによる切り分け。境界修正前。 |
| [画像境界修正・CPU buffer control](phase-4-plus-option-scale-2/image-boundary-buffer.md) | 画像境界の契約是正、修正後27条件、CPU bufferと出力行選択のcontrol。 |
| [CPU kernel・key padding・修正後A/B](phase-4-plus-option-scale-2/cpu-kernel-padding-baseline.md) | 修正後契約のE4B BF16/Q4 CPU baseline、kernel/key軸診断、2026-10-02再レビュー。 |
| [C/D CPU評価・再レビュー](phase-4-plus-option-scale-2/cd-cpu-evaluation.md) | 事前採用基準、E4B画像27条件のA/B/C/D比較、CPU Flash内部精度とdecode shapeの切り分け。 |
| [CUDA検証・384/512候補容量](phase-4-plus-option-scale-2/gpu-validation.md) | 2026-10-07のCUDA回帰、F16/Flashとreferenceの照合、容量・画像順序依存、未採用判断。 |
| [512候補の実用採用方針](phase-4-plus-option-scale-2/practical-512-adoption.md) | 200件超BOM/HD・FHD webcam用途、早期対応の優先順位、初期公開に必要な資源・品質確認。 |
| [画像先頭/末尾promptのCUDA比較](phase-4-plus-option-scale-2/image-placement-gpu.md) | 固定prefix再利用に向け、既存27画像条件と384/512候補の画像配置を比較。 |
| [初期512対応の採用・公開契約](phase-4-plus-option-scale-2/runtime-512-adoption.md) | CUDA F16/FA採用、512 label/core/CLI/HTTP/UI、HD/FHD資源・予算拒否・反復と回復。 |
| [phase Findings履歴](phase-4-plus-option-scale-2/plan-findings-history.md) | 2026-09-30〜2026-10-02のphase要約。詳細の正本は上記テーマ別記録。 |

## 日付・旧節から探す

旧ファイルの節リンクも、以下の索引から本文へ辿れる。

| 当時の節 | 移管先 |
|---|---|
| <a id="2026-09-30-gpu-全載せを前提としたメモリ削減案"></a>2026-09-30: GPU 全載せを前提としたメモリ削減案 | [本文](phase-4-plus-option-scale-2/memory-prefill.md#2026-09-30-gpu-全載せを前提としたメモリ削減案) |
| <a id="2026-09-30-step-1-prefill-microbatch-implementation-and-cpureference-checks"></a>2026-09-30: Step 1 Prefill microbatch implementation and CPU/reference checks | [本文](phase-4-plus-option-scale-2/memory-prefill.md#2026-09-30-step-1-prefill-microbatch-implementation-and-cpureference-checks) |
| <a id="precision-baseline-review"></a><a id="2026-09-30-量子化差を基準にする採用方針の検討"></a>2026-09-30: 量子化差を基準にする採用方針の検討 | [本文](phase-4-plus-option-scale-2/precision-baseline.md#precision-baseline-review) |
| <a id="2026-09-30-step-4-cpu-precision-baseline"></a><a id="2026-09-30-step-4-cpu-ab-baseline-partial"></a>2026-09-30: Step 4 CPU A/B baseline (partial) | [本文](phase-4-plus-option-scale-2/precision-baseline.md#2026-09-30-step-4-cpu-precision-baseline) |
| <a id="step-4-interim-review"></a><a id="2026-09-30-step-4途中結果のまとめと考察"></a>2026-09-30: Step 4途中結果のまとめと考察 | [本文](phase-4-plus-option-scale-2/precision-review-provenance.md#step-4-interim-review) |
| <a id="step-4-provenance-and-quality-fixtures"></a><a id="2026-09-30-unsloth公開履歴の照合と正解付きfixtureの補強"></a>2026-09-30: Unsloth公開履歴の照合と正解付きfixtureの補強 | [本文](phase-4-plus-option-scale-2/precision-review-provenance.md#step-4-provenance-and-quality-fixtures) |
| <a id="step-4-structured-vision-fixtures"></a><a id="2026-10-01-step-4-structured-vision-fixtures"></a>2026-10-01: Step 4 structured vision fixtures | [本文](phase-4-plus-option-scale-2/vision-screening.md#step-4-structured-vision-fixtures) |
| <a id="vision-multi-model-cpu-comparison-2026-10-01"></a>2026-10-01: structured vision 4 GGUF CPU比較 | [本文](phase-4-plus-option-scale-2/vision-screening.md#vision-multi-model-cpu-comparison-2026-10-01) |
| <a id="vision-cpu-interim-review-2026-10-01"></a><a id="2026-10-01-structured-vision-cpu結果の再レビュー"></a>2026-10-01: structured vision CPU結果の再レビュー | [本文](phase-4-plus-option-scale-2/vision-screening.md#vision-cpu-interim-review-2026-10-01) |
| <a id="2026-10-01-e4b-q4-cpu-referenceとtext-control"></a>2026-10-01: E4B Q4 CPU referenceとtext control | [本文](phase-4-plus-option-scale-2/vision-reference-controls.md#2026-10-01-e4b-q4-cpu-referenceとtext-control) |
| <a id="vision-reference-control-review-2026-10-01"></a><a id="2026-10-01-reference--text-controlの再レビュー"></a>2026-10-01: reference / text controlの再レビュー | [本文](phase-4-plus-option-scale-2/vision-reference-controls.md#vision-reference-control-review-2026-10-01) |
| <a id="vision-boundary-fix-2026-10-01"></a><a id="2026-10-01-再レビュー対応--画像境界tokenの保持"></a>2026-10-01: 再レビュー対応 — 画像境界tokenの保持 | [本文](phase-4-plus-option-scale-2/image-boundary-buffer.md#vision-boundary-fix-2026-10-01) |
| <a id="vision-boundary-buffer-review-2026-10-01"></a><a id="2026-10-01-境界修正とcpu-buffer切り分けの再レビュー"></a>2026-10-01: 境界修正とCPU buffer切り分けの再レビュー | [本文](phase-4-plus-option-scale-2/image-boundary-buffer.md#vision-boundary-buffer-review-2026-10-01) |
| <a id="cpu-kernel-alignment-2026-10-01"></a><a id="2026-10-01-追加レビュー対応--cpu-buildbufferkernel照合と修正後27条件ab"></a>2026-10-01: 追加レビュー対応 — CPU build/buffer/kernel照合と修正後27条件A/B | [本文](phase-4-plus-option-scale-2/cpu-kernel-padding-baseline.md#cpu-kernel-alignment-2026-10-01) |
| <a id="cpu-kernel-padding-review-2026-10-02"></a><a id="2026-10-02-cpu-kernelkey-paddingと修正後abの再レビュー"></a>2026-10-02: CPU kernel/key paddingと修正後A/Bの再レビュー | [本文](phase-4-plus-option-scale-2/cpu-kernel-padding-baseline.md#cpu-kernel-padding-review-2026-10-02) |

## 追記先

次の検証は関連するテーマ別記録へ追記し、新しいテーマや長い履歴は別ファイルにする。
この索引とphaseの短いFindingsも更新し、raw JSON/JSONL/TSVは既存の場所で保持する。

## 2026-10-02: Step 4 C/D CPU測定

[事前基準と結果](phase-4-plus-option-scale-2/cd-cpu-evaluation.md)、
[raw/comparison](phase-4-plus-option-scale-2-cd-e4b-q4-cpu-2026-10-02.jsonl)、
[測定source・再現手順](phase4-step4-cpu-reference/README.md)。

## 2026-10-07: マージ後CUDA検証

[結果と採用判断](phase-4-plus-option-scale-2/gpu-validation.md)、
[再現手順・source・raw measurements](option-scale-2-gpu-2026-10-07/README.md)、
[集計と全候補reference差](option-scale-2-gpu-2026-10-07/report.json)。
現行回帰は通過。8 GiBで384/512候補の測定workloadは成立したが、
画像判断とreference差、F16/FlashのE2B追加誤答が残り、公開上限・runtimeは変更していない。

## 2026-10-07: 初期512対応

[採用実装・結果](phase-4-plus-option-scale-2/runtime-512-adoption.md)、
[再現資料・production raw measurements](option-scale-2-runtime-512-2026-10-07/README.md)。
利用者の早期提供方針により、現行画像先頭promptでCUDA F16/Flashと512件を採用。
高精度CUDA比較・実BOM品質・固定prefix cacheと新配置は追加調査として残す。
