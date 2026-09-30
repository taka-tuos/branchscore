# Phase 4+ option-scale - larger Choice option sets（調査打切り）

## Status / Position

**2026-09-30、調査途中で打ち切り。完了扱いではない。**
現在の作業計画は[option-scale-2](phase-4-plus-option-scale-2.md)。
旧Step 1/2の調査を実施した後、本番のE4B単一NVIDIA GPU全載せとメモリ削減を
優先する方針に変更したため、旧計画の継続を止めて後続計画へ引き継いだ。

旧CPU resource sweepとF16/Flash/query-only chunkは調査用probe。
本番CUDA peak VRAM、拡張labelの384/512件判断品質、request budget、runtime採用判断は未完了。
Step 3の契約・入口変更は未着手で、productionは2–16件・A–Pの単一token readoutのまま。
512件を断念した判断ではなく、後続計画では512件を目標、384件を比較点にする。

## Read First

- 次の作業は `docs/phases/phase-4-plus-option-scale-2.md`
- 旧調査の参照は `docs/records/phase-4-plus-option-scale.md`
- GPU全載せのsource調査・容量推算は `docs/records/phase-4-plus-option-scale-2.md`

## Handoff

旧計画の調査結果・問題点一覧・暫定512labelの説明・打切り時点の到達点は
[旧調査記録](../records/phase-4-plus-option-scale.md#2026-09-30-旧計画の打切りと引き継ぎ)へ移管した。
CPUのraw JSONLとlabel TSVは元のrecordsのファイル名で保持する。
これらは歴史的証拠であり、現在の実装順や採用条件はoption-scale-2に従う。

後続の実装順は、Prefill全体のtoken microbatch化、F16 KVとtext FA、
必要ならSWA cache短縮、その後に本番GPU計測・label/資源予算・入口の採用判断。
従来F32 graphへのbit一致を必須とせず、llama.cpp相当条件で照合する方針も後続へ引き継ぐ。

## Notes / Findings

2026-09-30: 本文書を旧計画の打切り・引き継ぎの案内に整理した。
日付付きのtokenizer/CPU計測、既存probeの限界はrecordsへ保持し、
GPU全載せの追加調査をoption-scale-2 recordsへ移管した。
[後続計画](phase-4-plus-option-scale-2.md)は計画作成済み、実装は未着手。
