# Phase 4+ - Larger Choice option sets: investigation record index

[旧 option-scale](../phases/phase-4-plus-option-scale.md) は2026-09-30に調査途中で打ち切った。
本記録はtokenizer、初期品質screening、CPU resource/F16/attention probeの歴史的証拠を保持する。
現在の作業計画は[option-scale-2](../phases/phase-4-plus-option-scale-2.md)。
直近のGPU全載せに向けたsource調査と容量推算は
[option-scale-2調査記録](phase-4-plus-option-scale-2.md)へ移管した。
暫定label mappingは[512-label候補表](phase-4-plus-option-label-candidates.tsv)に保持する。

2026-10-02整理: 打切り時点の本文をテーマ別Markdownへ分割した。
CPU probeは実験記録で、productionへの採用判断を表すものではない。

## テーマ別記録

| 記録 | 読む内容 |
|---|---|
| [ラベル形式・初期判断品質](phase-4-plus-option-scale/labels-and-quality.md) | 2026-09-29のtokenizer照合、2例の品質screening、暫定512ラベル候補。 |
| [CPU資源・F16/Flash/query-chunk probe](phase-4-plus-option-scale/cpu-memory-probes.md) | 2026-09-29のCPU資源sweep、F16 mask/KV、Flash、query-only chunkとupstream参照。 |
| [旧計画の打切り・引き継ぎ](phase-4-plus-option-scale/handoff.md) | 2026-09-30の打切り時点の状態、当時の問題点と後続計画への引き継ぎ。 |

## 日付・旧節から探す

旧ファイルの節リンクも、以下の索引から本文へ辿れる。

| 当時の節 | 移管先 |
|---|---|
| <a id="notes--findings"></a>Notes / Findings | [本文](phase-4-plus-option-scale/labels-and-quality.md#notes--findings) |
| <a id="cpu-resource-sweep"></a>CPU resource sweep | [本文](phase-4-plus-option-scale/cpu-memory-probes.md#cpu-resource-sweep) |
| <a id="f16-mask-probe"></a>F16 mask probe | [本文](phase-4-plus-option-scale/cpu-memory-probes.md#f16-mask-probe) |
| <a id="f16-kv-cache-probe"></a>F16 K/V cache probe | [本文](phase-4-plus-option-scale/cpu-memory-probes.md#f16-kv-cache-probe) |
| <a id="flash-attention-cpu-probe"></a>Flash Attention CPU probe | [本文](phase-4-plus-option-scale/cpu-memory-probes.md#flash-attention-cpu-probe) |
| <a id="f32-query-chunk-attention-cpu-probe"></a>F32 query-chunk attention CPU probe | [本文](phase-4-plus-option-scale/cpu-memory-probes.md#f32-query-chunk-attention-cpu-probe) |
| <a id="llamacpp-gemma-4-text-attention-reference"></a>llama.cpp Gemma 4 text attention reference | [本文](phase-4-plus-option-scale/cpu-memory-probes.md#llamacpp-gemma-4-text-attention-reference) |
| <a id="2026-09-30-旧計画の打切りと引き継ぎ"></a>2026-09-30: 旧計画の打切りと引き継ぎ | [本文](phase-4-plus-option-scale/handoff.md#2026-09-30-旧計画の打切りと引き継ぎ) |
| <a id="調査結果と問題点"></a>調査結果と問題点 | [本文](phase-4-plus-option-scale/handoff.md#調査結果と問題点) |
| <a id="暫定-512-ラベル候補"></a>暫定 512 ラベル候補 | [本文](phase-4-plus-option-scale/labels-and-quality.md#暫定-512-ラベル候補) |

## 追記先

新しい検証は[option-scale-2記録](phase-4-plus-option-scale-2.md)の関連テーマへ記録する。
旧調査の本文は打切り時点の履歴として保持し、訂正時は日付と根拠を明記する。
raw JSON/JSONL/TSVは既存の場所で保持する。
