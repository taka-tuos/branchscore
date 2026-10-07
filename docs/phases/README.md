# Phase plans

作業対象のphaseを開き、Read Firstから担当Stepに必要な文書だけ読む。
phaseは範囲・契約・手順・完了条件・現在の進捗の正本。
日付付きの実行結果・数値表・長いレビュー履歴は[records](../records/README.md)へ置く。
Notes / Findingsには到達点・未解決事項と詳細記録へのリンクを短く残す。

| Phase / 対象 | 計画 | 詳細記録 |
|---|---|---|
| Phase 1: 調査・最小設計 | [Phase 1](phase-1-research-design.md) | 計画のFindingsとresearch参照 |
| Phase 2: 単一backend | [Phase 2](phase-2-single-backend.md) | [実行記録](../records/phase-2-single-backend.md) |
| Phase 2+: 契約の明確化 | [Phase 2+](phase-2-plus-contract-hardening.md) | [実行記録](../records/phase-2-plus-contract-hardening.md) |
| Phase 3: 継続尤度の評価・観測 | [Phase 3](phase-3-evaluation-debug.md) | [旧baseline](../records/phase-3-continuation-baseline.md) |
| Phase 3+: 現行categorical契約 | [Phase 3+](phase-3-plus-categorical-readout.md) | [実行記録](../records/phase-3-plus-categorical-readout.md)、[実装レビュー](../records/phase-3-plus-implementation-review.md) |
| Phase 4: 計測に基づく境界 | [Phase 4](phase-4-backend-separation.md) | [計測記録](../records/phase-4-backend-measurements.md) |
| Phase 4拡張: Vision Flash Attention | [Vision FA](phase-4-vision-flash-attention.md) | 計画内の記録リンク |
| Phase 4+: 逐次HTTP server | [HTTP server](phase-4-plus-http-server.md) | [実行記録](../records/phase-4-plus-http-server.md) |
| Phase 4+: 旧候補拡大調査（打切り） | [旧option-scale](phase-4-plus-option-scale.md) | [テーマ別索引](../records/phase-4-plus-option-scale.md) |
| Phase 4+: 現在のメモリ削減・候補拡大 | [option-scale-2](phase-4-plus-option-scale-2.md) | [テーマ別索引](../records/phase-4-plus-option-scale-2.md) |
| Phase 5: request単位のworker | [Phase 5](phase-5-logit-workers.md) | 着手条件・記録先は計画を参照 |
| Phase 6: request pipeline | [Phase 6](phase-6-pipeline-scheduler.md) | 着手条件・記録先は計画を参照 |
| Phase 7: scheduler調整 | [Phase 7](phase-7-scheduler-tuning.md) | 着手条件・記録先は計画を参照 |

将来phaseの案内は着手を意味しない。各phaseの前提条件に従う。
