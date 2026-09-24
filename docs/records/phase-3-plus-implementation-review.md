# Phase 3+ - Implementation review record

This is the implementation-after-review section extracted from
[`research/direction-review-2026-09-20.md`](../research/direction-review-2026-09-20.md).
It preserves the resolved R1/R2 findings and their evidence separately from
the direction review's design rationale.

## Phase 3+ 実装後レビュー（2026-09-20）

主要な採点経路は設計と整合しており、今回の確認で明確な採点バグは見つからなかった。
実装後レビューで見つかったR1/R2は、fixture修正と後続Phaseのhandoff更新で対応した。
以下は元の指摘と対応結果を追跡可能に残す記録である。

検証済みの範囲:

- E2B CPU model-backed CTest を再実行し10/10成功。
- 16候補のE2B CPU warm-loaded runで、A–Pとsemantic IDの対応、確率の正規化、
  raw scoreと選択の一致、run/decision/aggregateのschema 2を確認した。
- CUDA版はビルド成功。ただしレビュー環境ではGPUが検出されず、model-backedの
  2件はbackend初期化で失敗した。CUDA推論の正否をこの結果から判定しない。

### R1 [P2][Resolved] 判断対象の命題がない Yes/No fixture が残っている

**根拠・対象**

- [fixtures/phase3-text.jsonl](../../fixtures/phase3-text.jsonl) の3行目に残っていた、
  `id=text-single-token`。
- state は `The service is healthy.`、question は `Which answer is correct?`、
  options は `Yes` / `No` のまま。
- [Phase 3+](../phases/phase-3-plus-categorical-readout.md) Step 3+.4.2 は、
  この行を明確な命題に置き換えることを明示している。
- READMEの短いbenchmark例もこのfixtureを入力に指定している。

**影響**

Yes/Noが何への肯定・否定なのか決まらず、どちらを選んでも正答・誤答として
評価できない。処理完走や有限logitの確認には使えるが、移行後の意味判断を
確認した証拠にはならない。名称も旧方式のdescription token数に由来しており、
すべての回答を単一ラベルで読む新方式での目的が伝わりにくい。

**修正内容**

1. questionを `According to the state, is the service healthy?` に変更した。
   期待semantic ID `yes` と根拠をfixtureの補助fieldに記録した。
2. 行IDを `text-explicit-yes-no` に変更し、旧成果物との混同を避けた。
3. 修正版fixtureをE2B/E4Bで実行し、結果を ignored な
   `out/phase3plus-categorical-text-explicit-{e2b,e4b}.jsonl` に保存した。
   旧成果物は変更していない。
4. Phase 3+ Findingsにprompt identity、selected ID、raw score、相対確率を記録した。

benchmarkは未知の補助fieldを読み飛ばすため、production request型や自動採点frameworkは
拡張していない。

**対応結果**

- 読み手がstateとquestionから期待回答を一意に説明できる。
- 新しいIDとprompt identityで結果を識別でき、E2B/E4Bの実測を記録した。
- prompt identityは両モデルで
  `gemma4-categorical-v1/sha256:4b020de3e50743c1487560d536e3d1ab08938dcead59546626fc4d4202017bd4`。
  E2Bはselected `yes`、raw `11.3818397522/-11.0629043579`、relative probability
  `0.999999999821/1.78801634759e-10`。E4Bはselected `yes`、raw
  `23.6086540222/5.744384288`、relative probability
  `0.999999982556/1.74440058628e-8`。
- E2B/E4Bとも期待回答 `yes` を選択した。これはfixture smokeの観測であり、必ず正解することや
  特定の確率閾値を移行の完了条件には追加しない。
- 旧方式との比較研究や全777件の再実行は不要。

### R2 [P2][Resolved] 後続Phaseの旧worker前提が残ったまま完了を宣言している

**根拠・対象**

- [Phase 3+](../phases/phase-3-plus-categorical-readout.md)冒頭は「設計・実装完了」と
  宣言する一方、Step 3+.5.2ではPhase 4–7の再整理をPhase内の作業として定めている。
- [Phase 4](../phases/phase-4-backend-separation.md)冒頭は「handoff再整理が必要」と
  保留しているだけで、Step 4.4.1には削除済み`OptionScorer`の独立配置が残る。
- [Phase 5](../phases/phase-5-logit-workers.md)には`OptionScoreJob`、option batch、
  候補並列化を完了条件とする記述が残る。
- [Phase 6](../phases/phase-6-pipeline-scheduler.md)はLogit queue/workerへのstate fan-out、
  [Phase 7](../phases/phase-7-scheduler-tuning.md)はmax option batchの調整を前提にしている。

**影響**

保留注記は誤着手を防ぐが、新方式に沿った次工程のhandoffにはなっていない。
新方式では候補ごとにTransformerを動かさず、一度のgatherで全候補を読むため、
旧計画に従うと不要な候補worker・cache転送・queueを再導入することになる。
「Phase完了」と「そのPhaseが担当するhandoffは未完了」が併存し、次の担当者が
どの計画を実装してよいか判断しにくい。

**修正内容**

1. Phase 4–7のGoal、Prerequisites、Stages、Deliverables、Completion Criteriaを
   categorical readout前提で書き換えた。旧計画を実行対象のStageと混在させていない。
2. 各Phaseを次の方向で再整理した。後続機能そのものは今回実装していない。

| 対象 | 除く前提 | 修正後に明確にする内容 |
|---|---|---|
| Phase 4 | Vision/Prefill/Logitを必ず別backendへ配置、OptionScorerの移送 | 新方式の計測を開始条件とし、Vision分離などの有用な境界だけを選ぶ。小さいgatherの独立backend化を必須にしない |
| Phase 5 | 候補単位のOptionScoreJob、option batch、候補workerへのKV配布 | 並列化が必要なら質問またはrequestを仕事の単位にする。質問は全候補を含む一つの判断であり、candidate分岐ではない |
| Phase 6 | Logit queue/workerとstate fan-outが必ず存在するpipeline | 実測で選んだstageと仕事単位を使う。所有権・寿命・同期を定義し、独立readout queueを強制しない |
| Phase 7 | max option batch調整、旧Logit worker構成が固定 | 実際に導入した質問/request batch・queue・cacheを対象とし、まだ導入していない機構を調整要件にしない |

3. 次の改善候補である「単一backend・逐次の質問間state-prefix再利用」を、
   実装済み機能やPhase 3+の追加必須実装にせず、着手前に別Stage/Phaseで境界と計測条件を
   定めるhandoffを追加した。
4. `AGENTS.md`、`docs/README.md`、Phase 3+のStatus/Findingsを、実装完了・検証残件・
   後続計画・次の着手先が一致するよう更新した。

**対応結果**

- 実行対象の後続計画に、削除済みOptionScorerや候補continuation workerを要求する
  Stage・依存関係・完了条件が残っていない。
- 質問/request並列化と、同じ質問の候補logit読み出しを明確に区別している。
- 次の担当者が、未測定の最適化を実装必須と解釈せずに次の設計・検証へ進める。
- Statusと実際の残件が一致し、R1のfixture結果とR2のhandoff更新をFindingsから追跡できる。
