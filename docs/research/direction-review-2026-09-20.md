# Direction review: branchscore / SemIf / Jev

2026-09-20。Phase 1 Stage 1.1/1.3 の設計判断と Phase 3 の実装・評価を確認し、
ユーザーによる SemIf 方式への移行決定と、Phase 3+ の主要実装を反映して改訂したレビュー。
移行仕様は [Phase 3+](../phases/phase-3-plus-categorical-readout.md) に集約する。
以下では移行前の実装事実と移行後の実装を区別する。
実装後レビューでは fixture と後続計画に未完了項目を確認した。
具体的な修正内容は末尾の「Phase 3+ 実装後レビュー」を参照。

## 対象と根拠

- branchscore: `28afbb3`。初期設計 `../archive/ggml_jevlike_project_spec.md`、
  現行 requirements/architecture、Phase 1–5 の関連計画、実装、保存済み結果。
  これに加え、レビュー後のユーザー判断と Phase 3+ 実装を反映した。
  実装後レビューの対象差分は `28afbb3..78d3dc9`。
- SemIf: ローカル `ca3ba65f142967030ecb453346e94d6f476a69df` の
  `core.py`, `direct.py`, `serial.py`, `shared.py`, METHOD/RESULTS。
  [公開リポジトリ](https://github.com/TheoLeeCJ/SemIf)も確認。
- llama.cpp: 既存調査対象 `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f`。
  Phase 3+ の同一 E2B categorical promptで token IDs と回答 logits を再照合した。
- Jev: [Introduction](https://docs.typesafe.ai/introduction)、
  [System One](https://docs.typesafe.ai/concepts/system-one)、
  [Choice](https://docs.typesafe.ai/primitives/choice)、
  [Confidence](https://docs.typesafe.ai/confidence)、
  [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)。
  公開仕様・提供者の説明との比較であり、live endpoint の測定ではない。

## 総評

独立した ggml/Gemma マルチモーダル基盤は初期方針に忠実で、移行にも再利用できる。
今後の目的は、候補を提示して意味判断を返すローカルエンジンであり、採点方式の
比較研究ではない。この目的と将来の Diffusion 系対応に合わせ、Phase 3+ で
SemIf-style categorical readout に一本化する判断は妥当である。

優先するのは新契約の実装と小さな動作・数値検証。旧方式への優位性を証明する
A/B 研究や、両方式の恒久保守は要件にしない。方式を合わせただけで意味品質や
校正が保証されるわけではないため、実際の判断例と既知の限界は記録する。

### 1. 初期設計の OpenJev 解釈と実際の SemIf は異なる

初期仕様 3.2 は option continuation を OpenJev 系の思想として説明していた。
Phase 1 の調査は、SemIf が候補説明を入力し A–P の単一回答トークンを
採点することを正しく発見した。その後も continuation を明示的に採用して
いるため、実装の逸脱ではなく、設計上の選択である。ただしユーザーは今回、
Phase 1 では現方式の説明を求めたものの、SemIf との差まで把握していたわけでは
ないと明示した。以前の採用を、両方式の違いを踏まえた継続希望とは扱わない。
今回はこの認識差を解消したうえでの採点契約の変更であり、git revert ではない。

SemIf の主眼は公開モデルで runtime-defined な意味判断、生成不要の回答、
同一 state に対する複数質問の計算共有を再現すること。RESULTS は今後の
方向として意味判断と calibration の targeted training を挙げるが、
branchscore の現行非目標である training を追従する必要はない。

### 2. 候補説明を判断材料にする契約へ変更する

`gemma4_decision_engine.cpp` は state/question だけを renderer に渡し、
description を `tokenize_option` に渡す。`option_scorer.cpp` はその全 token の
log-probability を加算する。詳しい定義や除外条件を追記すると採点対象が
長くなり、説明を読む分類器とは違う影響が出る。

sum は長さ・表現の事前確率に影響される。mean に変えるだけで意味判断や
calibration が得られるわけでもない。EOS を含めないため、同じ token 列を
prefix に持つ候補では短い方の sum が必ず長い方以上になる。文字列の prefix
だけでは token の prefix が保証されない点には注意が必要。

Phase 3+ では全候補の説明を prompt に入れ、同一回答位置の A–P の logits に
softmax を適用する。説明文の尤度加算、mean/sum の選択、候補別 continuation は
通常経路から外す。説明長は入力 Prefill 費用には影響するが、長い説明を採点対象として
加算する性質はなくなる。意味や候補順序による影響までなくなるわけではない。

旧 scorer を比較用の実行modeとして残す必要はない。過去のcommit・測定結果を
履歴として保存し、新しいprompt/readout/schemaで出力を区別する。
既存 loader / Vision / Prefill と最終位置logitsを再利用でき、E2B/E4BのA–Pについて
単一token・重複なし・提案text promptの境界維持も確認済み。新rendererの完成後に
実際のtext/image入力で契約を検証する。

### 3. state 共有の粒度が違う

現行 evaluate は毎回 state + question を Prefill し、候補間だけで KV を共有。
戻り時に request state を保持しない。SemIf serial/shared は question より前の
state prefix を保持し、複数質問の suffix を処理する。Jev は同一 state への
独立した複数質問を一括評価する公開契約を持つが、内部を KV 共有と断定できない。

保存済み shape777 CUDA JSONL を再集計した。両モデルとも 777 decision 行。
E2B は 1.0924 decisions/s、平均 Prefill 628.70 ms / request 915.25 ms。
E4B は 0.6688 decisions/s、平均 Prefill 1052.30 ms / request 1495.06 ms。
約 69–70% が Prefill。ただし、これは旧continuation方式の測定値で、新方式では
候補説明がPrefillに入り、continuation費用がなくなる。新方式の比率や高速化率に
そのまま読み替えない。

Phase 3+ 自体は1 requestのfresh評価を維持し、質問間共有は実装しない。
移行後の改善候補として、単一backend上のstate-only prefix逐次再利用を設計し、
token境界・position・mask・画像・branch isolationを保ってfreshとの差と速度を測る。
これは移行完了の前提でも、今回追加された実装要件でもない。

### 4. Logit worker の仕事量を再定義すべき

現行 `PrefillEngine::continue_one` は全 transformer 層を実行する。
m-token 候補には m-1 回の continuation forward が必要で、sampling はないが
自己回帰依存の計算は残る。「Logit」を軽量な出力 head と解釈できない。
別 backend に置くなら、現方式では text weights と必要な各層 KV が必要。
最終 hidden state だけでは任意長 continuation を継続できない。

単一ラベル方式では候補ごとの continuation worker 自体が不要になる。
Phase 3+では一度の回答logits読み出しと小さいsoftmaxで結果を得るため、
旧候補workerを別backendへ移す計画は引き継がない。Phase 4–7は移行時のhandoffで
再整理し、質問/リクエスト単位の並列化やVision分離を、その後の測定に基づき選ぶ。
新しいreadoutのためだけに複数workerや汎用schedulerを導入する必要はない。

### 5. 移行検証は小さく、旧結果と新方式を区別する

shape777 は semantic gold を持たない systems fixture。E2B/E4B の一致は
320/777 (41.18%)、yes 件数はそれぞれ 204/615 と再確認した。どちらが正しいか、
原因がモデル能力・量子化・prompt・実装のどれかは、この結果から決まらない。
CPU/CUDA の既知の反転も量子化だけが原因と断定しない。これらは旧方式の観測であり、
新方式の品質予測や移行の採否判断には使わない。

`fixtures/phase3-text.jsonl` の single-token 行は「Which answer is correct?」で
命題が特定されず、multi-token 行と質問も違う。処理の smoke test にはなるが、
single/multi-token の意味品質比較には使えない。候補順序不変も現方式では
構造的に期待されるため、意味判断の正しさを証明しない。

Phase 3+ではこの曖昧な行を明確な命題へ置き換え、少数のtext/image問題で
labelとIDの対応、候補のprompt内提示、有限確率、tie、境界、context上限を確認する。
新方式では順序変更でpromptも変わるので、結果不変を必須テストにしない。
同じPrefillの全語彙debug値とgather値を照合し、実際の短い判断promptの回答logitsを
pinned llama.cpp と条件を合わせて比較する。複数continuation位置の比較は不要になる。

SemIfのauthored144 / perturbations108は後で評価を増やす際の候補だが、
全件実行は移行条件ではない。authored144は合成・モデルレビュー済みで、人手裁定済み
外部goldと同等ではない。旧777件、TypeSafe102件、量子化横断比較も必須にせず、
短いwarm-loaded runで新schema・計測境界を確認し、判断誤りは事実として記録する。

### 6. Jev の学習とサービス特性は別の比較軸

Jev は Choice/Score/Noul と複数質問を提供し、RLCD による校正を説明している。
公開資料だけで学習・内部 sampler の再現や、用途ごとの校正性能を確認できない。
confidence は公式には probabilities の分布形状から計算する統計値であり、
独立した正解確率 head と考える根拠はない。

branchscore の相対確率を confidence と呼ばない方針は適切。SemIf も未校正を
明示する。API 型を増やすだけではこの差は埋まらない。
Jev の現行公開入力は text のみなので、branchscore の画像入力・ローカル GGUF・
直接 ggml による観測可能性は独自の有用な軸になる。

### 7. Diffusion 系とは判断の入出力を共有する

候補説明を入力し、候補順のscore/probabilityと選択IDを返す形は、将来の
穴埋め型readoutにも対応させやすい。Phase 3+はこの意味的な契約を整える。
Gemmaではgeneration prefixの次位置を読む。Diffusion系で1step穴埋めが成立するか、
mask位置・token・attention・step/scheduleをどう扱うかは、そのモデル導入時に検証する。
同じtoken ID、causal KV、Prefill手順、単一forwardを共通前提にしない。

今回Diffusion実装や共通engine基底クラスを追加する必要はない。スコアの由来を
scoring_basis/readout_idで区別し、Gemma固有metadataを共通意味から分けておけばよい。

## 決定済みの次の順序

1. Phase 3+の候補提示・単一回答ラベルreadoutへ一本化し、結果・計測・出力契約を移行。
2. 小さなtext/image fixture、回答logitsの数値照合、短いbenchmarkで移行を確認。
3. 移行完了時に旧Phase 4–7を再整理する。その後の改善候補は単一backendでの
   質問間prefix再利用とし、別途設計して測定する。
4. 新方式で残るbottleneckに応じ、質問/リクエスト並列化やbackend分離を判断する。

旧方式との比較研究・恒久的な二方式保守・旧評価の全完了をこの順序に挿入しない。

独立 ggml、sequential-first、モデル固有処理の局所化、相対確率の明示、
prompt identity、時間境界、無断 truncation 禁止は維持する。
2026-09-20 の Phase 3+ 実装では、候補提示・A–P readout・schema 2・text/image smoke・
 E2B/E4B benchmark・focused CTestを完了した。pinned llama.cppとの照合ではtoken IDsは
 完全一致したが、branchscore CPUのA/B logits `14.7927408/-1.7233130` と llama.cpp
 既定F16 KVの `15.1408/-1.39846` は一致しなかった。llama.cppをF32 KV/no-repackに
 しても `14.754/-1.57566` であり、両者のggml revision・cache・graph/backend差を含む
 ため、raw logitの厳密互換は主張しない。この数値差と選択Aの一致をPhase 3+ Notesに記録した。

## Phase 3+ 実装後レビュー（2026-09-20）

主要な採点経路は設計と整合しており、R1/R2の指摘は fixture 修正と後続Phaseの
handoff更新で対応済みである。検証範囲と対応結果の詳細は [分離した実装後レビュー記録]
(../records/phase-3-plus-implementation-review.md) に移した。
