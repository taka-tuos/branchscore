# Phase 3+ - SemIf-style categorical decision readout

## Status / Position

2026-09-20 設計・実装完了。現在の Phase 3 continuation baseline の後、
Phase 4 より前に挿入した独立した移行 Phase。通常の Gemma 4 engine は
categorical readout に移行済みで、旧 continuation scorer は削除した。

この文書が新しい判断方式への移行仕様。requirements / architecture は
移行後の実装へ更新済みである。
旧 Phase 3 の未完了の全量評価・量子化比較を本 Phase の開始条件にはしない。
Phase 4–7 の旧 option-worker 前提は、本 Phase 完了時に再整理するまで着手しない。

## Goal

Gemma 4 E2B/E4B の判断方式を、候補説明文の continuation 尤度から、
SemIf と同じ「候補を提示し、単一回答ラベルの logits を読み出す方式」へ移行する。
研究用に方式を増やすことではなく、日常的に使う意味判断の実装を単純化する。

将来の Diffusion 系でも、state・question・候補提示・候補別スコア・選択結果を
同じ意味で扱えるようにする。ただし、そのモデルの 1-step 穴埋めが実際に
成立するかは当該モデル導入時に検証し、今回の実装には含めない。

## Background / Decision history

1. 初期基本設計は OpenJev 系の考え方として option continuation を想定した。
2. Phase 1 は SemIf の実際の direct path が A–P の単一回答ラベル方式であることを
   発見したが、最初の scoring contract として sum continuation を明示的に残した。
3. Phase 2/2+/3 はその契約どおりに実装・計測した。今回の変更は実装ミスの修正や
   過去のコードへの git revert ではなく、採点契約の切り替えである。
4. 2026-09-20 の方向性レビューで、候補説明を「判断材料」として読む方式との違い、
   説明長の影響、質問間と候補間の state 共有の違いを整理した。
5. ユーザーは、Phase 1 時点では両方式の違いまで把握していなかったこと、比較研究は
   主目的でないこと、将来の Diffusion 系も考え候補提示方式に統一したいことを明示した。
   よって現方式との A/B 研究を採用判断の条件にせず、SemIf 方式への移行を設計する。

前回レビューの「旧 scorer を比較 baseline として保持」は、この判断により
恒久的な実装要件ではなくなる。過去の結果・commit は履歴として残せばよい。

## Prerequisites

- Phase 2+ の独立 ggml / Gemma4DecisionEngine / 固定 renderer が利用できる。
- Phase 3 の逐次 JSONL runner と計測境界を再利用できる。
- E2B/E4B のローカル GGUF と少数の検証入力が利用できる。
- SemIf の方式を参照するが Python / Transformers / llama runtime を依存にしない。

## Read First

- `docs/requirements.md`（現行と移行先を区別）
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/research/openjev.md`
- `docs/research/direction-review-2026-09-20.md`
- `docs/phases/phase-2-plus-contract-hardening.md`
- `docs/phases/phase-3-evaluation-debug.md` の benchmark 契約と Notes / Findings

初期経緯を追加確認する場合だけ archive の 3.2 と Phase 1 Stage 1.1 を参照する。

## Scope decisions

- 1 model / 1 backend / 1 request / 2–16 options の逐次実行を維持する。
- 新しい通常経路は categorical readout のみ。恒久的な scoring-mode 切替は作らない。
- Vision、GGUF loader、Gemma text graph、backend 管理、JSON 入力形状を再利用する。
- request は string state、question、ordered `{id, description}`、任意の画像1枚を維持。
  SemIf の構造化 state 対応まで広げることは今回の要件ではない。
- 既存の reserved template-file no-op、reasoning-disabled 方針を維持する。
- 候補の説明を提示し、ラベルだけを採点する。説明文の尤度・長さ正規化・PMI は使わない。
- 同一 state の質問間再利用、並列 suffix、server、backend 分離、学習・校正は対象外。
- 旧 777 件の再現、102-row TypeSafe 評価、広い量子化比較は移行完了の必須条件にしない。
- SemIf と方式を合わせる。異なるモデル・chat format の数値互換や品質同等は主張しない。

## Target decision contract

### Prompt and answer slots

renderer ID は `gemma4-categorical-v1`。Gemma の既存 native turn/image token を
使い、generation prefix の直後を単一回答位置とする。A–P を入力順に割り当てる。
option ID は結果対応用で、モデルに提示しない。description は省略せず提示する。

system 指示は SemIf の次の文面を採用する:

```text
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.
```

user 本文は以下の固定構造とする。これは SemIf の JSON payload と byte 同一ではない。
候補配列は既存 JSON serializer で compact に出力し、改行・引用符を正しく escape する。

```text
[画像ありの場合だけ、既存の画像markerと改行]
State:
{state}

Question:
{question}

Options:
[{"letter":"A","description":"..."},{"letter":"B","description":"..."}]
```

配列内順序は入力順。object key は既存 serializer の決定的な順序を使い、実装時に
exact-render テストで固定する。候補一覧より前に state を置くことで、将来の
state-only prefix 再利用を妨げない。今回 cache handle や共有 API は作らない。

各リクエストで使用する回答ラベルについて以下を検証する:

- standalone tokenization が1 token、逆変換が同じ1文字、token ID が相互に異なる。
- `tokenize(rendered_prompt + label) == tokenize(rendered_prompt) + [label_id]`。
- special token や EOS を回答ラベルとして使わない。
- 不成立なら明示エラー。multi-token continuation や別ラベルへ自動fallbackしない。

既存の prefix-boundary 検証はラベル用に再利用できる。実際の候補説明を含む prompt を
検証対象とし、説明文の token 数を回答 token 数と混同しない。
画像展開後の prefix 長を context 判定に使用し、候補を含む入力を勝手に truncate しない。

### Computation

```text
optional Vision
  → Prefill(state + question + displayed options)
  → gather last-position logits for A ... selected last letter
  → stable softmax over these 2–16 values
  → map to semantic option IDs
```

`z_i` はモデルの既存 final norm / output projection / softcap 適用後の回答ラベル logit。

```text
relative_probability[i] = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
selected_index          = first argmax(z)
```

温度は1に固定。全語彙 log-softmax は不要。sampling・回答 token 消費・EOS採点・
候補別 continuation forward を行わない。候補説明の長さは入力 Prefill 費用に影響するが、
説明文の log-probability 加算はしない。順序・表現の影響や意味誤りは依然あり得る。

既存 Prefill は最終語彙 logits を backend 上に保持しているため再利用可能。
回答 token ID に対応する N 個だけを ggml graph で gather してホストへ転送し、
既存と同じく小さい N の stable softmax / 結果集約はホストで行う。
全語彙 download は明示 debug のみ。全語彙 projection 自体の削減は今回不要。

### Result and timing migration

未安定 API のため、旧フィールドの意味を書き換えて互換を装わず、明示的に更新する。

| 対象 | 新契約 |
|---|---|
| request / option ID | 現行を維持、IDを返し、入力順を保持 |
| option score | `input_index`, `option_id`, `raw_score`, `relative_probability` |
| raw score の意味 | `scoring_basis=answer_slot_logit`。異なるmodel/readout間で直接比較しない |
| answer slot debug | ラベル文字と token ID を Gemma 固有 metadata として保持 |
| readout identity | `readout_id=gemma4-next-token-categorical-v1` |
| prompt identity | 新 renderer と候補を含む実際の prompt の hash |
| selection / tie | 最初の最大値、exact tie を明示、未校正の相対確率 |
| terminator | `terminator_scored=false` を継続 |
| old option fields | `sum_logprob`, `mean_logprob`, description の token logprobs/count を通常出力から除去 |
| per-option timing | 個別推論がないため削除。共有時間をN分割して出力しない |
| readout timing | `readout_ms` とその copy/sync/node 数。全候補 gather・必要な同期とhost readを包含 |
| normalization timing | N候補のsoftmaxと選択を計測 |
| request total | validation開始から結果完成まで。load/warmup/file writeを除外 |

JSONL の新出力には `schema_version=2` を run / decision / aggregate 行に記録。
旧 version 無しの成果物は legacy continuation として保存し、新方式の集計に混ぜない。
旧結果を書き換えない。benchmark 入力形状は維持し、CLI 表示と内部C++契約も同時更新する。
Vision / Prefill / tokenization の既存計測と prompt metadata は維持する。

### Boundary for future diffusion engines

共通化するのは「意味付き候補を入力し、候補順の score/probability と選択IDを返す」契約。
`raw_score` の由来は `scoring_basis` / `readout_id` で識別する。

Gemma の回答位置は generation prefix の次位置。将来の Diffusion 系は、適切な
位置の mask を1stepで埋められるならその位置の候補スコアを読む。mask token、
位置、attention、step/schedule、label tokenization は当該モデルの実装責務。
単一 forward / 同じ token ID / causal KV / Prefill が全モデル共通とは仮定しない。

今回は共通 engine 基底クラス、model registry、mask実装、汎用cache interfaceを作らない。
Gemma固有の prompt/token/timing 情報を共通結果の必須意味にしないことで拡張余地を残す。

## Stages

### Stage 3+.1 - Prompt and label contract

#### Step 3+.1.1
renderer が ordered options を受け取り、上記の固定 prompt と新 identity を生成する。
text/image、2/16候補、引用符・改行・日本語説明を exact-render で確認する。

#### Step 3+.1.2
A–P の label mapping / 単一token / round-trip / 境界検証を追加する。
E2B/E4B の実 tokenizer で確認し、invalid boundary / collision は明示エラーとする。

### Stage 3+.2 - Categorical readout and orchestration

#### Step 3+.2.1
既存 backend-resident final logits から N回答logitsをまとめて gather する小さい
categorical readout を追加する。非有限値を拒否し、stable softmax と first-max を適用。

#### Step 3+.2.2
Gemma4DecisionEngine を新経路に切り替える。Vision/Prefill 各1回、readout1回。
通常経路から OptionScorer の候補ループ、reset_branch、continue_one 呼び出しを除く。
Prefill の tail reserve は0にできるよう検証する。消費しない回答token用に
contextを予約しない。既存KV構造の抜本的最適化やcache-free graph化は行わない。

### Stage 3+.3 - Result, CLI, benchmark migration

#### Step 3+.3.1
上記 result/timing 契約を実装し、CLI・JSONL codec/runner・集約・helpを同時更新。
入力互換は維持し、schema/readout/prompt ID で旧結果との取り違えを防ぐ。

#### Step 3+.3.2
旧 continuation 専用の公開score型、production scorer、専用テストを削除または置換。
下位のKV/graph機構は新経路や現在のfocused検証に必要な部分だけ残す。
将来使うかもしれないという理由で旧modeや二重engineを恒久保守しない。

### Stage 3+.4 - Focused verification

#### Step 3+.4.1
少数候補logitsの安定softmax、tie、有限値検査、ID対応を確認する。
model-backed では gather値を同じPrefillのdebug全語彙logitsと照合する。
実際の短い判断promptについて pinned llama.cpp の同じモデル・prompt・backend条件で
回答logits/分布を比較し、許容誤差と条件を記録する。差を量子化だけと決めつけない。

#### Step 3+.4.2
質問・正解が明確な小さな fixture を用意し、E2B/E4B で text と image を実行する。
候補逆順はIDを対応させて観測し、意味判断が必ず不変というテストにはしない。
token ID とraw logitから結果が正しく対応すること、descriptionがprompt内にあること、
候補数が増えてもcontinuation forwardが増えないことを検証する。
曖昧な旧「Which answer is correct? + Yes/No」行は明確な命題に置き換える。

#### Step 3+.4.3
短い warm-loaded JSONL run で新schema・row count・有限確率・計測境界を確認する。
no-op template、context overflow拒否、画像経路、既存の関連focused testsを実行する。
判断が誤る例も記録するが、旧方式に勝つことや大規模accuracy目標は完了条件にしない。

### Stage 3+.5 - Documentation and downstream handoff

#### Step 3+.5.1
requirements / architecture / README / AGENTS を移行後の実装に同期し、候補説明・
回答label・未校正確率の意味を統一。旧Phaseと結果は日付付き履歴として残す。

#### Step 3+.5.2
Phase 3 の次の評価を新readoutへ向け直す。続く改善候補は単一backendのままの
state-prefix質問間再利用とし、必要なら別の小さいStage/Phaseとして設計する。
Phase 4–7 から候補continuation worker必須という前提を外し、質問/リクエスト単位の
並列化とVision分離を計測で選ぶ計画に再整理する。今回は実装しない。

## Suggested implementation locations

- `include/branchscore/decision.hpp`
- `include/branchscore/gemma4_prompt_renderer.hpp`, `src/gemma4_prompt_renderer.cpp`
- `include/branchscore/tokenizer.hpp`, `src/tokenizer.cpp`
- `src/gemma4_decision_engine.cpp`, `src/prefill_engine.cpp`
- 必要なら小さい `categorical_readout.hpp/.cpp`（一般frameworkにしない）
- `tools/branchscore.cpp`, `tools/branchscore_bench.cpp`
- 対応する既存 tests と `fixtures/phase3-text.jsonl`

## Deliverables

単一のSemIf-style通常判断経路、新prompt/readout/schema契約、更新済みCLI/benchmark、
focused verification の記録、現行仕様と後続Phaseの更新。

## Completion Criteria

- 候補説明が入力され、全候補が同一回答位置のラベルlogitから評価される。
- E2B/E4Bで単一token・境界検証に成功。失敗を旧方式へfallbackしない。
- 候補別continuation、sampling、EOS採点を通常経路で実行しない。
- N候補の結果・ID・tie・相対確率が明示され、説明長正規化の旧fieldが残らない。
- 新prompt/readout/schemaと共有readout timingがCLI/JSONLで識別できる。
- text/imageのfocused検証と短いbenchmarkが完了し、数値差・意味誤りを記録。
- 既存の独立ggml・単一backend・画像機能・no-truncation契約を維持。
- 後続計画が旧option-worker前提に戻らない。Diffusion実装は未着手と明示。

## Notes / Findings

- 2026-09-20: ユーザーの方向転換に基づき本Phaseを設計。方式比較研究・
  旧scorerの恒久維持は要求しない。設計後にcategorical readout移行まで実装した。
- 2026-09-20: 方向性レビューも本Phaseの移行決定を前提に改訂。旧方式の観測は
  履歴として区別し、比較研究の推奨を撤回。移行検証を単一回答位置のlogitsへ向け、
  質問間共有・並列化は移行後の改善候補として整理した。
- 2026-09-20: SemIf `ca3ba65f142967030ecb453346e94d6f476a69df` の
  `core.py` / `direct.py` で、候補提示・単一ラベル検証・候補logitsへのsoftmaxを再確認。
- 2026-09-20: 現行 `PrefillState::logits()` は最終位置のbackend tensorを公開しており、
  loader/Vision/text graphを作り直さずreadoutを置き換えられる。移行後のPrefillは
  prompt長だけをcache capacityとして確保し、回答token用のtail reserveを持たない。
- 2026-09-20: 既存 `branchscore-tokenize` とローカル E2B/E4B Q4_K_M GGUF を使い、
  A–P の16文字すべてについて standalone が単一token、piece が元の文字、IDが重複しない
  ことを確認。上記案に沿った text prompt の末尾へ追加した場合も同じ1tokenとなり、
  prefix境界検証に成功した。これは移行可能性の確認であり、完成した新rendererの
  全リクエスト検証・画像検証・モデル推論の代わりにはならない。
- 2026-09-20: Stage 3+.1 を実装。renderer を `gemma4-categorical-v1` に更新し、
  system 指示と候補配列を含む実際の prompt を SHA-256 identity に反映した。
  JSON serializer の deterministic な object key 順 (`description`, `letter`) と
  引用符・改行・日本語を exact-render test で固定した。
- 2026-09-20: Stage 3+.1/.2 の E2B/E4B tokenizer/model-backed 検証で A-P 全ラベルの
  standalone 単一 token、piece round-trip、相互非重複、prompt 境界を確認した。
  `ggml_get_rows` による一回の categorical gather は Prefill の同じ全語彙 logits の
  対応値と一致し、非有限値・語彙外 ID は拒否する。
- 2026-09-20: Stage 3+.2/.3 を実装。通常 engine は Vision/Prefill/readout を各1回
  実行し、候補 continuation、sampling、EOS 採点、branch reset を呼ばない。
  `OptionScorer`、旧 `OptionScore` fields、per-option timing、専用 summary test を
  削除し、Prefill cache tail は prompt 長と同じで追加予約しない。
- 2026-09-20: Stage 3+.3 の JSONL E2B/E4B text fixture は schema 2、
  `readout_id=gemma4-next-token-categorical-v1`、`scoring_basis=answer_slot_logit`、
  有限 probability、run/decision/aggregate の row count を確認した。E2B の4行は
  約2.04--2.65秒/request、E4B も完走した。4候補まで増えても readout graph は
  小さい単一 gather で、continuation forward は増えない。
- 2026-09-20: E2B CPU の image CLI smoke も完走した。640級の小画像で Vision
  約2.15秒、Prefill 約4.58秒、readout 約0.03ms。image marker と候補説明を含む
  prompt identity、有限な A/B raw logits/probability、selected ID を確認した。
- 2026-09-20: focused CTest は 10/10。stable softmax の極端値、first-max tie、
  非有限値拒否、renderer/tokenizer/Prefill/engine を含む。旧 continuation の
  model-backed 結果は新 schema 集計に混ぜていない。
- 2026-09-20: pinned llama.cpp `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f` の
  `llama-debug` と同じ E2B categorical promptを照合した。86 token IDs は完全一致し、
  branchscore CPU の A/B raw logits は `14.7927408/-1.7233130`、llama.cpp既定の
  F16 KV cache は `15.1408/-1.39846` で、選択Aは一致した。llama.cppをF32 KV・
  no-repackにしても `14.754/-1.57566` となり、branchscoreとは完全一致しない。
  llama.cpp側はbranchscoreのggml v0.24.0とは別の同梱ggml revisionであり、cache型・
  graph/backendの差を含むため、数値差を量子化だけとは扱わず、raw logitの厳密互換は
  未主張とした。
