# Phase 4+ extension - larger Choice option sets (investigation and plan)

## Status / Position

2026-09-29 tokenizer/品質screening、CPU resource sweep、F16/attention probeを実施。
request-level budget と採用判断は未完了。現行の 2–16 件契約は維持し、上限変更は未実装。
目標は 512 件程度までの選択を検討すること。これは HTTP の件数定数だけの変更ではなく、
Phase 3+ の単一 A–P token を読む採点契約の変更を含み得る。
1 model / 1 backend / 1 request の逐次実行を維持し、backend 分離や option worker は導入しない。

## Read First

- `docs/requirements.md`
- `docs/architecture.md`
- `docs/development-rules.md`
- `docs/phases/phase-3-plus-categorical-readout.md`（現行の採点契約）
- `docs/phases/phase-4-plus-http-server.md`（Choice 入出力と UI）
- `docs/phases/phase-4-backend-separation.md` と
  `docs/records/phase-4-backend-measurements.md`（Prefill baseline）
- 必要時のみ `docs/research/openjev.md` と `docs/research/llama-ggml.md`

## 調査結果と問題点

| 項目 | 現状と 512 件時の懸念 |
|---|---|
| 採点ラベル | `Gemma4PromptRenderer` は A–P のみを割り当て、`GemmaTokenizer::tokenize_answer_label` も A–P の単一通常 token と prompt 境界を要求する。512 件を 3 桁番号で表す案は、手元の Gemma 4 語彙で `000`、`001`、`255`、`511` がいずれも 3 token。現行の「次の 1 token の 1 logit」をそのまま適用できない。 |
| Prompt と文脈 | `PrefillEngine` は GGUF の `gemma4.context_length` を超えた入力を拒否する。調査済み E2B/E4B GGUF の metadata は 131,072 token だが、長い option 説明と画像 token は同じ枠を消費する。HTTP の 16 MiB 制限だけでは token 数を制御できない。 |
| Prefill 時間・メモリ | 現実装は全 prompt 長 `T` に対して full/sliding の F32 mask をそれぞれ `T×T` で作り、通常 attention graph も使う。CPU測定の固定workloadは512件でtext 9,295 token、image 9,297 prompt token+77 visual token。textの二つのmask値だけで合計約659.2 MiB、host vectorとbackend tensorが共存する時は約1.29 GiBになり得る。graph bufferの他の値、KV、モデル重み、画像は別。上限131,072 tokenは実行可能なメモリ量を保証しない。 |
| F16 Prefill | 現行の mask と KV cache は F32。F16 mask は `0` と `-∞` をそのまま表現でき、text 512件で二つのmaskの値の容量をhost/backendそれぞれ約659→330 MiBにできる。実測ではlogitは一致したがRSS/速度の改善は限定的。F16 KVではcache容量を半減できるが、logit数値差が出た。一方、通常attentionの`ggml_mul_mat`はF32 scoreを作るため、F16入力だけでは二乗サイズのscoreを消せない。 |
| Tokenization | `evaluate` はラベルごとに完全な prompt と `prompt + label` を再 tokenization して境界検証する。ラベルだけ増やすと件数と prompt 長の双方で検証コストが増える。 |
| Readout | `ggml_get_rows` は要求 token ID をまとめて取得し、512 個の F32 値は 2 KiB。候補数に応じた graph node 増加は現行構造からは見込まれず、主な性能問題は readout より Prefill/検証にある。実測は必要。 |
| 品質 | ラベルの token 事前確率、選択肢順序、遠く離れた説明を参照できるかが未検証。既存 SemIf 調査でも順序反転で 36 件中 10 件の argmax が変わった。512 件の条件付き softmax は校正された確信度ではない。 |
| HTTP / UI | `SystemOne` adapter、core、renderer、browser UI の各所に 16 件上限がある。HTTP は 1–16 問を逐次処理するので、1 問 512 件を許すだけでもリクエスト全体の処理量が大きくなる。UI の手入力 512 行は運用しにくい。TypeSafe Choice の 255 件上限を超えるため、512 件は branchscore 固有拡張として明記する必要がある。 |

## 暫定 512 ラベル候補

[512 件の二文字ラベルと token ID](../records/phase-4-plus-option-label-candidates.tsv)
を調査用に保存した。`index` は 0 始まりの入力位置候補で、caller の
semantic option ID ではない。英大文字 2 文字で統一し、先頭文字と末尾文字が
それぞれ A–R で 20 回、S–Z で 19 回現れるように選んだ。その条件の下で
token ID の合計が小さい集合を選び、先頭文字ごとに巡回して並べた。
token ID は BPE 語彙上の値であり、モデルの回答 logit や判断品質の代用ではない。
この表は Phase 3+ の A–P ラベルや 2–16 件の本番契約を変更しない。
現行の `tokenize_answer_label` は A–P 以外を拒否し、system 指示も 1 文字の
回答を求めるため、本番で使うには新しい renderer/validator の契約が必要。

## 変更の進め方

### Step 1 - ラベル方式の判断を先に行う

1. E2B/E4B の実 GGUF tokenizer に対し、候補となる 32、64、128、256、512 個の
   ラベルの単一 token・piece round-trip・重複・prompt 境界を機械的に調べる。
   自動選定された語彙 token が「選択肢の番号」としてモデルに理解されるとは
   仮定しない。E2B/E4B では上記の暫定集合の形式検証まで完了した。pinned
   ggml の project build でも、2026-09-29 に暫定 512 ラベル全件を現行
   renderer の回答位置で照合した。単一 normal token、piece、重複なし、prompt
   境界維持、保存済み token ID のすべてが E2B/E4B とも 512/512 で一致した。
2. 少数の実際の判断入力で、16 件 baseline と拡張ラベルの選択、順序入替え、
   無関係候補の追加、E2B/E4B 差を測る。`raw_score` と `selected_id` を残し、
   結果の不安定さを件数ごとに記録する。ラベル集合が採点に適さなければ
   単一 token 方式の上限拡大は止める。
3. 単一 token で 512 件の意味付けが成立しない場合、固定長の番号を全候補に付け、
   **全 512 件を一つの prompt に表示したまま**、番号 token 列の条件付き
   log probability を逐次 branch 採点する案を別契約として設計する。
   prefix trie で共通番号 prefix を再利用できるか検証する。これは複数 forward、
   KV 分岐、末端境界、異なるスコア/確率意味、追加時間を伴うため、現行
   `answer_slot_logit` と同じ `readout_id` や schema 意味で返さない。
   16 件ずつ分割して勝者を集める方式も比較案に留める。分割間 logits は直接
   比較できず、全 512 件の条件付き確率と見なせない。

### Step 2 - 長い prompt の資源計測

1. 同じ説明長分布で 16、64、128、256、512 件を作り、text と image、E2B と
   E4B、利用可能な CPU/CUDA で token 数、tokenization、mask 作成、Prefill、
   readout、peak host RAM/VRAM、失敗段階を記録する。モデルロード後の空き容量と
   実行中 peak を区別する。画像は visual token 数も併記する。
2. 最初のメモリ改善候補として、F16 の full/sliding mask を選択 backend と
   pinned ggml で検証する。host の mask も F16 で直接作り、backend の型変更
   だけで host メモリを残さない。`ggml_soft_max_ext` の backend 対応を実際の
   graph で確認し、F32 mask 時と回答 logits・選択 ID・時間・peak memory を
   比較する。mask の値は `0` と `-∞` のまま保つ。
3. 次に `StateCache` の K/V を F16 にした小さな試験を行う。既存の F32
   出力から cache への変換・再読込が CPU/CUDA で動くか確認し、E2B/E4B の
   小・中規模 prompt で最大/平均 logit 差、選択 ID、peak memory、速度を
   F32 baseline と比較する。mask と KV の試験は別々に測り、全 activations を
   一括 F16 にする変更は前提にしない。
4. まず現行 graph と上記 F16 候補で実行可能域を確かめる。長い入力に残る
   F32 attention score が制約なら、text attention の Flash 対応 probe または
   小さな逐次 chunk の設計を別 slice として検証する。いずれも数値一致、
   KV 位置、sliding window、final-position logits を短い prompt で照合して
   から拡張する。Vision Flash Attention の実装結果を text 側へ無条件に適用
   しない。`gemma4.context_length` だけを根拠に 512 件を公開しない。
5. リクエスト全体の token/資源予算を決め、上限超過時はモデル実行前に明示
   エラーを返す。候補数の上限と prompt token 上限は別に扱う。

### Step 3 - 契約と入口の変更

1. 採用したラベル方式ごとに prompt renderer ID、scoring/readout ID、
   probability の意味と schema 変更要否を確定する。16 件既存入力の意味を
   黙って変更しない。共通の option 件数定数を core/adapter/UI に適用するのは
   この決定後とする。
2. `Gemma4DecisionEngine`、renderer、tokenizer/readout または新しい番号列
   scorer、`SystemOne` adapter、CLI/JSONL、HTTP/UI の順で小さく変更する。
   512 件時の UI は行を大量追加する前提にせず、貼り付け/一括入力を検討する。
   JSON の key 辞書順と CLI/JSONL の入力順の違いも結果 metadata に残す。
3. HTTP では件数超過と token/資源予算超過を区別して 422/413 相当の既存
   エラー契約へ対応付ける。1–16 問という既存上限との積で極端な処理量に
   ならないよう、リクエスト合計の予算も測定に基づいて設定する。

### Step 4 - focused verification と採用判断

- 16 件の既存 prompt/logit/ID を回帰確認し、64→128→256→512 件の各段階で
  tokenization、ID 対応、同点順序、確率和、context 超過、メモリ不足の
  エラー処理、失敗後の次リクエストを確認する。
- 事前に用意した少数の意味判断 fixture で、候補順序入替え・無関係候補追加時の
  top-1 安定性と正解率、p50/p95 latency、peak memory を記録する。
  達成できた最大件数をモデル/backend/説明長別に示す。
- 512 件の実測と品質が成立した場合だけ公開上限を 512 にする。
  成立しなければ検証済みの上限を採用し、512 件を妨げた条件を記録する。

## Notes / Findings

日付付きの tokenizer 調査、prompt 長・メモリ推算、参照実装との照合は
[Phase 4+ option 拡大の調査記録](../records/phase-4-plus-option-scale.md) に移した。
E2B/E4B で暫定 512 ラベルの単一 token・ID・回答位置の境界は確認済み。
現行上限は 2–16 件のまま。初回の16/32/64件 model-backed screening は
記録したが、512件の判断品質・latency・peak memory は未測定で、公開上限を
変更する根拠にはまだ足りない。

2026-09-29: 本 checkout の pinned ggml (`456172ec`) を使った形式照合を完了した。
初回 model-backed 比較は、ユーザーが追加した E2B/E4B の対応 mmproj GGUF を使い、
CPU・逐次で実施した。2つの手作り text fixture について、production A–P 16件、
順序反転、二文字候補ラベル16件、順序反転、無関係候補追加32/64件を比較した。
全28条件の option 別 `raw_score` と `selected_id` は
[保存した JSONL](../records/phase-4-plus-option-scale-scores-2026-09-29.jsonl) を参照。
小規模な screening であり、512件の品質や公開上限を決める完了測定ではない。

2026-09-29: ユーザー指定により CPU backend のみを使い、E2B/E4B×text/image の
16/64/128/256/512件 resource sweep を完了した。全20条件は成功。短い固定長説明の
512件 prompt は9,295/9,297 text tokens（画像条件はさらに77 visual tokens）で、
Prefill は E2B 568/591秒、E4B 843/855秒、peak RSS はそれぞれ8.55/8.65 GiB、
10.64/11.33 GiBだった。単発 CPU 計測であり、p50/p95 や 512件の意味的な判断品質を
示さない。詳細・計測上の注意は[CPU resource 記録](../records/phase-4-plus-option-scale-resource-cpu-2026-09-29.jsonl)
にある。現在の 2–16件契約と公開上限は変更しない。

同日、pinned ggml の CPU softmax が F16 mask を受け付けることを試作 graph で確認した。
E2B/E4B の text 128/256件で F32/F16 mask の全候補 logits が一致し、winner も同じ。
256件では F16 によって mask upload が約半分、host build は遅くなり、観測 peak RSS は
約168 MiB減った。F16 K/V cache も E2B/E4B の text 128/256件で動作したが、raw logits
には差が出た（最大差1.06–4.97、平均絶対差0.26–3.02）。winner はこの1 fixtureでは
維持した。数値差と再現性の確認が残るため、どちらの F16 候補も production には採用していない。
詳細 raw scores は[F16 mask](../records/phase-4-plus-option-scale-f16-mask-cpu-2026-09-29.jsonl)、
[F16 K/V](../records/phase-4-plus-option-scale-f16-kv-cpu-2026-09-29.jsonl)に保存した。
2026-09-29: Flash Attention CPU probe はE2B/E4Bの128件でbaselineとraw scoreが異なり、
E4Bではwinnerも変わったため採用しない。F32 query軸128-token chunkは16件E2Bと
128件E2B/E4Bでraw scoreが一致した。512件ではbaselineのwinnerを維持し、peak RSSは
下がったがprocess swapを使い、PrefillはE2Bで3.4%、E4Bで10.6%遅かった。256件以上には
一時probeでgraph容量65,536が必要だった。品質確認とrequest token/resource budgetは残り、
Step 2を継続する。詳細は[attention CPU記録](../records/phase-4-plus-option-scale-attention-chunk-cpu-2026-09-29.jsonl)
と[Flash CPU記録](../records/phase-4-plus-option-scale-flash-cpu-2026-09-29.jsonl)。現行契約・公開上限は変更しない。

2026-09-29: llama.cpp の Gemma 4 text attention 実装を確認した。既定は Flash `AUTO` で、
対応 device では Flash、非対応なら通常経路を使う。Flash graph は既定 F16 K/V cache と
F32 accumulation を使い、prompt は既定512-token microbatchで処理する。今回の
branchscore CPU Flash probeとは条件が異なり、llama.cpp での判断結果のズレは未測定。
参照 revision とコード上の詳細は[調査記録](../records/phase-4-plus-option-scale.md#llamacpp-gemma-4-text-attention-reference)。
