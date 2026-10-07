# Phase 4+ option-scale: 旧計画の打切り・引き継ぎ

[記録索引](../phase-4-plus-option-scale.md) · [phase計画](../../phases/phase-4-plus-option-scale.md)

2026-09-30の打切り時点の状態、当時の問題点と後続計画への引き継ぎ。
本文は当時の測定条件・判断を保持する。現在の契約・進捗はphase計画を参照。

## 2026-09-30: 旧計画の打切りと引き継ぎ

CPU probeを中心とした旧計画を途中で打ち切り、本番のE4B単一NVIDIA GPU全載せを
前提とするメモリ削減の実装へ進むため、option-scale-2へ移行した。
旧計画の完了や512件の公開上限採用を意味しない。

| 旧Step | 打切り時点の状態 |
|---|---|
| Step 1: ラベル方式 | E2B/E4B暫定512ラベルの形式照合済み。16/32/64件の2例で初期screening済み。384/512件の判断品質は未完了。 |
| Step 2: 資源計測 | CPUの16/64/128/256/512件 sweepとF16/FA/query-only chunk probe済み。本番CUDA peak VRAM、request budget、runtime採用判断は未完了。 |
| Step 3: 契約・入口変更 | 未着手。productionは2–16件・A–Pのまま。 |
| Step 4: 採用検証 | 未完了。公開上限変更の根拠は未確定。 |

以下の調査整理とlabel候補説明は旧phase文書から移管した打切り時点の内容。
現在の実装順・採用基準はoption-scale-2のphase文書を参照。

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
