# Phase 3+ - Categorical readout: execution record

This is the dated design, implementation, review, and verification record
extracted from
[`phases/phase-3-plus-categorical-readout.md`](../phases/phase-3-plus-categorical-readout.md).
The phase is complete; its current contract and handoff remain in the phase
document, while the detailed chronology lives here.

## Notes / Findings

- 2026-09-21: `kishida/llama.cpp` の `jev` branch
  (`b779b6ac2ca517e83de42010e54fbe67acf1bb3c`) を調査した。`/v1/systemone`
  は候補を表示し、次位置の単一 label token logits を読むため、branchscore
  の現行 categorical readout と計算方式は一致する。公開文書の評価は
  2--8候補の1,191問、別の358問で温度をNLL fitし、accuracy・T=1/校正後ECE・
  50逐次requestの平均時間を報告する。これはbranchscoreの2--16候補、raw
  label logits、warm-loaded逐次runnerで実行可能な形である。一方、同branchには
  文書が参照する `scripts/bench_models.py` と評価/校正データが含まれず、公開物
  だけでは表を再実行できない。またforkはmodel固有chat template、異なる固定指示/
  option layout、A-Z/a-z/0-9、temperature、cyclic permutationを使い、branchscore
  はGemma 4固定prompt、A-P、T=1である。従ってデータ入手後も、同一benchmark
  入力上のbranchscore評価と、forkの表の厳密再現を区別する。文書中のSystem One
  `choice` 例をoption名込みdescriptionへ変換したE2B CPU smokeは既存runnerで
  完走し、3候補の有限確率と`technical`選択を確認した。
- 2026-09-20: 修正commit `041a2db` をR1/R2の完了条件に照らして再レビューし、
  両件の解消を確認した。修正版fixtureと保存済みE2B/E4B JSONLを照合し、全4行のID、
  schema 2、修正質問、prompt identity、選択yes、logitsから再計算した確率、集計件数が
  整合していた。Phase 4–7の実行対象は計測に基づく境界と質問/request単位へ更新され、
  旧候補workerの必須条件は残っていない。今回の変更範囲に追加指摘はない。確認は差分・
  文書・保存済み成果物を対象とし、モデル推論とCTestは再実行していない。
- 2026-09-20: `28afbb3..78d3dc9` の移行実装をレビュー。候補提示、単一位置のgather、
  stable softmax、旧continuation経路削除、schema 2の主要実装は設計と整合。CPU
  model-backed CTestを再実行し10/10成功。別の16候補E2B CPU warm-loaded runでもA-P/ID
  対応、有限正規化、選択とraw scoreの一致、run/decision/aggregateのschema 2を確認。
  CUDA版はビルド成功したが、このレビュー環境ではGPU未検出のためmodel-backedの2件は
  backend初期化で失敗し、CUDA推論は未検証。
- 2026-09-20: 実装後レビューR1を対応。fixture 3行目を `text-explicit-yes-no` と
  明確なquestionへ変更し、`expected_selected_id` と `expected_reason` の補助fieldを
  追加した。修正版のE2B/E4B schema 2結果は、ignoredな
  `out/phase3plus-categorical-text-explicit-{e2b,e4b}.jsonl` に保存し、両方で `yes` を
  選択した。prompt identityは両モデルで
  `gemma4-categorical-v1/sha256:4b020de3e50743c1487560d536e3d1ab08938dcead59546626fc4d4202017bd4`
  だった。E2Bはraw `11.3818397522/-11.0629043579`、相対確率
  `0.999999999821/1.78801634759e-10`、E4Bはraw `23.6086540222/5.744384288`、相対確率
  `0.999999982556/1.74440058628e-8`。補助fieldはbenchmarkのproduction request型を
  拡張しない。
- 2026-09-20: 実装後レビューR2を対応。Phase 4–7の文書を、計測主導のbackend境界、
  質問/request単位のworker・pipeline・tuningへ更新した。削除済みOptionScorer、候補
  continuation worker、候補KV fan-out、max option batchを後続Phaseの必須条件にしていない。
  Phase 3+は完了のまま、後続機能は未実装である。
- 2026-09-20: ユーザーの方向転換に基づき本Phaseを設計。方式比較研究・旧scorerの
  恒久維持は要求しない。設計後にcategorical readout移行まで実装した。
- 2026-09-20: 方向性レビューも本Phaseの移行決定を前提に改訂。旧方式の観測は履歴として
  区別し、比較研究の推奨を撤回。移行検証を単一回答位置のlogitsへ向け、質問間共有・
  並列化は移行後の改善候補として整理した。
- 2026-09-20: SemIf `ca3ba65f142967030ecb453346e94d6f476a69df` の `core.py` /
  `direct.py` で、候補提示・単一ラベル検証・候補logitsへのsoftmaxを再確認した。
- 2026-09-20: 現行 `PrefillState::logits()` は最終位置のbackend tensorを公開しており、
  loader/Vision/text graphを作り直さずreadoutを置き換えられる。移行後のPrefillは
  prompt長だけをcache capacityとして確保し、回答token用のtail reserveを持たない。
- 2026-09-20: 既存 `branchscore-tokenize` とローカル E2B/E4B Q4_K_M GGUFを使い、
  A–Pの16文字すべてについてstandaloneが単一token、pieceが元の文字、IDが重複しない
  ことを確認した。上記案に沿ったtext promptの末尾へ追加した場合も同じ1tokenとなり、
  prefix境界検証に成功した。これは移行可能性の確認であり、完成した新rendererの全
  リクエスト検証・画像検証・モデル推論の代わりにはならない。
- 2026-09-20: Stage 3+.1を実装。rendererを `gemma4-categorical-v1` に更新し、system
  指示と候補配列を含む実際のpromptをSHA-256 identityに反映した。JSON serializerの
  deterministicなobject key順 (`description`, `letter`) と引用符・改行・日本語を
  exact-render testで固定した。
- 2026-09-20: Stage 3+.1/.2のE2B/E4B tokenizer/model-backed検証でA-P全ラベルのstandalone
  単一token、piece round-trip、相互非重複、prompt境界を確認した。`ggml_get_rows` による
  一回のcategorical gatherはPrefillの同じ全語彙logitsの対応値と一致し、非有限値・語彙外
  IDは拒否する。
- 2026-09-20: Stage 3+.2/.3を実装。通常engineはVision/Prefill/readoutを各1回実行し、
  候補continuation、sampling、EOS採点、branch resetを呼ばない。`OptionScorer`、旧
  `OptionScore` fields、per-option timing、専用summary testを削除し、Prefill cache tailは
  prompt長と同じで追加予約しない。
- 2026-09-20: Stage 3+.3のJSONL E2B/E4B text fixtureはschema 2、
  `readout_id=gemma4-next-token-categorical-v1`、`scoring_basis=answer_slot_logit`、
  有限probability、run/decision/aggregateのrow countを確認した。E2Bの4行は約2.04--2.65秒/
  request、E4Bも完走した。4候補まで増えてもreadout graphは小さい単一gatherで、continuation
  forwardは増えない。
- 2026-09-20: E2B CPUのimage CLI smokeも完走した。640級の小画像でVision約2.15秒、Prefill
  約4.58秒、readout約0.03ms。image markerと候補説明を含むprompt identity、有限なA/B raw
  logits/probability、selected IDを確認した。
- 2026-09-20: focused CTestは10/10。stable softmaxの極端値、first-max tie、非有限値拒否、
  renderer/tokenizer/Prefill/engineを含む。旧continuationのmodel-backed結果は新schema集計に
  混ぜていない。
- 2026-09-20: pinned llama.cpp `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f` の
  `llama-debug` と同じE2B categorical promptを照合した。86 token IDsは完全一致し、branchscore
  CPUのA/B raw logitsは `14.7927408/-1.7233130`、llama.cpp既定のF16 KV cacheは
  `15.1408/-1.39846` で、選択Aは一致した。llama.cppをF32 KV・no-repackにしても
  `14.754/-1.57566` となり、branchscoreとは完全一致しない。llama.cpp側はbranchscoreのggml
  v0.24.0とは別の同梱ggml revisionであり、cache型・graph/backendの差を含むため、数値差を
  量子化だけとは扱わず、raw logitの厳密互換は未主張とした。
