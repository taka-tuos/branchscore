# branchscore

日本語 · [English](README.md)

`branchscore` は、Gemma 4 E2B/E4B の GGUF モデルを使う独立した C++/ggml
判断エンジンです。テキストの状態、任意の画像、質問、選択肢を渡すと、選択結果、
生のスコア、候補間の相対確率、処理時間を返します。

目的は、提示された選択肢から選ぶための小さなローカル実行環境を作ることです。
具体的な用途の一つは、カメラに映した部品パッケージを部品表（BOM）と照合することです。
AI 支援を多く用いて開発している趣味の PoC で、安定した API や本番運用のサポートは
保証していません。

## ビルドして試す

以下は `branchscore` リポジトリのルートで実行します。必要なのは C++17 対応の
コンパイラ、CMake 3.20 以上、選択する ggml backend の開発ツールです。
例では Ninja を使いますが、別の CMake generator でも構いません。

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

CPU は標準で有効です。GPU backend をビルドする場合は、configure コマンドに
`-DBRANCHSCORE_CUDA=ON` または `-DBRANCHSCORE_VULKAN=ON` を追加します。
サーバーも標準でビルドされ、同梱の llhttp C ソースを使います。通常のビルドに
Node.js、npm、pkg-config、システムの llhttp パッケージは不要です。
CLI だけをビルドする場合は `-DBRANCHSCORE_BUILD_SERVER=OFF` を指定します。

モデルの重みは同梱していません。開発では、[THIRD_PARTY.md](THIRD_PARTY.md#model-assets)
に記載した Unsloth Gemma 4 E2B/E4B の Q4_K_M テキスト GGUF と、対応する
`mmproj-F16.gguf` を使っています。テキストだけの入力でも両方のファイルが必要です。

```sh
./build/branchscore --list-backends
./build/branchscore --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --state "The service is healthy." \
  --question "Which action should be taken?" \
  --option keep="Keep it running" \
  --option stop="Stop it"
```

PNG/JPEG 画像を1枚追加するには `--image FILE` を指定します。backend の指定は
デバイス名または種類を受け付け、大文字・小文字を区別しません。
`--backend auto` は最初の GPU または統合 GPU を選び、なければ最初の利用可能な
デバイスを選びます。

重みを必要とするテストは、モデルのパスが未設定ならスキップされます。実行するには
`-DBRANCHSCORE_TEST_MODEL=/path/to/text.gguf`、
`-DBRANCHSCORE_TEST_MMPROJ=/path/to/mmproj.gguf`、
`-DBRANCHSCORE_TEST_BACKEND=cpu`（または選択した backend）を configure 時に指定し、
再ビルドして CTest を実行します。

## ブラウザー UI と HTTP API

モデルをロードしたまま使えるサーバーを起動します。

```sh
./build/branchscore-server --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --host 127.0.0.1 --port 8080
```

[ブラウザー UI](http://127.0.0.1:8080/ui) を開き、**Connect** を押して、状態・質問・
選択肢を入力します。このループバック設定では bearer token 欄は空で構いません。
BOM は品番を1行ずつ一括貼り付けできます。説明を付ける場合はタブの後に書きます。
空の説明は `null` として送信し、モデルには品番だけを提示します。

エンドポイントは `GET /ui`、`GET /healthz`、`POST /v1/systemone` です。
テキストだけの最小リクエスト例は次のとおりです。

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data '{"state":"The service is healthy.","model":"branchscore-local","questions":{"action":{"type":"choice","instructions":"Which action should be taken?","criteria":{"keep":"Keep it running","stop":"Stop it"}}}}'
```

API は TypeSafe の Choice envelope に対応します。1リクエストに1–16問、各問に
2–512候補を受け付け、逐次評価します。モデル ID は `branchscore-local` のみです。
文字列の criterion は `key: description`、`null` は key だけをモデルに提示します。
現在の JSON codec により、質問・候補は key の辞書順で処理されます。
CLI/JSONL の候補は入力順を保ちます。画像と255件を超える候補は branchscore 独自の
拡張です。Score と Noul には対応していません。

LAN で使う場合は、起動前に環境変数 `BRANCHSCORE_BEARER_TOKEN` を設定し、
ループバック以外の `--host` を指定します。`/healthz` と `/v1/systemone` には
`Authorization: Bearer ...` が必要になります。`/ui` はフォームを開けるように
認証なしで配信し、UI は token をメモリ内だけに保持します。サーバー自体は平文 HTTP
なので、信頼できないネットワークでは TLS を終端するプロキシを使ってください。

応答には選択された key、相対確率、`branchscore` 診断情報が含まれます。診断情報は
生の logits、実際の候補順、prompt/readout の識別情報、処理時間などです。
`confidence: 1.0` は互換用の固定値で、確信度の測定結果ではありません。
`usage.input_tokens` はレンダリングされたテキストの token 数で、展開後の visual token
を含みません。`usage.output_tokens` は0です。`branchscore.prefill_positions` は
全質問の画像展開後の position 数です。画像の入力形式、認証、エラー、応答の詳細は
[HTTP 契約](docs/phases/phase-4-plus-http-server.md)を参照してください。

## 採点の仕組み

1. 任意の画像、状態、質問、**全候補の説明**を、一つの固定 Gemma 4 prompt に並べます。
2. 画像があれば Vision を実行し、prompt 全体を Prefill します。
3. 同じ次 token 位置から、各候補の回答ラベルに対応する logit を読み出します。
4. その logits に温度1の softmax を適用し、最大値の候補を選びます。同点なら先頭です。

2–16候補では `gemma4-categorical-v1` が A–P を割り当てます。17–512候補では
`gemma4-categorical-512-v1` が固定の二文字ラベル集合を使います。各ラベルは実際の
回答境界で単一の通常 token になる必要があり、検証に失敗したリクエストは拒否します。
表示上の文字数と token 数は別です。

候補の説明は判断材料です。説明文を続けて生成する尤度は採点せず、回答 token の
sampling や消費も行いません。最終の全語彙 projection は backend 上で計算し、
小さな ggml gather graph で候補ラベルの logits だけをホストへ転送します。
確率は提示した候補集合内の相対値で、**校正された確信度ではありません**。
候補の追加、順序、表現によって変わり得ます。

Prefill は一つの論理的な prompt 評価を512-token microbatch に分けて実行します。
CUDA は F16 K/V・mask、Flash Attention、F32 accumulation を使い、CPU/Vulkan は
F32 K/V と通常の Prefill attention を使います。これは現行実装の方式であり、
将来の全 backend に同じ型を要求するものではありません。prompt の正確な内容、
採点式、component の所有関係、実行の詳細は[architecture](docs/architecture.md)
を参照してください。

## 目的に基づく範囲と設計方針

- 独立した C++ プロジェクトで ggml を直接使います。llama.cpp と SemIf は
  実装・研究の参照元であり、その runtime をラップしません。
- 主目的は候補からの選択です。汎用チャット、長文生成、学習、校正、Jev の出力の
  完全再現はプロジェクトの対象外です。
- 現段階の基準は、1モデル・1 backend・1判断の逐次実行です。request worker や
  scheduler は計測で必要性が示された場合の後続作業です。Transformer の layer split
  と tensor parallel は対象外です。
- 直接回答を指示する、バージョン付きの組み込み categorical prompt を使います。
  GGUF の chat template は診断情報です。`--chat-template-file` は予約済みの no-op で、
  パスだけを記録し、ファイルの読み込みや適用は行いません。
- 対応予算を超えた入力は、切り捨てずに拒否します。

これらの方針と現行の入力契約の正本は [requirements](docs/requirements.md) です。
内部の `branchscore_core` target は、安定した API やインストール可能なライブラリとして
提供するものではありません。

## 現在の上限と残っている課題

以下は現行実装・検証範囲の制限で、モデルやプロジェクトの恒久的な制約ではありません。

| 入力 | 現在の上限 |
|---|---|
| 1判断の候補 | 2–512件 |
| 1判断の画像 | PNG/JPEG を最大1枚 |
| 1判断の画像展開後 Prefill positions | 16,384、かつモデルの context 以内 |
| HTTP の質問数 / Prefill positions 合計 | 16問 / 32,768 |
| HTTP request body | 16 MiB |
| HTTP の元画像 | 一辺8,192 pixels、総数8 megapixels |

展開後の positions は visual token と prompt の全テキストを含みます。512候補対応でも
任意の長さの説明が入るわけではなく、モデルの context metadata が実行メモリの余裕を
保証するわけでもありません。HTTP は全質問を推論前に preflight し、position 超過は
413、513候補以上は422を返します。`/healthz` で候補・position 上限を確認できます。

現在の prompt は画像を先頭に置きます。K/V cache は1判断が所有し、終了後に破棄します。
画像や質問を繰り返しても prefix は再利用しません。画像配置と cache 再利用は後続の
調査課題です。サーバーは1リクエストずつ処理し、応答後に接続を閉じます。

大入力の資源確認は、主に RTX 2060 SUPER（8 GiB）での E4B Q4_K_M / CUDA が対象です。
CPU/Vulkan で同じ512候補の資源 workload は未検証です。この GPU では、合成した
20桁品番512件と白い FHD 画像の入力が約11.5秒でした。資源確認全体での最大 VRAM
観測サンプルは6,352 MiBです。これは資源測定であり、パッケージ OCR 精度や一般的な
ハードウェア要件を示す値ではありません。

既知の品質差として、CUDA F16/Flash で E2B の僅差 fixture が1件悪化し、E4B の
大候補数・画像 fixture には候補順依存が残っています。高精度重みとの CUDA 比較、
実 BOM・画像の評価は後続作業です。日付付きの結果と検証範囲は
[512対応の採用記録](docs/records/phase-4-plus-option-scale-2/runtime-512-adoption.md)
を参照してください。

セキュリティ強化、安定した API、長期互換性の保証はありません。ローカルのモデル・
入力 parser と画像 decoder は、悪意ある入力を対象とする fuzzing を行っていません。

## ベンチマークと診断

`branchscore-bench` はモデルをロードしたまま JSONL の各行を逐次評価します。
空行以外は `id`、`state`、`question`、
`options: [{"id": "...", "description": "..."}]` が必須です。追加の field は無視します。
任意の `image` パスは入力ファイルからの相対パスです。

```sh
./build/branchscore-bench --backend cpu \
  --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --input fixtures/phase3-text.jsonl \
  --output /tmp/branchscore-e2b.jsonl
```

出力先は未作成のパスを指定します。標準の warmup は最初のリクエストを1回実行し、
`--warmup COUNT` で変更できます。schema 2 の出力には run 行、入力ごとの decision 行、
p50/p95 latency と decisions/second を含む aggregate 行が入ります。
モデルのロード、warmup、ファイル書き込みはリクエストの測定区間に含めません。

`branchscore-tokenize` はモデルの重みをロードせずに token ID と回答境界を調べます。

```sh
./build/branchscore-tokenize --model /path/to/model.gguf --text "Hello world"
./build/branchscore-tokenize --model /path/to/model.gguf \
  --prefix $'<|turn>model\n' --answer-label A
```

画像の projection 後の embeddings を取得するには、判断 CLI に `--vision-dump FILE` と
`--image FILE` を指定します。推論後にファイルを作成または上書きし、int32 の2次元
（`tokens`、`width`）、続いて row-major の float32 値を書きます。
HTTP 応答には prompt 全文、token ID、Vision dump を含めません。

## ドキュメントとライセンス

文書の入口は[ドキュメントマップ](docs/README.md)です。現行計画は
[phase 索引](docs/phases/README.md)、日付付きの実験結果や開発環境は
[records](docs/records/README.md)にあります。HTTP parser の再生成は
[llhttp の説明](third_party/llhttp/README.md)を参照してください。

プロジェクトのコードと文書は [MIT License](LICENSE) です。依存ソフトウェア、
研究の参照元、別ライセンスのモデル資産は [THIRD_PARTY.md](THIRD_PARTY.md)
に記載しています。
