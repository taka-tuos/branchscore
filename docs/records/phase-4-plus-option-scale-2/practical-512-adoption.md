# 2026-10-07: 512候補の実用優先方針

[phase計画](../../phases/phase-4-plus-option-scale-2.md) · [GPU検証](gpu-validation.md)

## 利用目的と優先順位

利用者から200件超の候補を使う相談があり、余裕を持った512候補対応を早期に進める。
用途はHD～FHD Webカメラの画像による連続リクエストで、Digikey等から到着した部品
パッケージをBOMリストと照合すること。高度に厳密な判断精度より、軽量モデルによる
実用的な判断と必要な候補容量を優先するというユーザー方針を記録する。
候補descriptionは品番だけで、枝番の末尾まで含めて20文字程度になることがある。
実際のBOM行・画像と要求latencyはまだ取得/確定していない。

E4B Q4_K_M/CUDAを初期512対応の主評価対象とし、F16 KV/mask＋Flash Attentionを
メモリ削減の第一候補として調査し、利用者の早期提供方針により初期runtimeへ採用した。
[採用実装・検証](runtime-512-adoption.md)が現行の公開契約。以下の比較観測は判断材料として保持する。
E2B/CPUの追加研究と高精度CUDA A/Bは初期512対応の一律の公開条件から外す。
過去のCPU事前screeningの未達判定とGPU未採用の測定時点の判断は保持する。

## 利用者が確認済みの成功例

利用者によれば、現在の運用では10文字を超える品番、BOM側に途中の余計な文字列が
挟まるケースでも対応し、末尾1文字だけ違う候補も区別できている。
説明用の仮例はパッケージ`ABX5C110XYZ`、BOM`ABX5C200110XYZ`。
これは実際の文字列ではなく、余分な途中文字列の関係を示す例として記録する。
カメラは480pセンサ画像をFHDへ拡大したもの。利用者は現状の品質を十分以上と評価した。
この観測を現在の実用品質の根拠として尊重する。実画像・BOM・既存結果と、
使用model/backend/経路はまだこの記録へ取得していないため、本GPU probeの結果とは区別する。

初期採用の品質確認は、この成功例に対応する少数の回帰入力を優先し、FAと512件への
拡張でも期待するsemantic IDと末尾違いの区別が維持されるかを観測する。
BOMの途中文字列を機械的に削除する新しい正規化規則や完全一致判定へ置換しない。
実運用のstate/questionによる照合基準を保持する。実画像の取得までは成功例を合成画像で
置き換えて検証済みと称さず、既知のGPU resource/label確認を独立に進められる。
解像度拡大はセンサの情報量を増やさないが、runtimeへ渡すFHD画像の処理費用は
実際の前処理で確認する。GPU資源評価では入力寸法と元センサ解像度を区別する。

## 実用上の判断材料

- CPU E4Bの同一embedding 27条件では、Q4の非FA reference→削減版branchscoreの
  centered差はBF16→Q4量子化差より全27条件で小さく、fixture別比の中央値は約22%。
  選択は27/27一致。ただし確率差の例外とreference実装差は残る。
- GPU E4Bではquality正解数24/24を維持、vision 13/27→14/27。
  短文512候補は最大VRAMサンプル7,060→5,976 MiB、Prefill約9.0→6.4秒。
  画像384/512でも容量は成立したが候補順序依存を観測した。
- F16 KV/FAが変更する数値差と、512候補・二文字label・画像の読取りによる
  判断差を別に観測する。小さいモデルというだけで全差を許容したことにはしない。
- パッケージの品番の可読性と類似品番の取り違えが用途の重要な評価点となる。
  既存board/status/table fixtureはこのBOM用途の品質を直接保証しない。

## 初期公開の確認と継続評価

数値のbit一致やCPUのC→D centered 0.01/確率0.001基準をCUDA公開の必須条件にはしない。
高精度GPUがないこと、少数fixtureのwinner変化があることだけで実装を止めない。
CPU BF16比較は補助証拠、同条件CUDA referenceと非FA controlは実装診断として維持する。

利用者が画像配置/cacheの研究をいったん区切り、現状の512対応を先に出す方針を示したため、
初期公開は下記1–3・5と既存品質回帰を確認した。4の実BOM/画像品質は利用後の継続評価。
追加研究の完了を公開条件にしない。用途上の制約と既知の差を採用記録に残す。

1. actual extended promptで512ラベルのsingle normal token/ID一意性/境界とsemantic ID対応。
   既存2–16/A–P rendererの契約は維持し、拡張rendererを識別する。
2. E4B Q4とmmprojの単一GPU全載せ、FA node対応、padding遮断、absolute position、
   shared KV、画像splicing、finite logits、final-position readout。
   これらの実装不具合を品質許容の名目で容認しない。
3. 実際のBOM descriptionとHD/FHD画像に近いrequestのtoken量、VRAM、Vision/Prefill/total
   latency。512件とtoken/資源予算を別に定め、測定済み範囲を超える入力は実行前に拒否する。
4. 少数のパッケージ/類似品番で、正解BOM ID、非FAとの差、候補順序/無関係候補追加を
   観測し、用途上困る取り違えの反復・明確な品質悪化がないかを評価する。
   これは初期公開後の評価項目とし、追加誤答の許容数値は実例と利用者方針から将来の変更判定前に定める。
   既測定結果に後から新しい事前基準を適用したとは扱わない。
5. 逐次の繰返しrequestで資源の解放、失敗後の次request、時間・VRAMの増加を確認する。
   continuous requestは1 model/backend/requestの逐次契約を維持する。

追加のtokenizer実測では、20文字の合成品番を512件表示した画像promptは
英数字11,085、区切り入り11,175、数字のみ13,909 rendered tokens。256件は5,606–6,997。
[合成token sizing](../option-scale-2-gpu-2026-10-07/mpn-token-counts.json)は画像placeholder
1 tokenを含むだけで、実画像のvisual token展開・VRAM・判断品質の測定ではない。
20文字という上限だけからtoken数を固定しない。既測定9,808-position短文より長いため、
BOM想定の11k–14k textにHD/FHDのvisual tokensを加えた資源確認を実施した。
採用版はHD 11,509 positionsで約7.9秒、FHD 14,854 positionsで約11.5秒。
単問16,384/HTTP合計32,768 positionsを公開予算とした。使用画像は白い資源control。

今回の512短文測定はPrefillだけで約6.4秒。HD/FHDのVisionと他stageを含むtotal時間は
実運用入力で別測定し、video frame rateで判断できるという保証はしない。
小規模PoCの実用対応を優先し、広いbenchmarkやscheduler/workerの開発を前提にしない。

## 連続画像requestの固定prefix再利用候補

利用者からcacheと画像配置の検討が挙がった。BOMと判定条件がフレーム間で固定なら、
そのtext prefixのKVをGPUへ保持する小さな逐次再利用が次の有力候補となる。
現行rendererはsystem→画像→State→Question→Optionsの順。causal attentionでは
画像以降のBOMのKVも画像に依存するため、画像が替わるとその部分を再利用できない。
現行`StateCache`はrequest所有で全Prefill後にfreezeし、既存cacheへのappendや
次requestに向けたtail resetの公開契約はない。既実装の機能とは扱わない。

検討する拡張rendererは、固定system/State/Question/BOMを先に置き、その後に今回の
画像spanと回答prefixを置く。文字列境界だけでcacheを切らず、最終rendered promptの
token列で固定prefixと動的tailの安全な境界を確定する。画像のnative begin/end、
absolute position、shared KV、SWAとpadding maskの扱いを保つ。
旧2–16/A–P rendererは維持し、順序変更を識別できるrenderer IDを用いる。

単一model/backendで固定prefixを一つ保持し、毎回その直後へ今回の画像と回答prefixを
計算する。前フレームのtailは次のattentionへ含めず、失敗時も次の実行を汚染させない。
固定prefixは保持中に変更しない。prefix token列、候補の順序/label/semantic ID対応、
判定条件、model/backend、cache型とattention/renderer設定が変われば再構築する。
画像のVisionは新しいフレームごとに実行し、projected embeddingsはその実行の寿命とする。
画像の保存や複数BOM cache、並列workerをこの案の前提にはしない。

これにより固定BOMのPrefillを初回だけにできる可能性があるが、固定KVのVRAMは必要で、
画像tailのattentionもBOMを参照する。cacheはF16/FAの容量削減を置き換えない。
初回/warm時のVision・tail Prefill・total latencyと、prefix＋最大画像tailの容量を
実測して判断し、既測定の単発Prefill時間から速度向上を保証しない。

まず現行画像先頭promptと新しいBOM先頭promptをcacheなしで比較し、既存の末尾違い・
途中文字列の成功例が維持されるか確認する。その後、同じ新promptの全Prefillと
prefix再利用を比較し、順序変更による判断差とcache実装差を分ける。
[architectureのLater direction](../../architecture.md#later-direction)に沿う別の小さな
契約・検証単位とし、512件/FA公開の実装済み範囲には含めない。

その後の[既存サンプル画像配置比較](image-placement-gpu.md)では、単純な画像末尾移動は
27条件の合計正解数を大きく変えない一方、15–17条件の選択変更と384/512 probeの悪化があった。
末尾配置を採用決定とは扱わず、固定候補/BOM→画像→判定条件・質問を次の比較候補とする。
この次配置とcache再利用はまだ未測定。

## llama.cppの通常経路との関係

確認したlocal llama.cpp `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f`と、今回のreference
`19e28a27702117d8f2eb16b825b9a308111f67d9`では、context既定はFA `AUTO`、K/V `F16`。
現行common CLI/serverもFA AUTO・F16 K/Vが既定。AUTOは実graph/backend対応を調べて
選択するため、全model/backendが必ずFAになるという意味ではない。
[公式server引数](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)、
[source](https://github.com/ggml-org/llama.cpp/blob/60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f/src/llama-context.cpp)。

`llama_get_logits_ith`はcontextの同期と保存済みlogitsの取得で、読むときにattentionを
非FAで再計算しない。FAでdecodeしたcontextなら取得するlogitsもFA経路の結果である。
今回の非FA/F32 referenceは比較controlとして明示指定した経路であり、通常設定での
利用を再現したという主張ではない。llama.cppがFAを通常候補にする事実は採用の実用的な
根拠になるが、branchscoreのkernel/padding/位置処理の正しさまで代わりに証明しない。
