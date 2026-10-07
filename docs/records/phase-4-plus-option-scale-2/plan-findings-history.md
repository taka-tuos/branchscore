# Phase 4+ option-scale-2: phase Findings履歴

[記録索引](../phase-4-plus-option-scale-2.md) · [現在のphase計画](../../phases/phase-4-plus-option-scale-2.md)

2026-09-30〜2026-10-02のphase Notes / Findingsを移管した履歴。
当時の要約を保持し、現在の進捗や実行順はphase計画から読む。

2026-09-30: 旧option-scaleはStep 1/2の調査途中で打ち切り、メモリ削減を優先する本計画へ移行。
旧tokenizer照合・初期品質screening・CPU resource/F16/FA/query-chunk probeは
[旧調査記録](../phase-4-plus-option-scale.md)に保持する。
重み容量の訂正、llama.cppのmicrobatch/SWA、CUDA FAの型・padding条件、
容量式と数値差の再解析は[option-scale-2調査記録](../phase-4-plus-option-scale-2.md)に移管した。
512件を目標、384件を比較点とする。GPU peak VRAM・拡張label品質・request budgetは未確定。

2026-09-30: Step 1を完了。最大512 tokenの全layer microbatch、全長F32 KV、
absolute position、画像spanのchunk分割、最終位置readoutを実装した。
E2B/E4BのCPU境界testとllama.cppのtext/image照合を実施し、今回のfixtureではtop-1が一致。
相対logits/probability差は残り、拡張候補数の品質とCUDAの許容差は未確定。
容量値・比較条件・全数値表・vision差の切り分けは
[Step 1実行記録](memory-prefill.md#2026-09-30-step-1-prefill-microbatch-implementation-and-cpureference-checks)へ移管した。

2026-09-30: 重み量子化差をruntime最適化の追加差の物差しにする方針をユーザーと合意。
追加誤答と同条件referenceとの実装照合を併せて評価するStep 4を追加した。
従来Step 4–6はStep 5–7へ繰り下げた。判断の背景と未測定事項は
[採用基準の検討記録](precision-baseline.md#precision-baseline-review)に保持する。

2026-09-30: Step 4のCPU A/B初期測定を実施。synthetic 16-option short/long、3-option ambiguity、
E2B固定visual embeddingを比較し、tokenizer一致とraw/centered logits・候補softmax差を記録した。
top-1変更はこの7 paired comparisonsで0件だが、全fixtureに評価用の正解labelがないため誤答率・採用基準は未確定。
CUDA C/D、Step 5 GPU測定は未実施。詳細・由来の未確認事項は
[Step 4 CPU A/B記録](precision-baseline.md#2026-09-30-step-4-cpu-precision-baseline)を参照する。

2026-09-30: 旧option-scale→Step 1→Step 4の経緯と途中結果を再解析した。
Q4 GGUF本体でimatrix metadataを確認し、初期記録の記載漏れと画像fixtureの説明を訂正。
同一checkpoint由来・imatrix実体/再現設定は未確定。Step 1との差はcentered最大差で
A/Bより小さい5条件があるが、確率差では逆転やA/B差の約54–60%の例もあり、
十分小さいとの採用判定には至らない。全7表を再検算し、新規inferenceは行っていない。
比較条件の限界、意味品質とCUDA/384・512件の未測定範囲、残りの確認順は
[Step 4途中レビュー](precision-review-provenance.md#step-4-interim-review)を参照する。

2026-09-30: Unsloth公開hash/commitとlocal時刻を照合。E2B BF16/Q4は7月版、
E4B BF16は7月版、Q4は5月版に一致。E4B Q4新旧ヘッダの差はchat templateのみで
全720 tensor記述が一致。BF16/Q4の同型非量子化tensorもE2B 283個/E4B 339個でbyte一致。
元checkpointの厳密なrevisionと全量子化recipeは未確定だが、公開品との対応・世代差は説明できた。
正解・根拠を固定した8シナリオ/24条件を追加し、全4 GGUFでtoken/label境界一致を確認。
現行E4B Q4/CPU screeningは正解24/24、順序/候補追加の選択変更0件。
最小marginは3.668でnear-tieは未獲得。正解付きreference A/BとCUDA C/Dは未実施。
詳細と新fixtureの位置づけは
[由来照合とquality fixture記録](precision-review-provenance.md#step-4-provenance-and-quality-fixtures)を参照する。

2026-10-01: Step 4用の複雑な画像証拠をImageMagickで追加。盤面の色/形の計数、
状態画面の複数条件、表の適格性/最小値比較の3系統に、証拠1か所の変更と配置変更を用意した。
正解付き9画像/27条件だが、独立27標本ではなく6証拠シナリオと3配置変形。
全4 GGUFでprompt/回答ID一致、両mmproj設定でdecode/前処理、変更pixelの範囲、
同環境での再生成hash一致を確認。E4B BF16/Q4とE2B BF16/Q4をCPUでscreenし、正解数は
13/27、13/27、8/27、6/27。GPUは使っていない。strict shared-embedding/reference A/BとCUDA C/Dは
未実施でStep 4は未完了。
設計・正解・利用方法は[vision fixture説明](../../../fixtures/phase4-step4-vision/README.md)、
入力検証と[CPU比較記録](vision-screening.md#vision-multi-model-cpu-comparison-2026-10-01)、
個別raw出力は記録内にリンクした4 JSONLに保持する。

2026-10-01: structured-visionの108出力を再集計。E4BのBF16→Q4は正解→誤答2件と
誤答→正解2件が相殺し、同じ13/27でも選択は6件変化。E2Bは正解→誤答3件、
誤答→正解1件で8/27→6/27（比較記録の変化方向を訂正）。
証拠変更前後を両方正解した組はE4B各1/9、E2B各0/9。
E4B Q4のmargin 0.025936は上位2候補が双方誤答の数値stress例。
BF16でも残る誤答をモデル能力不足と断定せず、少数例のshared-embedding/reference照合と
事実を文章にしたcontrolで切り分ける。新規推論は行っておらずStep 4は未完了。
詳しい集計と考察は[CPU結果の再レビュー](vision-screening.md#vision-cpu-interim-review-2026-10-01)へ記録した。

2026-10-01: E4B Q4の5条件を同一CPU GGUFでllama.cppと照合し、画像encoder共有のtext prefill、通常mtmd経路のend-to-end、画像事実を文章化した5 controlを実行。text controlはbranchscore 5/5、llama.cpp 4/5。画像条件はいずれも各経路1/5で、上位選択はbranchscoreとllama.cppのE2Eで1/5一致。同一300-token embeddingを固定したllama.cpp Q4 prefillはbranchscoreの選択と4/5一致。branchscore埋め込みを通常mtmd経路へ差し替えるとllama.cpp mmproj実画像経路と4/5でwinner一致し、平均cosine 0.99999986、relative RMS差0.000529。near-tieのboard-relayoutでは画像encoder差で誤答winnerが入れ替わった。同一埋め込み・promptでllama.cpp BF16/Q4をA/Bし、双方1/5正解、winner3/5一致。tableのbase/reversed間では正解↔誤答が反転した（GGUF由来世代差があり量子化のみの効果とは断定しない）。通常mtmd経路にはbranchscoreの画像token列にない境界token 2個が加わる。GPUは使っていない。詳細・raw scoresは[CPU reference/control記録](../phase-4-plus-option-scale-2-vision-reference-controls-e4b-q4-cpu-2026-10-01.jsonl)、text-only runは[raw JSONL](../phase-4-plus-option-scale-2-vision-text-controls-e4b-q4-cpu-2026-10-01.jsonl)、入力は[text control fixture](../../../fixtures/phase4-step4-vision-text-controls.jsonl)。Step 4は未完了のまま。全27件のA/B、最適化C/D、CUDA品質・VRAM実測は残る。

2026-10-01: reference/controlを再レビューし、保存embeddingの300×2560 shape/hashと
各経路のtoken hash・scores・正解数を再検算した。現行spliceはupstreamの`<|image>` / `<image|>`
境界を保持しておらず、画像入力契約の実装差として優先確認する事項を特定。
同じ埋め込みでも境界なし→通常mtmdで選択4/5変化。ただしdecode helperも異なるため
全てを境界tokenのみの因果効果とは確定しない。純text status controlでもcentered最大差
0.706434で正解/誤答が分かれ、画像差とは別に調べる必要がある。
画像reference/branchscore差はこの5例ではBF16/Q4差より小さいが、許容済みとはしない。
次は小さい境界splice修正と固定5→27条件の照合を優先候補とする。新規推論/runtime変更なし。
詳しい根拠・数値とcontrolの限界は
[reference/controlの再レビュー](vision-reference-controls.md#vision-reference-control-review-2026-10-01)に保持する。


2026-10-01: 末尾再レビューへ対応し、画像rendererにupstreamと同じ`<|image>` / `<image|>`
境界を保持する修正とengineのtoken境界検証を追加した。27条件で変更は境界2tokenのみ、
回答ID不変・prompt hash整合を確認。固定decoderの境界あり5条件は既存mtmd共有embedding
結果と全候補logit一致し、境界有無でwinner4/5変化。修正branchscoreとreferenceは4/5一致。
27条件のE4B Q4 CPU正解は13/27→12/27、winner4件変化、正解→誤答2件・誤答→正解1件。
純text statusは同cache/attention/chunkでREPACK有無×最終/全位置出力を比較し、REPACKを
無効にするとlyra誤答→vega正解、行選択ではwinner不変。通常CPU bufferでもcentered差
0.422551が残り、完全数値一致・許容済みとはしない。focused testsを確認、GPU未使用。
境界修正、raw出力と残る確認は[再レビュー対応記録](image-boundary-buffer.md#vision-boundary-fix-2026-10-01)
に保持する。Step 4は未完了で、修正後全27条件reference A/BとCPU kernel照合、最適化・CUDA評価が残る。

2026-10-01: 境界修正とbuffer controlを再レビュー。27条件の入力・score差・正解数、
5条件referenceと2×2 controlを再計算し記録と一致。固定decoderにより境界有無での
選択4/5変化を確認でき、helper差の交絡を除いた。正解13→12は契約是正の取消理由にしないが、
証拠変更前後を双方正解した組は1/9→0/9で、意味品質の改善も認定しない。
純text statusのbuffer/kernel差を特定した一方、画像board-baseは修正後にも
正解/誤答が分かれるmargin 0.012167の例として残る。旧BF16と修正後Q4を同入力のA/Bにしない。
次は通常CPU buffer/build条件を揃えたstatusと画像5条件、その後に修正後27条件A/Bを優先する。
新規推論/runtime追加変更なし。詳しい根拠は
[境界修正とCPU bufferの再レビュー](image-boundary-buffer.md#vision-boundary-buffer-review-2026-10-01)に保持する。

2026-10-01: 追加レビューへ対応し、CPU referenceの通常bufferとLLAMAFILE OFF buildを用意。
reference vendor ggml 0.25.3とproject ggml 0.24.0のsource差、新版の既定tiled matmulを分離した。
純text statusはtiled OFFでcentered差1.790469→0.249105へ縮小し、診断用のkey軸paddingも
揃えると候補logitsと213中間tensorがbyte一致。画像5条件も候補logitsがF32一致し、
board-baseの正誤差をこのCPU shape controlで解消した。runtimeへpaddingを追加採用はしていない。
そのreference経路を固定した修正後27条件shared-embedding A/BはBF16/Q4とも13/27、
選択変更3件、正解→誤答/誤答→正解各1件。非padding branchscore Q4は12/27で、
referenceとの選択不一致/追加誤答はboard-baseの1件。centered差は27/27、確率差は25/27で
A/B差より小さいが、追加誤答を含むため許容/採用判定はしない。GPU未使用。
長文/chunk境界、E2B修正後基準、最適化B/C/D、CUDA・384/512・VRAMは残りStep 4は未完了。
rawデータ・診断source・比較条件は
[CPU kernel/key軸照合と修正後27条件A/B](cpu-kernel-padding-baseline.md#cpu-kernel-alignment-2026-10-01)
に保持する。

2026-10-02: CPU kernel/key paddingの追加検証を再レビュー。27条件の入力・scores・正解・差と
30経路比較を再計算し、診断6条件を別記録のreference F32 scoresと照合した。
statusの213 tensorは保存比較記録で全一致、現在残るreference実dumpも全hash一致。
project側status dumpは後続画像runで上書きされており、今回の両側実dump再比較はできていない。
kernel/key軸を揃えた画像5条件では分割差があっても候補が一致し、この範囲のCPU残差は説明できた。
paddingは未採用で、全27条件・523 positions / 2 chunk・長文/window/CUDAへ一般化しない。
BF16/Q4 reference各13/27は正誤変化各1件の相殺。証拠変更前後の双方正解はBF16 0/9、
Q4 1/9で、runtime照合の解決を画像判断の品質達成とはしない。次はpadding採用候補の
全27条件/境界focused検証と時間・メモリ差を確認し、品質基準を固定してB/C/D評価へ進む。
trace出力先の上書きとmanifestの空build_flagsも再現性上の補足として記録。新規推論なし。
根拠と残る範囲は[CPU kernel/padding再レビュー](cpu-kernel-padding-baseline.md#cpu-kernel-padding-review-2026-10-02)
に保持する。Step 4は未完了。
