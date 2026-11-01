# LightVC 現行方針

> 更新: 2026-11-01（研究リセット・本線を synth_nam_vc へ。製品要件・採用ゲートは維持）。製品要件と設計索引の正本。
> 研究・設計・学習・評価の前に、本書と [RESEARCH.md](RESEARCH.md) を読む。

## 1. 目的と採用状況

ASMR・官能バ美肉向けのリアルタイムVCを作る。普通の発声から望む美少女音声・deliveryを生成し、人間が出力を聴きながら距離感・息・喉の質感・響き・抑揚を調整できることを目指す。

- 製品E2Eは **p95 <30ms**、理想20ms級、**p95 ≥50msは失格上限**。50ms未満だけでは合格ではない。
- 内容と発声への追従を保ち、sourceの話者性は目標の声へ変える。
- 息・囁き・小声、滑らかさ、近さ、長く聴ける快適さを重視する。
- 自動萌えと人間の連続制御を両立する。目的の演技を操作者が完成させることを前提にしない。
- zero-shotのtarget設定を最終目標として維持する。固定targetによる診断成功をzero-shot達成と呼ばない。
- 声・style・textureの分離と説明可能な編集を目指す。保存する編集の意味を不透明なlatentの座標だけに依存させない。

**2026-11-01 研究リセット(オーナー)**: これまでの出力は「適当にメルが近くなっただけ」の水準に留まった。本線を [synth_nam_vc.md](synth_nam_vc.md)(構音の構造のアナログモデリング・PROPOSED rev1)へ移す。理論の土台 = 構音の構造を変更すれば任意の話者に変換できる。フロンティア = 低レイテンシで実現可能なモデリング。学習データは Irodori-TTS v4.1 の生成・zero-shot を維持・目標の声は架空と同意済みの実在の両方。**リセット前の記録は全て撤回**で、以下の構音VC系の記述と資料も本線の根拠にしない。

**採用済みの完成VCは確認できていない。** B4 Dual-Path Kansei VCの目的の下で、直近に検証したのは **構音VCの前段・変換器 + rvoc系レンダラ**。表の引きと操作者の動き、参照を読む回帰型R、分布型Fを試したが、変換の明瞭さ・自然さ・目標話者性の同時合格には至っていない。部品の再合成・Rust一致・計算速度の証拠を、製品変換・萌えの制御・E2Eの合格と区別する。

Y-S1/CFMは以前に検証した経路であり、旧女声再合成の耳PASSだけを現在の品質天井にしない。現状の判定と欠測は [RESEARCH.md](RESEARCH.md)、観測・撤回・未判定の詳細は [現状監査](../results/research_audit_2026-10-13/README.md)。今回の整理で設計をADOPTEDへ昇格させていない。

旧freebig等の当時の採用記録は歴史として保存した。現製品経路の採用を意味しない。

## 2. 設計と現役資料

以下は目指す分担であり、採用済みモデルの配線図ではない。研究実装で各経路の分離が成立したかは別に検証する。

```text
入力 → 自作の因果content / 明示prosody ─┐
target参照 → voice / style / texture ──┤
人間の制御 → 条件の写像 ──────────────┤
                                      ↓
                      条件付き変換・生成 → レンダラ → 出力
                                                        ↓
                                             試聴・操作・選択
```

| 資料 | 役割 |
|---|---|
| [RESEARCH.md](RESEARCH.md) | 確認済みの到達点、証拠の所在、未解決の問い |
| [ROADMAP.md](ROADMAP.md) | 未決の問いと、判断に不足する証拠 |
| [synth_nam_vc.md](synth_nam_vc.md) | **本線(2026-11-01 リセット)**: 構音の構造のアナログモデリング・低遅延で実現できるモデリングがフロンティア（PROPOSED rev1） |
| [artic_vc.md](artic_vc.md) | 構音VCの設計・静的表と動きの診断経過（PROPOSED・製品未達） |
| [converter.md](converter.md) | 前段・変換器の設計、Rust経路、過去の比較（PROPOSED） |
| [renderer.md](renderer.md) | rvoc系出力部、再合成の耳記録、Rust一致（PROPOSED） |
| [artic_inv.md](artic_inv.md) | 声道の物理座標の逆推定とその限界（PROPOSED） |
| [paper_mf.md](paper_mf.md) | 論文の骨格: 男声 → 女声を女声だけの学習で一般化させる条件(界面の不変性・観測されない帯の外挿)・以後の腕はここの予測から導く(PROPOSED・2026-10-30) |
| [artic_morph.md](artic_morph.md) | 構音写像VC: 操作者の身振り × 目標の声道と動き方・固定の半音で Graillon 相当に勝つ（PROPOSED・2026-10-25） |
| [ref_reader.md](ref_reader.md) | 参照を注意で読む回帰型Rと診断結果（PROPOSED・製品未達） |
| [dyn_flow.md](dyn_flow.md) | 分布型Fと短時間診断のBREAK（PROPOSED） |
| [causal_codec.md](causal_codec.md) | 以前に検証したY-S1の詳細設計・ABI（PROPOSED・未採用） |
| [cfm_ys1.md](cfm_ys1.md) | 以前のCFM/Y-S1接続契約と診断（PROPOSED・未採用） |
| [kansei_control.md](kansei_control.md) | 官能評価、萌えのパラメタ化、再生側UX（PROPOSED） |
| [design_laws.md](design_laws.md) | 設計三層則と起動前検査L/I/C（規範・全学習腕必須） |
| [FAILURE_NOTES.md](FAILURE_NOTES.md) | 再試行前に確認する失敗・撤回・帰属の限界 |
| [EVALUATION.md](EVALUATION.md) | 評価記録とレイテンシ計測の共通契約 |

現行アーキテクチャの詳細は各ネットワークの一ファイルへ集約する。新しい仮説・数値・run状態をREADMEへ積み上げない。

## 3. 実装・学習の境界

- 推論はRust/Candle、学習はPyTorchのuv環境。Python runtime不要、conda禁止。
- 製品推論は自作のネットワークを使う。SSL/ASR・話者認識等の補助モデルは学習・評価・表現監督用とし、出荷時依存にしない。
- contentはSSL/ASR系、prosodyは明示特徴、voice/style/textureは別経路。DAC latentからspeaker-free contentを学ぶStage1へ戻らない。
- 別VCの変換音声をtargetにするteacher蒸留は禁止。実音声・同一内容ペア・検証された信号処理由来の教師を使う。
- GANは信号・内容・制御が成立した後のtexture fine-tune。失敗したcontentを救済しない。
- 製品経路の未来参照・発話全体正規化は禁止。dataset固定統計と因果stateを使う。
- PythonとRustの重みキー・正規化・時刻規約を一致させる。境界のsample rateとhopを明記する。
- Intel GPUはXPU。groups=1標準convを前提とし、既知のXPU backward問題を回避する。
- MIT、GPLv3依存禁止。VST3はMITのclap-wrapper経由。ASIO SDKをコミットしない。

データの用途を混同しない。codecと変換音の主教師・評価は実音声。TTSは補助用途を明記し、実音声の品質証拠に置き換えない。本番のデータ被覆と、切り分け用overfitを区別する。現存量・split・キャッシュは実行時に確認する。

## 4. 研究の進め方

人間の選択が最終的な判断材料になる。話者類似度・loss・音響proxyだけで昇格しない。診断の責任は研究側が持ち、評価者へ欠点の言語化を必須にしない。

実験前に仮説、対照、データ、ABI、判断条件、次の分岐を固定し、[design_laws.md](design_laws.md) の検査L/I/C（損失最適解・識別可能性・最ラク性）を記入する — 記入できない腕は起動しない。長時間学習の前に因果性と静的遅延台帳を確認する。失敗や保留をPASSへ読み替えない。既存の閾値変更は過去の明示承認と区別して記録する。

昇格は「RESEARCHの仮説 → PROPOSED設計 → overfit・耳・必要な実装ゲート → ADOPTED設計 → 本書の採用索引」。資料整理だけで採用状態を変えない。

## 5. 退役資料

2026-09-08以前の混在資料は [.archive/docs_2026-09-08/README.md](../.archive/docs_2026-09-08/README.md) に案内を残し、整理直前の23ファイルを完全保存した。

退役資料の「現在」「必須」「次に実行」は当時の記述であり、現行指示ではない。通常の作業で全履歴を読み直す必要はない。再試行を検討するときだけFAILURE_NOTESから該当記録を調べる。
