# LightVC 現行方針

> 更新: 2026-11-28（段 P126 の耳の関門通過に伴い、採用構成と製品の既定書き出しを更新）。製品要件と設計索引の正本。研究の正本は [paper_snv.md](paper_snv.md)。
> 研究・設計・学習・評価の前に、本書と [RESEARCH.md](RESEARCH.md) を読む。

## 1. 目的と採用状況

ASMR・官能バ美肉向けのリアルタイムVCを作る。普通の発声から望む美少女音声・deliveryを生成し、人間が出力を聴きながら距離感・息・喉の質感・響き・抑揚を調整できることを目指す。

- 製品E2Eは **p95 <30ms**、理想20ms級、**p95 ≥50msは失格上限**。50ms未満だけでは合格ではない。
- 内容と発声への追従を保ち、sourceの話者性は目標の声へ変える。
- 息・囁き・小声、滑らかさ、近さ、長く聴ける快適さを重視する。
- 自動萌えと人間の連続制御を両立する。目的の演技を操作者が完成させることを前提にしない。
- zero-shotのtarget設定を最終目標として維持する。固定targetによる診断成功をzero-shot達成と呼ばない。
- 声・style・textureの分離と説明可能な編集を目指す。保存する編集の意味を不透明なlatentの座標だけに依存させない。

**2026-11-01 研究リセット(オーナー)**: これまでの出力は「適当にメルが近くなっただけ」の水準に留まった。本線を [synth_nam_vc.md](synth_nam_vc.md)(構音の構造のアナログモデリング・PROPOSED rev1)へ移す。理論の土台 = 構音の構造を変更すれば任意の話者に変換できる。フロンティア = 低レイテンシで実現可能なモデリング。学習データは Irodori-TTS v4.1 の生成・zero-shot を維持・目標の声は架空と同意済みの実在の両方。**リセット前の記録は全て撤回**で、構音VC系の旧資料は 2026-11-10 の棚卸しで削除した。

**採用(2026-11-28): 段 P126 の K3G-v4**([paper_snv.md](paper_snv.md) 段 P123〜P126・耳の関門はオーナーの判定「うーん、ええんちゃう」): 因果 2 系統高さ追跡(`v4_dual`)と瞬時 STFT マスク(`inst_mask = true`)を持つ前段(`results/snv_p126_gate_export`)→ 前段 `v4` 全コーパス特徴で学び直した 3 組の平均と錨(`results/snv_p126_k3g16kv_export`)→ 左寄せ包絡・時刻同期(`delay = 480`・`env_smooth = 0.25`)の出力部 `snv_p123_s`(`results/snv_p123_s/export_20k`)。因果写し合成 PESQ `3.5404`(旧 `T` 2.832)、開発の組 `devA` で CER `−0.0050`・`M +0.0351`(`k=0,7`)および **`k=12` `M +0.0307`(`0.3905 → 0.4238`)**、伴奏・拍手単体 `pad_db = −24.61 / −33.41 dB`、CPU/Rust 未来不変性 0 標本・静的遅延台帳 `25.0 ms`(予備なし実遅延 `15.0 ms`)・4 スレッド実時間関門 S `p99 = 4.89 / 4.67 ms < 5.0 ms` を通過した。それまでの採用(2026-11-19〜11-26): 段 P93 の K3G(オーナー耳判定「p11、k3gでは目標音声に対する話者の同一性が、人間の耳ではk3gが圧倒的に優れている。」・段 P95 で出力部平滑を復帰・段 P100/P104 で前段 v3.1 定数を採用・書き出し `results/snv_p45_gate_t80_export` + `results/snv_k3g16kv_export` + `results/rvoc3hi/export_330k` は保持)。未解決の問いは [RESEARCH.md](RESEARCH.md)。

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
| [RESEARCH.md](RESEARCH.md) | 確認済みの到達点、棄却仮説(negative results)の圧縮一覧、未解決の問い |
| [synth_nam_vc.md](synth_nam_vc.md) | **本線(2026-11-01 リセット)**: 構音の構造のアナログモデリング・低遅延モデリングの基本概念と初期較正（rev2 圧縮版） |
| [paper_snv.md](paper_snv.md) | **本線の論文・研究の正本(2026-11-06〜・rev248 圧縮整理版)**。オーナー方針(11-06)「では論文からのみ学習／評価を行うこと」: 命題 1・2・3、現行採用法(`K3G-v4` + `v4_dual` + `inst_mask` + `snv_p123_s`)、棄却仮説の体系表、評価規約(§7)、オーナー判断台帳(§9) |
| [windows_handoff.md](windows_handoff.md) | 段 P126 `K3G-v4` の製品経路を Windows 実機で検証・配布するための引き継ぎ |
| [renderer.md](renderer.md) | rvoc系出力部(採用済み描画器 `snv_p123_s` および基盤 `rvoc3hi`)の設計（ADOPTED・旧耳記録はリセット前で撤回） |
| [kansei_control.md](kansei_control.md) | 官能評価、萌えのパラメタ化、再生側UX（PROPOSED） |
| [design_laws.md](design_laws.md) | 設計三層則と起動前検査L/I/C（規範・全学習腕必須） |
| [ml_method.md](ml_method.md) | ML の計画法（単一の最適化指標・端から端の基準・誤差の分析） |

現行アーキテクチャの詳細は各ネットワークの一ファイルへ集約する。新しい仮説・数値・run状態をREADMEへ積み上げない。

## 3. 実装・学習の境界

- 推論はRust(`crates/lightvc-core` の自作の実装・入口は `snv_stream::SnvStream`)、学習はPyTorchのuv環境。Python runtime不要、conda禁止。
- 製品の経路は paper_snv の段 P126 の K3G-v4(2026-11-28 採用): 門と瞬時重みつきの前段(`results/snv_p126_gate_export` = `v4_dual`[`W_YIN` 600・`ENV_CAP_MS` 7.5・`TAU_UV_HZ` 500] + `inst_mask = true`)→ 符号と回路の 3 組の平均と錨(`results/snv_p126_k3g16kv_export` = `snv_p125{a,b,e}_c`・重みと注意の鍵と値は f16・`target_v4 = true`)→ 出力部 `snv_p123_s`(`results/snv_p123_s/export_20k`・48 kHz・hop 240・`delay = 480`・4 スレッド・因果 3 点平滑 `env_smooth = 0.25` + 出力の番 `out_guard`)。全客観関門・耳の関門(オーナー 2026-11-28「うーん、ええんちゃう」)・Rust 一致と実時間(関門 S: 1 ブロックの p99 最悪 4.89 / 4.67 ms < 5.0 ms)を通った。それまでの製品の経路は 段 P93 の K3G(`results/snv_p45_gate_t80_export` → `results/snv_k3g16kv_export` → `results/rvoc3hi/export_330k`)、さらに前は 前段 v3 → E / C(`results/snv_p11_r`)→ rvoc3hi。研究の候補(RESEARCH.md・成績表 results/LEADERBOARD.md。2026-11-12 から研究の基準の系は `snv_p30_c` の型 = paper_snv §7 rev86)は Rust に移していない。
- 製品推論は自作のネットワークを使う。SSL/ASR・話者認識等の補助モデルは評価の物差し・参考値まで。学習の目標・蒸留・表現の監督として直接使わない(オーナー判断 2026-11-01「既存アルゴリズムはベンチマークの参考値にしても良いが、直接教師にしない」)。出荷時依存にしない。
- prosodyは明示特徴、voice/style/textureは別経路。contentをSSL/ASRの表現で作る旧方針は上の判断で置き換えた。DAC latentからspeaker-free contentを学ぶStage1へ戻らない。
- 別VCの変換音声をtargetにするteacher蒸留は禁止。実音声・同一内容ペア・検証された信号処理由来の教師を使う。例外は Irodori-TTS v4.1 の生成音声(オーナー判断 2026-11-01・加工は入力側だけ)。
- GANは信号・内容・制御が成立した後のtexture fine-tune。失敗したcontentを救済しない。
- 製品経路の未来参照・発話全体正規化は禁止。dataset固定統計と因果stateを使う。
- PythonとRustの重みキー・正規化・時刻規約を一致させる。境界のsample rateとhopを明記する。
- Intel GPUはXPU。groups=1標準convを前提とし、既知のXPU backward問題を回避する。
- MIT、GPLv3依存禁止。VST3はMITのclap-wrapper経由。ASIO SDKをコミットしない。配布物の重みの学習データと第三者の許諾は [NOTICE](../NOTICE)。

データの用途を混同しない。
- 学習データは Irodori-TTS-v4.1-Small の生成(入力側も出力側も・VoiceDesign の架空の声・オーナー判断 2026-11-01)。
- 入力側の乱しにだけ、許諾の明らかな本物の非音声の音源を使う(CC0 / CC BY / PD・人の声を除く・オーナー判断 2026-11-08)。
- 評価は本物の男声の入力(外郎売・namikawa・VCTK)。JVNV / JNV は較正と評価の入力の側だけに使う。TTS の結果を本物の音声の品質の証拠に置き換えない。
- female-dataset は公開しない(オーナー判断 2026-11-10)。
- 本番のデータ被覆と、切り分け用overfitを区別する。現存量・split・キャッシュは実行時に確認する。

詳細は AGENTS.md の Data と `data/README.md`(ローカルだけ)。

## 4. 研究の進め方

人間の選択が最終的な判断材料になる。話者類似度・loss・音響proxyだけで昇格しない。診断の責任は研究側が持ち、評価者へ欠点の言語化を必須にしない。

実験前に仮説、対照、データ、ABI、判断条件、次の分岐を固定し、[design_laws.md](design_laws.md) の検査L/I/C（損失最適解・識別可能性・最ラク性）を記入する — 記入できない腕は起動しない。長時間学習の前に因果性と静的遅延台帳を確認する。失敗や保留をPASSへ読み替えない。既存の閾値変更は過去の明示承認と区別して記録する。

昇格は「RESEARCHの仮説 → PROPOSED設計 → overfit・耳・必要な実装ゲート → ADOPTED設計 → 本書の採用索引」。資料整理だけで採用状態を変えない。

## 5. 退役資料

リセット前の資料・`.archive/`・撤回された結果・旧経路の Rust 実装は 2026-11-09〜10 の棚卸しで削除した。また 2026-11-28 の整理で、棄却済み実験段の逐次ログを `paper_snv.md`・`RESEARCH.md`・`synth_nam_vc.md` 内で要約表へ圧縮し、孤立診断スクリプトと閉じた段のキャッシュを削除台帳(`results/DELETED_2026-11-28.txt`)に記録して整理した(圧縮前の逐次ログ全文とスクリプト本文は git コミット `b749b59` に保存)。再試行を検討するときだけ履歴から該当記録を調べ、撤回資料を本線の根拠にしない。
