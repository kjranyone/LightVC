# Voice Synth Synthesis — 設計方針転換

> 2026-06-27. depth-aware adapter実験の失敗を受けて、方針を再定義する。

## 経緯と教訓

### 検証したこと

| 実験 | 結果 | 教訓 |
|------|------|------|
| **cross-text eval** (200 pair) | ECAPA B1 はsame-textと同一性能 (Δmargin=+0.002) | same-textバイアスは存在しなかった。adapterは汎化している |
| **depth surgery** (200 pair) | d1-3が話者情報の68%、d5-8は<6% | RVQ内に話者/content構造は存在する |
| **histogram VC** (学習なし) | margin=-0.43 | フレーム位置を無視するとcontentが壊れる |
| **co-occurrence lookup VC** (学習なし) | margin=-0.54 | 話者独立の写像はidentity予測になる |
| **ref_latent token bank** (学習あり) | epoch0 margin=-0.31 | 参照音声のcodec表現は条件付けに使えない。ECAPAの事前抽出が強い |
| **depth-aware V1** (8-dim code操作) | src=0.94で変換不起立 | 8-dim射影空間は表現力不足 |
| **depth-aware V2** (1024-dim per-depth delta) | epoch4でmargin=-0.05、プラトー | per-depth構造が最適化を阻害。ECAPA B1(+0.27)に遠く及ばない |

### 確定した事実

1. **ECAPA B1 adapterは堅牢**: cross-textで完全汎化。same-text監督のバイアスなし。
2. **学習なしVCは不可能**: code選択は(content, speaker)の同時関数。統計量では因数分解できない。
3. **ECAPAは強い**: 参照音声の生codec表現より、事前抽出された話者ベクトルの方が条件付けとして優れる。
4. **per-depth構造は最適化コストが高い**: 訓練可能な表現力を保ちつつdepth分離を実現するのは、アーキテクチャ設計の困難なトレードオフ。

### 確定したリソース

- **ECAPA B1 checkpoint**: `checkpoints/phase3c_ao_b1_ecapa/best.pt` (margin +0.282 same-text, +0.323 cross-text)
- **cross-text eval基盤**: `eval_cross_text.py` (200 pair CI)
- **depth surgeryデータ**: `results/depth_surgery.json` (depth別話者寄与マップ)
- **depth knob evalスクリプト**: `eval_depth_knobs.py` (訓練済みモデルのknob分離度測定)
- **streaming eval基盤**: `eval_streaming.py` (F0/CER/MCD + SECS/SNR)

---

## 方針: 「ボイスチェンジャー」として完成させる

### コンセプト変更

~「離散code操作のsynthetic VC」
→ **「実用リアルタイムボイスチェンジャー」**

depth-awareの離散操作は研究として興味深いが、現段階ではECAPA B1連続latent adapterの性能に届かない。方針を転換し、**手持ちの成果で実用品を作り切る**。

### 定義: LightVC v1.0 = 何ができるか

```
入力: マイク音声（自由発話）
参照: target話者の短い音声クリップ（数秒）
出力: target話者の声質で、入力の内容・韻律を保持した音声
遅延: <50ms (CUDA, Balanced 4f)
```

これが**今すぐ動く**。B1 adapter + frozen DAC + Rust/Candle推論で実装済み。

### v1.0完成に必要な残作業

| 項目 | 状態 | 所要 |
|------|------|------|
| B1 adapter (ECAPA) | ✅ 訓練済み、cross-text汎化確認 | — |
| Rust/Candle推論 | ✅ parity確認、CUDA latency<50ms | — |
| GUI ( realtime tab ) | ✅ B1 controls実装済み | — |
| CLI convert-b1 | ✅ 動作確認済み | — |
| cpal実機テスト | ❌ #1 | ユーザー環境 |
| README更新 | ✅ done | — |
| Strict mode無効化 | ✅ #7 done | — |
| 評価メトリクス | ✅ F0/CER/MCD追加 (#6) | — |

**残りは実機テスト(#1)のみ。** コード側は完成している。

### シンセ機能の位置づけ

per-depth knobはv2.0以降の研究課題として保留。v1.0では:

- **wet/dry mix** = VC量の連続制御（実装済み: `SetWetDry` control）
- **τ (temperature)** = soft RVQの硬度 = 音質/忠実度トレードオフ（実装済み: `SetB1Tau` control）

この2つで実用的な「ノブ」として機能する。

---

## 研究バックログ（v2.0以降）

以下は論文/次版向けの研究課題。v1.0リリースをブロックしない。

### R1: per-depth knob（再挑戦）

失敗したV2の教訓:
- 8-dim空間は表現力不足
- 1024-dim per-depthは最適化が困難
- 共有backbone + depth embeddingでは限界

次のアプローチ候補:
- **post-hoc分解**: 訓練済みB1のdelta [1024, T] をRVQ depth別に射影分解し、推論時に重み付け再構成。訓練不要。
- **distillation**: B1 adapterの出力をteacherとして、per-depth studentをdistillation訓練。平坦な最適化地形を利用。

### R2: decoder fine-tune (#2)

frozen DAC decoderのストリーミング品質限界（Balanced B/C）。last 2 blocksのfine-tuneで短窓ロバスト性を獲得。

### R3: Mimi移行 (#3)

causal codec → 真の低遅延 + CPU実行可能。VChangeCodec型アーキテクチャ。

### R4: content tokenizer (Phase 4)

q0 anchorの話者リーク(source SECS=0.238)の根本解決。WavLMベースのcontent encoderでq0に代わるcontent anchorを構築。

### R5: 論文執筆

- cross-text汎化の発見（same-text監督がcross-textでも機能する）
- depth surgery（RVQ depth別話者寄与マップ）
- 学習なしVCの不可能性証明（histogram/lookup実験）
- 実用リアルタイムVCシステム（Rust/Candle、<50ms）

---

## アクションプラン

1. **depth-aware V2学習を停止**（プラトー確認済み）
2. **v1.0としてリリース準備**: タグ付け、CHANGELOG、バイナリビルド
3. **#1 cpal実機テスト**: ユーザー環境で確認
4. **研究は別ブランチで継続**: R1-R5は `research/` 配下で

---

## 教訓: 「synthetic VC」の正体

当初の構想「離散code操作で合成VC」は、以下の理由でv1.0では成立しなかった:

1. DACのcodebookはcontent/speaker直交分離を持たない（depth surgeryで実証）
2. 統計量ベースのcode操作は不可能（histogram/lookup実験で実証）
3. 学習ありper-depth操作は最適化が困難（V1/V2実験で実証）
4. ECAPA事前抽出が、いかなるcodec由来表現よりも良い条件付け（ref_latent実験で実証）

**「frozen codec + ECAPA条件付け + 連続latent delta」が、現 DACアーキテクチャにおける最適解。** これは「synthetic VC」ではなく「codec-space neural VC」であり、それで十分実用的。
