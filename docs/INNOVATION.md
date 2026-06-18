# LightVC — 革新価値と実験意義

> LightVC は SOTA VC システム（Seed-VC / VEVO / Takin-VC / EZ-VC）の「追従」を目指すプロジェクトではない。DAC 潜在空間での軽量 codec-space VC が**段階的・分離的な話者特性制御**を実現できるかを検証する実験プロジェクトである。

---

## 1. 問題意識：SOTA の限界

2024-2025 年の SOTA zero-shot VC は、ほぼ全システムが同一パラダイムに収束した：

```
[全SOTA共通パイプライン]
  wav → SSL features (WavLM/HuBERT) → content tokens
                                     ↓
       speaker embed (CAM++/WavLM-SV) → CFM → Mel spectrogram → BigVGAN → wav
```

この「Mel 全体を生成し直す」アプローチは高品質だが、本質的な制約がある：

| 制約 | 原因 | 影響 |
|---|---|---|
| **制御粒度が粗い** | end-to-end で Mel を生成するため、音色・ブレス・抑揚を独立に制御できない | 「音色だけ変えたい」「ブレスだけ寄せたい」が不可能 |
| **ストリーミング困難** | Mel 生成に大 chunk（または全文）が必要。CFM は 10-32 step の ODE loop | リアルタイム変換の algorithmic latency が 300ms+ |
| **重い** | 300M+ params の generator + BigVGAN 122M。Python ランタイム必須 | デスクトップ単独動作が困難 |
| **ブラックボックス** | 何を変えて何を変えていないかが不透明 | VC の「倫理的制御」(何を変換したか明示) が困難 |

## 2. LightVC の仮説

> **DAC 潜在空間（1024-dim, 86Hz）は、話者特性を階層的に保持している。  
> この階層構造（RVQ depth）を軸に変換を分離すれば、  
> 音色・ブレス・抑揚を独立に制御できる軽量 VC が実現する。**

### 2.1 なぜ DAC 潜在空間か

DAC (Descript Audio Codec) は 44.1kHz 音声を 1024-dim × 86Hz の連続潜在表現に圧縮する。9 codebook の Residual Vector Quantization (RVQ) により、情報が depth 方向に階層化されている：

| RVQ depth | 情報の性質 | VC での意味 |
|---|---|---|
| codebook 1-3 (coarse) | 音素識別・ピッチレンジ・話者大枠 | **音色変換**（性別らしさを含む） |
| codebook 4-6 (mid) | スペクトル包絡詳細・フォルマント | **ブレス・話質**（声の質感） |
| codebook 7-9 (fine) | 微細ノイズ・周期性の揺らぎ | **質感・息もれ**（ささやき・ざらつき） |

SOTA の Mel 生成アプローチでは、これらが 1 つの 128-dim スペクトログラムに平坦化され、分離不可能。DAC latent では、depth 軸が**構造的な分離軸**として存在する。

### 2.2 制御の粒度

LightVC の推論式：

```
z_out = z_src + velocity_scale × Σ_i ( depth_strengths[i] × v_i(z_src, speaker_embed) )
```

3 つのランタイム制御軸：

1. **velocity\_scale**: 変換の全体強度。0.0 = identity（変換なし）、1.0 = 通常、>1 = 強調（diffusion CFG と同義）
2. **depth\_strengths\[i\]**: RVQ depth group 別の変換強度
   - `(1, 0, 0)`: coarse のみ → 音色だけ変換、ブレス・質感は source 維持
   - `(1, 1, 0)`: coarse + mid → 音色 + ブレス、質感は source
   - `(0, 1, 1)`: mid + fine のみ → 音色は source、ブレス・質感だけ target
3. **ProsodyMode**: 抑揚（F0/エネルギー輪郭）の取り扱い
   - PreserveSource: 元の抑揚を維持
   - Blend: source/target の抑揚を混ぜる（prosody\_blend で比率）
   - ImitateTarget: target 話者の抑揚を模倣
   - FlattenPrivacy: 抑揚を平坦化（プライバシー保護用途）

この組み合わせで、例えば「男性アナウンサーの声色を女性に変えつつ、落ち着いた抑揚は維持し、ブレス感だけ女性寄りにする」という**繊細な制御**が可能になる。SOTA ではこのような制御は構造的に不可能。

### 2.3 ストリーミング・軽量性

| 性質 | LightVC | SOTA (Seed-VC/EZ-VC) |
|---|---|---|
| Algorithmic latency | **12-186ms**（mode 切替可） | 300ms+（ODE loop 必須） |
| Params | **100-120M** (converter + DAC) | 400M+ (generator + vocoder) |
| Runtime | **Pure Rust (Candle)** | Python + PyTorch |
| 推論 step | **1-NFE**（rectified flow） | 10-32 step（CFM ODE） |

12ms の Strict mode は、DAC encoder/converter/decoder の各 conv の受容野のみで決まる。Mel 生成アプローチは原理的にこの遅延を達成できない。

## 3. 実験の核心問い

LightVC の成否を分ける 3 つの問い：

### Q1: 品質 — DAC latent で意味のある話者変換ができるか？

SECS > 0.50（held-out speakers）が最低ライン。Phase B (SpeakerEncoder 蒸留) + Phase C (speaker-sim 直接最適化) で検証する。

→ 09-01 で「DAC SpeakerEncoder が WavLM-SV を cos=0.95 で再現できる」ことを確認済み。DAC latent に話者情報は十分ある。

### Q2: 分離性 — depth\_strengths で音色・ブレス・質感を独立に制御できるか？

これが **LightVC の研究価値の核心**。Phase E で検証する：

- coarse のみ変換 → 音色が変わり、ブレス・質感は維持されるか？
- fine のみ変換 → 質感が変わり、音色は維持されるか？
- 人間評価（ABX）で「女性らしさ」「ブレス感」「自然さ」が独立に変化するか？

分離性が確認できれば、**「潜在空間での階層的話者特性制御」が初めて実証**される。これは Mel 生成アプローチにはない独自能力であり、学会発表の核になる。

### Q3: 実用性 — デスクトップで real-time 動作するか？

- 12ms (Strict) / 93ms (Balanced) / 186ms (Quality) の 3 mode
- CPU/GPU 両対応（Candle の device 抽象）
- Pure Rust バイナリ（依存なし、double-click で起動）

これは SOTA システムにはない**デプロイ性**。オフライン・プライバシー重視・低遅延の用途（配信・ゲーム・オンライン会議）で唯一の選択肢になる。

## 4. SOTA との差別化マップ

```
                    制御粒度（細かい ↑）
                         │
          LightVC ●      │
          （目標）         │
                         │
    ─────────────────────┼─────────────────
                         │
                         │     ● Seed-VC
                         │     ● VEVO
                         │     ● Takin-VC
                         │     ● EZ-VC
                         │
                    制御粒度（粗い ↓）
    
    軽量・高速 ←─────────────────────→ 重い・低速
          ●                        ●
       LightVC                  SOTA勢
```

LightVC は「軽量 × 高制御粒度」の未開拓領域を狙う。品質が SOTA の 70-80% でも、制御性とデプロイ性で独自の価値を持つ。

## 5. 失敗シナリオとその意義

### 5.1 Q1 品質で失敗（SECS < 0.50）

DAC latent の表現力が不足。→ Mel 生成移行（凍結中の γ）を再評価。

ただし、この失敗自体が **「codec latent では VC が困難である」というネガティブリザルト** として価値がある。次の研究（どの codec なら可能か？どの圧縮率まで耐えるか？）の出発点になる。

### 5.2 Q1 成功・Q2 分離性で失敗

DAC latent は話者変換できるが、RVQ depth で音色/ブレス/質感が分離できない。→ depth\_strengths は意味を持たず、単一の velocity\_scale で「全部まとめて変える」SOTA と同じになる。

この場合でも、Pure Rust real-time codec-space VC としては実用価値がある。ただし LightVC の「実験プロジェクトとしての新規性」は薄くなる。

### 5.3 Q1・Q2 成功・Q3 で課題

分離制御はできるが、Rust/Candle の推論速度が実用的でない。→ CUDA kernel の最適化、または ONNX 経由の GPU アクセラレーションで解決可能。アーキテクチャ上の問題ではない。

## 6. ライセンス・倫理

- **MIT ライセンス**: 全コンポーネント（DAC, WavLM, 本プロジェクト）が MIT/Apache。商用利用可能。
- **VC teacher 不使用**: 別 VC モデルの出力を教師としない。target = 実音声の DAC エンコードのみ。
- **変換の透明性**: depth\_strengths と ProsodyMode が「何を変えたか」を明示。これは deepfake 懸念に対する技術的アプローチ。
  - FlattenPrivacy mode は、 speaker anonymization 用途を想定
  - PreserveSource prosody は、「内容の抑揚は変えない」という倫理的制約を技術的に保証

## 7. ロードマップ概要

```
Phase 0  事前検証           ← 09-01 PASS (cos=0.95), 09-02a 完了
Phase A  診断
Phase B  SpeakerEncoder 蒸留
Phase C  speaker-sim直接最適化
Phase E  ★分離性・表現力検証★  ← 研究の核心
Phase D  WavLM補助 fine-tune
```

Phase E の結果が、このプロジェクトの論文・発表の結論を決める。

---

## 更新履歴

- 2026-06-18: 初版作成。Phase 0 検証結果と設計レビューを踏まえた革新価値の整理。
