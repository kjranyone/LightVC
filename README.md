# LightVC

リアルタイム声質変換（開発中）。本パッケージは v2f 因果ボコーダの評価版です。

## 使い方（CLI）

```
lightvc-app v2f input.wav -o output.wav          # ボコーダ resynthesis
lightvc-app vc  input.wav --shift 15 -o out.wav  # 声質変換（男声→目標話者、実験段階）
```

44.1 kHz mono WAV を入力してください。重み・行列は `models/` に同梱
（`vc` は `models/e1.bin`＝content encoder と `models/g1.bin`＝voice cartridge を使用。
`--shift` は半音単位の音高ノブで、声の高さに合わせて調整してください）。

## 遅延

- 設計目標: E2E p95 < 30 ms（ASIO io=64 で 27.4 ms、実測台帳は latency_decision.md）
- **Windows の WASAPI 共有モードでは 54.5 ms（失格域）になります。30 ms 級には
  ASIO ドライバが必要です**（ASIO SDK は再配布不可のため同梱していません）。
- 512/1024 サンプルのホストバッファは失格域（50 ms 超）です。

## 品質の現状（正直な開示）

- 因果・低遅延制約下の resynthesis 品質: PESQ ~1.98（eval 集合、394k step）/ ~1.64（域外男性声）
- 非因果の診断系（出荷不可）は同条件で ~2.9 であり、**差は縮小中**（学習継続中）。
- 本版は昇格判定（PESQ ≥ 非因果系）を**まだ満たしていません**。評価版です。

### 声質変換（vc）の現状

- 目標話者類似（ECAPA SECS）: 0.72（合格線 0.50、本人実録の上限 0.76）
- 明瞭度: 大きな音高シフト（+14〜21 半音）では機械書き起こし CER ~0.7 と
  未成熟です。同レジスタ入力（シフト小）では CER ~0.11。
- 付属 cartridge は単一目標話者の評価版です。
