# models/ — 推論用重み（配布）

realtime app / CLI が読み込む Candle 推論重み。**大きい重みは git に入れない**（履歴肥大回避）。

## freeC ボコーダ（低レイテンシ 5.8ms 合成 / streaming ~39ms E2E）

- `freeC.safetensors` (~107MB) — **git 管理外**（`.gitignore`）。別端末では以下で入手:
  1. **GitHub Release アセット**（推奨）: `gh release download <tag> -p freeC.safetensors -D models/`
  2. **Git LFS**: `git lfs pull`（LFS 設定時）
  3. **直接転送**: scp / クラウド / USB で `models/freeC.safetensors` に置く
- `mel_basis_44k_2048_128.safetensors` (513KB) — **git 追跡済**（clone で付属、固定 slaney mel filterbank）。

### 生成元（再エクスポート手順）
`training/checkpoints/freeC/foundation_lowlatency_5p8ms.pt` の `gen`（`window` buffer 除く）→ safetensors。
mel_basis = librosa slaney mel（sr44100, n_fft2048, n_mels128, fmin0, fmax=None）。

## 実行

```bash
# CLI 再合成（ファイル）
cargo run -p lightvc-app --release -- resynth \
  --input in.wav --weights models/freeC.safetensors \
  --mel-basis models/mel_basis_44k_2048_128.safetensors --output out.wav --k 4

# GUI（realtime タブの "FreeVocoder Resynth" ボタンでロード → mic ライブ）
LIGHTVC_FREEC_WEIGHTS=models/freeC.safetensors \
LIGHTVC_MEL_BASIS=models/mel_basis_44k_2048_128.safetensors \
  cargo run -p lightvc-app --release -- gui
```

grid: freeC = causal, n_fft/win=256, hop=128, mel 解析窓=2048/hop=128。E4 parity: SNR 88.5dB(vocoder) / 62dB(E2E resynth)。
