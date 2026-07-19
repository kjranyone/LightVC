# models/ — 推論用重み（配布）

realtime app / CLI が読み込む Candle 推論重み。**大きい重みは git に入れない**（履歴肥大回避）。

## freeC ボコーダ（低レイテンシ 5.8ms 合成 / streaming ~39ms E2E）

- `freeC.safetensors` (~107MB) — **git 管理外**。**Hugging Face で配布**: <https://huggingface.co/mus8tte/lightvc>
  ```bash
  huggingface-cli download mus8tte/lightvc --include 'vocoder/*' --local-dir models_dl/
  # -> models_dl/vocoder/{freeC.safetensors, mel_basis_44k_2048_128.safetensors}
  ```
  （app には `--weights models_dl/vocoder/freeC.safetensors --mel-basis models_dl/vocoder/mel_basis_44k_2048_128.safetensors` で渡す、or `models/` にコピー）
- `mel_basis_44k_2048_128.safetensors` (513KB) — **git 追跡済**（clone で付属）＋HFにも同梱。

> LightVC の重みは Hugging Face の **monorepo `mus8tte/lightvc`** に component 別サブフォルダで集約（`vocoder/` 済、`content-encoder/` `vc/` `voices/` は今後）。

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
