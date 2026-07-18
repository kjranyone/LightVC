# freebig (FreeVocoder) Rust/Candle 推論移植スコープ

> status: **PROPOSED**（設計スコープのみ・実装未着手）
> 最終更新: 2026-07-18
> 対象: 出荷ボコーダ `freebig` = `training/free_vocoder.py::FreeVocoder`、重み `checkpoints/freebig/foundation_bigvgan_parity.pt`（key `'gen'`）
> ルール遵守: 推論は全て Rust(Candle)・Python ランタイム不要・**PyTorch↔Rust 推論キー名は完全一致**・groups=1 標準 conv 前提（XPU depthwise 回避）
> 位置づけ: `current/vocoder.md` 枝B（BigVGAN/Vocos 系 自作 weights）の Rust ランタイム。既存 DAC decoder 経路とは別系統の新規モジュール。

---

## 0. 要約

- freebig は mel[128]→ISTFT-head の Vocos 型（F0 なし・自由位相）。既存 Rust 推論経路は **DAC latent 系**（`dac_model.rs`/`converter.rs`）で、**mel 入力ボコーダも STFT/iSTFT も未実装**＝新規モジュール `free_vocoder.rs` を起こす。
- Candle 側の部品（Conv1d / LayerNorm / Linear / gelu_erf / cos/sin/exp/clamp）は**全て既存 0.10.2 に有り**、conv/LN/Linear のロード規約も既存コードに前例がある。
- **唯一の欠落 = FFT**。candle-core 0.10.2 に rfft/istft は無い（現物確認済み）。iSTFT は **行列型 DFT（`ltv_render._build_ola_mm` と同型の GEMM）で自前実装**する。複素 dtype も無いので mag·cos/mag·sin を実 tensor 2 本で持つ。
- streaming（config C, causal, 5.8ms）は `torch.istft(center=True)` を使わず、`ltv_render._ola_fold` 相当の**カスタム causal OLA を逐次フレームで**回す。

---

## 1. 既存 Rust 推論経路の把握（現状）

`crates/lightvc-core/`（candle-core/nn/transformers 0.10, features: cuda/metal/hf-hub）:

- **モデル定義規約**（前例として流用できる）:
  - `weights.rs::load_varbuilder` — `unsafe VarBuilder::from_mmaped_safetensors(&[path], dtype, device)`。safetensors を mmap ロード。
  - `dac_model.rs::conv1d_plain` — `vb.get((out,in,k),"weight")` + `vb.get((out,),"bias")` → `Conv1d::new(w,Some(b),cfg)`。**Conv1d の重みレイアウト [out,in,k] は PyTorch と一致**（そのまま渡せる）。
  - `converter.rs::linear_layer` — `vb.get((out,in),"weight")`+`(out,)"bias"` → `candle_nn::Linear`。**Linear も [out,in] で PyTorch 一致**。
  - `converter.rs::layer_norm_layer` — `vb.get((dim,),"weight"/"bias")` → `candle_nn::LayerNorm::new(w,b,eps)`。
  - GELU は `Tensor::gelu_erf()` を使用（`flow_converter.rs`/`utte_adapter.rs` に前例）。**PyTorch の `nn.GELU()` 既定 = erf 型なので gelu_erf が正解**（`gelu()` は tanh 近似なので使わない）。
  - `CausalConv1d`（`converter.rs`）— 左パディングのみ `pad_with_zeros(D::Minus1, pad, 0)` の因果 conv 前例あり。freebig causal 版の padding に流用可。
- **mel 生成・STFT・iSTFT・窓関数は Rust に一切存在しない**（grep 済み）。現行 vocoder = DAC decoder（ConvTranspose 上采样, `dac_model.rs`）で、mel を経由しない。→ freebig は入出力とも新規。
- **入力 mel の供給元**は本スコープ外（上流 VC/prosody stage が mel[128] を出す前提）。ただしパリティ検証では Python が吐いた mel を Rust に読み込ませる（§5）。

定数（Python 側, `kansei_vocoder.py`/`ltv_render.py`）: `n_mels=128, dim=512, n_layers=8, NFFT=WIN=2048, NB=1025, HOP=512, SR=44100`。config C は `nfft/win/hop` 可変・`causal=True`（別途 grid 値を checkpoint/config から取得＝**要現物確認**）。

---

## 2. アーキ対応表（FreeVocoder → Candle）

forward（`free_vocoder.py:70-84`）の順で。PyTorch tensor は `[B,C,T]`（C=チャネル）。

| # | PyTorch 層/演算 | 形状 | Candle 実装 | state_dict キー（完全一致） |
|---|---|---|---|---|
| 0 | 入力パディング `F.pad(mel,(3,3))` 非causal / `(6,0)` causal | `[B,128,T]`→`[B,128,T+6]` | `Tensor::pad_with_zeros(D::Minus1, l, r)`（causal: `(6,0)`／非causal: `(3,3)`） | — |
| 1 | `embed = Conv1d(128,dim,7,padding=0)` | →`[B,512,T]` | `conv1d_plain(128,512,7,cfg{pad:0})` | `embed.weight` `[512,128,7]`, `embed.bias` `[512]` |
| 2 | `blocks.{i}` = ConvNeXtBlock1d ×8 | `[B,512,T]` | §2.1 参照 | `blocks.{i}.*`（下記） |
| 3 | `norm` = LayerNorm(dim) on transpose | `[B,512,T]`→(→`[B,T,512]`→LN→戻す) | `x.transpose(1,2)`→`LayerNorm(eps=1e-5)`→`transpose(1,2)` | `norm.weight` `[512]`, `norm.bias` `[512]` |
| 4 | `head = Linear(dim, 2*nb)` on transpose | →`[B, 2050, T]` | `x.transpose(1,2)`→`Linear`→`transpose(1,2)` | `head.weight` `[2050,512]`, `head.bias` `[2050]` |
| 5 | `mag = clip(exp(h[:, :nb]), max=1e2)` | `[B,1025,T]` | `h.narrow(1,0,1025)?.exp()?.clamp(f64::NEG_INFINITY,1e2)`（下限なし=`minimum`不要、`clamp_max` 相当） | — |
| 6 | `p = h[:, nb:]`（自由位相角） | `[B,1025,T]` | `h.narrow(1,1025,1025)` | — |
| 7 | `S = mag*(cos p + j sin p)` | 複素 `[B,1025,T]` | **複素 dtype 無し** → `re = mag*p.cos()`, `im = mag*p.sin()` の実 tensor 2 本 | — |
| 8 | `_istft(S)` center=True | `[B, T*hop]` | §3 自前 iSTFT/OLA | — |
| — | `register_buffer("window", hann_window(win))` | `[win]` | **重みからは無視**、Rust 側で hann を生成（§3・§4） | `window`（buffer, ロード時スキップ可） |

### 2.1 ConvNeXtBlock1d（`kansei_vocoder.py:37-62`, causal=False 既定）

`x:[B,512,T]`、`k=7, mult=3, dim=512`, 残差 `r=x`。

| 演算 | Candle | キー |
|---|---|---|
| pad: 非causal `(pad, k-1-pad)`=`(3,3)` / causal `(6,0)` | `pad_with_zeros(D::Minus1,...)` | — |
| `dw = Conv1d(512,512,7,groups=1)` | `conv1d_plain(512,512,7)` ※**groups=1**（depthwise ではない, XPU 安全） | `blocks.{i}.dw.weight` `[512,512,7]`, `.dw.bias` `[512]` |
| `transpose(1,2)` → `[B,T,512]` | `transpose(1,2)` | — |
| `norm = LayerNorm(512)` | `LayerNorm(eps=1e-5)` | `blocks.{i}.norm.weight`/`.bias` `[512]` |
| `pw1 = Linear(512, 1536)` | `Linear` | `blocks.{i}.pw1.weight` `[1536,512]`, `.pw1.bias` `[1536]` |
| `act = GELU()` | `gelu_erf()` | — |
| `pw2 = Linear(1536, 512)` | `Linear` | `blocks.{i}.pw2.weight` `[512,1536]`, `.pw2.bias` `[512]` |
| `transpose(1,2)` → `[B,512,T]` | `transpose(1,2)` | — |
| `return r + x` | `(&r + &x)` | — |

注: dw conv は `padding=0` で forward 内で明示 pad しているため、Candle 側も `Conv1dConfig{padding:0}` にして手動 pad を合わせること（`converter.rs::CausalConv1d` と同流儀）。

### 2.2 完全キー一覧（`gen` state_dict）

```
embed.weight, embed.bias
blocks.0..7 . { dw.weight, dw.bias, norm.weight, norm.bias,
                pw1.weight, pw1.bias, pw2.weight, pw2.bias }
norm.weight, norm.bias
head.weight, head.bias
window            ← buffer（hann, ロードしない or 検証にのみ使用）
```

Rust 実装は VarBuilder で上記キーをそのまま `vb.pp("blocks").pp(&i.to_string()).pp("dw")` 等で辿れば PyTorch と**キー完全一致**になる。IF/GCI 版（`mag_head`/`dphi_head`/`tau_head`/`res_head`）は freebig(=FreeVocoder) では**使わない**ので移植対象外。

---

## 3. iSTFT / OLA 実装方針

### 3.1 前提

- candle-core 0.10.2 に **FFT なし・複素 dtype なし**（現物確認: `candle-core-0.10.2/src` に rfft/fft/istft/fourier いずれもヒットせず）。
- よって iSTFT は **行列型逆 DFT + overlap-add** を自前実装。`ltv_render._build_ola_mm`/`_ola_fold`（backend="mm"）が**そのまま設計テンプレート**（XPU 安全 GEMM・cos/sin テーブル・fold）。

### 3.2 非causal（center=True）版 — オフライン/パリティ基準

Python `torch.istft(S, nfft=2048, hop=512, win=2048, window, center=True)` を再現。

1. **逆 DFT 行列**（定数, 起動時 1 回）: `[NB=1025] → [win=2048]` の実信号復元。
   - フレーム `f` の時間信号 `y_f[n] = (1/nfft) * Σ_k Wk * (re_k*cos(2πkn/nfft) - im_k*sin(2πkn/nfft))`、`Wk`=片側スペクトル重み（k=0 と k=nfft/2 は 1、他は 2）。
   - Candle: `re[B,NB,T]`,`im[B,NB,T]` を `[B*T, NB]` に reshape → GEMM `Y_cos = (re∘Wk) @ Cos^T`、`Y_sin = (im∘Wk) @ Sin^T`、`y_frame = (Y_cos - Y_sin)` → `[B*T, win]`。`Cos[n,k]=cos(2πkn/nfft)`, `Sin[n,k]=sin(2πkn/nfft)`、n∈[0,win)。
2. **窓掛け**: `y_frame *= hann_window[n]`（合成窓）。
3. **overlap-add + 窓正規化**: `torch.istft` は合成窓の二乗和で割る（NOLA 正規化）。`out[t] = Σ_f y_frame_f * w / Σ_f w²`。分母 `wsum2[t]=Σ_f w[n]²` は定数配列（フレーム位置のみ依存）→ 事前計算。
   - Candle に `F.fold` は無い → **手動 index-add**（`Tensor::slice_scatter`/`narrow`+加算のループ、または `[B, win, T]`→`pad`→ストライド和）。フレーム数 T は小さいのでループ可。
4. **center=True の縁**: 先頭 `nfft/2` サンプルを捨てる（PyTorch center パディング分）。出力長 `= T*hop`。

**要現物確認**: `torch.istft` の正確な窓正規化（`window` 二乗和で割る・NOLA）と、末尾フレーム端の扱い。パリティが取れるまで Python と数値照合（§5）で詰める。

### 3.3 causal streaming（config C）版 — カスタム OLA

`FreeVocoder._istft` のコメント通り、streaming では center=True istft を使わず **causal OLA**（`ltv_render._ola_fold` 相当）を回す。設計:

- モデル本体（embed/blocks/head）は既に因果（左 pad のみ）→ **1 フレーム mel in → 1 スペクトルフレーム out** の逐次。
- 各フレームで §3.2 の逆 DFT（`win` 長合成）→ **リングバッファへ overlap-add**。窓正規化分母 `wsum2` は定常状態では周期定数（`win/hop` が割り切れる格子）→ 事前計算した循環配列で割る。
- **確定できる出力サンプル**は「今フレームまでで OLA テールが閉じた分」= 先頭 `hop` サンプル/フレーム（tail `win-hop` は次フレーム待ち）。`latency_ms`（`free_vocoder.py:55-60`）= causal で `win/sr`（win=win, tail+1block）。config C の実 grid（nfft/win/hop, 5.8ms 目標）は checkpoint/config 依存＝**要現物確認**。
- 状態管理: `[win-hop]` 長の OLA 残差バッファ + conv 各層の左 context バッファ（ConvNeXt causal は k-1=6, embed は 6）。`streaming.rs` の既存 StreamingCodec の状態保持パターンを参考にできる。

### 3.4 レイテンシ内訳（目安, 要 grid 確定）

- アルゴリズム: config C causal で `win/sr`（例 win@5.8ms grid）。非causal は `+win/2` 先読み。
- 計算(RTF): backbone は Conv1d/Linear の GEMM のみ（8 blocks×dim512）＋ iSTFT GEMM（`[NB×win]` 行列）。CPU 単スレッドでも軽量想定だが **RTF 実測が必要**（§6）。目標 E2E <50ms は上流込み。

---

## 4. 重み変換（PyTorch .pt → safetensors）

1. Python 側（学習環境, uv）で `sd = torch.load("checkpoints/freebig/foundation_bigvgan_parity.pt", map_location="cpu")["gen"]`。
2. **キー名はそのまま**（§2.2, リネーム不要 = 完全一致ルールを自然に満たす）。`window` buffer は含めても Rust 側でロードしなければ無害（または export 時に drop）。
3. dtype: 学習は fp32/AMP。**推論 safetensors は fp32 で出す**（iSTFT は Python 側でも `h.float()` 済み＝fp32 前提）。Rust ロードも `DType::F32`。将来 CPU 高速化で f16/bf16 を検討する場合は §5 パリティを dtype 別に取り直す。
4. `safetensors.torch.save_file(sd, "checkpoints/freebig/freebig.safetensors")`。contiguous 化（`.contiguous()`）を各 tensor に適用。
5. Conv1d `[out,in,k]`・Linear `[out,in]` は **PyTorch と Candle で同レイアウト**＝転置不要。

※ 変換スクリプトは新規（`training/export_freebig_safetensors.py` 等）だが本スコープでは実装せず手順のみ規定。checkpoint は読み取りのみ。

---

## 5. パリティ検証計画（E4 パリティゲート）

同一 mel 入力で Python(freebig) と Rust(Candle) の出力波形一致を確認。

1. **入力固定**: 代表 mel `[1,128,T]`（実発話由来）を `.npy`/`.safetensors` で吐き、両実装に同じものを食わせる。乱数・F0 は無関係（F0-free）。
2. **段階 anchor**（バグ切り分け用, 各段の中間 tensor を Python から dump し Rust と照合）:
   - a. `embed` 出力 → b. `blocks.7` 出力 → c. `head` 出力 `[B,2050,T]` → d. `mag`/`p` → e. iSTFT 前 `re,im` → f. 最終波形。
   - LayerNorm eps・GELU(erf) 型・pad 幅の食い違いはこの段階照合で即座に露見する。
3. **合否指標**:
   - 中間 tensor: 相対 L∞ `< 1e-4`（fp32）。
   - 最終波形: **SNR ≥ 60dB**（`10log10(Σy_py²/Σ(y_py-y_rs)²)`）かつ **相互相関ピーク ≥ 0.9999 @ lag0**。istft の窓正規化差が主因になりやすいので §3.2 を最優先で詰める。
4. **causal streaming パリティ**: フレーム逐次 Rust OLA 出力 vs Python の（同 grid・カスタム OLA 参照実装）を比較。境界 hop の連続性（クリック無し）を確認。参照は `ltv_render.ltv_ola(backend="mm")` の OLA 部を mag/phase 合成用に流用した Python 版を別途用意。
5. **既存ハーネス流用**: `crates/lightvc-core/examples/parity_test.rs`・`snr_diagnostic.rs`・`decoder_finetune_parity.rs` が SNR/相互相関の前例。freebig 用に mel 入力版を追加。
6. **耳ゲート**は別（`current/vocoder.md` Gate V）。パリティは「Python と bit 近い」ことの数値ゲートで、音質採用ゲートではない。

---

## 6. ギャップ / リスク

- **[高] FFT 欠落**: candle にネイティブ iSTFT 無し → 行列 DFT 自前。数値正規化（NOLA 窓二乗和）で Python と一致させるのが最大の実装リスク。`ltv_render` mm 実装が実証済みテンプレートなので致命ではないが、**要現物確認**（`torch.istft` 内部の窓正規化を精読して式を確定）。
- **[高] `F.fold` 相当が無い**: overlap-add を手動実装。T ループ or ストライド和。性能とパリティの両立に注意。
- **[中] 複素数**: candle に複素 dtype 無し → re/im 実 tensor 2 本で全経路。cos/sin/exp/clamp は unary op として存在（確認済み）。
- **[中] CPU RTF / <50ms**: iSTFT GEMM `[1025×2048]`×T と 8 ConvNeXt blocks。XPU は depthwise 回避済み（全 groups=1）で安全だが、**CPU/XPU 実測 RTF が未知**＝ベンチ必須（`bench_pipeline.rs` 流儀）。
- **[中] streaming 状態管理**: OLA テールバッファ + conv 左 context の整合。causal grid（config C の nfft/win/hop）が checkpoint に紐づくか config 外挿かが**未確認**＝現物で確定してから実装。
- **[低] `clamp` 下限**: Python は `clip(max=1e2)` のみ（下限なし）。Candle `clamp` は両端要求なら下限に `-inf`/十分小さい値、または `minimum`(定数) 相当を使う＝実装時に片側 clamp API を現物確認。
- **[低] window buffer**: safetensors に `window` を残すと Rust の未知キー扱い。VarBuilder は get したキーのみ読むので無害だが、export で drop するのが綺麗。
- **[情報] config C grid**: 5.8ms・可変 nfft/win/hop の具体値は本 read 範囲で未確定。checkpoint メタ/学習 config を**要現物確認**。

---

## 7. 実装ステップ順（提案）

1. **重み export**（Python, §4）: `.pt['gen']` → `freebig.safetensors`（fp32, キー不変）。
2. **`free_vocoder.rs` 骨格**: VarBuilder ロード + embed/ConvNeXt×8/norm/head を実装（§2）。iSTFT はまず**非causal center=True**（§3.2）。
3. **中間 anchor パリティ**（§5-2 a〜d）: backbone を Python と一致させる（iSTFT 前で L∞<1e-4）。
4. **iSTFT パリティ**（§5-2 e,f）: 行列 DFT + OLA 正規化を詰め、波形 SNR≥60dB。
5. **RTF ベンチ**（CPU/XPU, §6）: 目標 grid で計測、閾値未達なら GEMM/バッファ最適化。
6. **causal streaming OLA**（§3.3）: config C grid 確定後、逐次フレーム化 + 状態管理 + streaming パリティ（§5-4）。
7. **耳ゲート**（`vocoder.md` Gate V）: 数値パリティ通過後、Rust 出力を人間の耳で確認して採用判定。
