# ハイブリッド DSP + 薄ネット ボコーダ 実装調査（一次資料）

> 状態: RESEARCH 付随資料（揮発）。最終更新 2026-07-14。
> 目的: `current/RESEARCH.md` の「DSP+薄ネット ハイブリッドボコーダ」フォークの設計判断に使う実装粒度の文献根拠。
> 設計対象: **DSP(harmonic+noise 励起 + FIR/LPC 時変フィルタ) + 薄ネット(高域 MVF/aperiodicity/位相 の時変予測)**。
> 要件: CPU realtime <50ms × 44.1kHz × ASMR(息・囁き・弱基音) × 自作 weights。
> 狙い: min-phase A/S(WORLD/NSF-LTV)が耳で死んだ天井を、**学習位相 + 薄ネット**で突破しつつ、BigVGAN(122M)級より遥かに軽く CPU-RT 化する。
>
> 表記: 数値・帯域・値域は原典引用。原典に無い論理的帰結・二次情報は「(推定)」と明記。

---

## 0. ライセンス早見（プロジェクト規約: MIT/BSD/Apache 可、GPL 系禁止）

| 実装/資産 | ライセンス | 可否 |
|---|---|---|
| LPCNet (xiph/LPCNet) | BSD-3-Clause | 可 |
| FARGAN (xiph/opus 内 `dnn/torch/fargan`) | BSD-3-Clause | 可 |
| Meta DDSP Vocoder (arXiv:2401.10460) | 論文 CC BY 4.0 / 公式コード有無未確認(推定) | 論文可, コード要確認 |
| torchlpc (yoyololicon/torchlpc) | MIT | 可 |
| GOLF (yoyololicon/golf) | MIT | 可 |
| DDSP-SVC / pc-ddsp (yxlllc) | MIT | 可 |
| **SawSing (YatingMusic/ddsp-singing-vocoders)** | **AGPL-3.0** | **✗ コード流用不可**(設計参照のみ) |
| WORLD / D4C (mmorise/World) | modified-BSD, 特許なし | 可 |
| h-NSF / NSF (nii-yamagishilab) | BSD-3-Clause | 可 |
| iSTFTNet / HiFTNet(yl4579) / Vocos(gemelo-ai) / SiFiGAN(chomeyama) | MIT | 可 |
| NHV 参照実装 (k2kobayashi) | MIT | 可 |
| DDSP (magenta) | Apache-2.0 | 可 |
| CREPE (marl/crepe) | MIT | 可 |
| librosa (yin/pyin, ISC) | ISC | 可（pYIN の GPL 回避経路） |
| **pYIN Vamp plugin (c4dm/pyin)** | **GPLv2** | **✗ 禁止** → librosa.pyin(ISC)で代替 |
| **SEDREAMS / COVAREP** | **GPLv3** | **✗ 禁止** → ZFF/ZFR 独自実装で代替 |
| SHS (Praat 実装) | Praat=GPL | ✗ → Hermes 1988 論文から自前実装 |
| GlottDNN | 未確認(推定) | 要現物確認 |

**回避策**: (1) pYIN は c4dm/pyin(GPLv2) ではなく librosa.pyin(ISC)。(2) GCI 系は SEDREAMS(GPLv3) ではなく **ZFF/ZFR 独自実装**(0Hz 共振器+トレンド除去、単純数式で論文から実装可)。(3) SawSing のアイデア(鋸波 source + LTV-FIR + UV マスク)は再実装なら可、コード取り込み不可。

---

## 1. LPCNet — 「LPC は DSP、RNN は残差励起だけ」

出典: Valin & Skoglund, ICASSP 2019, arXiv:1810.11846 / 実装 https://github.com/xiph/LPCNet (BSD-3)

- **役割分担**: 線形予測(LPC)を信号処理側で担い、ニューラルは**励起 = 予測残差 e_t のみ**をモデル化。μ-law 領域で動作。
  - `p_t = Σ_{k=1..M} a_k·s_{t-k}`（DSP 側で LPC から計算）
  - sample-rate 網が `P(e_t)` を softmax 出力しサンプリング → `s_t = p_t + e_t`
- **frame-rate network**（10ms=160サンプル毎、軽い）: 入力 **20次元** = 18 Bark ケプストラム + 2 ピッチパラメータ(period, correlation)。`conv 1×3 → conv 1×3`(前後2フレーム文脈) + residual → `FC → FC` → 条件ベクトル f。
- **sample-rate network**（16kHz 毎サンプル）: `GRU_A`(**N_A=384**, block-sparse 16×1, 密度 d=0.1) → `GRU_B`(**N_B=16**, 密) → dual FC → softmax で **Q=256** μ-law レベル上の分布。前サンプル/前励起/予測/条件を embedding 経由で入力。
- **LPC 係数はネットが予測しない。DSP で計算**: 18 Bark ケプストラム → 線形 PSD → 自己相関 → Levinson-Durbin。次数 **M=16**(アーキ図読み取り, 推定込み)。プリエンファシス α=0.85。
- **複雑度**: 本文明示 **≈2.8 GFLOPS**(N_A=384, N_B=16, Q=256, Fs=16000 から算出)→「<3 GFLOPS」。**Apple A8 1コア / 2.4GHz Broadwell 1コアの20% で RT**。8-bit dot product(AVX2/NEON)前提。
- **品質**: 原論文は **MUSHRA**(MOS でない)。同一 dense-equivalent で LPCNet > WaveRNN+。※世間の「MOS≈3.2」は後発 FARGAN 論文側の再評価値。
- **サンプルレート**: 16kHz。**44.1k への単純スケール ≈7–8 GFLOPS(推定)** → 量子化なしでは CPU-RT 厳しい。
- **本設計との親和**: LPC(source-filter)を DSP に外出しし残差励起のみ学習 = current の source-filter 方向と思想一致。ただし sample-AR で 44.1k には重い。

---

## 2. FARGAN — LPCNet 後継、framewise AR + GAN

出典: Valin, Mustafa, Büthe, IEEE SPL 2024, **arXiv:2405.21069**(依頼の 2405.21077 は誤り) / 実装 xiph/opus `dnn/torch/fargan/fargan.py`(BSD-3) / デモ https://ahmed-fau.github.io/fargan_demo/

- **LPCNet 比の改善**: sample-AR + 密度推定をやめ **subframe 単位 framewise AR + GAN**。teacher-forcing 排除(露出バイアス回避)、明示ピッチ長期予測、~5倍低複雑度。
- **subframe 構造**: 10ms(16k)を **4 subframe = 2.5ms = 40サンプル**に分割し AR 生成。**波形サンプルを直接合成**(LPCNet の残差ではない)。出力に一次 IIR de-emphasis(α=0.85)。
- **frame 条件網**: FC + conv 3×1 + 転置conv(4×upsample)。入力 20次元音響特徴(18 BFCC + ピッチ周期 + 有声度) + **12次元ピッチ埋め込み = 32次元**。
- **subframe 合成網**: **GLU 複数層**(出力以外 tanh)、条件から得た gain 乗算、前 subframe 直接フィードバック。**820k params**(小型版 500k)。
- **ピッチ長期予測**: T≥40 で1周期前、T<40 で2周期前を参照。最大ピッチ 500Hz(最小周期32)。無声はゲート抑制。
- **識別器**: **6個の magnitude-STFT 識別器**(対数振幅入力, STFT サイズ 2^(k+5), 75% overlap, 2D 周波数 sin-cos 埋め込み)。LS-GAN + feature matching。
- **複雑度/品質**: **0.6 GFLOPS**(小型 0.35)、CPU 1コア <1% で RT。MOS FARGAN(large)≈3.7(CARGAN/HiFi-GAN v1 と同等)、LPCNet≈3.2、PESQ 3.298 vs LPCNet 2.539。16kHz。**44.1k スケール ≈1.6 GFLOPS(推定)** → 単一コアで十分 RT。
- **本設計との差**: 波形直接生成型で LPC 分離がない。DSP で source/filter を切り分けたい本設計とは思想が異なる(=DSP 事前分解の軽さを取りに行くなら Meta DDSP 系のほうが近い)。ただし「framewise AR + GAN + 明示ピッチ + 8-bit 量子化で CPU-RT」の工学は全面的に流用可(BSD)。

---

## 3. Meta 超軽量 DDSP Vocoder — 15 MFLOPS, MOS 4.36

出典: Agrawal, Koehler, Xiu, Serai, He (Meta), ICASSP 2024, arXiv:2401.10460 / 論文 CC BY 4.0

- **正体**: "Framewise WaveGAN" ではなく **微分可能 DSP(DDSP)source-filter ボコーダ**。ボコーダ部は行列積を持たず FFT/iFFT のみ。
- **DSP 部(学習パラメータ 0)**: 励振源 = インパルス列(F0駆動) + 白色雑音を周期性 P で混合。フィルタ = 声道 log-magnitude V(257次元)。合成 `iFFT([P·E_imp(F0) + (1−P)·E_noise] · V)` を周期/非周期別処理。
- **frame network が予測(270次元)**: F0(1) + 周期性 P(12) + 声道 V(257 log-mag)。波形から end-to-end 学習(抽出特徴を使わない)。
- **ニューラル部(=音響モデル, Emformer ベース = streaming Transformer)**:
  ```
  Linear(512→128)+Tanh+Dropout(0.1)
  Emformer ×4 (dim=128, ffn=512, memory=4, seg=32)
  Linear(128→199)+Tanh+Dropout(0.1)
  Linear(199→270)
  ```
- **なぜ 15 MFLOPS で MOS 4.36**: 波形生成を FFT/iFFT が担い、source-filter の強い事前分解で薄ネットは低次元フレームパラメータを出すだけ。sample-AR 不要。音響モデルと DSP を joint 微分最適化し DSP 劣化を学習補償。**MB-MelGAN 比 340倍 FLOPS 効率**。
- **パラメータ**: sr **24kHz**(44.1k でない)、FFT 512、hop 128 ≈ **5.3ms フレーム**、分解能 ~47Hz。vocoder RTF **0.003**、overall 0.044(1スレッド 2GHz Xeon)。
- **本設計への含意**: **これが本フォークの最短参照**。current の kansei が既に「DSP 励起 + 神経スペクトル」で A/S 天井を破った実測があり、それの CPU-RT 版がまさにこの構造。ただし Emformer(memory=4, seg=32)の lookahead が <50ms 予算では削減対象。44.1k 化・公式コード有無は要確認(推定)。

---

## 4. 微分可能 LPC/FIR と source-filter 実装の filter 表現・学習安定性

### 4-1. torchlpc / Differentiable All-pole Filters
出典: Yu et al., DAFx 2024, arXiv:2404.07970 / https://github.com/yoyololicon/torchlpc (MIT)
- **手法**: 時変 all-pole `y(n)=x(n)−Σ a_i(n)y(n−i)`。核心定理: **backward も同じ時変 all-pole を時間反転して通したもの**。custom op 化で BPTT 展開を回避。
- **制約(最重要)**: **pole 半径 r→1 で破綻**。|p|<1 でも **p(n)>0.98 付近で学習中に発散**を実測。対策 = **倍精度(double)で動かす**。
- **train-test mismatch**: 本手法の売りは逆で、frame ベース FIR 近似(周波数サンプリング法)と違い、all-pole 実装で学習した系は**サンプル逐次 RT へ変換しても汎化問題が出ない**と主張。
- **バックエンド**: 現行は C++/OpenMP + CUDA。**XPU/SYCL 記述なし → XPU では既存高速カーネル不動(推定)**。XPU で使うなら backward を自前 XPU テンソル演算で再実装が必要(推定)。API `sample_wise_lpc(x, A, zi)`。

### 4-2. GOLF
出典: Yu & Fazekas, ISMIR/TISMIR 2023, arXiv:2306.17252 / https://github.com/yoyololicon/golf (MIT, torchlpc 利用)
- **filter 表現 = 2次セクション(biquad)カスケード**(反射係数/LSF でなく複素共役極ペア)。次数 M=22。`LTVMinimumPhaseFilterPrecise` で最小位相再構成オプション。安定性は各 biquad を安定域に制約して保証(生の直接形 LPC を使わない設計思想)。
- **source**: transformed-LF 由来の声門流ウェーブテーブル(K=100×L=2048、2D 線形補間)。
- encoder: **Bi-LSTM 3層(~0.7M)**、mel 入力、200Hz でパラメータ出力。合成 = frame LPC + OLA。loss = MRSTFT(512/1024/2048) + F0 + V/UV BCE。

### 4-3. DDSP-SVC / pc-ddsp
出典: https://github.com/yxlllc/DDSP-SVC, https://github.com/yxlllc/pc-ddsp (MIT)
- 2モデル: **Sins**(正弦加算, pitch-shift でフォルマント動く) / **CombSub**(combtooth 減算, フォルマント不変)。harmonic+noise。
- **filter = 周波数サンプリング法の linear-phase FIR**(`core.py frequency_filter`): `irfft(magnitudes)` → `roll(ir_size//2)` 中央化 → **Hann 窓対称**(→linear-phase) → STFT ベース OLA(50% overlap の unfold→FFT乗算→Fold)。CombSub は `half_width=1.5·sr/f0` の**ピッチ適応窓**で LTV-FIR。
- 学習改良: volume aug, random-scaled STFT loss, UV 正則化, phase prediction。**CPU 軽量**(GTX-1660 級で 44.1k 可、SO-VITS より桁違いに軽い)。

### 4-4. SawSing（⚠ AGPL-3.0, コード流用不可・設計参照のみ）
出典: Alonso & Erkut 系, ISMIR 2022, arXiv:2208.04756 / https://github.com/YatingMusic/ddsp-singing-vocoders (AGPL-3.0)
- **sawtooth source + LTV-FIR**。FIR 係数を mel から NN 推定、位相連続性強制でグリッチ回避。source-filter の inductive bias で**3録音・3時間でも収束**。SawSinSub は aliasing 対策に正弦事前合成で鋸波近似。既知課題: 減算系は無声で buzzing → **UV マスク**で抑制。FIR は DDSP 系 linear-phase 周波数サンプリングと同系統(推定)。

### 4-5. FIR+OLA 微分可能学習の一般手法 & 安定性の定石
- **周波数サンプリング法**: NN が log-magnitude 予測 → `irfft` で IR → 中央 roll → 窓掛け(Hann=linear-phase / min-phase が欲しければ log-mag→Hilbert ケプストラム法) → STFT ベース OLA 畳み込み。
- **LSF/反射係数が直接形 LPC より安定な理由**(DSP 定石):
  - 直接形 a_i は安定領域が箱型でなく、補間で根が単位円外に出得る(torchlpc の p>0.98 発散はこの脆さ)。
  - **反射(PARCOR)係数**は lattice で |k_i|<1 が安定の必要十分 → tanh 束縛だけで安定保証・補間安全。
  - **LSF** は交互配置・単調順序を守れば安定、量子化/時間補間に頑健、フォルマントと対応良。
- **3系統の設計選択**: GOLF=biquad カスケード+最小位相 / torchlpc=倍精度で緩和 / DDSP 系=全極を使わず **FIR(常時安定)** で回避。**本設計は FIR 系(DDSP-SVC の frequency_filter)が最も安定・軽量で第一候補**。

---

## 5. MVF(Maximum Voiced Frequency)予測 — 息・囁きの表現核

### 5-1. 定義・役割(Stylianou HNM)
- MVF = 有声音で**周期成分と非周期成分を分ける単一のスペクトル境界**。**MVF 以下 = harmonic(正弦和)、以上 = noise(ランダム位相)**。time-varying で、初期 vocoder は固定 ~4kHz を使った。
- **次元 = フレームあたりスカラー1値**(周波数境界の時系列 contour)。

### 5-2. Drugman & Stylianou 適応 MVF 推定
出典: IEEE SPL 21(10):1230-1234, 2014, arXiv:2006.00521
- フロー: Windowing(**4周期長 Hanning 窓**) → FFT → 各倍音候補の peak-picking(p·ω0 の ±10Hz 最大) → 3特徴抽出 → ML 判定 → 時間平滑。
- 特徴: **AS**(局所 HNR, 主ローブ vs 残りの dB 差) / **IHPC**(群遅延の隣接倍音間差, 倍音=コヒーレント) / **ICPC**(T0 シフト位相差, FFT 追加1回)。**IHPC 単独が最良**。
- 判定: ML 基準 `MVF = argmax_m [Π_{k≤m} p(x_k|H1)·Π_{l>m} p(x_l|H0)]`(境界以下=全倍音, 以上=全非倍音)。特徴分布はガウス近似。高ピッチ声で従来法(P2V, SLM)に CMOS +1.8。Matlab 実装は tcts.fpms(COVAREP 収録)。

### 5-3. ニューラルで MVF 時変予測(sinc-h-NSF)
出典: Wang & Yamagishi, SSW 2019, arXiv:1908.10256 / nii-yamagishilab (BSD-3)
- base-h-NSF は harmonic/noise を**固定 FIR** + U/V で 2択 MVF 切替(voiced: LP5k/HP7k, unvoiced: LP1k/HP3k)。sinc-h-NSF はこれを**時変・学習可能**化。
- **予測対象 = 正規化カットオフ f_t^c ∈(0,1) のフレームごとスカラー**(=MVF/Nyquist)、5ms(200Hz)。帯域ベクトルでなく単一境界。
- 予測経路: condition module に Bi-LSTM + tanh conv で残差 r_t∈(−1,1)、U/V を v_t∈{0.7,0.3} にマップ。融合 `f_t^c = a·v_t + b·r_t + c`。
  - **sinc1: `f=v_t+0.2·r_t` が最良**。sinc2(U/V 無し)=voiced で MVF 過小推定し失敗。sinc3(sigmoid)=飽和し f=1.0 固定で noise 消失し失敗。
  - **教訓: 「U/V を prior にして残差を足す」が最安定。スクラッチ位相/MVF 予測は不安定**。
- windowed-sinc(Hamming, **M=31**)で LP/HP 係数を解析生成、f まで解析的逆伝播。16k, mel80+F0。MOS は WaveNet 同等。
- ※Sinsy 等 SVS が MVF スカラーを回帰する確証は未取得(推定)。明確な NN-MVF は sinc-h-NSF 系。

### 5-4. 低 MVF での breathy/whisper & F0 寛容性
- **MVF↓ → harmonic 帯域が狭まり高域が全て noise 化** → 息・気息を自然表現。whisper(F0=0)は正弦励起がガウス雑音に置換され実質 MVF→0(全帯域 noise)。Drugman 論文の「高ピッチで MVF 過小推定 → noise 過剰」は「MVF↓⇒noise↑」の直接証拠。ASMR ではこれが望ましい方向。
- **低 MVF 時に F0 誤差の可聴影響が減る根拠(機構は強固だが定量は推定)**: F0 誤差は harmonic 帯域内の倍音配置ズレとしてのみ可聴。MVF 以上はランダム位相 noise で正確な周波数配置を持たない → MVF を下げ harmonic 帯域を狭めるほど F0 依存倍音の本数が減り、F0 ズレ寄与が縮小。→ **ASMR/囁きの低 MVF 運用は F0 に寛容な領域**。current の「この声は F0 追跡不能」問題を吸収できる読み。

---

## 6. aperiodicity(ap/BAP)予測

### 6-1. WORLD D4C
出典: Morise, Speech Communication 2016 / https://github.com/mmorise/World (modified-BSD, 特許なし)
- ap = **信号全体パワーと非周期成分パワーの比**(帯域依存, 複数帯域)。group-delay ベースの temporally-static 表現。
- **値域**: 各ビン **0<ap≤1**(1=完全 aperiodic/noise、小=周期的)。silent は −60dB フロア。
- **符号化帯域数(codec.cpp `GetNumberOfAperiodicities`)**: `n = min(15000, fs/2−3000)/3000`(**3kHz 刻み、上限15kHz**)。fs=16k→**1帯域**、fs=48k→**5帯域**。符号化 `20·log10(ap)`(dB) + 粗補間。
- WORLD = F0(DIO) + 包絡(CheapTrick) + ap(D4C)。合成時 ap で各帯域の周期(パルス):非周期(noise)混合比を決める。

### 6-2. ニューラルで BAP 回帰
出典: Merlin (CSTR-Edinburgh), NNSVS 等
- 標準構成: DNN/LSTM が **MGC + BAP + log-F0 (+ V/UV)** を連続値回帰。**BAP 次元 = fs 依存**(16k→1, 48k→5, 3kHz 帯域ハードコード)。

### 6-3. harmonic:noise 比の時変制御 & MVF との対比
- WORLD 合成: 帯域 ap が noise 割合、(1−ap)相当が harmonic 割合。**帯域別に混合比を時変制御**(多帯域励起)。
- **粒度トレードオフ**: **MVF 系(HNM/h-NSF) = 単一境界で二分(スカラー)** / **ap 系(WORLD/multiband) = 帯域ごとに連続混合(ベクトル)**。息の連続制御は後者が細かく、RT 軽量には前者が単純。Drugman §I の「2大戦略(multiband vs MVF)」に対応。
- h-NSF は harmonic/noise を別 source-filter で生成し LP/HP 合成(=MVF 二分)。帯域別連続 ap を NSF に持たせる実装は調査 3論文には無い(WORLD 特徴を条件入力する NSF なら可能, 推定)。

---

## 7. 位相を「励起位相からの残差」で予測して安定化

### 7-0. なぜ位相直接回帰が崩壊するか & 対策
- 位相は ±π で wrap し、生の角度差は円周上の真距離でない(−3.1 と +3.1 で L1≈6.2)。平均は物理的に無意味 → L1/L2 回帰はノイズへ崩壊(=AFHN で実測した RMS 崩壊)。
- 対策 = **anti-wrapping 関数** `f_AW(x)=|x − 2π·round(x/2π)|` で3損失を活性化(APNet 系, arXiv:2211.15974 / 2311.11545):
  - IP(瞬時位相): `E[f_AW(P̂−P)]` / GD(群遅延, 周波数軸差分): `E[f_AW(Δ_DF P̂−Δ_DF P)]` / IAF(瞬時角周波数, 時間軸差分): `E[f_AW(Δ_DT P̂−Δ_DT P)]`。

### 7-1. iSTFTNet(元祖, arXiv:2203.02395, MIT)
- HiFi-GAN 末尾数層を削り iSTFT。最終 conv を `(n_fft/2+1)×2` ch にし **振幅 + 位相**。活性化 **振幅=exp、位相=sin**。C8C8(2段)=MOS 4.22 が実用点。**位相基準(正弦源)なし → 不安定の温床**。

### 7-2. HiFTNet(本命, arXiv:2309.09493, MIT) — iSTFTNet + NSF 調波源
- **「励起位相からの残差」の本命構造**。復元(models.py):
  ```
  spec  = exp(x[:, :n_fft/2+1, :])
  phase = sin(x[:, n_fft/2+1:, :])
  waveform = iSTFT(spec, phase)   # gen_istft_n_fft=16, hop=4
  ```
- **STFT パラメータ(config_v1)**: sr=22050, upsample=[8,8](積64), iSTFT n_fft=16/hop=4(総 64×4=256=分析 hop 一致)。分析 n_fft=1024/hop=256/mel=80。
- **NSF 正弦源(SourceModuleHnNSF, 位相基準の核心)**:
  ```
  f0_i=i·f0 (i=1..8); rad=(f0/sr)%1; phase=cumsum(rad)·2π; sines=sin(phase)
  sine_amp=0.1, 無声ノイズ amp=0.1/3; sine_merge=tanh(linear(sines))
  ```
  調波源を **STFT で時間周波数へ変換して注入**(`har_spec, har_phase = stft.transform(har_source)`, concat, noise_convs で加算)。→ ネットは 0 から位相回帰せず「正しい周期位相の下敷きからの補正」を学ぶ。**current の kansei「出力位相を励起位相にアンカー」と同一思想**。
- F0 推定器 = 事前学習 JDC(mel→F0)。LSTM 除去 ablation で CMOS −0.475。

### 7-3. Vocos(arXiv:2306.00814, MIT)
- 学習アップサンプルを全排除、終始低時間解像度で iSTFT で一気に波形化。バックボーン **ConvNeXt**(depthwise 大カーネル→LN→pointwise inverted bottleneck→GELU)。dim=512/8ブロックは公式実装標準値からの推定。
- ISTFT head: `mag=exp(m)`(clip 1e2), `S=mag·(cos p + j sin p)`, `istft(S)`。**magnitude+angle パラメータ化**(real/imag 直接でない)。cos/sin で単位円へ写し無界位相回帰の不安定を回避。n_fft=1024/hop=256/mel=100/sr=24000。正弦源なし。

### 7-4. NHV(Neural Homomorphic Vocoder, Interspeech 2020) — mixed-phase の本命
出典: Liu et al., ISCA DOI 10.21437/Interspeech.2020-3188(**arXiv 版は存在しない, 推定**) / https://github.com/k2kobayashi/neural-homomorphic-vocoder (MIT)
- **source-filter(H+N)**: `sh=(w·p)*hh`(調波: F0 インパルス列を時変 IR で畳込), `sn=u*hn`(雑音), `s=(sh+sn)*h`(学習 FIR 仕上げ)。インパルス列は余弦和(~200正弦, alias-free)。
- **NN が予測 = 複素ケプストラム ĥ**(IR そのものでなく): `h=IDTFT{exp(DTFT{ĥ})}`(N=1024)。理由: `log X=log|X|+j∠X` の実部(対数振幅)と虚部(位相)を同時符号化。
- **mixed-phase を明示採用**: 「linear/min-phase でなく mixed-phase、位相特性をデータから学習」。複素ケプストラムは**正ケフレンシ(因果=min-phase)と負ケフレンシ(反因果=max-phase)両方**に係数を持てる → min-phase 再構成では不可能な混合位相を表現。低ケフレンシのみ予測、各フレーム10ms 複素ケプストラム2本(調波/雑音)。
- **Filter Estimator**: frame レベル、log-mel80 → `256Conv(k3)ReLU×3 → 222Conv(k3)` → 1/|n| スケール → 複素ケプストラム。**~0.6M params, 15 kFLOPs/sample**(b-NSF の ~1/270)。
- 損失: MRSTFT + hinge 敵対(Discriminator=log-mel 条件 non-causal WaveNet)。**adversarial 必須**(MUSHRA 62.7→85.9)。

### 7-5. mixed-phase / 最大位相(声門開放相)で min-phase の robotic 臭を除く
- 理論(Doval/D'Alessandro/Henrich 2003): 声道+声門**閉相**=最小位相(因果)、声門**開相**=最大位相(反因果)。音声は本質 mixed-phase で、**反因果成分は声門開相のみ由来**。min-phase のみ合成はこれを欠落 → ブザー/ロボ声。
- 複素ケプストラムによる分離(Drugman 系, arXiv:1912.12602): `x̂(n)=IDTFT{log X}`、**負ケフレンシ=最大位相=声門開相(源)、正ケフレンシ=最小位相=声道**。分離は**リフタリング(時間ゲート)のみ**。ZZT 等価だが FFT/IFFT のみで高速(60Hz で 1837ms→17ms)。要点: FFT 4096, α=0.72, GCI 中心 2周期窓。
- **本設計の含意**: 正弦励起が既に自然位相を供給していれば min-phase 問題は大きく緩和。残る robotic 感には NHV 流 mixed-phase(複素ケプストラム低ケフレンシ予測)で源の位相構造を回収。**位相を出すなら anti-wrapping 損失(IP+GD+IAF)を必須併用**。

### 7-6. 位相を直接回帰しない4戦略まとめ
| 戦略 | 予測量 | 位相の出所 | 代表(License) |
|---|---|---|---|
| (a) 瞬時周波数 IAF | dφ/dt | cumsum 時間積分で復元 | DDSP(Apache-2.0) |
| (b) 群遅延 GD | −dφ/dω | 周波数積分 | anti-wrapping 系 |
| (c) 調波残差 | 整形フィルタのみ | **正弦励起が位相基準供給** | NSF/HiFTNet/SiFiGAN(BSD/MIT) |
| (d) 複素STFT | mag·e^{jp} | mag/phase 結合で暗黙決定 | iSTFTNet/Vocos/APNet(MIT) |

DDSP(magenta, Apache-2.0)は位相を**予測せず** F0 の瞬時周波数から累積合成(`φ_k(n)=2π Σ f_k(m)/sr`, `x=Σ A_k sin φ_k`)で回帰問題を根本回避。

---

## 8. 44.1kHz CPU-RT の FLOPS 予算・streaming 化

### 8-1. 複雑度一覧(すべて 16kHz 報告)
| ボコーダ | 複雑度 | CPU/RTF | MOS | 備考 |
|---|---|---|---|---|
| Meta DDSP(2401.10460) | **0.015 GFLOPS** | RTF 0.003 | 4.36 | 24k, source-filter |
| FARGAN(2405.21069) | 0.6 GFLOPS | 0.8% | ≈3.7 | 16k, 820k params |
| LPCNet | 2.8 GFLOPS | 20%/1コア(AVX2, 5×RT) | ≈3.2 | GRU sample-rate |
| Framewise WaveGAN(2212.04532) | 1.2 GFLOPS | — | 3.68 | |
| HiFi-GAN v1 | 38.1 GFLOPS | — | 4.02 | |
| CARGAN | 65.9 GFLOPS | — | 4.06 | |

### 8-2. 44.1k スケール試算(推定)
- FLOPS はほぼ sr 比例(sample-AR)またはフレームレート比例(framewise)。44.1k/16k≈2.76×。
  - DDSP 系: 0.015 → ~0.04 GFLOPS(FFT サイズ増でやや増) → CPU 1% 未満で余裕。
  - FARGAN: 0.6 → ~1.6 GFLOPS → 単一コアで十分 RT。
  - LPCNet: 2.8 → ~7–8 GFLOPS → 量子化なしでは厳しい。
- **結論(推定): 44.1k CPU-RT には DDSP系(source-filter)または FARGAN(framewise AR)が最有力。sample-rate GRU(LPCNet型)は重い。**

### 8-3. frame-rate パラメータ予測: 薄Transformer vs 因果TCN vs GRU
- **薄Transformer(Emformer)**: Meta DDSP 採用。rolling KV/memory bank で長文脈を安価保持、frame レート(~5ms)動作でコスト小、44.1k でも文脈品質◎。lookahead で品質↑/レイテンシ↑のトレードオフ。
- **因果TCN/Conv**: FARGAN の条件網。**ring-buffer と相性最良・実装最単純・jitter 最小**。
- **GRU**: 状態1本で streaming が素直だが sample-rate 化すると重い(LPCNet)。frame-rate 条件網なら軽量・因果。
- **指針(合成)**: RT・低 jitter 最優先 = 因果TCN、長文脈品質優先 = 薄Transformer(lookahead 管理必須)、状態最小逐次 = GRU。

### 8-4. <50ms E2E とレイテンシ内訳・streaming 勘所
- 内訳例(推定合成): lookahead(Emformer memory=4 で ~80ms が大)、overlap-add ~20ms、block/hop 5.3ms(DDSP)〜10ms(FARGAN)。**<50ms 予算では lookahead を 1–2 フレームに抑制が必須**(Emformer memory=4/seg=32 は削減対象)。
- streaming 化: **ring buffer**(conv を state で包み最新のみ畳込、過去活性キャッシュ) / **rolling KV cache**(直近 ~2秒, future peek 制限) / **overlap-add** / **mel 経由せず波形直接**(iSTFT 系)。

### 8-5. CPU SIMD・量子化
- LPCNet: **8-bit dot product**(AVX/AVX2+FMA/NEON)、`pmaddubsw`(x86 に signed 8bit 積命令が無いため)、quantization-aware 学習必須、~4倍速。
- FARGAN: **820k params(<1MB, L2/L3 収容) + 8-bit 量子化で SIMD 4倍** → CPU 1コア <1%。
- 関連: SIMD-size aware weight regularization(arXiv:2211.00898)。

---

## 9. 弱基音・ASMR/萌え声の F0 堅牢性

### 9-0. なぜ弱基音で octave error
- missing fundamental: 基音弱く倍音強いと、**自己相関/YIN系は 1/2倍誤り(2T ラグの偽ピーク)、調波和/HPS系は 2倍誤り(2f0 を基音誤認)**。

### 9-1. YIN(arXiv 無, de Cheveigné 2002)
- `d(τ)=Σ(x_j−x_{j+τ})²`, CMNDF `d'(τ)=d(τ)/[(1/τ)Σ_{j≤τ}d(j)]`。絶対閾値(0.10–0.15)を割る**最小 τ** で 2T 飛びを抑制。放物線補間。
- **弱基音/気息の失敗**: d' が浅くどのラグでも閾値を割らず 2T(1/2倍)を拾う。ささやきは全域高止まりで F0 未定義。

### 9-2. pYIN(Mauch & Dixon 2014)
- beta 事前 + 複数閾値で候補確率化 → HMM/Viterbi で F0+voicing。**基盤は YIN 差分関数なので 2T 誤りの根は残る**。気息が持続すると誤オクターブに安定ロックしうる(推定)。**実装: librosa.pyin(ISC)で GPL 回避、c4dm/pyin(GPLv2)禁止**。

### 9-3. CREPE(arXiv:1802.06182, MIT)
- 16k 波形(1024≈64ms)→6層1D-CNN→**360次元 cent 分類**(C1–B7, 20cent, ガウス σ=25cent, BCE)。confidence=最大活性を voicing 代理。
- **失敗**: 無声/ささやきを明示論じず、未学習音色に弱く、**フレーム独立(時間追跡なし)で気息区間のオクターブ飛び**。ASMR 囁きは訓練分布外で低 confidence+不定 cent(推定)。

### 9-4. 弱基音に強い調波和系(octave 抑制の核心)
- **SWIPE′(Camacho 2008)**: 入力に最も一致する鋸波の基本周波数。**核から非素数倍音を除去し第1・素数倍音(1,2,3,5,7)のみ使う** → サブハーモニクス偽整合が崩れ gross error 半減。スペクトル整合ゆえ**基音が弱くても倍音列全体で F0 復元**(missing fundamental に原理的に強い)。※Praat 実装は GPL、Camacho 論文から自前実装。
- **SHS(Hermes 1988)**: 対数周波数軸でスペクトルを 1/2,1/3… へシフトし重み h^n で加算(spectral compression) → 弱基音でも基音位置にエネルギー集積。電話帯域でも動作。※Praat 流用不可、論文から自前実装。
- **共通限界**: 倍音構造の存在が前提。ささやきは倍音構造自体が崩れ、調波系でも F0 低信頼/未定義。

### 9-5. voicing/非周期性で F0 を低重み化(breathy/whisper)
- WORLD D4C の band aperiodicity or CREPE confidence or pYIN voicing を信頼度代理にし、気息フレームで F0 重みを下げる or 補間。**ささやきは voicing≈無 → 推定器を信用せず明示 voicing gate でマスク**。
- **ASMR 設計指針(推定)**: 単一 F0 推定器に頼らず、①調波系(SWIPE′/SHS)で弱基音を強く取り、②aperiodicity/MVF で気息・ささやきをゲート/低重み化する**二段構え**。**h-NSF の学習可能 MVF がこの原理をニューラルで直接実装**しており ASMR 息の一級市民化と最も整合。

### 9-6. F0 非依存に寄せる合成(GCI ベース / F0 入力なし)
- **ZFF/ZFR(Murty & Yegnanarayana 2008) — 最も再実装向き・GPL 回避可**:
  ```
  0Hz 共振器(z=1 二重極): y[n]=x[n]+2y[n-1]−y[n-2]  (前段 x[n]=s[n]−s[n-1])
  2回カスケード → トレンド除去: ŷ[n]=y[n]−(1/(2N+1))Σ_{k=n−N}^{n+N} y[k]  (N≈平均ピッチ)
  正のゼロ交差 = GCI、傾き = 励起強度(SoE)
  ```
  F0 推定器不要、0Hz 近傍のみ、雑音/残響/電話帯域に頑健。※完全ウィスパーは SoE 弱く GCI 単独不可 → 雑音経路併用(推定)。SEDREAMS(COVAREP=GPLv3)は流用禁止 → ZFF 独自実装。
- **F0 入力なしで倍音を作る**: HiFi-GAN/Vocos は mel のみから周期性を暗黙生成(倍音間隔にピッチ情報)。**「測って間違える」失敗はしないがピッチ可制御性を失う**(SiFiGAN が指摘)。弱基音では mel 倍音手がかりが薄くピッチ不定になりやすい(推定)。
- **h-NSF 学習可能 MVF**(§5-3)= HNM 原理のニューラル化。非周期の強い区間は MVF を下げ周期側帯域を狭め雑音側を広げる → current の nsf-ltv「雑音を一級市民化」方針と整合。
- **NSF 無声自動切替**: `e_t=α sin(...)+n_t`(有声) / `e_t=(α/3σ)n_t`(f=0 無声)。cyclic-noise NSF(arXiv:2004.02191)は正弦を減衰ノイズで畳み連続混合(息漏れ jitter 向き, 推定)。
- **SiFiGAN(ICASSP 2023, MIT)**: source-net(正弦→励起)+filter-net。QP-ResBlock/PDCNN で拡張率を時変化 `d_t=⌊Fs/(f_t·a)⌋·d`(F0 でカーネル間隔伸縮=明示ピッチ可制御)。源正則化 `L_reg=E[‖log ψ(S)−log ψ(Ŝ)‖_1]`。CPU-RT 可。正弦入力依存ゆえ弱基音では F0 精度が品質を律速(推定)。

---

## 10. 本フォーク(LightVC B4 / kansei)への設計含意 総括

1. **骨格 = Meta DDSP 型 source-filter(§3) + HiFTNet 型 iSTFT 位相注入(§7-2)**。current の kansei が「DSP 励起 + 神経スペクトル + 出力位相を励起位相にアンカー」で A/S 天井を破った実測は、この 2 系の交点そのもの。CPU-RT 版は Meta DDSP(15 MFLOPS/MOS 4.36)が下限の存在証明。
2. **filter 表現 = 常時安定な FIR 系(DDSP-SVC frequency_filter, §4-3/4-5)を第一候補**。LPC 直接形は補間で不安定(torchlpc は p>0.98 発散, 倍精度必要, XPU 不動)。極が要るなら GOLF の biquad カスケード。
3. **位相**: 直接回帰は RMS 崩壊(AFHN 実測)。**(c) 正弦励起で位相基準を供給 + 残差だけ学習**を基軸に。位相を明示的に出すなら **anti-wrapping 損失(IP+GD+IAF, §7-0)必須**。robotic 感が残れば NHV 流 **mixed-phase(複素ケプストラム負ケフレンシ=声門開相, §7-4/7-5)**。
4. **ASMR/息 = MVF + ap の時変予測**。**MVF はフレームごとスカラー1値**を「U/V prior + 残差」で予測(sinc-h-NSF sinc1 が最安定, §5-3)。細かい息制御は band aperiodicity(WORLD, §6)。**低 MVF は F0 に寛容な領域**で current の F0 追跡不能問題を吸収。
5. **F0 堅牢性(§9)**: 弱基音は調波系(SWIPE′/SHS 自前実装)で強く取り、aperiodicity/MVF で気息を低重み化する二段構え。GCI が要るなら ZFF/ZFR 独自実装(GPL 回避)。萌え声は harmonic 下地を F0 に頼りすぎない設計へ(mel 由来暗黙周期 or 低 MVF)。
6. **44.1k CPU-RT(§8)**: 因果TCN 条件網 + 8-bit 量子化 + ring buffer streaming。lookahead 1–2 フレームに抑制で <50ms。sample-AR(LPCNet型)は 44.1k に重すぎ回避。
7. **ライセンス地雷**: SawSing(AGPL)/pYIN c4dm(GPLv2)/SEDREAMS・COVAREP(GPLv3)/SHS Praat(GPL)は**コード流用不可**。代替は全て permissive で揃う(§0)。

---

## 参照 URL(主要)

- LPCNet: https://arxiv.org/abs/1810.11846 / https://github.com/xiph/LPCNet
- FARGAN: https://arxiv.org/abs/2405.21069 / https://gitlab.xiph.org/xiph/opus/-/blob/spl_fargan/dnn/torch/fargan/fargan.py / https://ahmed-fau.github.io/fargan_demo/
- Meta DDSP: https://arxiv.org/abs/2401.10460
- SIMD-aware regularization: https://arxiv.org/abs/2211.00898
- torchlpc: https://arxiv.org/abs/2404.07970 / https://github.com/yoyololicon/torchlpc
- GOLF: https://arxiv.org/abs/2306.17252 / https://github.com/iamycy/golf
- DDSP-SVC: https://github.com/yxlllc/DDSP-SVC / https://github.com/yxlllc/pc-ddsp
- SawSing(AGPL, 参照のみ): https://arxiv.org/abs/2208.04756 / https://github.com/YatingMusic/ddsp-singing-vocoders
- MVF 推定: https://arxiv.org/abs/2006.00521
- sinc-h-NSF: https://arxiv.org/abs/1908.10256 / https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
- WORLD/D4C: https://www.sciencedirect.com/science/article/pii/S0167639316300413 / https://github.com/mmorise/World
- Merlin(BAP): https://cstr-edinburgh.github.io/vocoder-feature-extraction/
- iSTFTNet: https://arxiv.org/abs/2203.02395
- HiFTNet: https://arxiv.org/abs/2309.09493 / https://github.com/yl4579/HiFTNet
- Vocos: https://arxiv.org/abs/2306.00814 / https://github.com/gemelo-ai/vocos
- anti-wrapping/APNet: https://arxiv.org/abs/2211.15974 / https://arxiv.org/pdf/2311.11545
- NHV: https://www.isca-archive.org/interspeech_2020/liu20_interspeech.pdf / https://github.com/k2kobayashi/neural-homomorphic-vocoder
- mixed-phase/複素ケプストラム: https://arxiv.org/abs/1912.12602
- DDSP(magenta): https://github.com/magenta/ddsp
- SiFiGAN: https://arxiv.org/abs/2210.15533 / https://github.com/chomeyama/SiFiGAN
- CREPE: https://arxiv.org/abs/1802.06182 / https://github.com/marl/crepe
- pYIN(librosa, ISC): https://librosa.org/doc/latest/generated/librosa.pyin.html
- YIN: https://www.researchgate.net/publication/11367890
- SWIPE′: https://www.semanticscholar.org/paper/6748c0c9a47808f82272be85f6fa7f81601d223e
- SHS: https://pubs.aip.org/asa/jasa/article-pdf/83/1/257/11672770/257_1_online.pdf
- ZFF/ZFR: https://arxiv.org/pdf/2206.13420 / GCI レビュー https://arxiv.org/abs/2001.00473
- NSF 基本形: https://arxiv.org/abs/1904.12088 / cyclic-noise NSF: https://arxiv.org/pdf/2004.02191

## 主な訂正・未確認(実装前に再確認)

- FARGAN の正しい番号は **arXiv:2405.21069**(依頼の 2405.21077 は誤り)。
- NHV に **arXiv 版は無い(推定)**、ISCA DOI 10.21437/Interspeech.2020-3188。
- Meta DDSP 公式コード有無・44.1k 対応・16k→44.1k FLOPS スケール・E2E レイテンシ内訳は**全て推定**、実装前に要確認。
- torchlpc の **XPU 非対応は推定**(SYCL 記述なしから)。XPU 学習は自前 backward 再実装が必要になりうる。
- Vocos dim=512/8ブロック、LPCNet LPC 次数 M=16 はコード/図読み取りの推定込み。
- GlottDNN のライセンスは未確認。
