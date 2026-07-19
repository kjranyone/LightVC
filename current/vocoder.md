# Vocoder フル詳細設計

> status: **ADOPTED = freebig（F0非依存 薄型 Vocos型, §0.1）— 2026-07-17 耳ゲート通過（未学習萌え声で BigVGAN 耳同等）**。低遅延派生 config C（freeC, causal 5.8ms）は訓練継続中。
> 旧: NSF-LTV 不採用（2026-07-13 耳ゲート死亡・設計族ごと棄却, §3-8 は歴史記録）／ freebig は §5 枝B（BigVGAN/Vocos 系, 自作weights）の実現。
> 最終更新: 2026-07-17
>
> **★2026-07-13 決定的 negative result（耳駆動ボコーダ疎通診断, [[as-source-filter-ceiling]]）**: 同一発話・同一gainで並置し人間の耳で判定 — orig/gt/**istft(STFT→ISTFT 完全再構成, SNR71dB)=合格**、**world(標準WORLD A/S)=不合格 / oracle(うちのNSF-LTV, GT由来完全包絡)=不合格**。帰属確定: 再生系・データ・読込・STFT往復は無罪。成熟WORLDすら落ちる = 犯人はうちの実装バグでなく **A/S source-filter 分析再合成という方式クラスそのものが、このターゲット女声で耳基準を超えられない**。oracle(完全包絡)がゴミ = 予測器/GAN/包絡教師の改善は全て無関係、天井が低い。§4 の自認「LTV≈DDSP-SVC 生出力水準・非RT BigVGAN級は狙わない」が耳で確定。§6 kill-switch 成立 → 枝B(§5)へ pivot。NSF-LTV 系譜(NSF-HN→AFHN→HN2→HN3→LTV=全て過度に最小な自作フィルタ)は再開しない。新方針: 実績ある高容量ニューラルボコーダ構造を自作weightsで正面学習し、istft級の透明さを耳で通す(Gate V)。証拠=`results/e2_triage/*_{orig,gt,istft,world,oracle}.wav`, `training/voc_sanity.py`。
> 位置づけ: B4/SFRV（`current/README.md`）の合成バックボーン（HN2 スロット）。
> ルール: これは「1ネットワーク=1ファイル」のフル設計。採用は人間の耳ゲート通過時のみ。
> v1.1: 敵対的設計レビュー（コード検証 + 文献3系統、arXiv ID 全 verify）の指摘 F1–F6 を反映。レビュー全文はメモリ `nsf-ltv-design-review`。
> v1.2: **E0 オラクル実測（2026-07-11, `results/e0_oracle_ltv.json`）を反映。** 客観ゲート全5発話 PASS（0.975 vs WORLD 0.980）。修正2点＝ (i) clamp を区分型に（±8 nats tanh は公称の遥か手前で効きフォルマントを −9.5dB 圧縮していた＝E0 が捕まえた設計バグ）、(ii) 励起 unit-RMS を per-frame 実測から閉形式 √(Σg_k²/2) に（frame 格子 AM の混入根絶）。K_v=1024/Nb=1025 が実測確定。実装＝`training/ltv_render.py`（共有レンダラ）+ `training/e0_oracle_ltv.py`。
> v1.5 (2026-07-13): **E0 耳ゲート通過（ユーザー確認: pawB でアーティファクト解消）**。最終レシピ＝hpv 包絡（倍音ピーク上側/谷床下側、**ピッチ適応窓 3T₀**）+ OU 周期ジッタ σ0.3% + od·q⁴ 変調 + n_off−0.6。全 gate 指標 WORLD 同等以上。二値 MVF は負の結果（純 A/S では HF 穴、arXiv:1908.10256 は GAN 込みで成立）→ 段階的コヒーレンス崩壊（ジッタ）が正。**新設計原則: 低レベル音量（息・whisper）の学習を一級市民に**（§3.5 参照、ASMR=商品の核）。
> v1.4 (2026-07-12): 耳「ケースB（ltv<world）」→ 耳相関指標 mod_8_16k_dist を確立（gate v2）→ **7ラウンド改善ループで収束**。包絡教師 TE→**CheapTrick(q1=−0.30)**（F5 改訂）、変調 cos¹→**q⁴+オラクル d_t**（F2 改訂）。winner: mod8 0.708 / mod2 0.553（world 0.82/0.635 超え）/ contrast 0.969 per-utt 5/5 PASS。詳細＝RESEARCH「改善ループ」節。
> v1.3 (2026-07-12): **耳ゲート第1回 = FAIL（bad_tag: metallic）** + コードレビュー（49 agents + 人間精読）反映。レンダラ修正＝ (i) `backend="mm"` で LTV 畳み込みも GEMM 化（depthwise conv は XPU backward 既知死亡パターン）、(ii) causal モード（制御補間 [t−1,t] アンカー＝先読みゼロ・応答遅れ半フレーム、hold_f0 の backfill 廃止）、(iii) nb_in 既定 1025 + shape assert（ゼロパディング＝11kHz 以上ユニティゲイン素通りの罠を封鎖）。E0 ハーネス＝決定論シード・per-utt ゲート永続化・diag 出力分離・wexc/hop アームのバグ修正後に**全証跡を再取得**。

---

## 0. 現状サマリ

- **★採用中の vocoder = freebig（F0非依存 薄型 Vocos型, §0.1）。** 2026-07-17 耳ゲート通過（女声41k発話で本気訓練 → 未学習の弱基音萌え声で BigVGAN 耳同等）。以下 §2–§8 の NSF-HN/AFHN/NSF-HN2/HN3/NSF-LTV は**全て不採用の歴史記録**（A/S source-filter 方式クラスの天井, [[as-source-filter-ceiling]]）。NSF-HN3 不採用（§2.3、1サンプル暗記でもゴミ音質＝表現限界）。AFHN 不採用（§2.2、causal 制約違反＋位相予測不安定）。
- **提案中 = NSF-LTV v1.2**（§3）。系譜上は **NHV（Neural Homomorphic Vocoder, Interspeech 2020, DOI 10.21437/Interspeech.2020-3188）の min-phase・causal・44.1kHz 版** ＝ 実証済みの物理 + 本リポ AFHN P-B（周波数整列注入）の因果・時間領域版。新発明ではない＝失敗モードが文献に載っている（§3.0/§4 に反映済み）。
- **E0 オラクルゲート通過（2026-07-13）**: 客観 = 全5発話 per-utt PASS + gate v2/v3 指標（mod/lsd/lsharp）全て WORLD 同等以上。耳 = **pawB（v1.5 レシピ）でアーティファクト解消をユーザー確認**（metallic の顛末: 第1回 FAIL → 帰属 = 測定ハーネス gain 変調 + 包絡教師 + tanh clamp + 二重コム spec 等、`current/RESEARCH.md` の改善ループ全 19 ラウンドに記録）。kill switch は不発、設計族生存。
- **v1.5 レシピ（E1 の既定条件）**: 包絡教師 = hpv（倍音ピーク上側/谷床下側包絡、ピッチ適応窓 3T₀）、OU 周期ジッタ σ0.3%、od·q⁴ ピッチ同期 noise 変調、noise 床 −0.6 nats。実装 = `training/ltv_render.py` + `training/e0_oracle_ltv.py`。
- **次アクション = E1 overfit gate**（`vocoder.md` §6 E1。gate 閾値は E0 レンダの noise floor で較正、**低レベル音量の一級市民化を loss/gate に反映**＝§3.5）。
- E2 recon 耳ゲートを通れば status を ADOPTED に上げ、`current/README.md` §1採用がここを指す。

## 0.1 ADOPTED: freebig（F0非依存 薄型 Vocos型 foundation ボコーダ）

> status: **ADOPTED（2026-07-17, 耳ゲート通過）**。採用根拠＝**耳のみ**（未学習萌え声で BigVGAN 耳同等, ユーザー確認）。proxy 昇格ではない（帯域MAE 等 proxy は大改善を捉えられず＝「判定=耳のみ・proxy 単独昇格禁止」原則の実証）。
> 実装＝`training/free_vocoder.py`（`FreeVocoder`）+ `training/free_train_universal.py`。勝ち重み＝`training/checkpoints/freebig/foundation_bigvgan_parity.pt`。証拠wav＝`results/e2_triage/*_{freebig,bigvgan,gt,istft}.wav`。

### アーキテクチャ（Vocos 忠実, 自作weights=キメラ禁止 遵守）

```text
mel（128, 上流解析窓 n_fft2048/hop512/win2048=71dB 往復グリッド, 高解像度条件）
 → embed conv1d → ConvNeXtBlock1d ×8（dim512, groups=1=XPU安全, causal可）
 → LayerNorm → Linear head → [mag=exp(h), 自由位相 φ]（複素 STFT ヘッド）
 → S = mag·(cosφ + j·sinφ) → ISTFT（時間 upsample 無し＝ジリジリ構造的不在）
```

- **F0非依存**（harmonic源なし・noise源なし）。mel がピッチを内包し、F0制御は上流 VC/prosody 段の責務。→ 弱基音萌え声で「F0を測って間違える」failure が構造的に消滅。
- 28.8M（BigVGAN 122M の 1/4）。groups=1 標準 conv のみ（XPU 安全）。ISTFT グリッドは実証済み 2048/512/71dB。

### 採用に至る決定的事実（詳細＝`current/RESEARCH.md` R-proto-A）

- **品質ギャップの主因は「未訓練/データ被覆」で確定**（arch/損失/位相/source-filter は無罪）。診断の系列: かすれ→多話者ユニバーサル学習で解消（帯域MAE 不動でも耳は大改善）／定位感のブレ→ターゲット声の学習被覆（汎化ギャップ）で解消／そして**女声大コーパス（female-dataset 41,554発話, 188k step）で本気訓練し BigVGAN 天井に到達**。
- **反証済み（本線の根拠にしない）**: (a) A/S source-filter / mixed-phase = GT由来完全包絡の oracle ですら耳不合格＝方式クラスの天井（NSF-HN→AFHN→HN2→HN3→LTV 打ち切り, [[as-source-filter-ceiling]]）。(b) 位相の明示監督（GCI/IF 位相ヘッド, anti-wrapping IP+GD+IAF, sharpness ℓ4/ℓ2, env-stab）＝全て無効ないし悪化（定位感のブレは位相コヒーレンスでなく包絡の低速ドリフト＝汎化ギャップだった）。

### config C（freeC）— 低遅延派生（status: 訓練継続中）

- **出力合成グリッド（win/nfft256, hop128）を mel解析窓（2048据置＝条件解像度は上流で維持）から分離**し causal 化。causal 時のアルゴリズム遅延 = win/sr = 256/44100 = **実測5.8ms**（`FreeVocoder.latency_ms`）。it70k で耳「劣化なし」（freebig 比, ユーザー確認）。300k訓練中。
- **★2026-07-19 訂正（app 配線 streaming 検証）**: この 5.8ms は**合成窓側のみ**の数字。freeC の訓練 mel は centered（BigVGAN mel_spectrogram、win/2 先読みを内包）と判明し、真causal 左寄せ mel では streaming が SNR 1.26dB に崩壊（**mel 起因・vocoder 無罪**）。matching 品質 streaming の**真遅延 ≒ 合成 5.8 + mel 解析 ~23 + buffer ≒ 30ms**（<50ms・Beatrice/paravo 級）。**path A（centered streaming mel）採用**（実装進行中）、path B（causal-mel 再訓練で ~5.8ms）は将来。詳細＝§3.7 / `candle_vocoder_port.md` §0.5。
- ckpt=`training/checkpoints/freeC/last.pt`、証拠wav=`results/e2_triage/*_freeC.wav`。
- **「品質を保ったまま <50ms causal」の実証アーム**。CPU 推論 RTF0.046（単スレ 22倍速）＝計算は無罪。残課題は Rust/Candle 移植 と streaming 窓遅延台帳（§3.7 の台帳規律を継承）。

### 残タスク

- config C の 300k 収束後に耳再確認 → freeC を streaming 既定に。Rust/Candle パリティ（torch≡candle ≤1e-4）と causality CI（§3.7）。
- 息/whisper 低レベル音量の一級市民化（§3.5 の原則）は本線でも継承（golden の whisper/breath coverage gap は継続）。

---

## 1. 不可侵制約（`current/README.md` の目標・P1–P5 を継承）

- causal・E2E<50ms・streaming。Rust/Candle 推論、XPU（**groups=1 標準 conv のみ**、depthwise 禁止）。
- MIT／キメラ禁止（事前学習モデルは学習時 teacher/discriminator/perceptual-loss のみ、推論グラフに残さない）。
- VC teacher 蒸留禁止。判定は最終的に人間の耳。
- **息・ウィスパー・非周期テクスチャは商品**（消す対象でなく積極設計する）。
- 非回帰生成（音響ターゲットへ L1/L2 回帰しない＝muffle 回避, P1/P2）。texture は敵対のみ（P3）。

---

## 2. 不採用の系譜（hard-won lessons, 再開しない）

### 2.1 NSF-HN (M1/M2, `nsf_hn*.py` 初代) — 時間 upsample のジリジリ
- 原理: F0 harmonic 励起 + noise、条件で変調、6段 ConvTranspose で frame→sample アップサンプル。
- 欠陥（[[jirijiri-interharmonic-aa]]）: **時間方向 ConvTranspose がフレーム格子同期の checkerboard（86Hz 変調）**、段間非線形が aliasing で倍音間を埋め金属質。息/無声で格子がむき出し＝ジリジリ。loss/軽量改修では副作用なしに消せない。

### 2.2 AFHN (2026-07-09, `afhn.py`) — aliasing-free 複素スペクトログラム
- 賭け: 時間 upsample を廃し、複素スペクトログラム生成 + iSTFT（Wavehax 原理の自前実装）。harmonic prior + 専用 aperiodic 枝 + freq×time 2D conv。
- **生きている知見（NSF-LTV に継承）**:
  - **P-A 出力 head を zero-init 禁止**（|STFT|² loss が信号0で dead gradient）。small-normal init 必須。
  - **P-B freq-aware conditioning が収束の決定打**: content→nbins のスペクトル包絡を**周波数整列**注入（`env_proj`）。← **本プロジェクトで繰り返し確認される核心。NSF-LTV の設計根拠そのもの。**
  - **P-C 収束は plateau→breakthrough 型、早期停止禁止。**
  - **P-D foundation は no-GAN warm-up、GAN は後段 texture のみ（P3 と一致）。**
- 不採用理由: **非因果（時間対称 conv）＋ iSTFT 窓遅延で causal<50ms 制約違反**、位相予測 STFT の最適化不安定（2026-07-11 に scratch overfit で RMS collapse を再確認）。原理（freq 整列注入・harmonic prior）は NSF-LTV に継承。

### 2.3 NSF-HN2 → NSF-HN3 (2026-07-10, `nsf_hn2.py`/`nsf_hn3.py`) — sample-rate source-filter
- 賭け: 時間領域に留まり、**信号経路から upsample を排す**（全長 sample-rate、条件は滑らか FiLM 注入のみ）。→ **86Hz 息トレモロを構造的に解決（mod~0.001, 息 Gate PASS）**。causal, groups=1, ~1.5–4.5M。
- **★不採用の確定（2026-07-11）＝スカラー FiLM の表現限界**:
  - 固定1サンプル overfit（GAN/DataLoader/乱数排除, warm）で **mel-L1 0.26/mrs 2.44 床打ち、出力振幅 GT 半分で stuck、耳=ガビ**。
  - 鋭さ contrast（WORLD 再合成=1.02 上限）が**全変種 0.78–0.82 頭打ち**。ChanLayerNorm 除去・AA-GELU・noise 床除去は全て無効（副次容疑者を反証で否定）。
  - 根本原因（コード確認 `nsf_hn3.py:146-147,205-209`）: 条件が信号に触れるのは FiLM ch毎スカラー gain/bias + control-net 加算 + 初段 concat のみ。**時間混合 conv 係数は全て条件非依存の固定重み**＝「固定基底×スカラー混合」＝任意中心周波数の高Q共鳴を張れない。SGD が遅いのでなく**表現できない**（厳密には GELU 歪み積経由で原理上到達可能だが、それはジリジリ源そのもの＝使ってはならない機構。実用最適化では到達不能）。
  - 詳細診断・反証ログ: `current/RESEARCH.md`、`results/overfit_one_*`、[[nsf3-muffled-fork]]。

**共通教訓**: 条件は「フィルタそのもの」を作らねばならない。スカラー変調（HN3）でも非因果 STFT（AFHN）でもなく、**因果・時間領域・条件依存の時変フィルタ**が要る。

---

## 3. PROPOSED: NSF-LTV v1.1 decoder

**唯一の変更**: 条件 →（スカラー gain でなく）**時変フィルタ係数そのもの（envelope = impulse response）**。他（非回帰励起・harmonic+noise 二分岐・causal・groups=1・GAN texture）は継承。

### 3.0 レビューで確定した v1.0 → v1.1 の diff（根拠つき）

| # | 指摘 | v1.1 の修正 |
|---|---|---|
| **F1** | **K=256 は 44.1kHz で物理的に短すぎ**。帯域幅 B[Hz] の共鳴 IR は −27.29·B·t dB で減衰。B=50Hz（女声 F1 級）は K=256 で **−7.9dB 時点で切断**＝帯域広がり＝「こもり」を FIR 打ち切りとして再導入。−40dB には K ≥ D·fs/(27.29·B) ≈ **1230 taps**。causal フィルタの尾は**先読み遅延ゼロ**（計算コストのみ） | **K_v=1024**（E0 で 2048 と比較）、K_n=256 |
| **F1'** | **包絡グリッドも独立制約**: B を解像するには N ≥ 2fs/B（B=50Hz→N≥1764）。低次ケプストラム code 出力は quefrency 打ち切り＝帯域下限を作る | **H を 1025 点線形グリッド（Δf=21.5Hz）で直出し**+平滑正則化。低次元 code 化禁止 |
| **F2** | **86fps noise 包絡では breathy 有声の息が「別ストリーム」に聞こえる**。Hermes 1991 / Pantazis & Stylianou (ICASSP 2008): 定常 noise は harmonic と知覚的に分離（stream segregation）、**ピッチ同期 AM された noise だけが融合して breathy として知覚**。glottal 周期 4.5ms@220Hz < 格子 11.6ms。しかも本設計は学習パラメタが包絡のみ＝**GAN でも救済不能**（判別器が指摘してもハンドルが無い）。DDSP/NHV/SawSing/DDSP-SVC 全先行が未解決 | **ピッチ同期 noise 変調 §3.1**（決定論・因果・数FLOPs）+ **サブフレーム noise gain**（hop/4=2.9ms; 破裂・口内音の時間版 F1）。ASMR の商品本体かつ neural 系初の差別化点 |
| **F3** | **「GAN texture 継承」の前提が崩れている**。HN3 では GAN 勾配が sample-rate 重みに流れた。v1.1 の学習対象は frame-rate 包絡のみ→判別器はサブフレーム構造を直せない。NHV 実測: GAN 無し MUSHRA 85.9→**62.7 崩壊**、min-phase+GAN無し再実装は「**robotic timbre**」評 (arXiv:2406.05128) | GAN は「包絡回帰ボケの鮮明化」として必須（Phase B）。テクスチャ last-mile は F2 変調が構造として担う。それでも不足時のみ段階3: 小型 causal AA-refiner（§4） |
| **F4** | **レイテンシ台帳が楽観**。「≤1frame」は vocoder コアのみ。特徴窓・F0 窓・I/O ブロック込みの現実は §3.7 | 台帳 §3.7 + **causality CI テスト必須ゲート**（前科: center=True 系 0.73s 混入） |
| **F5** | **包絡教師の平滑バイアス**。CheapTrick は F0 適応平滑（幅 2F0/3）で F0=300Hz の B=50Hz ピークを **~5dB 潰す**（萌えレジスタ F0 250–350Hz が最悪領域）。D4C は whisper で縮退（周期成分前提） | **v1.4 改訂（E0 改善ループ実測）: 教師 = CheapTrick(q1=−0.30)**。TE（上側包絡）は HF で noise ピークに乗り H_n を汚す＝耳の劣化の主因だった（mod_8_16k_dist 1.068→0.778）。平滑バイアスは q1 強化（−0.15→−0.30）で contrast 回収（0.975、per-utt 5/5）。−0.45 以上は過鋭で破綻。warm-up 限定・anneal→0 は不変 |
| **F6** | **harmonic/noise 分解の不定性**。励起位相≠GT のため時間域 loss 不可、mag loss だけでは高域分解が曖昧（hoarse/buzz。SawSing の buzz=常時全帯域 harmonic 励起の残留、と同族） | D4C BAP 補助教師（anneal）+ v/uv ソフト化 + Nyquist ロールオフ + 無声中も F0 連続発振（§3.1 励起衛生） |

### 3.1 信号フロー（v1.1）

```text
励起（非回帰・非学習, P2）:
  harmonic e_h[n]: HN3 HarmonicSource 改（励起衛生 4点）
    - unit-RMS 正規化は閉形式 √(Σ_k g_k²/2)（rolloff ゲインから解析的に。F0 依存エネルギー除去、
      ゲインは H_v が持つ。v1.2: per-frame 実測 RMS は frame 格子 AM を再導入するため禁止）
    - Nyquist 近傍 raised-cosine ロールオフ（倍音数切替 pop 根絶。hard mask 廃止）
    - 無声中も F0 を hold+slew で連続発振＝位相連続（hard uv ゲート `nsf_hn3.py:70` 廃止。
      レベルは H_v が殺す。GOLF 式ソフト voicing f̂=v·f0 も可）
    - 位相 frac(k·frac(p))（float32 cumsum の既知課題、Candle 移植前提）
  noise e_n[n] = randn
    - ★ピッチ同期変調（F2）: e_n ← e_n · m[n]
        m[n] = ((1−d_t) + d_t·q(φ[n])) / E[q]              … unit-mean 正規化
        q(φ) = ½(1 + cos(φ − φ0))                          … glottal 位相同期の滑らか包絡
        ※v1.4（E0 改善ループ実測）: **q は q⁴ に強化**（cos¹ 形は無効を実測）。正規化は
          E[q^p]=C(2p,p)/4^p（`ltv_render.py` mod_p 実装済）。d_t のオラクル= D4C HF 帯 ap 平均。
          効くのは包絡教師が CT(q1=−0.30) で一貫している時のみ（TE 上では無効、RESEARCH 参照）
        d_t ∈ [0,1]: frame 毎予測（sigmoid）。φ0: 学習スカラー（init π）。φ = e_h の基本波位相
    - ★サブフレーム gain a_t^(j), j=1..4（hop/4=128smp=2.9ms 格子、線形補間、mean-1 正規化）

条件 → フィルタ（★核心）:
  C_t = (content c_t, timbre s, F0_t, energy_t) @86fps
  causal frame-rate net（§3.4, groups=1）→ frame 毎:
     H_v[t] ∈ R^1025 : voiced log-mag 包絡（線形グリッド 0–22.05kHz, Δf=21.5Hz）
     H_n[t] ∈ R^1025 : noise/breath log-mag 包絡
     d_t, a_t^(1..4), v_t（ソフト voicing）
  min-phase FIR 化（§3.2）: b_v[t]（K_v=1024）, b_n[t]（K_n=256）

時変フィルタ（NHV 方式 OLA, 因果・先読みゼロ）:
  励起を hop=512 の非重複矩形セグメントに区切り、セグメント毎に b[t] と直畳み込み。
  尾 (K−1) は未来サンプルへ加算（＝フレーム間の自然クロスフェード。min-phase 同士の
  係数補間は位相差で phasing するため、出力加算方式が正解）
  y[n] = LTV(e_h, b_v) + LTV(e_n · m · a, b_n)

post-net: 既定なし（フィルタに全スペクトル負荷。段階3 の任意 AA-refiner は §4）
```

### 3.2 min-phase 構成（実ケプストラム法。全て固定行列＝両側パリティ）

frame 毎・2枝、Oppenheim–Schafer 標準手順:

1. 1025 点 → N_c=4096 グリッド (2049 点) へ固定補間行列で refine（log-mag 線形補間）。**v1.3: 手順は refine→clamp の順**（実装・E0 較正ともこの順。逆順で Candle を実装するとパリティが max 0.69 nats ずれて必ず FAIL — レビュー実測）
2. clamp（**v1.2 修正、E0 実測根拠**）: 区分型 `x = H−mean(H)`, `|x| ≤ 8` は恒等、超過分のみ `sign(x)·(8 + 4·tanh((|x|−8)/4))` で飽和（上限 ±12 nats）。**unit-circle 上の深い notch はオーバーサンプルで消えない→clamp 必須**。~~v1.1 の `8·tanh(x/8)` は禁止~~: tanh は公称 ±8 の遥か手前で効き、実音声の包絡偏差（male p95=7.6 nats、breathy p50=4.9）を −9.5dB 圧縮＝フォルマント平坦化（bw +50Hz、contrast 0.83 頭打ち）を E0 で実測。区分化で male 0.834→1.014、全発話 PASS に回復
3. 実ケプストラム `c = IDFT_4096(mirror(H))`（偶対称 mirror; IDFT=固定 fp32 行列 matmul）
4. fold（min-phase 窓）: `ĉ[0]=c[0], ĉ[n]=2c[n] (1≤n<2048), ĉ[2048]=c[2048], 残り 0`
5. （任意 lifter）`ĉ[n] ← γ^n·ĉ[n]`（γ≈0.9999、共鳴減衰上限の保証。既定 off、E1 でリンギング時のみ on）
6. `B = exp_complex(DFT_4096(ĉ))`（**complex dtype 不使用**: (cos,sin) 実数対、exp(a)·(cos b, sin b)）
7. `b = IDFT_4096(B)[0:K]`（K_v=1024 は B=50Hz を −31.7dB、B=80Hz を −50dB まで収める）

- N_c=4096 ≥ 4×K_v: ケプストラム時間エイリアスの余裕（緩和材料: breathy 女声は glottal chink で B1 が 100–200Hz+ に広がる [Hanson 1997] ため要件が緩む。張った声・歌唱で効く制約→E0 実測）。
- **学習（XPU）**: torch.fft の XPU 対応は 2.8+ の oneMKL・fp32 のみ・既知バグ複数（torch-xpu-ops #4279/#3955/#3646）→ **既定 matmul-DFT**（微分可能・XPU 安全・86fps なら GPU コスト無視可能）。
- **推論（Rust）**: Candle に FFT op は無く Intel GPU バックエンドも無い（CPU/CUDA/Metal/WASM）→ **推論本番は rustfft**（MIT/Apache-2.0 デュアル、4096 点×~4変換×86fps×2枝 ≈ 0.2 GFLOP/s）。matmul-DFT はリファレンス/パリティ基準（4096 matmul を CPU 常用すると ~23 GFLOP/s で不可）。**torch(matmul) ≡ candle(matmul) ≡ rustfft の三点パリティ ≤1e-4 をゲート化**。
- LTV 畳み込み自体は時間領域直畳み込み（512×K の matmul）＝ FFT 不要・両側ビット一致が容易。

### 3.3 なぜ HN3/AFHN の壁を破るか（+ 系譜上の位置）

- **vs HN3（スカラー FiLM）**: 条件がフィルタ係数そのものに → mel→包絡→係数は周波数整列・全解像の写像＝任意の鋭い共鳴（AFHN P-B の因果・時間領域版）。「包絡→係数」が固定線形写像である事は問題ではない — それは学習ボトルネックでなく厳密な DSP 変換で、表現力は H に宿る。落とし穴は写像でなく K/Nb/notch clamp/ケプストラムエイリアス（→F1/§3.2 で対処済み）。
- **信号経路が線形 → activation aliasing（ジリジリ）が構造的に不在**。HN3 が条件依存スペクトルを作る唯一の機構は GELU 歪み積＝ジリジリ源だった。この矛盾を根絶する。
- **vs AFHN（位相予測）**: 位相は harmonic 発振＋min-phase から決定論的。pitch-shift 外挿にも頑健（位相回帰は分布外外挿。APNet2 自身「位相予測はフレームシフトに極めて敏感」と明記）。
- **系譜**: ≒ NHV（GAN 込み copy-synthesis MUSHRA 85.9 ≈ PWG、1.5e4 FLOPs/sample、複素ケプストラム 10ms×2本・DFT N=1024・非重複矩形フレーム OLA）の min-phase・causal・44.1k 版。ケプストラム→min-phase フィルタの微分可能実装は diffsptk（Yoshimura+ ICASSP 2023, arXiv:2211.11222）が学術的裏書き。直近同型 = FIRNet（ICASSP 2024; 「素朴な FIR 予測+BAP 混合では品質不足→pitch 依存構造と周期/非周期分離で改善」という改訂履歴が指針）。
- **時変 all-pole/LPC（GOLF/torchlpc）を選ばない理由**: torchlpc カーネルは Numba CPU+CUDA のみ＝**XPU 学習不可**。周波数サンプリングでの IIR 学習は pole 半径 r→1 で数学的破綻（エイリアス尾が r^N でしか減衰せず、r=0.999 は N=1024 でも −9dB ＝**狭帯域こそ学べない**）。frame-wise IIR 近似は train-test mismatch（TV-LP 論文が実証）。→ FIR+OLA は XPU 制約下の正解。all-pole は v2 オプション（反射係数/LSF+出力クロスフェード）。
- **min-phase の音色コスト（正直な差分）**: NHV は意図的に mixed-phase（声門開放相=最大位相の群遅延を学習）。min-phase+GAN 無し再実装は「robotic」評。→ last-mile は F2 変調 + Phase B GAN が担う設計。E0 の耳で早期確認。

### 3.4 frame-rate net（包絡予測器）

- 入力: cond（content + logF0 + energy）+ timbre s broadcast concat。
- 本体: causal TCN、**groups=1**、ch=384、10 blocks（k=3, dil [1,2,4,8,16]×2）、ChanLayerNorm + GELU。**frame-rate の非線形は音声信号に触れない**（包絡軌跡を整形するだけ）→ aliasing 無関係、AA 不要。~0.8 GMAC/s。
- head: 1×1 → 2×1025 + 1(d) + 4(a) + 1(v)。**zero-init 禁止（P-A 同型）: bias=データ平均 log 包絡、weight std 0.02**。H に隣接 bin 平滑正則化（弱、λ小）。
- timbre の FiLM/AdaIN 注入はこのネット内では**可**（包絡予測器の条件付けであり、信号のスカラー変調ではない。HN3 の失敗とは別物）。

### 3.5 Losses（P1–P3 整合、Phase 制）

- **★低レベル音量の一級市民化（2026-07-13, ユーザー要件）**: ASMR では息・whisper・小声＝商品であり、それらは**低 RMS 区間に住む**。エネルギー比例の素の MRSTFT/mel loss は静区間の寄与がほぼゼロ＝商品を学習しない。E1 以降の必須要件: (i) **レベル補償 loss**（per-frame ラウドネス正規化 or 静区間の明示重み上げ）、(ii) **gate/eval に息区間マスク指標**（無声×低レベル、E0 で計測盲点だったことを実測済: active mask が可聴フレームの 7–30% を除外していた）、(iii) 学習データのサンプリングも静かな発話を過小評価しない。
- **★Phase A 改訂（v1.6, 2026-07-13, E2 耳AB実測）: Phase A は包絡教師回帰のみ。mel/MRSTFT の spec loss は使わない（有害を実測）**。E2 対照実験: λ_env anneal→0 =完成度10% / floor0.15=40% / **純教師回帰=55%**（耳、単調用量反応）。機構: mel(帯域平均)はコム/ノイズ・per-bin 包絡誤差に盲目のまま勾配を注ぎ、無声倍音リーク（H_v +11nats）・振幅半減（生振幅0.51、gain補正が隠蔽）・盲点ドリフトを生む。spec 系の役割は Phase B（GAN + FM）に集約。旧記述（λ_env anneal→0 + MRSTFT 主体）は E1 の1サンプル過学習でのみ成立する構成だった。
- ~~**Phase A（no-GAN warm-up, P-D）**~~（旧設計、記録として保持）:
  `L = λ_env(t)·[L1(H_v,Ĥ_v) + L1(H_n,Ĥ_n) + L1(d,d̂)] + L_mrstft + λ_mel·L_mel`
  - MRSTFT: **6 構成**（win 128/256/512/1024/2048/4096、75% overlap、lin+log L1）。NHV「解像度を増やすほど artifact 減」（NHV は 12 構成）。
  - 包絡教師 Ĥ: **True Envelope 第一候補**（CheapTrick は −5dB 平滑バイアス既知、比較用に併走）。voiced: TE 包絡 + D4C BAP で H_v/H_n 分配。**unvoiced/whisper: STFT 包絡直教師**（D4C 縮退回避）。d̂: BAP の高域変調度からの粗 proxy。
  - **λ_env は cosine で →0（全 step の 20–30%）**。P1 との整合: 回帰先は音響ターゲットでなく**フィルタパラメタ**、かつ最終目的関数から消える。包絡→音の間に平滑化機構が無い（尖った H は尖ったフィルタとしてそのまま鳴る）＝「回帰 muffle」の機構が構造的に不在。
- **Phase B（texture, P3）**: + MPD + MRD + FM（既存実装流用）。GAN の役割 = 包絡回帰の regression-to-mean ボケの鮮明化。**NHV 警告「性能は判別器構造に敏感」→ 判別器変更も 1実験1仮説**。
- 補助モデル（BigVGAN mel teacher / WavLM / ECAPA）は loss・eval のみ（P5）。VC teacher 禁止継続。

### 3.6 real-time 予算（**推論本番は CPU**。Candle に XPU バックエンドは無い）

| 要素 | 概算 |
|---|---|
| 励起（≤340 osc、LUT sin） | ~0.1 GFLOP/s |
| LTV 直畳み込み（512×(1024+256)×86fps×2op） | ~0.23 GFLOP/s |
| min-phase（rustfft 4096×~4変換×86fps×2枝） | ~0.2 GFLOP/s |
| frame net（ch384×10blk） | ~1.6 GFLOP/s |
| head / 変調 / misc | ~0.2 GFLOP/s |
| **計** | **~2.3 GFLOP/s** |

公表 CPU-RT 実証帯: FARGAN 0.6 / Framewise WaveGAN 1.2 / LPCNet <3 GFLOPS（全て 1core RT）→ 44.1k 安全圏 3–6 GFLOPS の内側。品質天井の傍証: Meta ICASSP 2024（arXiv:2401.10460）= CPU 15 **M**FLOPS・RTF 0.003 で MOS 4.36（DSP+frame net 系）。実リスクは FLOPs でなく **86fps の小 op 群の jitter** → E4 で実測。

### 3.7 causal / latency 台帳（E2E、F4 反映）

| 要素 | lookahead / 遅延 |
|---|---|
| 入力デバイスブロック | 2.9–11.6ms |
| mel/特徴窓（左寄せの場合。ただし **★2026-07-19 訂正**参照＝採用中 freeC は centered mel 訓練ゆえ左寄せ不可） | 左寄せ訓練時 0（群遅延スミアのみ）／freeC 実態 **~23ms**（win/2 先読み、下記訂正） |
| 制御補間（f0/d/a、causal [t−1,t] アンカー） | 0 先読み（応答遅れ hop/2=5.8ms のみ、出力遅延に加算されない） |
| F0 窓（男声 80Hz → 20–25ms 左窓） | 0（同上。立ち上がり応答は鈍る） |
| content encoder LA | 0–23ms（B4 側予算） |
| frame 集約（hop） | 11.6ms |
| min-phase / LTV / OLA 尾 | **0**（causal フィルタの尾≠遅延） |
| 計算 wall-clock | 2–5ms 目標 |
| 出力デバイスブロック | 2.9–11.6ms |
| **vocoder 経路計** | **~20–35ms**（+encoder LA で <50ms。タイト） |

- **★2026-07-19 訂正（app 配線 streaming 検証で反証, `candle_vocoder_port.md` §0.5）**: 上表「mel 窓＝左寄せなら遅延0（群遅延スミアのみ）」の前提は **採用中の freeC で反証された**。freeC の訓練 mel = BigVGAN `mel_spectrogram`（`center=False` + reflect-pad `(n_fft−hop)//2=960`, n_fft/win=2048, hop=128）で、この pad が各フレームを**実質 centered** にする（原座標で frame t=[t·hop−960, t·hop+1088]、中心≈t·hop+64、**前方 ~1088≈win/2 の先読みを内包**）。ゆえに真causal 左寄せ（trailing 窓・先読み0）mel は freeC 訓練 mel を再現できず、streaming 出力は best-lag でも **SNR 1.26dB / xcorr 0.62 に崩壊**（数値ゲート）。**帰属分離＝同一 streaming mel を offline vocoder に通しても同値 → 劣化は 100% mel の causal-framing 起因、vocoder streaming は無罪**（vocoder 単独 90dB）。
- **正しい遅延内訳**: 「freeC causal 5.8ms」は**合成窓(256)側だけの数字**。matching 品質の streaming には **mel 解析側に ~win/2≈1024 サンプル≈23ms の先読み**が要る。→ **真の streaming E2E 遅延 ≒ 合成 5.8ms + mel 解析 ~23ms + buffer ≒ 30ms**（まだ <50ms・Beatrice/paravo 級）。
- **採用 path**: **(A)** centered streaming mel（win/2 先読みバッファで訓練 mel に一致）で ~30ms・良品質 ＝ **採用**（実装進行中）。**(B)** freeC を causal 左寄せ mel で**再訓練**すれば真 ~5.8ms（将来）。左寄せ台帳の「0 先読み」は *mel を左寄せで訓練した vocoder に限り*成立する → **framing はボコーダ毎に検証**する規律を追加。
- **causality CI テスト（必須ゲート）**: 時刻 T 以降の入力をランダム化 → T 以前の出力が**ビット一致**。center=True 混入（0.73s 前科）を構造的に捕まえる。
- **streaming ≡ offline パリティ**。streaming 状態 = {OLA 尾バッファ (K_v−1)、位相 acc（mod 2π）、subframe gain 補間、frame-net conv cache}。
- 予備カード: MS-Wavehax（arXiv:2506.03554）は「LA=1frame で非因果同等 MOS」→ 最後まで品質不足なら **LA=11.6ms を追加投資する選択肢**が予算内に残る。

### 3.8 製品ツマミとの結線（LTV パラメタ化の製品的勝ち筋）

包絡ドメインの推論時操作 ＝ **再学習ゼロのリアルタイム連続ツマミ**（86fps のベクトル演算、M2 の GUI 要件に直結）:

| GUI パラメタ | 実装 |
|---|---|
| breathiness（息） | H_n グローバルゲイン(+dB) & d_t バイアス |
| formant（声の小ささ） | H_v の周波数 warp（固定 warp 行列、α∈[0.85,1.15]） |
| softness | H_v/H_n への tilt（dB/oct）オフセット |
| register / liveliness | F0 経路（`render_m2.py` の register_st / exaggerate、実装済） |
| cuteness 強度 | M2 学習条件（後段） |

息＝商品の中核が d_t / H_n として**明示パラメタ化**される（「学習可能 breathiness ツマミ」目標の直接実装）。

---

## 4. リスク登記（正直な期待値）

- **品質天井**: LTV 単体 ≈「DDSP-SVC 生出力を min-phase 決定論位相 + ピッチ同期変調 + GAN で押し上げた水準」。非 RT BigVGAN 級 MOS は狙う軸ではない（勝つ軸は §7）。DDSP-SVC が enhancer（NSF-HiFiGAN/shallow diffusion）を事実上必須とした前例（生出力は metallic、位相予測歪み由来と pc-ddsp 明記）に対し、本設計は min-phase 決定論位相で当該原因を除去済み — それでも残る分は下記。
- **min-phase の音色コスト**（NHV mixed-phase との差分、「robotic」前例）→ E0 の耳で早期判定。
- **包絡回帰ボケ** → Phase B GAN + λ_env anneal。それでも残る場合**のみ**段階3: 小型 causal AA-refiner（`aa.py` 資産、GAN 学習、励起または出力に適用）— enhancer の in-model 版。既定では積まない。
- **分解不定性（F6）** → BAP 補助 + ソフト uv + ロールオフ。E2 で hoarse/buzz バッドタグ監視。
- **判別器敏感性**（NHV 警告）→ MPD/MRD+FM から開始、変更は単独実験。

## 5. 枝B（保険。E0/E2 が死んだ場合の pivot 先を先に温める）

- **B1: BigVGAN-v2 44k（コード+重み MIT、キャッシュ済）teacher → causal 小型 student 蒸留**。BigVGAN 原論文 = arXiv:2206.04658（ICLR 2023, anti-aliased AMP + MRD）; v2 は NVIDIA repo/model card のみ（arXiv 論文なし）。公表レシピ: arXiv:2408.11842（causal conv 化+蒸留+wav2vec2.0 特徴 loss で causal 版 PESQ 3.96 > 非因果小型 3.64）、Conan の pixel-shuffle 置換、DLL-APNet 蒸留。規約適合（補助モデル監督であり VC teacher 蒸留ではない）。**注意: 時間 upsample 復活＝ジリジリの土俵に戻る**（AA 活性+pixel-shuffle で緩和）。E0 で LTV 物理が死んだ場合は「無 upsample 原則」自体の再検討とセット。
- **B2: causal AFHN 再訪**（MS-Wavehax レシピ: causal conv+リングバッファ、LA0–1 で非因果同等）。AFHN の位相回帰不安定（RMS collapse）が再発するかが焦点。P-A/P-B/P-C/P-D 知見はそのまま有効。
- 参照: HiFTNet（MIT、22.05k、F0 条件 harmonic+noise+iSTFT）= 本設計に最も近い公開 MIT 実装（非因果）。AB 参照・teacher 用。
- **ライセンス地雷（使用不可）**: OpenVPI NSF-HiFiGAN（重み CC BY-NC-SA + リポ AGPL）、Seed-VC（GPL-3.0、archived）、Wavehax コード（LICENSE 無し＝原理の自前実装のみ可）、Mimi 重み（CC-BY-4.0、要帰属）。安全圏: BigVGAN / Vocos / HiFTNet / LLVC = MIT、LPCNet / FARGAN = BSD-3。

## 6. 検証ラダー（1実験1仮説・安い順。各ゲート実測をディスク永続化・GT 併記）

- **E0 オラクル analysis-synthesis（客観半分 PASS 済 2026-07-11・耳ゲート待ち）**: 実女声/breathy/男声 → True Envelope（+CheapTrick 比較）包絡 + D4C → §3 レンダラで再合成 → 耳 + contrast + フォルマント帯域幅（LPC 計測）。
  - sweep: **K_v∈{256,512,1024,2048} × Nb∈{257,513,1025,2049}**（こもり出現点＝F1 の実測）、**±ピッチ同期変調**（breathy 有声で息が融合するか＝F2 の実測、d 手動 0.5–0.8）、min-phase vs linear-phase 参照。
  - gate: contrast ≥ ~0.95×WORLD、耳でこもり/ガビ/robotic/**metallic** 無し、変調 on/off が breathy AB で可聴改善。
  - **gate v3（2026-07-12）**: 次数バケット線鋭さ `lsharp_dev`（k1-5/6-15/16-30/31-60 の peak−valley を **GT 係留・両方向乖離**で判定）。contrast_ratio は鋭さ不足のみ罰する片方向で、metallic の正体「過剰な決定論性」に盲目、かつ WORLD 自体が HF で過鋭（k16-30: GT 0.09 vs WORLD 0.45）のため WORLD 係留はこの欠陥を構造的に見逃す — アンカーは GT。
  - **gate v2（2026-07-12 追加、耳とペアで較正済）**: `mod_8_16k_dist`（8–16k 帯域包絡の変調スペクトル距離 vs GT）≤ WORLD 水準。耳が「world より劣化」と言うのに contrast が盲目だった穴を塞ぐ指標（5/5 発話で耳と同順、`results/e0_discriminator_hunt.json`）。補助: `mod_2_8k_dist` / `trajdist_b4_8k`。
  - **客観実測（`results/e0_oracle_ltv.json`）**: 全5発話 PASS（mean 0.975 vs WORLD 0.980）。K: 256→0.951 / 512→0.970 / **1024→0.975** / 2048→0.975（F1 の knee 実測確定）。Nb: 257→0.968 / 513→0.974 / **1025→0.975** / 2049→0.975（F1' 確定）。TE 0.975 > CheapTrick 0.924（F5 確定）。linear-phase 参照 1.001（min-phase コストは小）。±変調は客観 contrast 不変（F2 は知覚指標＝耳 AB で判定）。**whisper 明示カテゴリは golden 未収載（coverage gap）＝breathy/small_voice を代理**。
  - **耳ゲート第1回（2026-07-12）: FAIL — `ltv` に metallic**。帰属 AB（world/linphase/harm-only/noise-only）が次アクション。§0 参照。
  - **fail → 設計族ごと棄却、§5 枝B へ pivot（kill switch）** — 客観側は生存、耳側は帰属確定まで保留。
  - 実装: `training/e0_oracle_ltv.py` + **レンダラ共有モジュール `training/ltv_render.py`**（モデルと同一コード。E4 の Candle パリティ基準を兼ねる。fft≡matmul-DFT パリティ 2.7e-7・因果性ビット一致・静的包絡再現 ±0.4dB を自己テスト済）。
- **E1 overfit-one（trainability）: PASS（2026-07-13, `training/e1_overfit_ltv.py`）**。gate は同セグメントの E0 オラクルレンダ floor ×1.15 で較正（当初の推測値 mel-L1<0.1 は noise 実現を無視しており廃止）。実測: floor 0.834/3.63/0.795 に対し **anneal（λ_env cosine→0 + MRSTFT+mel[低レベル補償]）= 0.800/3.555/amp1.0/0.791 ＝ 教師 floor 超えで PASS**。env-only は floor 一致（表現力確認）。spec-scratch も 10k st で PASS（0.823/3.556/0.808、3k は plateau=P-C 再実証）。HN3 参照値（0.26/2.44/振幅半分/0.82=死）との対比で signal path の健全性確定。**次 = E2 単一話者 10min recon**。
- **E2 単一話者 10min recon**: vs `nsf3_gan` best、同一データ・同一計測（contrast / band share / centroid / 耳）。Phase A→B。**枝B を同一 gate で AB 並走**（1本勝負の再発防止）。
- **E3 ablation（各1仮説）**: ±ピッチ同期変調（breathy AB）/ ±subframe gain（/s/・破裂過渡）/ K 縮小の可聴性 / Nb 縮小の可聴性 / γ lifter / d_t の帯域分割（2-stream 化: 変調 noise と平坦 noise に別包絡）。
- **E4 causal+parity+RT**: causality CI / streaming≡offline / torch≡Candle≡rustfft ≤1e-4 / Rust RTF・block jitter 実測。
- **E5 B4 結線**: content c_t 条件（mel 教師なし）→ CIPT 出力側 identity 同時学習 → 男→萌え cross（srcshift）。既存 G-cipt / G-cross ゲート接続。

## 7. Frontier 位置づけ（何で SOTA を主張するか）

- **動作点の未占有**: 「causal × 44.1kHz × <50ms × CPU」は文献・OSS ともに空白（2026-07 調査、arXiv ID 全 verify）。全会計開示の公表実測最速 = RT-VC（ACL 2025 demo, arXiv:2506.10289）**61.4ms**/16k/CPU、次点 StreamVC 70.8ms・SynthVC 77.1ms。学術 streaming VC は**全て 16k 帯**、44.1k streaming vocoder/VC の公表例ゼロ。→ 台帳全開示の <50ms 実測はその動作点で文献初。
- **評価軸の未占有**: whisper/非 modal 発声を評価する streaming VC/vocoder 論文は皆無 → Kansei/ASMR HITL 評価系ごと新規性。
- **差別化技術**: ピッチ同期 noise 変調（neural vocoder 系で初。古典 HNM 知覚知見の移植）+ 包絡ドメイン実時間ツマミ（§3.8）。
- 品質主張は E0/E2 の耳ゲートが決める。数字で盛らない（proxy と耳の乖離前科）。

## 8. 未決の設計判断（E0/E1/E3 の実測で確定）

- ~~K_v / Nb~~ — **E0 で確定（2026-07-11）: K_v=1024, Nb=1025**（2048/2049 は客観無利得、256/257 は劣化。Nb の可聴下限のみ E3 耳 ablation 待ち）。γ lifter は既定 off 継続（E0 でリンギング不検出）。
- 包絡教師: TE 単独と hybrid（H_v=TE, H_n=CheapTrick）は clamp 修正後は同値。**既定 TE 単独**、noise 過多が耳で出た場合のみ hybrid を再訪。
- **フィルタ適用のサブ分割（hop 256/128、包絡は時間補間）**: E0 実測で 0.977→0.993→1.016（emotional/breathy/sibilant で一貫改善）。レイテンシ不変・計算 ×2/×4。**metallic 帰属の候補**（86fps 切替 splatter）かつ品質レバー。耳 AB（E3）で採否。
- d_t の帯域分割（v1=広帯域1本。2-stream 化は E3 の upgrade 候補）。
- 判別器構成（MPD/MRD 開始、変更は単独実験）。
- rustfft 置換タイミング（E4 プロファイル後。それまで matmul-DFT リファレンス）。
- 段階3 AA-refiner の要否（E2 の耳ゲート次第。既定は積まない）。
