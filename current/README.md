# LightVC 現行方針

> 状態: 生きた正典（採用＋設計索引）。最終更新 2026-07-17。
>
> **ドキュメント構造（設計・研究・評価作業の前に読む）**
> - `current/README.md`（本ファイル, 安定）= §1 採用アルゴリズム + §2 設計概観・索引。
> - `current/ROADMAP.md`（生きた計画）= データ×モデル×評価×製品の鳥瞰。**各段の着手前に依存物を確認する入口**。
> - `current/RESEARCH.md`（揮発）= 研究中の仮説・反証実験・進行 run・negative results。
> - `current/<network>.md`（フル設計, 1ネットワーク=1ファイル, status 付き）: `current/vocoder.md`（**ADOPTED: freebig** — F0非依存 Vocos型, §0.1）／`current/zeroshot_vc.md`（PROPOSED）／`current/gui_design.md`（PROPOSED）。
>
> **昇格フロー**: RESEARCH（仮説・PROPOSED 設計）→ overfit gate / 人間の耳ゲート → 勝てば設計を ADOPTED 化し §1採用がそれを指す。RESEARCH→採用の直行禁止。採用は耳で勝ったものだけ（proxy 単独昇格禁止）。証拠=`results/`、横断事実=`memory/`（重複させずポインタ）。

## 目標

LightVC は ASMR・官能バ美肉向けのリアルタイム声質変換を目指します。

- E2E レイテンシ 50ms 未満
- 滑らかで、近く、長時間聴いても疲れない声
- source speaker leakage の低さ
- Python ランタイム不要の Rust/Candle 推論
- Human-in-the-Loop Kansei 評価を必須ゲートにする

標準的なVC指標だけでは成功判定しません。Smoothness、Tenderness、Clarity、Embodiment、Control、Fatigue resistance の人間評価を通らない checkpoint は成功扱いしません。

## 実測で確定した方向（2026-07-04, 人間の耳で確定）

音声を実際に聴いて、以下が確定しました（`results/frontier_scoreboard.md`）。

- **oracle（自然な target latent → DAC decode）= 良い。** 天井は到達可能、DAC decoder は無罪。
- **本物 kNN-VC（WavLM content + kNN match + 学習済み coherent vocoder）= めちゃ良い。** これが到達すべき品質の参照。
- **DAC latent をフレーム単位で構築する系は全部カス。** 学習 adapter（大きな線形 delta）→ off-manifold で muffled。学習なし retrieval（DAC フレーム連結）→ 時間的に繋がらず rough。

→ **参照アーキテクチャは kNN-VC 型**：WavLM 系 content 特徴 → 固定 target プールへ kNN match → **coherence を与える学習済み vocoder**。DAC latent のフレーム構築は本経路から外す（decoder / streaming 実装は資産として残す）。「大きな線形 delta 禁止・target-manifold 生成」は、この文脈で「vocoder が時間的一貫性を作る」と読み替える。

### 「萌え」は音色ではなく演技（prosody / delivery）

kNN-VC は timbre（誰の声か）を変換するが、**delivery（抑揚・語尾・間・息の付け方＝萌えの本体）は source 側から保持**する。中立な男性 source を変換すると「女性の声だが萌えではない」になる。→ **萌えは声質変換だけでは出ない。** 路線は2つ:

- **受肉（採用寄り）**：操作者が萌えを演じ、VC は timbre/style を変える。`self_prosody` で操作者の演技を保持。今日の kNN-VC 成果はこの路線の土台。
- **自動萌え**：source が中立でも萌え prosody/style を明示付与（style/texture path）。難度高。

### 北極星：kNN-VC 品質のまま <50ms streaming / Rust

kNN-VC を3分解し、実時間化の難易度は以下。

- **content encoder（WavLM-large L6）**：非因果・重量級 → **因果 student へ蒸留**が必要（最難）。
- **vocoder（WavLM→HiFiGAN）**：因果/streaming 化（BigVGAN v2 44k がキャッシュにあり）→ 中難度。
- **matching（kNN）**：target は固定 → プール precompute + ANN index → **実時間化は容易**。

### Concept 整合性（MIT / Rust / zero-shot / 萌え）

- **zero-shot 維持 → 非パラ kNN を runtime に残す（route ①）。** target を焼き込む any-to-one 蒸留（route ②）は zero-shot を壊すので不採用。content encoder と vocoder は target 非依存で1回学習、kNN が任意 target プール（参照音声）を扱う。これで Phase D ルール「VC teacher の synthetic parallel distillation 禁止」とも整合。
- **MIT 維持 → 出荷 runtime に SSL を積まない。teacher 選択は HuBERT を第一候補。** ライセンス実確認：
  - runtime 非依存が大原則（AGENTS.md：補助モデルは loss・評価・蒸留に使ってよいが推論時依存にしない）。出荷は自作の因果 student encoder + 自作 vocoder。
  - **WavLM = CC BY-SA 3.0（コピーレフト）**。offline teacher 利用は許可されるが、蒸留 student が CC BY-SA の derivative かは法的 grey。→ **grey 回避のため HuBERT を第一候補**にする。
  - **HuBERT / wav2vec2 = Apache-2.0** → permissive、蒸留で作る自作 weights を MIT にできる。**content SSL は HuBERT 系を既定**。（ContentVec は HuBERT 派生で content 分離に向くが要ライセンス確認）
  - kNN-VC コード+HiFiGAN = MIT（`train.py` 同梱で再学習可）、BigVGAN v2 44k = MIT（原論文 arXiv:2206.04658, ICLR 2023; v2 は repo/model card のみ）。
  - **重要な帰結**：今日「めちゃ良い」の kNN-VC は **WavLM 依存＝上限リファレンス**。出荷版は HuBERT 特徴で encoder/vocoder を作り直して品質を再検証する（WavLM-L6 比で品質やや低下の可能性）。
- **Rust 維持**：runtime = 因果 student encoder + kNN(ANN) + 因果 vocoder を Candle 移植。プロファイル実測で計算は余裕（GPU で kNN-VC 全体 RTF≈0.01）。<50ms の壁はスループットでなく (a) 非因果性 (b) Rust移植 の2点。
- **萌え維持**：timbre は kNN-VC / 参照プール、萌え=delivery は受肉（操作者演技）+ style layer。zero-shot と両立。

## B4 ネットワーク設計（仮説駆動 / Frontier, 2026-07-04）

これは既存 kNN-VC の移植ではない。今日の実測エビデンスから**新しい仮説を立て、そこからネットワークを設計する**。kNN-VC は「retrieval+coherent vocoder が効く」ことの存在証明であって、そのまま使うものではない（非因果・WavLM=CC BY-SA・moe/prosody 非対応・境界 rough）。

### 実測エビデンス → 設計制約

- **E1** coherent latent は decode 良／構築 latent は悪 → 生成器は「latent 演算」でなく**時間的一貫性を自前で持つ生成 decoder**であるべき。
- **E2** retrieval→固定 target プール + 学習 vocoder = 良＆zero-shot → **非パラ retrieval が zero-shot の正解機構**。
- **E3** hard kNN 連結は境界 rough → **retrieval は平滑化が要る**。
- **E4** 萌え=prosody/delivery → prosody は操作者から保持 + 明示 style/F0/breath 制御。
- **E5** breath/高域は明示モデル要（RVQ 研究 / adapter が殺す）→ **明示 aperiodic/noise 分岐**。
- **E6** content encoder が因果性＋ライセンスの隘路 → **因果・蒸留・HuBERT(Apache)**。
- **E7** 計算は余裕（GPU RTF 0.01）→ 制約は flops でなく lookahead。ちゃんとした decoder を積める。

### 新仮説（設計の賭け, 各々反証可能）

- **HN1 微分可能ソフト retrieval**：target プールへの**学習 cross-attention** は、hard top-k kNN より滑らかで時間的に一貫した target-manifold 軌跡を出す（境界 rough を解消）。zero-shot は保持（プール=key/value, load 時差替）。検証：boundary discontinuity + human AB を soft-attn vs hard-kNN で比較。
- **HN2 source-filter / harmonic+noise decoder**：合成を「F0駆動の harmonic + 学習 aperiodic-noise」に分解し、retrieved timbre で条件付け。→ (i) breath/whisper を明示保持、(ii) formant/F0 を独立制御（萌え＆cross-gender pitch 修正）、(iii) coherence、(iv) 小型・因果・streamable。検証：CPP/H1H2/HNR + 「息が生きてるか」AB、cross-gender pitch 自然さ。**XPU 制約**：最終推論アプリは XPU 対応が要件。decoder は Candle で XPU-portable な標準 op に収め、**depthwise/grouped conv を避け groups=1 標準 conv 前提**で設計する（XPU で depthwise conv が失敗する既知問題）。
- **HN3 因果・二出力蒸留 encoder**：HuBERT から蒸留した1つの小型因果ネットが、**retrieval 用 content ベクトル**と**明示 prosody(F0/vuv/energy)**を同時出力。因果(lookahead ≤1–2 frame)でも content が kNN-match 可能に保つ。検証：retrieval 品質 + speaker/gender leakage を lookahead 依存で測る（レビュー H3）。
- **HN4 prosody policy 層**：既定は操作者 prosody 保持（受肉）。その上に**分離された moe-shaping**（F0 range/contour, breathiness, softness）を GUI 軸として乗せる。軸独立性を検証（レビュー H7 / cross-axis drift）。
- **HN5 参照プール zero-shot 条件付け**：target は数秒の参照音声から load 時に pool 埋め込み+retrieval key を構築。per-target 学習なし。検証：品質 vs 参照長、未知 target。

### ネットワーク: Streaming Factored Retrieval-Vocoder (SFRV)

```text
[操作者 wav, streaming, ~20ms hop]
   │  causal distilled encoder (HuBERT-taught, Apache→自作weightsはMIT)   ← HN3
   ├─► content c_t   (speaker-invariant, retrieval-ready)
   └─► prosody (F0_t, vuv_t, energy_t)
         │  prosody policy: 保持(受肉) + moe-shaping knobs               ← HN4
         └─► F0'_t, dynamics'
   c_t ─► soft differentiable retrieval  over target pool {K,V}          ← HN1/HN5
         (pool は数秒参照から load 時に構築; ANN + attention)
         └─► target-manifold embedding m_t  (滑らかな軌跡)
   m_t + F0'_t + energy' ─► source-filter / harmonic+noise
         streaming decoder (44k, causal, 小型)                           ← HN2
         └─► 出力 waveform (coherent / breath-alive / moe-controllable)
```

全ブロックが因果・小型・MIT/Apache → Candle 移植可。レイテンシ = encoder lookahead(≤1–2 frame) + decoder lookahead。計算は余裕(E7)。

**Frontier 要件の充足**：滑らか=HN1+HN2 / 萌え=HN2(F0・formant・breath)+HN4 / <50ms=全因果小型 / zero-shot=HN5 非パラ / MIT=HuBERT teacher + 自作 weights。

### 段階ゲート（ブロック単位で先に検証、安い順）

1. **G-enc**：HuBERT から因果 encoder 蒸留 → retrieval-match + leakage を lookahead 依存で gate（既存 proxy / h8 ハーネス再利用）。
2. **G-voc**：source-filter decoder を target 再構成で学習 → breath proxy + 「自然/息生存」AB（Gate 1 系）。
3. **G-retr**：soft-attn vs hard-kNN を boundary discontinuity + AB で（HN1）。
4. **G-cross**：cross-gender 男→萌え + F0 制御（HN2）— 今日崩れた所。
5. 統合 → streaming → Rust/Candle。

各ゲートは既存の `kansei_proxies.py` / `gate0_*` / `annot_gui.py` / `listen_gui.py` / `h8_retrieval.py` をそのまま計測系として使う。

## 第一学習の設計: 自作 NSF-HN VC network（2026-07-04, キメラ禁止）

**大原則（キメラ禁止）**：出荷推論グラフは**100%自作 weights**。事前学習モデル（ContentVec/HuBERT/ECAPA/BigVGAN）は**学習時の teacher・discriminator・perceptual-loss としてのみ**使い、推論経路には一切残さない（AGENTS.md と一致）。凍結 ContentVec＋BigVGAN fine-tune＋kNN を貼り合わせる RCAV は「出来合いのキメラ」なので不採用。retrieval も非パラ貼り合わせでなく学習表現に内包する。

### 全実測から導いた不可侵の設計原則

- **P1 回帰は muffle**：DAC latent も mel も L1/L2 予測は平滑化してカス。→ 音響ターゲットへ回帰しない。
- **P2 crisp の源は非回帰生成 + coherence を持つ decoder**：kNN-VC の教訓を、外部 vocoder 借用でなく**自作 decoder が非回帰励起から波形を作る**形で内在化する。
- **P3 texture は敵対のみ**：自作 decoder を GAN(MPD/MRD) で学習。失敗 content を GAN で救済しない。
- **P4 萌え=delivery**：timbre は target code、萌え表現は source prosody 保持(受肉)＋後段 style。
- **P5 MIT / runtime非依存**：推論は自作 causal encoder + 自作 decoder のみ。SSL/BigVGAN は蒸留・知覚損失の教師に限定。

### 自作ネットワーク（出荷グラフ = 全て自作 weights）

```text
operator wav (streaming)
 → 自作 causal content encoder  E   → content c_t   ← SSL(ContentVec/HuBERT)は蒸留teacherのみ
                                    → prosody F0_t,energy_t (受肉)
 target参照(数秒) → 自作 target encoder T → timbre code s   ← ECAPAは蒸留teacherのみ
 (c_t, s, F0_t, energy_t) → 自作 NSF-HN decoder G → 44kHz 波形
```

### 自作 decoder: Neural Source-Filter Harmonic-plus-Noise (NSF-HN)

> **2026-07-11: 世代交代済み。** vocoder の最新採用状況・フル設計は **`current/vocoder.md`**（NSF-HN3 不採用 / NSF-LTV PROPOSED）。以下は不可侵原則（P1-P5・非回帰励起・harmonic+noise）の記録。

回帰 muffle(P1)を原理的に回避する自作生成器。外部 vocoder は使わない。

- **励起生成（非回帰, P2）**：F0_t から正弦励起（基音+倍音）を合成 = harmonic source。voiced の crisp な周期性を回帰なしで得る。
- **noise 分岐（P4/E5, breath/ASMR）**：帯域制御した雑音励起 = aperiodic source。息・ウィスパを明示保持。
- **学習 filter network**：(c_t, s) 条件で harmonic+noise 励起を変調し波形化。**causal・groups=1 標準 conv（XPU 安全）**、小型・streamable に設計。
- **学習 loss**：多重解像度 STFT + mel 知覚(BigVGAN mel を教師) + 敵対(MPD+MRD) + feature matching。回帰項は補助、texture は敵対が担う(P3)。

### 学習手順（実音声のみ, VC teacher 不使用）

- moe コーパスで自己再構成。X について c=E(X), p=prosody(X), s=T(X 参照) → G(c,s,p) が X 波形を再構成。
- **encoder 蒸留**：E の content を ContentVec/HuBERT 空間へ整合（offline teacher）＋話者敵対で leakage 抑制。T を ECAPA へ整合。→ 推論は自作 E,T のみ。
- **content perturbation**：学習時に話者情報を落とす拡張（SR/formant shift）で、cross-gender でも崩れない content を学ぶ。

### 制約充足

| 要件 | 充足 |
|---|---|
| crisp(no muffle) | P2 非回帰 NSF 励起 + P3 敵対 decoder |
| 萌え | source prosody 保持 + noise 分岐(息) + 後段 style |
| zero-shot | 自作 target encoder が数秒参照→timbre code、per-target 学習なし |
| MIT / キメラ禁止 | 推論=自作 E+T+G のみ。事前学習は学習時 teacher/discriminator 限定 |
| <50ms / Rust / XPU | 全ブロック causal・groups=1 標準conv・小型 → Candle/XPU 移植 |

### マイルストーン（各段で human listen ゲート）

- **M1 decoder 単体**：自作 NSF-HN を実音声再構成で学習（content は暫定 ContentVec 入力）。**聴く**: 自作 decoder が moe を crisp に描けるか。← ここで decoder の素性を確定。
- **M2 男性→萌え**：自作 encoder で source=男性→timbre=萌え。**聴く**: 製品本体。全部不合格を越えるか。
- **M3 realtime**：causal 化・ANN・Rust/Candle、推論から事前学習モデルを完全排除。

## M2 conditional VC & auto-moe 制御（2026-07-05, 人間の耳で方向確定）

M1(自作 NSF-HN decoder)は moe を target 級 texture で再構成できることを確認（客観収束、耳では last-mile の vocoder 臭が残る＝MRD で継続改善）。次に **M2 = 条件付き VC** を実装:自作 timbre encoder T(参照→話者コード) + conditional NSF-HN decoder(content+F0+energy+timbre) + MPD/MSD/**MRD** 判別器、M1 から warm-start。

### 耳で確定した最重要方針:可愛さ=生成対象、受肉は不採用

- **可愛さ(萌え)= delivery/prosody**（抑揚・語尾・息・間・声質）であって、音色でも絶対ピッチでもない。中立男性を timbre 変換 + pitch-shift しても「pitch-shift した男性声」にしかならない（実測・耳で確認）。
- **受肉（操作者が萌えを演じ、VC は timbre のみ変える）は「自明」なので研究対象にしない。** 演じれば可愛いのは当然で、機械は何も生成していない。
- **採用 = 自動萌え(auto-moe)**：操作者が**中立に喋っても**、機械が**萌え delivery を生成付与**する。ただし操作者の生 prosody を土台に**上乗せ**する（full 生成で操作者を蚊帳の外にしない）。

### 製品定義:パラメタ制御 auto-moe のリアルタイム VC

**操作者が普通に喋る → 機械が萌え delivery を生成 → GUI ツマミでリアルタイム補正 → <50ms で発声。**

- **制御パラメタ = 焼き込まない。全て GUI のリアルタイム連続ツマミ**（ユーザ確認済み要件, 2026-07-05）:
  - **pitch register(高さ)** / **liveliness(抑揚=F0 動きの量)** … F0 変換の2ツマミ、実装済み(`render_m2.py` の register_st / exaggerate)。
  - **breathiness(息)** / **formant(声の"小ささ")** / **cuteness 強度** / **target 声(timbre)** … 未実装、decoder への条件入力として学習が必要（本命の追加）。
- **リアルタイム性**:ツマミ処理はフレーム毎の軽量演算＝自明にリアルタイム。**壁はツマミでなく非因果アーキ**。→ M3 で causal 化。

### 段階

1. **M2 継続学習**：timbre 成熟 + MRD で vocoder 臭低減。
2. **制御ツマミの学習化**：breathiness/formant/cuteness を decoder 条件入力として学習（F0 手動リマップ → 学習型 moe-prosody 生成へ）。
3. **M3 realtime**：ContentVec→因果 student 蒸留、decoder 因果化(groups=1)、streaming buffer、Rust/Candle、GUI 連続ツマミ配線。

## 専用 Vocoder（採用: freebig。詳細は `current/vocoder.md`）

vocoder のフル詳細設計・世代変遷と採用状況は **`current/vocoder.md`** に集約（1ネットワーク=1ファイル）。

- **★採用（ADOPTED, 2026-07-17, 耳ゲート通過）= freebig**: **F0非依存 薄型 Vocos型ボコーダ**（ConvNeXt groups=1（XPU安全）+ 複素STFTヘッド mag·exp(jφ) 自由位相 + ISTFT, mel入力, 28.8M＝BigVGAN の1/4）を女声大コーパス（41,554発話）で本気訓練 → **未学習の弱基音萌え声で BigVGAN と耳同等**（ユーザー確認）。**品質ギャップの主因は「未訓練/データ被覆」で確定**（arch/損失/位相/source-filter は無罪）。勝ち重み=`training/checkpoints/freebig/foundation_bigvgan_parity.pt`、証拠wav=`results/e2_triage/*_{freebig,bigvgan,gt,istft}.wav`。採用根拠は**耳ゲート合格のみ**（proxy 昇格でない。帯域MAE 等の proxy は大改善を捉えず＝耳のみ判定原則の実証）。
- **低遅延派生 = config C（freeC, status: 訓練継続中）**: 出力合成グリッド（win/nfft256, hop128）を mel解析窓（2048据置）から分離し causal 化 → **アルゴリズム遅延 実測5.8ms**、it70k で耳「劣化なし」。300k訓練中。ckpt=`training/checkpoints/freeC/last.pt`。**「品質を保ったまま <50ms causal」の実証アーム**（残課題は Rust/Candle 移植・窓遅延台帳、計算は無罪 RTF0.046）。
- **不採用（negative results, 現行の根拠にしない）**: 手作り DSP/A/S source-filter ボコーダ族（NSF-HN → AFHN → NSF-HN2/HN3 → NSF-LTV）は**方式クラスの天井で打ち切り**（[[as-source-filter-ceiling]]: GT由来完全包絡の oracle ですら耳不合格＝位相/予測器/GAN の改善は無関係）。位相の明示監督（GCI/IF ヘッド・anti-wrapping/sharpness/env-stab 損失）も反証済（定位感のブレは位相でなく汎化ギャップだった）。詳細＝`current/RESEARCH.md`。**旧経路は再開しない**（P-A/P-B/P-C/P-D・freq 整列注入・時間 upsample=ジリジリ の知見のみ `current/vocoder.md` §2 に保存）。

## 現在の判断

以前の B3 staged 設計は、主経路として不採用です。

不採用にした経路:

```text
DAC latent
-> 学習済み Stage1 content extractor
-> Stage2 generator
-> 任意の GAN fine-tune
```

理由:

```text
Stage1 は source speaker を消しながら、content、息、子音、小声、
タイミング、ASMR texture を保持する必要がある。

これは単一の learned codec-latent bottleneck に背負わせるには重すぎる。

content_dim=256 は微細情報を失う。
content_dim=1024 は微細情報を残すが source speaker が漏れやすい。
GAN fine-tune は、この失敗を「自然に聞こえる失敗」として隠してしまう。
```

したがって、以下を主依存にする作業を開始・継続してはいけません。

- `train_stage1_content.py`
- `train_stage2_generator.py`
- `train_stage2_adv.py`
- `train_b3.py`
- `b3_model.py`
- `kansei_render_b3.py`
- `checkpoints/stage1_*`
- `checkpoints/stage2_*`
- `checkpoints/b3_*`

これらのファイルは artifact としてリポジトリに残っていてもよいですが、現行設計ではありません。

## 新アーキテクチャ: B4 Dual-Path Kansei VC

DAC latent を無理に speaker-free content 表現へ変換しません。

経路を分離します。

```text
source audio
  -> SSL/ASR content path
  -> explicit prosody path
  -> source leakage monitor

target reference / target corpus
  -> target timbre path
  -> target style path
  -> ASMR/Kansei texture prior

content + prosody + target voice/style
  -> target-manifold latent generator
  -> streaming decoder
  -> output audio
```

### Content Path

言語内容は、事前学習済み SSL/ASR 表現から取得します。候補は WavLM、HuBERT、Whisper encoder features、または speaker に頑健な content model です。

content path は generator 学習の前に必ずゲートを通します。

- 単語とタイミングを保持する
- source speaker identity を保持しない
- 短文、フィラー、笑いに近い断片、息、ライブ発話を扱える
- 50ms 未満の streaming に必要な時間解像度を保つ

### Prosody Path

prosody は明示特徴として別経路にします。

- F0
- voicing
- energy
- rhythm
- pause
- attack / release timing

目的は、操作者の演技を残しつつ、source timbre は残さないことです。

### Target Voice / Style Path

target voice は単一の ECAPA vector だけで表現しません。conditioning は以下を表現できる必要があります。

- 年齢感
- 明るさ
- 距離感
- 息成分
- 柔らかさ
- 親密さ
- テンション
- ASMR comfort

target reference と target dataset から、timbre / style / texture 条件を分けて作ります。

### Generator

generator は target-manifold acoustic representation を生成します。source DAC latent から大きな線形 delta を足す設計にしてはいけません。

target 表現として codec latent を使ってもよいですが、source DAC latent を唯一の content source にしてはいけません。

この2つのルールには実測根拠があります（`results/frontier_scoreboard.md` A-20260704）。

- **8kHz曇り/muffledは codec でなく生成側で生じる**。自然音声の DAC roundtrip は Gate 0 を PASS するのに、phase3c 生成 latent を同一 decoder で decode すると hf_ratio が oracle 比で半減し、eight_k_cliff が 0.31→0.13 に崩壊する。しかもその生成は `delta_norm≈1.16` の**大きな線形 delta を source latent に足しており**、着地点が自然 manifold の外にあるため decoder が dull にレンダリングする。→ 「大きな線形 delta 禁止 / target-manifold 生成」の直接の根拠。
- **息・高域は RVQ の fine residual codebook を要する**。coarse latent（RVQ 1-2段）は HF を誤色付けし HNR を −1〜−3dB ずらす。HNR/高域が自然値に収束するのは 6-8段。→ generator が coarse な構造しか当てられないと breath_dead / 誤HF になる。generator は fine residual まで当てるか、**明示的な noise/aperiodic 分岐**で息を作るべき。

## SOTA Frontier 短期学習戦略

B4は汎用VCのSOTAを即座に主張する計画ではありません。短期の目的は、既存SOTAが正面から扱っていない **50ms未満・ASMR/官能・バ美肉・Human-in-the-Loop** の領域で、勝ち筋がある仮説だけを高速に残すことです。

最短で学習を進めるため、以下を徹底します。

### 1. 大規模学習より先にGateを通す

大きなtraining runは、Gate 0 / Gate 1 / Phase Aを通った後だけ許可します。

```text
Gate 0: codec / decoderが官能品質を保てるか
Gate 1: target autoencodingが聴ける品質か
Phase A: content/prosody featureがspeaker leakageなしに成立するか
```

ここを通らない状態で長時間学習してはいけません。過去の失敗と同じく、後段が「綺麗な失敗」を作るだけです。

### 2. Golden Mini Setを固定する

毎回同じ短い評価セットを使います。

最低限:

- 男性source: 通常発話、小声、早口、笑いに近い発話、息混じり、サ行が多い文
- target female/ASMR: 囁き、小声、柔らかい通常発話、感情発話、語尾が長い発話
- same-text検証: VCTK等の同一内容ペア
- negative samples: metallic、muffled、rough、source leak の既知失敗例

評価セットを毎回変えると、改善か偶然か判断できません。

この固定セットは `training/build_golden_set.py --seed 0` で再現生成し、`data/kansei_vc/golden_mini.tsv` に出力します。現状:target女性 18（TTSタグ breathy/soft/neutral/tension/warm + 実女性の small_voice/sibilant/long_tail/emotional をRMS・サ行かな数・durationでproxyラベル）、source男性 10、VCTK same-textペア 4、合成 negative 3。明示ラベルが無いカテゴリ（男性の 小声/早口/笑い/息/サ行、女性の explicit whisper）は捏造せず、`golden_mini_coverage.json` の `coverage_gaps_todo` に人間キュレーション待ちとして記録します。

### 3. 1実験は1仮説に限定する

同時に複数の要素を変えません。

良い実験:

```text
仮説: codec roundtripで囁きの高域が死ぬ
変更: codec候補A/Bだけを比較
評価: Gate 0の固定サンプルでAB
```

悪い実験:

```text
codec、content model、loss、dataset mix、GANを同時に変更する
```

### 4. SOTA部品は「比較対象」として使う

SOTAらしい外部要素は、まず比較対象・補助評価・上限確認に使います。いきなり本番依存にしません。

許可:

- SSL/ASR content feature の比較
- speaker / gender / ASR / MOS系の補助評価
- neural codec / decoder / vocoder候補のGate 0比較
- 既存VCを人間ABの参考ベースラインとして比較

禁止:

- 既存VCの変換音声をtraining targetにする synthetic parallel distillation
- ライセンス不明のモデルを本番依存にする
- Python runtime依存を最終推論経路に残す

### 5. RunPod等の高速GPUは「勝ち仮説」だけに使う

ローカルで以下を確認してから外部GPUへ送ります。

- 10-30分 smoke run が落ちない
- fixed eval packの出力が生成できる
- Gate指標が前回より悪化していない
- checkpointとmanifestが再現可能

外部GPUでは、長時間1本勝負ではなく、短い比較runを複数回します。

```text
例:
  A: content feature候補
  B: target representation候補
  C: decoder/vocoder候補
  各runは同じgolden mini setを必ず出力する
```

### 6. Frontier Scoreboardを作る

各実験は、最低限以下の表に記録します。

```text
run_id
hypothesis
changed_factor
checkpoint
dataset_mix
latency_estimate
Gate0_result
Gate1_result
content_result
source_leakage
target_likeness
human_AB_result
bad_tags
decision: continue / revise / archive
```

結果が悪いrunも消しません。bad tags付きで残し、同じ失敗を繰り返さないためのnegative evidenceにします。

記録は `results/frontier_scoreboard.md`（1行=1仮説、詳細JSONは同ディレクトリ）。

**第1行の所見（G0-20260704-female_real）**:実女性の DAC roundtrip は客観 Gate 0 を base・finetuned とも PASS。**過去の「8kHz曇り・metallic・崩壊」は codec roundtrip 自体ではなく、生成された latent（out-of-distribution）を decode する経路で生じている可能性が高い。** よって codec/decoder の作り直しに GPU を割かず、レバレッジは generator / latent 生成側に置く（finetuned decoder は HF でわずかに優位なので keep）。human AB は未了。

### 7. 短期の採用基準

短期で残す仮説は、以下を満たすものだけです。

- Gate 0またはGate 1を改善する
- source leakageを悪化させない
- high-band clarityを落とさない
- golden mini setで前回bestより人間ABが良い
- 50ms未満runtimeへ落とす見込みがある

これを満たさない仮説は、SOTAっぽく見えてもarchiveします。

## 学習計画

### Gate 0: Codec / Decoder 官能品質検査

B4は、いきなりVCを学習してはいけません。過去の出力では、8kHz程度に落としてからアップスケールしたような曇り、明朗さの不足、ざらつき、金属音、変換強度を上げた時の崩壊が繰り返し観測されています。

したがって、最初の合格条件はVC成功ではありません。まず、codec / decoder 経路だけで ASMR・官能品質が保たれることを確認します。

検査:

```text
実音声
-> encode
-> decode
-> 人間評価 + 自動解析
```

必須サンプル:

- 小声
- 囁き
- 息
- サ行・歯擦音
- 語尾の減衰
- 近接マイク感
- ライブ発話
- 感情発話

合格条件:

- こもらない
- 8kHz upsampled のように聞こえない
- 高域の明瞭さが残る
- 息が砂っぽいノイズにならない
- サ行が刺さらない
- 小声が潰れない
- 長く聴いて疲れない

このGate 0に失敗した場合、content pathやHITLを足しても「賢く調整された失敗」になるだけです。DAC/decoder前提を停止し、codec、decoder、target representation、またはvocoder設計から見直します。

#### Gate 0 自動解析（実装済み）

Gate 0 の「自動解析」半分は実装済みです。人間ABの前に、客観プロキシで既知の失敗を落とします。

```text
training/kansei_proxies.py   … 客観プロキシ計測ライブラリ（bad_tag → 測定量）
training/gate0_codec.py      … real audio -> DAC encode -> RVQ -> decode -> 自動解析
training/build_golden_set.py … 固定評価セット生成 -> data/kansei_vc/golden_mini.tsv
```

実行:

```text
cd training
uv run python gate0_codec.py --from-dir ../female-dataset --n 24 --seed 0 \
    --decoder both --tag female_real --export 6
# -> results/gate0_female_real.json（scoreboard） + results/gate0_female_real_ab/（human AB用trio）
```

3 decode 経路を比較します。`ceiling`（量子化なし = decoder上限）、`base`（stock DAC）、`finetuned`（R2 decoder）。これにより「劣化は RVQ 由来か decoder 由来か」を切り分けます。

各合格条件を、位相不変な分布系プロキシに写像しています（数値は暫定。最初のGate 0 runとhuman ABで較正）。

| Gate 0 合格条件 | プロキシ | 暫定閾値 |
|---|---|---:|
| こもらない | `centroid_delta_hz` / `hf_preserve` | > -350Hz / > 0.70 |
| 8kHz upsampledに聞こえない | `eight_k_cliff_ratio` | > 0.50 |
| 高域明瞭 | `brilliance_preserve` / `rolloff85_delta_hz` | > 0.55 / > -800Hz |
| 息が砂ノイズ・metallicにならない | `hf_flatness_delta` | < 0.15 |
| サ行が刺さらない | `sib_delta`（5–9kHz比の変化） | < 0.020 |
| breath/小声が潰れない | `cpp_delta_db` / `hnr_delta_db` | > -1.5 / > -3.0 |

`log_spectral_distance_db` と `mel_l1_db` は位相・整列に敏感なので**ゲートには使わず診断値**として記録します。合否は上表の分布系プロキシで判定します。

これらのプロキシは合成 negative anchor（`results/golden_negatives/`: muffled_8k / metallic_crush / rough_noise）で符号方向を検証済みです。ただし**プロキシは human AB を置換しません**。Gate 0 の最終合否は「自動プロキシ全PASS かつ human AB で許容」です。

### Gate 1: Target Manifold Autoencoding

Gate 0を通った後にのみ、female / ASMR / whisper / emotional / live speech の target-side autoencoding を検証します。

目的:

- target voice manifold がASMR・官能バ美肉用途に足りるか確認する
- VCではなく自己再構成で、滑らかさ・近さ・息・小声・高域明瞭さが保てるか確認する
- target manifold自体が弱いのに、cross-voice生成へ進むことを防ぐ

検査:

```text
target audio
-> target representation
-> generator / decoder
-> reconstructed target audio
-> 人間AB + 自動解析
```

合格条件:

- originalに対して、官能品質の劣化が小さい
- breath / whisper / soft voice が残る
- metallic / muffled / rough が増えない
- high-band clarity が保たれる
- Human ABで「自己再構成として許容」と判断される

Gate 1に失敗した場合、Cross-Voice Generationへ進んではいけません。

### Phase A: Feature Gate

content/prosody feature set が合格するまで、waveform generator を学習しません。

必須チェック:

- held-out utterance で ASR/WER または text consistency が保たれる
- content から source speaker が予測されにくい
- content から source gender が漏れにくい
- timing と pause が保たれる
- live utterance が崩れない

このゲートに失敗したら停止します。generator 学習へ進んではいけません。

### Phase B: Target Manifold Autoencoding

female、ASMR、whisper、emotional、live speech の target-side reconstruction を学習します。

目的:

- 自然な target voice manifold を学ぶ
- 高域の明瞭さを保つ
- 息と小声を保つ
- 金属音、こもり、8kHz upsampled のような質感を避ける

### Phase C: Cross-Voice Generation

source content/prosody と target conditioning から、target-like audio を生成します。

必須ゲート:

- source leakage が低い
- target similarity が十分
- content が保たれる
- conversion intensity を上げても roughness が増えない
- human AB preference で reject されない

### Phase D: Realtime Distillation

E2E 50ms 未満の runtime に向けて、モデルを圧縮・適応します。

ルール:

- 推論は Rust/Candle
- Python runtime 依存なし
- realtime path に diffusion / ODE loop を入れない
- VC teacher の synthetic parallel target を使わない

補助モデルは training loss・評価には使ってよいですが、runtime dependency にしてはいけません。

### Phase E: Texture Fine-Tune

GAN / adversarial training は、前段ゲートを通った後の最終 texture fine-tune としてのみ許可します。

ルール:

- 失敗した content 表現を GAN で救済しない
- GAN を使う場合は feature matching または同等の安定化 loss を入れる
- 同一サンプルで non-GAN baseline と比較する
- Human Kansei 評価が改善した場合だけ採用する

## データセット

現役データ層:

- `data/female_tts_corpus/`: Irodori TTS v3 生成女性コーパス
- `female-dataset/`: 実音声の女性演技コーパス
- VCTK / `data/phase3_10k`: source-side、same-text検証、content/prosody gate用コーパス
- male/mixed source speech: 操作者入力に近い source-side 音声
- emotional/live/whisper/ASMR utterances: 受肉用途に必須
- golden evaluation set: human AB 比較用の固定短尺サンプル

`data/` 直下は平坦なコーパス置き場ではありません。現行のデータ入口は `data/README.md` と `data/kansei_vc/manifests/` です。旧実験の生成物・cache・置換済み小規模データは `data/.archive/legacy_2026-07-04/` に退避済みであり、通常の学習入力として使ってはいけません。

Irodori TTS v3 生成コーパスは、発話テキストのパターンが少ない場合でも、target timbre / style / texture の初期学習には使える。ただし、25発話パターン程度では content generalization、ライブ発話、言い淀み、笑い、囁き、サ行・破裂音・長母音の網羅には不足する。したがって、Irodori TTSを全学習の主データとして扱ってはいけません。

Irodori TTSの正しい役割:

- target female/style manifold の初期化
- caption_keyごとの声質・距離感・柔らかさの条件付け
- Gate 1 target autoencoding の素材
- Kansei候補の初期探索

Irodori TTSだけでは不足する役割:

- content path学習
- cross-text generalization評価
- live speech robustness
- laughter / filler / hesitation / breath event
- 実音声由来の演技・崩れ・マイク近接感

短期方針として、25発話パターン版は残すが、次の拡張が必要です。

- phoneme-balanced textを追加する
- emotional / live / whisper / ASMR textを追加する
- 非言語音と短い反応発話を追加する
- 実音声 `female-dataset/` をtarget manifold評価に必ず混ぜる

この拡張の初期実データとして、`data/kansei_vc/japanese_live_vc_texts.tsv` を使います。これは128発話の日本語テキストバンクで、短い反応、相槌、言い淀み、ライブ発話、萌えpersona、ASMR/囁き、サ行・破裂音・長母音などを含みます。coverageは `data/kansei_vc/japanese_live_vc_coverage.json` に記録します。

VCTKは現役データですが、役割を間違えてはいけません。主用途は source-side入力、same-text validation、content/prosody feature gate、source leakage評価です。ASMR・官能バ美肉の target manifold を代表するデータとして扱ってはいけません。

### 日本語コーパス投入量の方針

男性入力から萌え声へ変換するライブVCでは、日本語コーパスの量を単純な時間数だけで決めてはいけません。重要なのは、どの経路に何を学ばせるかです。

役割分担:

```text
source-side日本語:
  男性/混合話者のcontent、prosody、ライブ発話、言い淀み、短い反応を扱う。

target-side日本語:
  女性/萌え声/ASMR/囁き/感情発話のtarget manifoldとstyleを扱う。

same-text日本語:
  content保持、source leakage、target similarityの検証に使う。

golden日本語:
  毎回同じ短文でHuman ABを行い、改善か偶然かを判定する。
```

短期の目安:

| 段階 | 量 | 目的 |
|---|---:|---|
| Golden Mini Set | 100-300発話 | 毎回固定のAB評価、bad tag検出 |
| Gate確認 | 1-5時間 | codec/decoder/content/prosodyの成立確認 |
| Core学習 | 10-50時間 | 日本語content/prosody、target styleの初期成立 |
| 実用探索 | 50-150時間 | 話者・発話型・styleのばらつき確保 |
| 大規模化 | 150時間以上 | GateとHuman ABで改善が続く場合のみ |

ただし、25種類程度の読み上げ文を話者数だけ増やしても、ライブVCの学習量としては頭打ちになります。それは声質・styleの面展開には使えますが、日本語の発話運用を学ぶには不足します。

増やすべきなのは時間数より先に発話型です。

必須発話型:

- 短い反応: 「えっ」「まじで」「うそ」「ん？」「あ、はい」
- 相槌: 「うん」「へえ」「なるほど」「そうなんだ」
- 言い淀み: 「えっと」「あの」「その」「ちょっと待って」
- ライブ発話: チャット確認、聞こえ確認、配信開始・終了、聞き返し
- 感情: 驚き、照れ、笑い、困り、安心、甘え
- 囁き・小声: 近接、内緒話、寝落ち、語尾の息
- 日本語音韻: サ行、シャ行、ツ、チ、ハ行、ラ行、撥音、促音、長母音、母音無声化
- 終助詞・語尾: 「ね」「よ」「かな」「かも」「だよ」「なの」「じゃん」

投入を止める条件:

- 読み上げ文を増やしてもGolden Mini SetのHuman ABが改善しない
- bad tagsが同じまま残る
- source leakageが下がらない
- whisper / breath / small voiceが改善しない
- high-band clarityが改善しない

この場合、さらに日本語読み上げを増やすのではなく、発話型、target manifold、codec/decoder、評価GUI、preference loopのどこが詰まっているかを見直します。

manifest は最低限以下を持ちます。

```text
utterance_id
path
path_type
layer
source_type
speaker_id
speaker_gender
text
text_id
caption_key
caption_text
duration_sec
split
license
quality_status
quality_flags
wav_path
latent_path
feature_path
```

現役のmanifest入口:

```text
data/kansei_vc/manifests/canonical_utterances.tsv  raw wav / latent / feature を束ねた完全台帳
data/kansei_vc/manifests/trainable_utterances.tsv  latent_pt / pair_pt かつ quality_status=ok の学習用manifest
data/kansei_vc/manifests/all_utterances.tsv        既存学習コード互換。trainable_utterances.tsv と同内容
```

生成・更新コマンド:

```text
cd training
uv run python build_canonical_manifest.py
```

`create_manifest.py` は互換入口であり、同じ canonical builder を呼ぶだけです。新しいAI作業者は `stem.split("_")` で caption を推定してはいけません。`low_tension`、`young_bright`、`intimate_close` のような underscore を含む caption が壊れるためです。

すべての utterance を単一の平坦な pool として扱ってはいけません。Target manifold、source content、live speech、ASMR speech、evaluation sample は役割が違います。

## Human-In-The-Loop ルール

候補ファミリは、成功扱いする前に必ず小さな AB set を出します。

このため、B4では推論GUIとは別に **学習用アノテーションGUI** を作ります。これは一般的なラベル付け画面ではなく、AIが生成した候補音声を人間が短時間で比較し、その選好を次の学習・探索へ戻すための評価ハーネスです。

必須機能:

- A/B 音声再生
- `A` / `B` / `tie` / `reject_both` の選択
- bad tag の複数選択
- 短いメモ入力
- persona / scene / relation / preset の選択
- candidate の生成条件とcheckpoint情報の表示
- 評価結果をJSONLまたはSQLiteへ保存

人間フィードバック形式:

```text
choice: A / B / tie / reject_both
bad_tags: metallic, muffled, rough, sibilant, source_leak, weak_vc,
          breath_dead, whisper_broken, tiring, latency_feel, uncanny
memo: short free text
```

AI はこの feedback を次回実行の制約として使います。SECS、margin、P(male)、STFT、mel loss だけを最適化してはいけません。

保存データには最低限以下を含めます。

```text
eval_id
timestamp
evaluator_id
preset
persona
scene
relation
candidate_a_path
candidate_b_path
candidate_a_checkpoint
candidate_b_checkpoint
candidate_a_controls
candidate_b_controls
choice
bad_tags_a
bad_tags_b
memo
```

このアノテーションGUIがない状態で「萌え」「ASMR comfort」「官能品質」を学習できたと判断してはいけません。

#### アノテーションGUI（実装済み）

推論用 Rust GUI とは別の独立ハーネスとして実装済み。追加依存なし（Python標準ライブラリのみのローカルwebアプリ）。

```text
training/build_ab_session.py … 候補音声 -> ブラインドA/Bペア manifest（JSON）
training/annot_gui.py         … ローカルweb GUI。判定を JSONL 追記（ブラインドA/B判定用）
training/listen_gui.py        … 任意のwavディレクトリを聴き比べる再利用ローカルアプリ（role接尾辞で自動グループ化 + 6帯域スペクトル + プロキシをライブ計算）
```

聴き比べ（判定ではなく素の確認）は `listen_gui.py`。焼き込みゼロで、どのcheckpointの出力でも同じツールで開ける。

```text
cd training
uv run python gate0_codec.py --from-dir ../female-dataset --n 12 --export 12 --tag t   # codec trio出力
uv run python gate0_generated.py --checkpoint <ckpt> --n 4 --export-audio 4             # 診断trio出力(source/oracle/gen)
uv run python listen_gui.py --dir ../results/gate0_t_ab      # -> http://localhost:8772
uv run python listen_gui.py --dir ../results/diag_<ckpt>
```

実行例（Gate 0 の base vs finetuned decoder を原音付きで比較）:

```text
cd training
uv run python build_ab_session.py gate0 --dir ../results/gate0_female_real_ab \
    --out ../results/ab_sessions/gate0_female_real.json --session g0_female
uv run python annot_gui.py --session ../results/ab_sessions/gate0_female_real.json --port 8770
# ブラウザで http://localhost:8770 -> results/ab_results/g0_female.jsonl
```

仕様に加えた保証:

- **ブラインド化をサーバ側で強制**：評価中は checkpoint / 候補の正体をブラウザに渡さない。A/B の表示順は (評価者, pair) のハッシュで決まり、resume 時も一貫。保存は真の候補（candidate_a/_b = manifest順）+ `shown_order` 形式なので解析が曖昧にならない。
- **resume**：評価者ごとに判定済み pair をスキップ。
- キーボード操作（<kbd>1</kbd>/<kbd>2</kbd>=A/B再生, <kbd>0</kbd>=原音, <kbd>a</kbd>/<kbd>b</kbd>/<kbd>t</kbd>/<kbd>r</kbd>=選択, <kbd>Enter</kbd>=保存）。
- 保存スキーマは本節の定義に準拠（eval_id, evaluator_id, candidate_a/b_path/checkpoint/controls, choice, bad_tags_a/b, memo, persona/scene/relation/preset）+ `choice_shown` / `shown_order` / `reference_path`。

これで HITL ループの step 4（人間評価）が実データで回せる。`gate0_female_real_ab/` の trio ×6 が最初の評価対象。

### 最適化ループ

Human-in-the-Loop は単なる主観評価ログではありません。B4では、評価を次候補生成に使う **preference optimization loop** として実装します。

1回のループ:

```text
1. 候補生成
   現在のcheckpoint、preset、control範囲から複数候補を生成する。

2. 自動フィルタ
   明らかな失敗を人間に聞かせる前に落とす。
   例: clipping、無音、source leakage高、high-band clarity低下、WER崩壊。

3. ペア選択
   人間が判断しやすく、かつ学習価値が高いA/Bペアを選ぶ。
   似すぎた候補、両方明確に悪い候補、既に結論済みの比較は避ける。

4. 人間評価
   アノテーションGUIで A/B/tie/reject_both、bad tags、memo を保存する。

5. Preference model更新
   人間選好から、presetごとの好ましさを予測するモデルを更新する。

6. 次候補提案
   preference model とbad tagsを使い、次に試すcontrol、checkpoint、loss重み、dataset samplingを提案する。
```

preference model は最低限、以下を入力にします。

```text
candidate_features:
  objective_metrics
  controls
  preset/persona/scene/relation
  checkpoint_id
  dataset_mix
  bad_tag_history

target:
  human choice
  reject_both
  bad_tags
```

最初は単純な集計・ランキングでよいですが、データが増えたら Bradley-Terry / logistic preference model / Gaussian process / Bayesian optimization のいずれかで候補選択を行います。

最適化対象は単一の `moe_score` ではありません。最低限、以下を多目的に扱います。

```text
maximize:
  human preference
  smoothness
  tenderness
  embodiment
  target likeness
  ASMR comfort

minimize:
  source leakage
  metallic
  muffled
  rough
  sibilant
  breath_dead
  fatigue
  latency
```

候補生成エージェントは、前回の bad tags を直接修正仮説に変換します。

```text
sibilant      -> sibilance guardを上げる / high-band texture lossを調整
muffled       -> high-band clarityを上げる / decoder or target manifoldを疑う
source_leak   -> content path or target conditioningを疑う
weak_vc       -> conversion_intensity or target conditioningを上げる
breath_dead   -> breath/texture priorとASMR target dataを増やす
tiring        -> high-band roughnessとdynamic shockを下げる
```

自動フィルタ（ループのstep 2）は、`training/kansei_proxies.py` の客観プロキシで、人間に聞かせる前に候補を落とします。各 bad_tag には検出プロキシが対応します。

```text
muffled       <- centroid低下 / hf_preserve低 / rolloff85低下
metallic      <- hf_flatness異常 / mel_delta_var増（ざらつき）
breath_dead   <- cpp上昇 / hnr上昇 / h1h2低下
sibilant      <- sib_ratio（5-9kHz比）増
rough/tiring  <- mel_delta_var増 / jitter / shimmer / hf_flatness増
source_leak   <- ECAPA cos(out,source) vs cos(out,target) margin, gender-probe（別途model要）
```

`bad_tag -> fix` は「人間評価後に何を直すか」、`bad_tag -> proxy` は「人間評価前に何を機械検出するか」です。両方を使い、人間の耳を疲れさせる前に自動で落とします。

採用条件:

- preference model上で改善している
- objective safety gateを破っていない
- human ABで前回bestに勝つ、または明確な改善タグを得る
- 同一bad tagで2回連続rejectされない

このループが実装されていない状態で、単にcheckpointを増やしたりlossを変えたりする作業を「Human-in-the-Loop最適化」と呼んではいけません。

## GUI パラメータ結線

現行Rust GUIには旧runtime向けの結線が残っています。これは実装済みですが、B4の正規UI意味論ではありません。

現在の主なlegacy結線:

| GUI表示 | 現在の送信先 | 現在の意味 | B4での扱い |
|---|---|---|---|
| `Strict / Balanced / Quality` | `RtControl::SetMode` -> `Backend::set_mode` | latency/chunk mode | 継続使用 |
| `Prosody: Imitate / Preserve / Blend / Flatten` | `RtControl::SetProsody` | 旧prosody混合 | B4 prosody policyへ再定義 |
| `velocity` | `RtControl::SetVelocityScale` | 旧flow/変換強度 | `conversion_intensity`へ置換 |
| `Load Reference` | `RtControl::LoadReference` -> `Backend::set_target` | 旧target voice参照 | target timbre/style抽出へ拡張 |
| `B1 Adapter / Tau / WetDry` | `SetB1Tau` / `SetWetDry` | 旧B1診断 | 研究・診断以外では非表示 |

B4 GUIは、低レベルの `tau`、`delta`、旧 `velocity_scale` をユーザー操作として前面に出してはいけません。ユーザーには官能・受肉の操作軸を見せ、内部で content/prosody/target/style/texture へ配線します。

正規B4パラメータ:

| GUIパラメータ | 範囲/型 | 接続先 | 目的 |
|---|---:|---|---|
| `conversion_intensity` | 0.0-1.0 | generator conditioning | 変換強度。旧deltaではない |
| `self_prosody` | 0.0-1.0 | prosody path | 操作者の抑揚・間・勢いを残す量 |
| `target_timbre` | reference/profile | target timbre path | 声色・話者性 |
| `age_brightness` | 0.0-1.0 | target style path | 若さ、明るさ、張り |
| `distance` | public/friendly/intimate | target style + texture path | 距離感、近接感 |
| `breathiness` | 0.0-1.0 | texture prior | 息成分 |
| `softness` | 0.0-1.0 | texture prior | 柔らかさ、刺さらなさ |
| `tension` | calm/normal/energetic | prosody + style path | テンション |
| `asmr_comfort` | 0.0-1.0 | texture prior + safety gate | 耳当たり、疲れにくさ |
| `latency_mode` | strict/balanced/quality | streaming runtime | 遅延・品質トレードオフ |
| `bypass` | bool | audio runtime | 変換バイパス |

B4 runtime control bus は現在の `RtControl` を拡張してよいですが、意味論は以下に揃えます。

```text
GuiControl
  SetLatencyMode(latency_mode)
  SetTargetProfile(profile_id or reference_audio)
  SetConversionIntensity(value)
  SetSelfProsody(value)
  SetStyle { age_brightness, distance, breathiness, softness, tension }
  SetAsmrComfort(value)
  SetBypass(bool)
```

GUIのプリセットは、上記パラメータの束として保存します。例:

```text
Preset: intimate_soft
  conversion_intensity = 0.85
  self_prosody = 0.70
  distance = intimate
  breathiness = 0.55
  softness = 0.80
  tension = calm
  asmr_comfort = 0.90
```

GUI結線の採用条件:

- `conversion_intensity` を上げても high-band clarity が落ちない
- `self_prosody` を上げても source timbre が漏れない
- `breathiness` / `softness` が metallic・muffled・sibilant を増やさない
- プリセット変更が50ms未満runtimeを壊さない
- Human ABでプリセット差が知覚でき、かつ不快方向に倒れない

## 停止条件

以下の場合は実行を止め、結果を archive します。

- 必須の upstream gate が失敗した
- content 保持が source speaker leakage によって成立している
- GAN が失敗モデルを滑らかに聞かせているだけ
- conversion strength を上げると high-band clarity が落ちる
- whisper、breath、小声が崩れる
- human AB が同じ bad tags で同一候補ファミリを2回 reject した

## Archive ルール

古い文書、レポート、plan は `.archive/` のみで保持します。これは歴史的証拠であり、指示ではありません。

迷った場合はこのファイルに従い、archive 済み設計は無視してください。
