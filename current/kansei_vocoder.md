# KanseiVocoder — 自作frontierボコーダ フル設計

> status: **PROPOSED（Gate V-1 学習中・耳ゲート未通過）** ／ ADOPTED: なし
> 最終更新: 2026-07-13
> 位置づけ: B4/SFRV（`current/README.md`）の合成バックボーン。NSF-LTV（A/S方式）が耳で死亡([[as-source-filter-ceiling]])した後の後継。
> 実装: `training/kansei_vocoder.py`（ネットワーク）＋ `training/kansei_train.py`（学習）。
> ルール: 1ネットワーク=1ファイルのフル設計。採用は人間の耳ゲート通過時のみ。

---

## 0. これは何か（一言で）

**原音の mel（声のスペクトル特徴, 128次元/フレーム）から、波形を復元するニューラルボコーダ。**
単一話者の自己再構成で学習し、「mel を入れたら原音そっくりの波形が出る」状態を目指す。
将来はこの mel の代わりに VC 経路が作った「萌え声の mel」を入れて、男→萌え変換の最終合成段になる。

判定は人間の耳のみ。到達目標＝既製 SOTA(bigvgan) と原音(gt) に並ぶ透明さ。

---

## 1. なぜこの設計か（今セッションで物理確定した事実→設計）

全て2026-07-13の耳駆動診断([[as-source-filter-ceiling]])と過去の失敗系譜(`current/vocoder.md` §2)から導出。思いつきではない。

| 確定した事実 | 設計への反映 |
|---|---|
| istft(波形/STFT保存)＝耳で透明、A/S(パラメトリック包絡)＝不合格 | **合成はISTFTヘッド**。71dB透明を実証した n_fft2048/hop512/win2048 グリッドをそのまま使う |
| 時間方向upsample(ConvTranspose)＝ジリジリ(旧M1) | **時間upsample廃止**。等時間解像度backbone(Vocos型)＝ジリジリが構造的に存在しない |
| AFHNの位相ゼロ予測＝RMS崩壊 | **F0駆動harmonic励起が「正しい位相の下地」を供給**(HiFTNet)。網は下地からの残差だけ学ぶ＝位相が安定 |
| bigvgan(SOTA neural)はこのデータで透明＝データは無罪 | ニューラル波形合成路線が正しいと実証済み。既製122Mでなく**小型自作**で並ぶ |
| ASMR＝息・囁き・小声が商品 | **明示noise枝**を励起に持つ。息を一級市民として扱う |
| XPUでdepthwise conv(groups=in_ch)が失敗 | backboneは**groups=1標準convのみ**(Vocos素のdepthwise ConvNeXtは不採用) |
| P1回帰=muffle / P3 texture=敵対のみ | 波形の質感は**GAN(MPD/MRD)が担う**。melは絶対振幅のアンカー |

**系譜上の位置**: HiFTNet(F0駆動 harmonic+noise → iSTFT, MIT, 非causal)の **causal・自作・ASMR/moe制御版**。
アーキは借り物ゼロ、全コード自作。既製の重みは天井参照(bigvgan)・mel知覚教師としてのみ使い、出荷グラフには載せない(キメラ禁止)。

---

## 2. ネットワーク設計図（信号の流れ）

```text
入力:  mel [128次元 × Tフレーム]      F0 [Tフレーム]（原音から抽出）
                                        │
                    ┌───────────────────┘
                    ▼
      ① 励起生成（非学習・決定論・自作HarmonicSource）
         harmonic:  F0から倍音サイン波を合成 → 正しい瞬時位相を持つ波形 e_h
         noise:     ランダム雑音 × 0.1（息・非周期成分＝ASMRの核）
         exc = e_h + noise
                    │
                    ▼
      ② 励起をSTFTでスペクトルに（位相の下地を供給）
         E = STFT(exc)  → 実部・虚部 [2×1025 × T]
                    │
        mel ────────┤
                    ▼
      ③ 入力射影  Conv1d(128 + 2050 → 512)          … mel と 位相下地 を合流
                    │
                    ▼
      ④ backbone: 等時間解像度ブロック ×8（時間長は一切変えない）
         各ブロック = Conv1d(512,512,k7,groups=1) → LayerNorm
                      → Linear(512→1536) → GELU → Linear(1536→512) → 残差加算
         （groups=1＝XPU安全。時間upsampleなし＝ジリジリ構造的不在）
                    │
                    ▼
      ⑤ 出力ヘッド  Linear(512 → 2×1025)
         mag   = clip(exp(前半1025))               … 振幅（フィルタ包絡）
         pres  = π·tanh(後半1025)                  … 位相残差（init 0）
         unit_src = Es / (|Es|+eps)                … 励起の位相（複素単位）
         S = mag × unit_src × (cos(pres)+i·sin(pres))
         ★出力位相を励起(harmonic)の位相にアンカー。網は振幅＋わずかな位相残差だけ。
           自由位相予測だと倍音がコヒーレントにならず谷が雑音で埋まる=かすれ
           (実測: 倍音間コントラスト 自由位相14.9dB vs gt19.9/bigvgan20.5)。
           アンカーで解消(2026-07-13 耳確認「かすれ解消」)。ソースフィルタの正しい形。
                    │
                    ▼
      ⑥ ISTFT（証明済み透明バックエンド, n_fft2048/hop512/win2048）
                    │
                    ▼
                出力波形 [T×512 サンプル @44.1kHz]
```

規模: 現状 dim=512・8層で **約29.5M params**（透明性検証優先。RT向けにV-2で縮小予定）。

### なぜ位相が安定するか（①②が肝）
目標波形 ≈ 励起 ∗ 声道フィルタ。位相 = 位相(励起) + 位相(声道)。
**位相(励起)はF0から解析的に既知**なので、網は滑らかで低分散な位相(声道)だけを当てればよい。
ゼロから全位相を回帰すると基本周波数由来の高速回転で誤差が発散する（AFHNのRMS崩壊）。それを回避。

---

## 3. 学習プロセス

- **データ**: 単一話者 af1ad5575a3fa383。F0キャッシュのある66発話（3発話をevalに保留、63で学習）。
- **入力の作り方**: 波形セグメント → BigVGAN の mel 抽出関数(128mel/2048/512, 知覚教師)で mel、F0はキャッシュから。
- **教師**: 原音そのもの（VC teacher蒸留は禁止＝別VCの変換音を目標にしない）。
- **損失（bigvgan train.py の結線を忠実踏襲）**:
  - `MultiScaleMel(原音, 生成) × 15` … 複数解像度melのL1。絶対振幅・スペクトル包絡のアンカー。
  - `MPD + MRD 敵対的損失` … 波形の質感(texture)を担う。回帰のボケを鮮明化(P3)。
  - `feature matching` … 判別器中間特徴の一致。GAN安定化。
  - 生成器 = mel + 敵対 + FM を最小化、判別器 = 本物/偽物を分離。**回帰項は補助、質感は敵対が担う**(P1回避)。
- **最適化**: AdamW(lr 2e-4, β 0.8/0.99)。batch 8、64フレーム(≈0.74秒)セグメント。
- **評価**: 1000 step毎に held-out 3発話を mel→波形で再構成 → `results/e2_triage/*_kansei.wav` に上書き → 8772で **kansei / bigvgan / istft / gt** を耳AB。判定=耳のみ。

---

## 3.5 潰したバグ（耳＋コード検証, 2026-07-13）

判定は耳、原因特定はコード/数値。1変数ずつ。

1. **F0オクターブ誤り**（[[f0-octave-weak-fundamental]]）: この声は基音が弱く第2倍音支配→検出器が全部2倍(~450Hz)に釣られる。真~220Hz。harmonic下地が真の基音・奇数倍音を外し「声の芯が薄い」。耳「普通の高さの女声」で発覚。→ `octave_correct` で発話一律オクターブ補正。合成バックエンドは無罪(istft round-trip 119.9dB)。
2. **F0ジャンプ→かすれ其1**: 最初のoctave補正をフレーム毎閾値でやったため閾値付近で隣接ピッチ2倍飛び(5-10%)→ピッチ不安定→かすれ。→ 発話ごと一律補正でジャンプ0-1%。
3. **位相非整合→かすれ其2**: 自由位相予測で倍音がコヒーレントにならず谷が雑音(コントラスト14.9dB)。→ 出力位相を励起にアンカー(§2 ⑤)。耳「かすれ解消」。
4. **残: F0/F1域の差**（2026-07-13 耳、大幅改善後の残差）: 低域の芯/第1フォルマントに gt との差。学習継続(mel loss低下中)+ source-envelope（弱基音声にflat harmonic下地）のミスマッチが候補。次に詰める。

## 4. 検証ラダー（安い順・1実験1仮説）

- **V-1 透明性ゲート（現在）**: 非causal(center STFT)で品質変数を隔離。単一話者を再構成し、耳で bigvgan/istft に迫れるか。
  - 合格 = 自作アーキがこのデータで透明に届くと実証 → V-2へ。
  - 不合格 = 方向は正だが未到達なら容量/学習量を1変数調整。構造的異音ならアーキ側(AA活性・ヘッド・source注入位置)を1変数修正。
- **V-2 causal + streaming + 小型化**: STFT/backbone/ISTFTをcausal化(左pad・center廃止)、<50ms/CPU-RTへ縮小。非causal V-1をteacher蒸留で立ち上げ(arXiv:2408.11842)。causality CIテスト必須。
- **V-3 男→萌えcross接続**: mel入力をVC経路(content+萌えprosody+target)出力に差し替え。F0/formant/breathinessの明示制御(§5)を配線。
- **V-4 Rust/Candle移植**: 出荷推論。事前学習モデル完全排除、自作weightsのみ。

各段で人間の耳ゲート。proxy単独昇格禁止。

---

## 5. 製品ツマミとの結線（バ美声ASMR制御, V-3で学習）

励起・条件ドメインの操作＝再学習ゼロのリアルタイムツマミ:

| GUIツマミ | 実装 |
|---|---|
| breathiness（息） | noise枝ゲイン ＆ 有声/無声バランス |
| register / liveliness（高さ・抑揚） | F0経路（既存 render_m2 の register_st/exaggerate） |
| formant（声の小ささ） | mel/包絡の周波数warp |
| softness | スペクトル tilt |

息＝商品の核が noise枝として明示パラメタ化される。

---

## 6. frontier位置づけ（何でSOTAを主張するか）

- **動作点の未占有**: causal × 44.1kHz × <50ms × CPU × 囁き/ASMR × Kansei-HITL は文献・OSSともに空白(`current/vocoder.md` §7)。
- **差別化**: F0駆動harmonic位相下地 + ISTFTヘッドをRT-causalで + 明示ASMR/moe制御 + 息の一級市民化。
- 品質主張はV-1/V-2の耳ゲートが決める。数字で盛らない。

---

## 7. リスク登記（正直な期待値）

- 29.5MはRT過大 → V-2で縮小時に品質が落ちるか（teacher蒸留で回復狙い）。
- causal化(V-2)で位相・透明性が劣化するか（ISTFT窓遅延・左pad）。
- 小型でbigvgan(122M)天井に届かないか（届かなければ容量を上げる、測れる結論）。
- F0推定器のcausal・低遅延化（V-3、別ネットワーク課題）。
