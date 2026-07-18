# 研究中（作業ノート）

> 揮発ファイル。今の問い・仮説・反証実験・進行 run・negative results を書く。
> 固まった設計は `current/vocoder.md` 等のフル設計へ、採用決定は `current/README.md` §1 へ昇格。
> 昇格は必ず overfit gate / 人間の耳ゲートの実測を添える。RESEARCH→採用の直行禁止。
> 最終更新: 2026-07-17

---

## ★決着（2026-07-17）: F0非依存 薄型Vocos型ボコーダ freebig が BigVGAN 耳同等 → ADOPTED

- **R-proto-A（F0非依存 ISTFTヘッド + 多話者ユニバーサル学習）の中核仮説が成立**。本気訓練（女声41k発話, 188k step, 28.8M）で**未学習の弱基音萌え声でも BigVGAN と耳同等**。品質ギャップの主因は「未訓練/データ被覆」で確定（arch/損失/位相/source-filter は無罪）。勝ち重み=`training/checkpoints/freebig/foundation_bigvgan_parity.pt`。
- **フル設計・採用状況は `current/vocoder.md` §0.1（ADOPTED）へ昇格済み**。低遅延派生 config C（freeC, causal 5.8ms）も同節。
- 以下の「ハイブリッドボコーダ研究」節は**この決着に至る改善ループの全記録**（残置）。反証済み経路は各節内 negative result として保持。

---

## 現在の最重要フォーク（2026-07-13〜）: DSP+薄ネット ハイブリッドボコーダ（研究中→2026-07-17 決着済み）

### 今セッションで確定した事実（耳＋コード検証、これが新方向の土台）

1. **A/S source-filter 方式は耳で天井死亡**（[[as-source-filter-ceiling]]）。同一発話・同一gainで並置: gt/istft(STFT往復,71dB)=合格 / **WORLD(標準A/S)=不合格 / NSF-LTV oracle(GT由来完全包絡)=不合格**。成熟WORLDすら落ちる=実装バグでなく方式クラスの天井。oracle(完全包絡)がゴミ=予測器/GAN/包絡教師の改善は全て無関係。→ NSF-LTV 系譜(NSF-HN→AFHN→HN2→HN3→LTV)は打ち切り。
2. **BigVGAN(F0非使用・事前学習)はこの声で透明**（耳確認）。データ無罪、neural波形合成は届く。ただしキメラ禁止で出荷不可＝天井参照。
3. **自作 KanseiVocoder(ISTFT + DSP harmonic励起 + 神経複素スペクトル, 敵対)= min-phase A/S 天井を突破**（`kansei_vocoder.py`/`kansei_train.py`）。eval発話で透明に接近・**メタリック/ジリジリ皆無**。潰したバグ: F0オクターブ誤り([[f0-octave-weak-fundamental]])→発話一律補正 / F0ジャンプ→かすれ / 自由位相→かすれ(倍音間コントラスト14.9dB vs gt19.9)→**出力位相を励起位相にアンカー**で解消。F0/F1の10%痩せ→MRSTFT loss追加で解消。
4. **「ジリジリ≠F0非依存」**: 過去のF0非依存が金属質だったのは**時間upsample(ConvTranspose)のエイリアシング**([[jirijiri-interharmonic-aa]])が原因で、F0の有無と無関係。kansei は ISTFT(時間upsample無し)で**一度もジリジリを出していない**。
5. **この声はF0が原理的に追跡不能**（弱基音＝萌え声の音色特性）。harvest/pyin/autocorr/CREPE 全て第2倍音(~400-450Hz)に釣られ有声率2-13%。→ F0駆動 harmonic下地は萌え/ASMRで本質的に脆い。

### 研究仮説（ユーザー提案 2026-07-13）

> **実装粒度の文献根拠は [[hybrid_dsp_vocoder_survey]]（`current/hybrid_dsp_vocoder_survey.md`）に集約**（LPCNet/FARGAN/Meta DDSP/torchlpc/GOLF/DDSP-SVC/MVF/D4C/HiFTNet/NHV/F0堅牢性 の層構成・予測対象・FLOPS・微分可能性・ライセンスを出典URL付き）。設計判断の一次資料。

**「DSP(harmonic+noise + FIR/LPC 時変フィルタ) + 薄いニューラル網(高域MVF/aperiodicity/位相 の時変予測)」の役割分担で、A/S天井を突破しつつ 軽量CPU-RT × 44.1kHz × ASMR を同時達成できるか。**

- 動機: bigvgan(122M)/kansei(29M)は**CPU realtimeに重すぎる**。製品(CPU<50ms×44.1k)には DSPで大半を担い薄ネットを位相/MVF/apに surgical に使う軽量系が正道（前例: LPCNet<3GFLOPS / FARGAN / Meta 15MFLOPS MOS4.36 = 全て DSP+薄ネット CPU-RT）。
- A/S天井突破の根拠: kansei が既に「DSP励起+神経位相(非min-phase)」で天井を破った実測がある。本提案はその insight の軽量RT版。min-phaseのrobotic臭を学習位相(mixed-phase)で除くのが核。
- **位相は「励起位相からの残差」で予測（ゼロ予測はAFHNのRMS崩壊、実測済み）**。
- ASMR: MVF/ap を時変予測し harmonic/noise を分割。低MVF(breathy)時はF0依存が低域に限局→**F0誤差の可聴影響が減る**読みで、5のF0脆弱性を吸収できるか検証。

### 検証計画（安い順・kansei を動く参照に）

- **参照**: kansei(動く・透明接近) を耳ABの上限側、bigvgan を天井、A/S(WORLD/oracle) を不合格側の係留。
- **R-lit**: 実装粒度の一次資料収集（LPCNet/FARGAN/Meta軽量・微分可能LPC/FIR・MVF/ap予測・位相残差・44.1k CPU-RT予算）＝進行中（agent）。
- **R-oracle**: A/S oracle を min-phase→**学習/mixed-phase**に替えた版が耳で天井を破るか（位相が主因かの決着。kansei が既に示唆）。
- **R-proto**: 薄DSPハイブリッド試作 → 単一話者 recon → **kansei と同一eval発話で耳AB**（薄い方が kansei並み品質を 1/N 計算で出せれば RT向けに勝ち）。
  - **R-proto-A（進行中, 2026-07-14, ユーザー選択）= F0非依存の薄型 Vocos型**（`training/free_vocoder.py` / `free_train.py`, tag=`free`）。研究が炙り出した非対称を突く: **BigVGAN(F0なし)=この声で透明 / kansei(F0駆動)=未知発話でかすれ** ⇒ **F0-harmonic源は弱基音声の負債**（vocoder段でF0は不要、melがピッチ内包、F0制御は上流VC/prosody段の責務）。F0非依存は「測って間違える」failureを構造的に持たない ⇒ 未知かすれの根因消滅（survey §7-3 Vocos/§9-6）。設計=Vocos忠実(mel→等時間解像度ConvNeXt→複素STFTヘッド mag·exp(jφ)→ISTFT, 自由位相)を実証済みISTFTグリッド(2048/512/71dB)・groups=1(XPU安全)で。**単一仮説: F0を捨てても自由位相ISTFTヘッドで耳がBigVGANに並ぶ**。容量は品質確認優先で据置(28.8M, BigVGANの1/4)、勝てば軽量化(dim/層削減)へ。eval 1000step毎 `results/e2_triage/*_free.wav`。判定=耳のみ。
    - **早期の設計含意（自由位相の再解釈）**: 過去「自由位相=かすれ(コントラスト14.9dB)」は**harmonic源が存在する文脈**での artifact。純Vocos(源なし)の自由位相はGAN+mel+ISTFT重畳整合でコヒーレント化する(Vocos/BigVGANが実証)。源を入れて半端に自由位相を残すと源と競合=かすれ、が正しい帰属。
    - **耳ラウンド1（step 5000, 未学習6発話, free vs kansei）: 「kanseiのかすれは減った」（ユーザーの耳）**。free は未学習でかすれず、残るのは全体の精度不足（未収束の粗さ・ボケ）のみ ＝ 別種の異音でない。**「F0-harmonic源＝この弱基音声の負債（未知発話かすれの元凶）」の帰属を耳で裏付け**。方向確定。まだ成果でなく方向シグナル（全体は粗い＝昇格せず）。→ step 継続（30k）して精度で再判定。次: 15k で未学習6を再レンダ・再耳AB。
    - **耳ラウンド2（step 15000, 未学習6, free vs bigvgan）: 55%（ユーザー）。ギャップの正体＝「かすれが残る」**。mel は 5k→15k で横ばい（0.75-1.0）＝**プラトー**（step だけでは埋まらない）。帰属: **残りかすれは F0非依存のせいでなく ISTFT自由位相ヘッド固有**（BigVGAN=F0非依存・時間領域アンチエイリアスで**かすれ皆無**＝F0非依存でもかすれゼロは可能、と対照）。かすれ＝倍音間位相の非コヒーレンス。→ **設計レバー＝位相の明示監督（anti-wrapping IP+GD+IAF, survey §7-0）**。
    - **位相アーム（free_ph, anti-wrapping IP+GD+IAF, warm 15k+8k）: 耳ラウンド3「変化無し」＝負の結果**。位相損失（振幅重み付け）は かすれを動かせず。
    - **機構の客観切り分け（計測=機構特定用, 判定は耳）**:
      - **comb深さ（倍音間コントラスト, 未学習発話）: 全アーム 13.0-13.5dB で同一（gt/bigvgan含む）** ＝ **かすれ ≠ 倍音間の谷埋め**。当初の「自由位相→谷ノイズ」モデルは棄却（位相損失が効かなかった理由と整合）。
      - **帯域別 対gt log-mag MAE（未学習6）**: free の劣化は**低〜中域(0-1kHz=声の芯)に集中**（0-300: free 6.76 vs bigvgan 3.74, +3dB／300-1k: 5.78 vs 2.95）。HF差は小(11-22k: 8.82 vs 7.31)。
      - **free-TRAIN vs free-UNSEEN**: 0-300 = 4.68(train)→6.76(unseen)＝**明白な汎化ギャップ**。かつ free-train(4.98@300-1k)ですら bigvgan-unseen(2.95)に未達。
    - **診断確定: free の未学習かすれ＝主に汎化/データ規模の敗北**（アーキ無罪寄り）。天井 bigvgan は**多話者ユニバーサル**、free は単一話者63発話。63発話では universal vocoder の汎化に勝てない。→ **正着＝free を多話者コーパスで自作ユニバーサル化**（bigvganの天井達成法そのもの・自作weightsでキメラ禁止も満たす）。
    - **ユニバーサル・アーム起動（freeuniv, 2026-07-14, fresh）**: `free_train_universal.py --buf 4000 --tag freeuniv`。コーパス=female-dataset(2776話者/41,554発話/35G)、**af1ad5575a3fa383 は完全holdout**（学習除外）。同一GANレシピ（位相損失なし=無効判明）。eval=af1ad の eval-3+未学習6（=かすれ試験集）。**単一仮説: 多話者学習で未知発話の汎化が付き、かすれが消え天井(bigvgan)に近づく**。対照=free単一話者(55%/かすれ)。判定=耳。af1ad holdout の早期checkpointで汎化が上がるか耳AB。
      - **12k 客観（機構確認, holdout帯域MAE）: freeuniv ≈ free（0-300: 6.50 vs 6.57 / 300-1k: 5.75 vs 5.72）＝まだ差なし**。想定内（ユニバーサルは収束にstep要）。**耳ゲートは出さない**（早期=無駄打ち回避）。方針: 収束させ holdout帯域MAEが free-train(4.68)/bigvgan(3.69)方向に落ちるか客観監視 → trend出れば耳、~6.5頭打ちなら汎化仮説棄却→時間領域合成レバーへ。次: 30k で帯域MAE再測。
      - **30k 客観: 弱いtrend**（0-300: 6.50→6.18↓, HF↓, だが**声の芯300-1k/1-3kは横ばい5.70/5.82**で bigvgan 2.99から遠い）。客観は曖昧＝**耳に仲裁を委ねた**。
      - **★耳ラウンド4（step 30000, af1ad完全holdout, freeuniv vs free）: 「めちゃ良くなりました」（ユーザー）＝汎化仮説を耳で確定**。多話者ユニバーサル化で free単一話者の かすれが大きく改善。**重要: 帯域MAEはほぼ不動（0-300 6.57→6.18）なのに耳は大改善＝proxyが捉えない perceptual gain を耳が捉えた**（「判定=耳のみ・proxy昇格禁止」原則の実証、[[as-source-filter-ceiling]] と同型の教訓）。→ **R-proto-A の中核仮説「F0非依存 薄型ISTFT + 多話者ユニバーサル学習で かすれ解消」成立方向**。ckpt保全=`freeuniv_30k_earwin.pt`。
      - **耳ラウンド5（step 60000, 対 bigvgan 天井）: 残ギャップ＝「定位感のブレ」**（音像の揺れ）。かすれは解決、残りは天井との差。
      - **★定位感のブレの正体（10軸triangulation, 2026-07-14, [[freeuniv-universal-vocoder]]）**: band-MAE/comb深さ/振幅・位相jitter/crest/kurt/GD分散/重心wobble/低速変調 は**全て盲目**（freeuniv≈bigvgan≈gt）。効いた軸=**同一発話再構成の帯域別 残差-vs-gt**: 低域(0-300Hz=基音)で **freeuniv 1.2dB vs bigvgan 0.6dB（2倍ズレ）**。band-MAE(6.5 vs 3.7)・位相誤差(1.60 vs 1.36)も同じ低域集中で三点一致。**機構**: hop512 > 基音周期(~200smp@220Hz)。各STFTフレームが基音周期を2.5個跨ぐ→ISTFT自由位相ヘッドがフレーム境界で基音の位相連続を保てず基音がビート→芯が揺れる＝定位感のブレ。bigvganは時間領域(サンプル連続)なので基音が固い。
      - **定位感の理論（2026-07-15, ユーザー主導の理論詰め）**: モノラル定位感＝**音源コンパクトさ＝倍音間位相コヒーレンス（共通GCIで全倍音を同時刻集中→鋭いパルス列→固い音像）**。ブレ＝その時間的ゆらぎ。freeuniv自由位相ヘッドは位相が劣決定（損失がほぼ振幅ドメイン、倍音を共通GCIに縛る構造制約なし）→残留分散が入力依存で揺らぐ。**かすれ(振幅ドメイン)は多話者汎化で解けたが、定位感(位相コヒーレンスドメイン)は直交ゆえ残る**——観測順序を理論が予言。
      - **DSP因果テスト2つとも失敗（実験設計の教訓）**: ①gt低域位相をランダム破壊→「ただ壊れた音」（破壊が雑すぎ、微妙なブレを再現せず）。②freeuniv/bigvgan間で振幅・位相swap→「別物なので違うとしか言えない」（複数変数同時変化＝単一変数分離になってない）。**教訓: 変数分離は単一基準から1変数だけ変える必要。別モデル間swapや過剰破壊は無効**。→ 正しい単一変数実験＝同一モデルで位相パラメータ化だけ変える（ヘッド比較）。
      - **理論からの損失選別（ユーザー提案の吟味）**: 自己相関損失＝**位相盲目（Wiener-Khinchин: 自己相関↔パワースペクトル）**で定位感に無力・除外。群遅延一致・白色化ℓ4/ℓ2鋭さ＝shift不変かつ位相依存で的中。
      - **★統合アーム起動（freegci, 2026-07-15, ユーザー指示「まとめて組み入れて訓練」）**: `FreeVocoderGCI`（**GCIアンカーヘッド**=per-frame共通τの線形位相−2πkτ/NFFTで全倍音を構造的にGCI整合＋小分散残差, comb振幅×共通τ=鋭い周期パルス）＋ **sharpness_loss**(白色化ℓ4/ℓ2, 位相のみ, λ30)＋ **gd_loss**(群遅延一致, 高振幅bin重み, λ2)。freeuniv 77kからbackbone+mag warm-start。1実験1仮説を外し理論一式を同時投入（実験コスト~12h/本を踏まえた指示、効けば後でablation）。**仮説: 構造(GCI)＋損失(鋭さ/群遅延)で位相コヒーレンスを直接締め定位感のブレが消える**。対照=freeuniv(77k, かすれ解決/定位感ブレ)。freeif(temporal-only)は理論的に弱く停止。
      - **freegci 統合の結果（負の結果）: 品質大幅劣化で判定不能（ユーザーの耳 @20k）**。sharpness損失が**発散**（sh 19→28→41、白色化ℓ4/ℓ2はバッチ分散巨大）→ mel 1.0→1.4に悪化、gen/fm上昇。grad clip 100 でも封じ込めきれず。**「まとめて全部」が裏目＝品質崩壊で定位感の検証が confounded**。教訓: 高分散損失は grad clip でも学習を汚す／1実験1仮説を外すと故障が分離できない。
      - **分離＝GCIヘッド単体（freegci2, lam0/0）**: mel 1.30→1.06 回復・帯域MAE≈freeuniv＝**ヘッド無罪・品質維持**（旧統合版100dB崩壊は sharpness損失が犯人と確定）。
      - **★耳ラウンド6（freegci2 20k, 定位感）: 「ブレは増えた」＝GCIアンカー仮説を反証**。予測(GCIで音像が締まる)と逆。
      - **★★反証の解析的測定（2026-07-15, ユーザー要求「物語でなく測定」）**:
        - [1] **freegci2 の予測 τ フレーム間ジッタ std = 44.1 サンプル**（基音周期~200の22%）＝melに絶対タイミング情報が無く τ が定まらず大ジッタ、を数値確認。
        - [2] **モデル非依存「全域時間シフト・ジッタ」**（Δ位相の対周波数傾きの時間分散, 振幅重み, ×10⁻³rad/bin）: **gt 0.546 ≈ bigvgan 0.559 ＜ freeuniv 0.638 ＜ freegci2 0.676**。**順序が耳と完全一致＝10軸全滅の中で定位感のブレを追う初の proxy**。
        - 結論(データ): **定位感のブレ ＝ この全域時間シフト・ジッタ**。freegci2 の絶対τは全域コヒーレント・ジッタを注入し悪化(測定確認)。治すべき量＝freeuniv 0.638→目標~0.55。mel由来の絶対アンカー予測は ill-posed で逆効果、**相対的な位相の時間連続性**が正しい軸。
      - **全域時間ジッタ proxy は失格**: freegci2 再測で 0.543<freeuniv 0.638 と出て耳（freegci2 悪化）と矛盾＝無効。撤回。
      - **★★定位感のブレを追う VALIDATED proxy 発見（2026-07-15, ユーザー「有意な音質差を解析的に観測できないはあり得ない」→スカラー集約をやめ構造化）**: **帯域毎 log包絡誤差(vs gt) の 低速(0.5-2Hz)変調エネルギー**（`measure_wobble.py`）。同一内容ゆえ gt 減算で自然変調が消え純アーティファクトが残る。**bigvgan 4.28（天井・ブレ無）＜ freeuniv 6.54（元のブレ）＜ freegci2 6.87（耳「ブレ増えた」と一致）＜ freeif2 9.74（最悪）＝順序が耳と完全一致**。band-MAE/jitter/crest/GD分散 等の集約スカラーが全盲目だったのは、これが**誤差の平均でなく誤差の低速ゆらぎ**だから。
      - **★大再解釈**: **定位感のブレ ＝ スペクトル包絡が ~1Hz でゆっくり真値からドリフト＝振幅(包絡)の時間安定性の欠陥。位相コヒーレンスではない**（追ってた方向が誤り）。**位相ヘッド実験（GCI・IF）は両方ともブレを悪化**（6.87, 9.74）＝位相方向は棄却。best=freeuniv(6.54)、目標=bigvgan(4.28)。
      - **env-stab損失＝的外れと判明**: warm-start時の学習データ上 es_raw 0.0022（低い）＝学習分布上のブレは既に小。
      - **★★診断: ブレは汎化ギャップ（かすれと同型, 2026-07-16）**: freeuniv wobble = **学習分布内話者 ~2.9（bigvgan 4.28より低い！）vs af1ad holdout 6.54**。freeuniv は学習分布の声には安定包絡、**af1ad(弱基音萌え声)を完全holdoutにしたためこの声だけ包絡ドリフト**。→ **損失やアーキでなくデータ**。af1ad holdout は過剰に厳しい条件（zero-shot to 未知萌え声）。
      - **★修正＝ターゲット声を学習に入れる（実製品の正しい設定）**: `--include-target --target-repeat`（af1ad を buffer に追加、eval3+未学習6 の9発話のみ held-out）。plain head・損失追加なし・freeuniv 77k warm。freetgt(af1ad 3.8%)=効かず→ **freetgt2(af1ad 49%, repeat25)**。
      - **★測定バグ発見・修正（2026-07-16, 厳密性）**: `measure_wobble.py` の相互相関アラインメントが**同一melレンダ(本来サンプル整合済)に偽ラグを注入・誤差水増し・freetgt2 vs freeuniv の判定を反転**させていた。→ **アラインメント除去**（同一mel由来は整合済み、共通長に切るだけ）。旧・全域時間ジッタ proxy の非再現性もこの種の不整合が原因。
      - **★★結果（クリーン測定・held-out af1ad, 指標）: freetgt2 が wobble を下げた**。同一mel/full長/整合済: bigvgan 1.64 ＜ **freetgt2(~3k) 2.79** ＜ freeuniv 3.47（eval-wav版でも bigvgan 2.36<freetgt2 3.69<freeuniv 4.44 で同順）。train af1ad 3.44 ≈ held-out 3.27＝**過学習でなく汎化**。→ **「定位感のブレ＝汎化ギャップ、ターゲット声を学習に含めれば解消」を測定が支持**（まだ3k、bigvganへ前進中）。次: freetgt2 収束→再スクリーニング→**耳で最終確認**。**CPU推論=実測RTF0.046(単スレ22倍速)で余裕**（残課題は causal化と窓遅延、計算でない）。

    - **★★★本気訓練＝freebig で BigVGAN 耳同等に到達（2026-07-17, R-proto-A 決着・耳ゲート通過）**: 診断が一貫して指した「主因は未訓練/データ被覆（アーキ/損失/位相/source-filter ではない）」に従い、**F0非依存 薄型 Vocos型（ConvNeXt groups=1 + 複素STFTヘッド mag·exp(jφ) 自由位相 + ISTFT, 28.8M＝BigVGAN の1/4）を女声大コーパス（female-dataset 41,554発話）で本気訓練**（`training/free_train_universal.py`, tag=`freebig`, 188k step）。**耳ゲート: 未学習 af1ad（弱基音萌え声）で BigVGAN と同等**（ユーザー確認）。かすれ・定位感のブレ（過去の全残ギャップ）が本気訓練で解消。**「品質ギャップの主因＝未訓練/データ被覆」を耳で確定**（かすれ→多話者汎化, ブレ→ターゲット声被覆, そして本量の訓練で BigVGAN 天井に到達＝一連の診断の統合的裏付け）。勝ち重み=`training/checkpoints/freebig/foundation_bigvgan_parity.pt`。証拠wav=`results/e2_triage/*_{freebig,bigvgan,gt,istft}.wav`。→ **昇格: freebig を ADOPTED foundation ボコーダに（`current/vocoder.md`・`current/README.md` §採用 更新済）。耳ゲート合格が根拠（proxy 昇格でない）。**
    - **★低遅延版 config C（freeC, 2026-07-17〜, 派生）**: 出力合成グリッド（vocoder win/nfft=256, hop128）を mel解析窓（2048据置＝上流の条件解像度は維持）から**分離**し causal 化 → **アルゴリズム遅延 実測5.8ms（win/sr=256/44100）**。it70k で耳「劣化なし」（freebig 比、ユーザー確認）。訓練継続中（300k目標）。ckpt=`training/checkpoints/freeC/last.pt`、証拠wav=`results/e2_triage/*_freeC.wav`。**「品質を保ったまま <50ms causal」の実証アーム**（残課題は Rust/Candle 移植と窓遅延台帳、計算は無罪）。

    - **反証済み（この F0非依存線で確定した negative results, 現行の根拠にしない）**:
      - **位相の明示監督は全て無効**: anti-wrapping IP+GD+IAF（free_ph, 耳変化無し）／GCIアンカーヘッド（freegci2, 耳「ブレ増えた」＝反証, sharpness損失は発散で品質崩壊）／IF cumsum ヘッド（freeif2, wobble 最悪 9.74）。**定位感のブレは位相コヒーレンス問題でなく、包絡の低速ドリフト＝汎化ギャップだった**（§上記診断）。→ 位相パラメータ化・位相損失（sharpness/gd/env-stab）は本線で不採用。
      - **A/S source-filter / mixed-phase（R-oracle）**: A/S oracle を min-phase→mixed-phase/学習位相に替えても耳は不合格。**GT由来の完全包絡（oracle）ですら音質が悪すぎ＝位相は主犯でなく A/S 方式クラスの天井**（[[as-source-filter-ceiling]]）。→ 手作り DSP ボコーダ（NSF-HN→AFHN→HN2→HN3→LTV 系譜）打ち切り確定、F0駆動 harmonic 源も弱基音声で負債。
- **R-f0**: 低MVF発話で「F0誤差が可聴か」を切り分け（F0堅牢化を別課題に立てるか、MVFで吸収するか決定）。**→ freebig（F0非依存）で BigVGAN 耳同等に到達したため、vocoder段のF0依存問題は構造的に消滅（F0制御は上流VC/prosody段の責務）。本課題はクローズ。**
- 昇格: ~~耳で kansei 同等以上 かつ CPU-RT見込み → `current/vocoder.md` へフル設計化~~ → **達成: freebig が BigVGAN 耳同等（proxy でなく耳）＋ CPU-RT実証（RTF0.046）＋ config C で低遅延 → ADOPTED（`current/vocoder.md` §0.1）。**

### 旧フォーク（NSF-LTV, 2026-07-11〜13）は打ち切り — 以下は歴史記録

---

## 【打ち切り】旧フォーク: vocoder バックボーン（NSF-LTV, A/S天井死亡で棄却）

**状態**: NSF-HN3 死亡 → NSF-LTV v1.1（設計レビュー）→ v1.2（E0 客観 PASS）→ v1.3（耳 metallic FAIL + コードレビュー反映）→ v1.4（gate v2 + 教師交代）→ v1.5（2026-07-13: E0 オラクルゲート通過）→ **E1 overfit gate PASS（2026-07-13, anneal レジーム）**。次 = spec-scratch の P-C 継続確認 → E2 単一話者 10min recon。

### E1 overfit gate（2026-07-13, `training/e1_overfit_ltv.py`, `results/e1_overfit_ltv.json`）: PASS

- **構成**: LtvFrameNet（§3.4 causal TCN, 7.1M, head bias=データ平均包絡=P-A 遵守）→ v1.5 レンダラ（励起固定実現キャッシュ、微分は包絡/d 経由）。HN3 と同一セグメント（female_real 209c94…/00038900 off200 seg128）。
- **ゲートは floor 較正**（同セグメントの E0 オラクルレンダ=完全予測床）: mel_l1 0.834 / mrs 3.63 / contrast 0.795。**設計時の推測値 "mel-L1<0.1" は noise 実現を無視した非現実値と判明**（床基準 ×1.15 に置換）。
- **結果**: env-only 800st = floor 一致（0.839/3.60）＝ net は包絡を表現・到達可能。**anneal（§3.5 設計レジーム）3000st = 0.800/3.555/amp 1.0/contrast 0.791 で PASS、教師 floor を超えた**（本命 loss が教師を超える設計命題の1サンプル実証）。HN3 の死（mel 0.26 で stuck・振幅半分・ガビ）と対照的に勾配は健全。
- spec-scratch は 3000st で 1.19/4.78 未達 → **10k st で 0.823/3.556/0.808 PASS**＝P-C（plateau→breakthrough、早期停止禁止）の再実証。教師なしでも本命 loss 単独で床到達（warm-up は収束加速の役割）。
- **低レベル補償 loss**（--lc、mel の逆ラウドネス重み）を既定 ON で実装（§3.5 の一級市民化）。

### E2 トリアージ（2026-07-13, 耳駆動・司令準拠）: 物理無罪・予測ギャップ確定 + ハーネス無効化

- **耳タグ（Step0, `results/e2_triage/`, 全アーム同一発話・同一seed・同一gain）**: gt=良 / **oracle=良 / oraclehop=良（物理・Phase A 天井は無罪）** / **net=聴けたもんじゃない** / nethop=聴けるが metallic+buzzy+muffled+rough+robotic+gritty。
- **帰属計測**: net の H_v フレーム間ジャンプ = oracle の 2.2〜2.5倍（|ΔH_v| 2.30 vs 1.02 nats/frame）。H_n 同等・d は2倍（絶対値小）。nethop（=時間補間で平滑）が聴ける事実と整合 → **本命仮説: net の H_v 軌跡ジッタ × 86fps フィルタ切替 = スプラッタ**。チャネル置換 AB（subhv/subhn/subd）で耳判定待ち。
- **コードレビュー12件（ユーザー、全て正当）**: 妥当性バグ #1 mel center=True=非因果入力（**E2 の従来 run は causal 検証として無効**）/#2 GT-mel条件はVC進捗の証拠でない/#3 評価がグローバルRNG破壊/#7 gain補正が振幅失敗を隠蔽/#10 resume非再現/#11 XPU無視/#12 best未保存 → **全て修正済**（因果mel: 左pad+center=False・フレーム厳密整列、評価RNG state 退避/復元、amp_ratio_raw+clip_rate 記録、RNG/total_steps を ckpt 保存、xpu>cuda>cpu、best.pt）。学習力学系 #4 noise統計loss /#5 a,v未接続 /#6 MRSTFT大音量支配 /#8 quiet層化サンプラ /#9 評価分割層化 → **次 run の単独アーム**として登録（同時変更禁止）。
- 旧 100k run は非因果条件のため停止・無効化。再学習は耳の置換AB結果を反映してから。
- **トリアージ第2–3ラウンド（耳+計測）**: 単独置換 subhv/subhn/subd **全てゴミ** ＝ 複数チャネルが独立に破壊。局在化計測: **H_v は無声フレームで +11〜12 nats（有声は±0.25で正確）＝無声中の倍音コム鳴りっぱなし**、H_n は per-bin p95 3–6 nats、d 有声誤差 0.28。**根本診断 = loss の計器不一致**: mel(128帯域・帯域平均) はコム/ノイズ・per-bin 包絡誤差に盲目。λ_env→0 anneal で唯一の per-bin 敏感項が消え、loss の盲点方向へ H がドリフト（E1 の1サンプル過学習では露呈せず、14分汎化で顕在化。「教師超え」は数字上のみ＝proxy 教訓の再演）。
- **対照実験起動（修正済ハーネス=因果mel、他条件同一）**: run A = λ_env→0（従来） vs run B = **λ_env floor 0.15**（耳承認済みオラクル教師への per-bin アンカー維持）。30k×2 本、`results/e2_ltv_eval_{a,b}.json`。
- **耳ラウンド4（only* = net チャネルを1つだけ残す）**: 良さ序列 **onlyd > onlyhn ≫ onlyhv** ＝ 破壊度 H_v ≫ H_n > d。局在化計測と完全整合。B が不十分な場合の次アーム = **v ゲート接続**（soft voicing ヘッドをレンダラへ、無声の倍音枝を明示 kill、レビュー#5）と確定。
- **耳ラウンド5（A vs B、差分= λ_env floor のみ）**: **netb(floor 0.15) > neta(anneal→0)。完成度 net 10% → netb 40%（ユーザー評価）** ＝ 教師アンカー仮説を耳で確認。客観は A/B ほぼ同値（この故障モードに盲目、既知）。新計装: **生振幅 = GT の 0.51–0.59**（gain 補正が隠していた HN3 同型症状、独立欠陥として登録）。
- **次: 用量反応ブラケット**: run C = floor 0.5 / run E = 純教師回帰（spec loss なし・λ_env=1 固定 = アンカー最大端点）。
- **耳ラウンド6（B vs E）: E（純教師回帰）= 55% > B(40%) > A(10%)** ＝ 用量反応が単調。**Phase A 確定: spec loss は使わない（有害）**。客観も E 全勝（lsd 0.864 / contrast 0.96 / 生振幅 0.76 vs B 0.51）。機構: mel/mrstft は per-bin 盲目のまま勾配を注ぎ、振幅低下と盲点ドリフトを引き起こす。**設計改訂（vocoder.md §3.5 v1.6）: Phase A = 包絡教師回帰のみ、texture/レベル最終調整は Phase B GAN 専任**。
- 次アーム（単独・優先順）: ①E レシピ 100k step（学習量の用量反応、env-only は高速）②v ゲート接続 ③容量。残ギャップ = nete 55% vs oracle（教師天井、耳良）＝純粋に回帰精度。
- **R: 学習量 300k → 50–75k で飽和**（best=75k、lsd 0.864→0.89 微悪化）。残差の局在化: **病理ゼロ**（無声リーク解消・ジッタ oracle 並み 0.98/1.02・全域一様 ±0.5 nats）＝純粋な回帰ボケ。仮説メモ: mel128 条件は HF で倍音間隔より粗く per-倍音振幅を運べない＝±0.5 nats は cond の情報床の可能性（cond 拡張アームは耳の分岐待ちで保留）。
- **耳ラウンド7: rege(75k)≈nete(30k)（プラトーの耳確認）、残ギャップの正体 = こもり・周波数劣化（回帰ボケ系、異音なし）** → 設計の予言どおり = regression-to-mean。**分岐確定: Phase B GAN へ**（cond 拡張は保留）。
- **Phase B 起動（`training/e2_phaseb_ltv.py`）**: e300 best から warm-start、loss = 0.3·env アンカー + LSGAN(MPD+MSD+MRD, train_m2 流用) + 2·FM、**spec loss なし（v1.6）**、seg32/batch8、30k。単一仮説「GAN が回帰ボケを鮮明化する」。
- **gan1（spec なし・env 0.3）= 発散（負の結果）**: lsd 0.87→1.82、生振幅 0.72→0.35。音声ドメインの絶対アンカー不在では D が生成を GT から引き剥がす。**v1.6 の「spec loss 有害」は D 不在の Phase A に固有**（文献の標準 = HiFiGAN/NHV は GAN 時に mel λ45 併走、と整合）。
- **gan2 起動（単一変更: +mel_lc λ45）**: Phase B の修正仮説 = 「D がテクスチャを担う時、mel は絶対アンカーとして機能する」。20k step。
- **耳ラウンド8（rege vs netgan=gan2 20k）: 57%（+2）、新規異音なし** ＝ GAN 方向は安全だが 20k では跳ねず。次分岐 = cond 情報床アーム（保留分）を先に検証。
- **espec アーム（cond: mel128 → 因果 log|STFT| 1025bin、env-only 75k）**: best@25k。客観混在 — lsd 0.893 ≈ 不変（**情報床は劇的には崩れず**）だが **生振幅 0.725→0.858・contrast 0.964・mod8 0.568** に改善。
- **耳ラウンド9: rege ≈ regspec ≈ oracle（「両方 oracle と同様」）** ＝ **net は教師天井に到達（E2 recon の実質達成）**。品質不満の対象は A/S 天井そのものへ移動 → 残りは Phase B スケールの担当（設計どおり）。
- **ganlong（200k, mel_lc アンカー, espec warm）: 10k で発散基準抵触・停止**（lsd 1.06 / amp_raw **0.425**<0.45 / lsharp 0.167）。**機構: lc 重みは大音量フレームの勾配を弱め、振幅情報の在処を D の圧力に明け渡す**（G が出力を静音化して敵対勾配から逃げる古典崩壊）。→ **Phase B の mel アンカーは lc=False が正**（低レベル原則はサンプラ層化で担保、loss 重みでの実装は GAN と非互換）。
- **ganlc0 起動（単一変更: mel アンカー lc=False、100k）**: 10k eval を発散基準で自動監視中。

### E2 単一話者 recon Phase A（2026-07-13, `training/e2_recon_ltv.py`）: 客観収束・**耳 FAIL**（上記トリアージへ）

- **構成**: 話者 af1ad5575a3fa383（実女声、train 14.3分 / held-out 3発話）。oracle キャッシュ（hpv-paw/d/f0, `data/e2_ltv_cache/`）→ LtvFrameNet 7.1M、anneal（λ_env→0@6k）+ mel_lc + MRSTFT、20k steps（47分、GAN なし=P-D）。ckpt=`training/checkpoints/e2_ltv/last.pt`。
- **held-out 軌跡（`results/e2_ltv_eval.json`、単調収束）**: mod8 1.88→**0.577** / mod2 1.26→**0.582** / traj 1.27→0.359 / lsd1-4k 1.57→0.876 / lsharp 0.368→0.149 / contrast 0.82→**0.997**。**texture 系で E0 オラクル水準（pawB 0.825/0.683）を held-out で超えた**＝E1 の「本命 loss は教師を超える」の汎化再現。
- 残gate: (1) **耳**（`results/e2_listen/` gt/early/net、WebUI 配信中）、(2) **nsf3_gan best との同一データ AB**（nsf3 は content 特徴が必要＝この話者の特徴抽出が前提、pending）、(3) 枝B 並走（pending）。

### E0 オラクル実測（v1.3 再取得 2026-07-12, `results/e0_oracle_ltv.json`, 客観 PASS ／ 耳 FAIL）

- **客観ゲート**: 全5発話 per-utt PASS を JSON に永続化（決定論シード・causal レンダラ）。ref 0.977 vs WORLD 0.980。per-utt: small_voice 1.118 / emotional 1.050 / sibilant 0.803(W=0.836, マージン0.009=最薄) / breathy 0.897(W=0.902) / male 1.019(W=1.037)。
- **耳ゲート第1回 = FAIL**: 本命 `ltv`（K1024/Nb1025/TE/min-phase）に**金属ノイズ（bad_tag: metallic）**。客観 contrast は WORLD 同等でも耳が落とした＝proxy と耳の乖離の再演。
- **耳ゲート第2回（2026-07-12）: metallic ほぼ解消**。第1回と第2回のレンダ差分は (a) E0 ハーネスの gain 整合を per-frame 強制（11.6ms 粒度＝86Hz ゲイン変調サイドバンド）→ 平滑 RMS 比 + peak ガードへ変更、(b) causal 制御補間、(c) 決定論シード。**主犯は (a) の測定ハーネス artifact が濃厚**（world が gain 整合でも無傷だったのは g≈一定のため、と整合）。教訓: 診断アームのクリップ（+10dB 過大）も同根で、帰属 AB を1ラウンド汚した。**レンダラ自体の metallic 容疑はほぼ晴れたが、確定は「ltv vs world 同等か」の耳判定待ち**。第2回で「GT 比の音質劣化」が新たに報告 → E0 の合格線は GT 透明性でなく WORLD 同等（oracle A/S の物理天井）のため、ltv vs world の直接 AB が判定点。
- **fork の実測**: K_v knee=1024（256→0.953 / 512→0.972 / 1024→0.977 / 2048→0.978）。Nb は 1025 で客観飽和（257 でも 0.970＝ゲート内。**可聴下限は E3 耳 ablation 待ち**）。TE 0.977 > CheapTrick 0.922。linear-phase 0.999。±変調は客観不変（F2 は耳でのみ判定可）。
- **★フィルタ更新レートは実レバーだった（2026-07-12 訂正）**: 旧「hop 512/256/128 は無効」は**配線バグによる偽アブレーション**（sub 未伝達＝全アーム hop512、レビュー#4）。修正後の実測: base 0.977 → **hop256 0.993 → hop128 1.016**。per-utt では emotional 1.05→1.23 / breathy 0.90→0.98 / sibilant 0.80→0.85 と動的・ノイズ系で一貫改善（small_voice のみ 1.12→0.98 と後退）。**86fps フィルタ切替の splatter は実在＝metallic の帰属候補の一つ**。レイテンシ不変・計算 ×2/×4 の設計レバーとして E3 へ登録。
- **★E0 が捕まえた設計バグ（v1.2、`ltv_render.py`）**:
  1. **±8 nats tanh clamp がフォルマントを圧縮**。実音声の包絡偏差は male p95=7.6 nats / breathy p50=4.9 で、`8·tanh(x/8)` は 6.4→5.31（−9.5dB）と公称の遥か手前で飽和。症状＝bw +50Hz・帯域 median +9dB・contrast 0.83 頭打ち。修正＝区分 clamp（±8 恒等・超過のみ tanh 尾、上限±12）で male 0.834→1.014。
  2. **per-frame 実測 RMS 正規化は frame 格子 AM を再導入**（HN2 で殺した 86Hz 系の回帰）。修正＝閉形式 √(Σg_k²/2)。
- **診断の negative results（v1.3 で有効性を再確認済のもののみ）**: noise 枝の谷埋め単独犯説・TE 上側包絡バイアス単独犯説・包絡時間ジッタ（tsmooth3/5 ≒不変）・gain-match 変調は反証。WORLD 励起注入（wexc、長さバグ修正後の初の有効実行）は 0.93＝励起置換では WORLD に届かない。
- **コードレビュー（2026-07-12, 49 agents+精読, 15件生存）→ v1.3 修正**: 致命=①ltv_ola が depthwise conv（XPU backward 死亡パターン）→ mm バックエンド（GEMM 円状畳み込み）追加、②制御経路の半フレーム未来参照+hold_f0 backfill → causal モード（[t−1,t] アンカー、応答遅れ 5.8ms・先読みゼロ）、③nb_in 既定 2049 の暗黙ゼロパディング（11kHz 以上素通り）→ 既定 1025+assert。証跡系=④hop偽アブレーション ⑤wexc 全クラッシュ ⑥非決定論シード ⑦mean-only gate ⑧diag が PASS 証跡を上書き — 全修正済。
- **共有レンダラ検証（v1.3）**: パリティ min-phase 3.8e-7 / linear-phase 2.3e-7 / ltv_ola conv≡mm 7.6e-7（全て E4 gate 1e-4 の内側）。因果性ビット一致は **f0/H_v/H_n/d/a の全経路**で assert（旧主張「因果性ビット一致」は包絡経路のみだった＝過大表示を訂正）。勾配は H_v/H_n/d/a/φ0 全学習入力に流通。静的包絡再現 ±0.4dB。
- **whisper 明示カテゴリは golden 未収載＝coverage gap 継続**。

### 耳ゲート第3回（2026-07-12）: ケースB確定 → 解析的 discriminator を発見・登録

- **耳判定**: metallic 解消後も「`ltv` は `world` より劣化」（ケースB）。contrast_ratio は盲目（0.977 vs 0.980）→ ユーザー要件「解析的に見分けられないといけない」。
- **discriminator hunt（`training/e0_discriminator_hunt.py`, `results/e0_discriminator_hunt.json`）**: 13 候補指標を gt/world/ltv ×5発話に一括適用し、耳と同順で割れる指標を系統探索。
  - **勝者 = `mod_8_16k_dist`（5/5 発話で world < ltv、0.884 vs 1.115）**: 8–16kHz 帯域包絡（689Hz サンプリング）の変調スペクトル log 距離 vs GT。**劣化の正体＝HF noise の時間テクスチャ**（我々の noise 枝は「フレーム定常の白色 noise × 固定フィルタ」、GT/WORLD はピッチ同期・過渡変調）。
  - 次点: `mod_2_8k_dist`（4/5, 0.661 vs 0.862）、`trajdist_b4_8k`（帯域エネルギー軌跡 L1、4/5, 0.288 vs 0.402）。
  - **crest_ratio は 0/5（ltv が GT に近い）＝ min-phase 位相/バズ系は無罪方向**。HF contrast（5–10k/10–16k peak−median）はほぼ盲目。
  - 指標は `e0_oracle_ltv.py` の measure() に配線済（以後の E0 run で自動記録）。
- **レバーの実測（同指標で採点）**:
  - **±ピッチ同期変調 d=0.65 は無効**（mod8_16k 1.115→1.132）。**設計の q(φ)=½(1+cos) は変調形として弱すぎる**（WORLD は周期毎の窓掛け＝はるかに鋭い時間集中）。F2 の原理でなく「形」の問題。
  - **hop128（フィルタ更新 ×4）は world 超え**（mod8_16k 0.688 < world 0.884）。contrast でも +0.04 だったのと整合。
- ~~次アーム~~ → **改善ループ実施・収束（下記、2026-07-12）**。

### 改善ループ（2026-07-12, 7ラウンド, `--modhunt`）: 包絡教師の交代で不合格を脱出

- **ループ様式**: 指標=mod_8_16k_dist（+mod_2_8k/traj4_8k/contrast の同時監視）、目標=world 水準、1ラウンド=1仮説、包絡キャッシュ（`results/e0_cache/`）で1回数分。証跡=`results/e0_modhunt.json`（上書き逐次）+ 最終は正典 `results/e0_oracle_ltv.json`。
- **勝者: 包絡教師 = CheapTrick(q1=−0.30)**（`ctq30_k1024_nb1025`: mod8 0.778 / mod2 0.582 / traj 0.425 / contrast 0.975、**contrast per-utt 5/5 PASS**）。さらに **+オラクル d_t + q⁴ 変調**（`_od_qp4`: mod8 **0.708** / mod2 **0.553** / contrast 0.969）で texture 両指標が world（0.82/0.635）を明確に超えた。
- **確定した機構理解**:
  1. **TE（上側包絡）は noise テクスチャを構造的に汚す**。HF は noise 主体で、上側包絡が noise ピークに乗る＝H_n が過大・誤形状。F5 の「TE 第一候補」は**撤回**: contrast は CT(q1=−0.3) で同等に取れる（q1=平滑補償を−0.15→−0.30 に上げるだけ。−0.45 以上は過鋭で traj/contrast>1 に破綻）。
  2. **F2 変調は「正しい包絡の上でのみ」効く**: TE 上では q⁴+オラクル d が無効（R1: 1.068→1.044）、CT-q30 上では 0.778→0.708。変調形は q⁴（`ltv_render.py` の mod_p、E[q^p]=C(2p,p)/4^p 正規化）+ d_t=D4C HF ap（オラクル）。cos¹形（v1.1 設計）は弱すぎで死亡確定。
  3. **オラクル a_t（GT HF 包絡の 2.9ms gain）は逆効果**（倍音パルス混入で noise 枝に誤テクスチャ）— 設計の a_t は「GT からの直接抽出」でなく学習予測 or 無声限定で再設計要。
  4. **計測の罠（3度目）**: 帯域別包絡のレベルを混ぜる系（hybrid/blend/apw）は per-frame 全体 gain と結合して trajdist/contrast が暴れる＝見かけの改善/悪化を作る。**包絡は単一推定器で一貫させる**のが正解。
- **negative results**: hybrid_l（スカラーレベル整合）、blend（周波数ランプ）、apw（ap 加重）、TE+hop 分割、TE+q^p 全滅。hop 分割は CT 系では未再検証（ctq30 単体で目標達成のため保留、E3 候補）。
- **耳ゲート第4回（2026-07-12）: winner も FAIL**（bad_tag 未取得）。→ ループ第2巡へ。

### 改善ループ第2巡（2026-07-12, R8–R9）: world vs winner の判別指標と誤差の局在化

- **拡張ハント（world vs winner, 20指標）**: **帯域別 per-frame LSD（lsd_b1_4k/b4_8k/b8_16k）と trajdist_b1_4k〜b8_16k が 5/5** で world < winner。mod 系は winner が勝っており、耳の残差は「テクスチャ統計」でなく「フレーム毎スペクトル精度」側。
- **誤差の局在化（bin×時間×倍音/谷 分解）**: winner の LSD 超過は**有声フレームの倍音ビンに集中**（breathy: world 0.63 vs winner 1.00 nats、emotional: 0.48 vs 0.85。谷ビンは同等）。
- **反証済み（R8–R9）**: hop 分割（h256/h128）は LSD 不変・mod 悪化（ctq30 系でも死亡＝フレーム追従説の再反証）。ソフト uv split（a0=0.5/0.7、倍音枝の √(1−ap²) 減衰を有声で解除）も LSD 不変＝**倍音ビン誤差（≈8.6dB）は split 減衰（≈3.6dB）では説明不能**。gain 平滑も無罪（厳密 per-frame gain の方が traj/LSD 悪化を実測）。
- **未解決仮説**: (i) GT 倍音線の自然ジッタ/位相揺らぎによる線形状差（=指標アーティファクトの可能性）、(ii) WORLD のピッチ同期 noise が倍音ビン上に再集中して線振幅を回復する機構の不足（d=1.0/q⁸ の強変調 `mh_dfull_qp8` は mod 最良 0.685/0.529 だが LSD 不変・contrast 0.946 に低下）。
- **判断: 指標単独の追跡はここで停止**（proxy 追いの past failure と同型のリスク）。HITL 規律に従い **bad_tag を取得してから修正仮説に変換する**（README「bad_tag → fix」表）。試聴 = gt/world/ltv/ctq30/winner/dfull。

### アンカー問題（2026-07-12, ユーザー指摘）: WORLD は GT 同等でない → 評価アンカーの再構築

- **指摘**: world 自体が GT より劣化しており、world 相対の評価では合否判定が成立しない。
- **R10: 直接スペクトル包絡オラクル `spec`**（H = GT の log|STFT| 2049bin そのまま、特徴量損失ゼロの振幅上限）を実装・実測:
  - **完全振幅オラクルでも bin-LSD は 1.072 ＝ winner と同じ床**（lsd1_4k。contrast 1.049=倍音コム二重化でやや過鋭）。包絡の質では床が動かない。
  - **noise 実現フロア較正（初実施）**: 同一アーム別シード間 LSD = 0.26–0.58 nats。→ **bin-LSD の約半分はフロア汚染**＝知覚透明性の判定器として弱い。フロア除去後の構造残差 ≈0.4 nats は実在（位相×窓の非定常結合 / GT ジッタ由来の線形状差が候補）。
  - **教訓: lsd_b* を gate に使う場合はフロア併記必須**。world(0.985) vs 我々(1.07) の差はフロア比で小さい。
- **アンカー梯子（評価の再定義案）**: gt ＞ spec（振幅完全・min-phase・レンダラ物理）＞ world（WORLD特徴量）＞ 学習到達点。**耳で spec ≈ gt なら**: spec を E0 アンカーに昇格（world 廃止）、E0 合格 = 「spec との耳同等」。**spec < gt なら**: min-phase+現励起の枠組み自体が可聴限界＝設計判断（mixed-phase 励起 hygiene / ジッタ注入 / AA-refiner / 枝B）。
- **耳判定（2026-07-12）: spec は gt より明確に劣化 — ロボ・ざらつき・こもり**。→ **振幅情報が完璧でも現枠組みは transparent でない**ことが確定（§4 登記リスク「min-phase 音色コスト」「robotic 前例」の実現）。E0 の合格基準を world 相対にしていた前提が二重に崩れた（world も spec も GT 未満）。

### 理論確定（2026-07-12, ユーザー指示「理論をまず固めろ」への回答。R12–15 で検証済）

- **T1: spec アーム（H=生 log|STFT|）は二重コムで不正**。コム振幅の min-phase IR は周期 T0 のパルス列になる。**P1 実測: spec の IR はエネルギー 99.9% が第1周期以降（自己相関@T0=0.896、第2周期+3.6dB）、ctq30 は compact（−21.6dB、0.006）**。ロボ=コムリンギング（23ms×5エコー）、ざらつき=フィルタ側コム（真の周期）と励起側コム（harvest F0）の δf ビート（k=20, δf=1Hz→20Hz AM＝ざらつき知覚帯）、こもり=コム IR+46ms 窓スミア。**spec の耳 FAIL は枠組みの天井ではなく実験系のバグ**。R11 の励起 hygiene 検証も壊れた土台上＝無効実験。
- **T2: 正しい分解 = (倍音ピーク上側包絡 → H_v, 倍音間谷床 → H_n)**。励起=フラットコム＋白色である以上、線振幅が GT に一致する唯一の条件。TE/CT/hybrid の全成績を統一説明（TE=H_v に正・H_n に過大、CT=谷に近く**ピーク過小**←測定済み倍音ビン誤差の正体）。**実装 `hpv_envelopes()`**（ピーク=±3bin max→周波数補間、床=中点±2bin p25→補間、無声=平滑スペクトル）。線/床の窓ゲイン差は較正定数 C_h,C_n + 解析的 f0 依存項 0.5·log(f0/200) で吸収。**予測どおり lsd1_4k で全アーム初の world 超え（0.985→0.92）**。
- **T3: ロボ残差=ゼロ位相コヒーレンス**（sin(kθ) 全同相=最大パルス性、パラメトリック合成の古典バズ）。古典解=群遅延分散+自然ジッタ/シマー（STRAIGHT 系譜）。実装済（jitter/shimmer/disp、`ltv_render.py`）。客観はほぼ盲目＝官能で判定する項目。
- **T4: こもり=特徴量の時間分解能**（46ms 窓は包絡を ~21Hz LPF。CheapTrick=3T₀≈10–15ms）。E1 特徴窓設計への制約。
- **R13–15 の追加知見**: 床の時間平滑は逆効果（床変動は本物の息ダイナミクス）。hn レベル −0.6 nats が contrast/lsd 同時最良。推定器混合(mix)は lsd を戻す（H_v/H_n は同一 |STFT| から取るのが正しい）。
- **パレート前線（客観、world 比）**: texA=ctq30+od+q⁴（mod8 0.71✓ / lsd 1.08✗）、lineB=hpv−0.6+od+q⁴（lsd 0.92✓ / mod8 0.97✗）、finalC=lineB+jitter/shimmer/分散（lsd 0.93✓ / mod8 0.89 / contrast 0.943✓）。**客観の全軸同時 world 超えは未達＝どの軸が知覚支配的かは官能評価の管轄**。

### なぜ GT に勝てないか — 最終診断（2026-07-12, 実測+文献裏取り済み）

**主因: 我々の信号モデル「全帯域コヒーレント倍音 + 定常ノイズ」は、自然音声の「周波数とともに確率化する倍音」を表現できない。**

- **実測（倍音線の鋭さ peak−valley, nats, 次数バケット別）**: male GT = k1-5: 2.17 / k6-15: 0.83 / **k16-30: 0.09 / k31-60: 0.03 ＝ 約1.6kHz以上でほぼ完全にノイズ**。breathy GT も同様の減衰。ジッタ σ の FM 理論で線幅 ∝ k·f0·σ となり高次で必然的にノイズへ遷移する。我々は 19.8kHz まで完全コヒーレントな線を張る（=ロボ/金属の一次源）。WORLD も k16-30 で 0.45 と過鋭（world も GT に負ける理由の一つ）。
- **文献アンカー（全て確認済）**:
  1. **HNM の maximum voiced frequency (MVF)**（Stylianou）: 時変 Fm を境に下=倍音・上=有色ノイズ。**NSF harmonic-plus-noise に trainable MVF を入れた前例が arXiv:1908.10256**（Wang & Yamagishi）＝我々のアーキテクチャ族に欠けている部品を名指しで追加した論文。
  2. **DSM（Drugman, arXiv:2001.00842 / 2001.01000）**: 残差=Fm 以下決定論+以上確率、確率成分は**エネルギー包絡で時間変調**（=設計の a_t の正当化。抽出は >Fm 帯から行うべき＝R1 の oracle a_t 失敗の正しい説明: 倍音帯から取ったのが誤り）。
  3. **STRAIGHT の群遅延操作**（Kawahara 2001, mixed mode excitation + group delay manipulation）: buzz 除去の正典手法。R16 の hfrand（3kHz 以上ランダム位相）は独立に同じ答えに到達していた。
  4. **mixed-phase 音声モデル**（Drugman & Bozkurt, CCD/ZZT, arXiv:1912.12843 系）: **声門開放相は最大位相**＝ min-phase 枠組みでは原理的に表現不能。二次因（pressed/buzzy の残差）。設計 §3.3 の「min-phase 音色コスト」の文献的実体。
- **副因**: 周期クローン（フレーム内の周期が完全同一。ジッタはフレーム粒度でなく周期粒度が要る）、T4 特徴窓。
- **帰結（v1.5 設計差分として提案、E0 検証待ち）**:
  (a) **時変 MVF の導入**: 倍音は Fm(t) 以下のみ、以上は noise 枝へ（oracle は線鋭さ実測から推定可、製品は 1908.10256 同様に学習可能パラメタ）。
  (b) a_t は >Fm 帯から抽出（DSM 準拠）。
  (c) Fm 以上の位相は STRAIGHT 式ランダム群遅延（hfrand 実装済）。
  (d) mixed-phase 化 or Phase B GAN による位相修復は fork（min-phase 純度 vs 音色の設計判断、v2）。
- **正直な期待値**: (a)-(c) は「A/S として world 超え・アーティファクト無し」までの部品。**GT 完全同等は振幅系 A/S 単体では文献的にも未達領域**で、last-mile は設計上 Phase B GAN の担当（P3）。E0 の採用ゲートは vocoder.md §6 の定義どおり「こもり/ガビ/robotic 無し」＝アーティファクト自由性であり、GT 不可識別ではない。

### R17–19（2026-07-12）: MVF 実装 → 負の結果 → 段階的コヒーレンス+T4 で全指標 WORLD 同等以上に到達

- **計測系追記（ユーザー指示のゲート再定義2点、実施済）**: ①E0 耳ゲート項目に metallic 明示（vocoder.md §6）。②`lsharp_dev`（次数バケット線鋭さ、**GT 係留・両方向**）を gate v3 として登録・measure() 配線。contrast_ratio の片方向性と WORLD 係留の構造的盲点（WORLD 自体 HF 過鋭 0.314）を文書化。
- **R17: 二値 MVF（HNM 式ハードゲート）は負の結果**。traj 0.41→0.9+/lsd 0.92→1.1+ に退行、lsharp_dev も 0.118→0.134（新指標が「切りすぎ=under-sharp」を正しく検出＝GT の HF は部分コヒーレント 0.05–0.2 nats）。DSM 式 noise 補填でも回復せず。**arXiv:1908.10256 の trainable MVF は学習フィルタ+GAN 込みで成立する構成であり、純 A/S では HF エネルギー穴が開く**（周波数 raised-cosine 500Hz・時間中央値平滑・hop512 固定・a_t/d_t 分離アームの注意点は全て遵守した上での結果）。
- **R18: 段階的コヒーレンス崩壊（OU 周期ジッタ, τ=5ms, サンプルレート実装）が正**。線幅 ∝ k·f0·σ の滑らかな遷移＝エネルギー穴なし。σ=0.3% が最良（mod2 0.814→0.690、lsharp 維持）。σ≥1% は lsharp 悪化（過広帯化）。フレーム粒度ジッタは廃止。
- **R19: T4（ピッチ適応窓 3T₀）が最後の大物**。hpv 包絡を win∈{2048,1024,512} のピッチ適応選択+窓ゲイン較正で抽出 → **`hpvpaw_k1024`: 全指標 WORLD 同等以上**（mod8 0.772 / traj 0.330 / lsd 0.887/0.866 / lsharp 0.126 / contrast 0.969 per-utt 5/5 PASS）。+j03+od·q⁴ 変種は mod2 0.683。
- **v1.5 レシピ（正典 run 登録済 `hpvpaw_k1024*`）**: hpv 包絡（ピーク/床、ピッチ適応窓）+ n_off−0.6 + OU ジッタ 0.3% + od·q⁴ 変調（変種）。
- **官能評価（2026-07-13）: pawB 好評**（アーティファクト系の明示 NG 報告なし）。同時に**息継ぎの劣化が world にも存在**との指摘 → 実測で3点確定:
  1. **息区間は現行計測から除外されていた**: active mask（rms>5%×p99）が「鳴っているのに静かな」フレームを 7〜30% 除外＝息そのもの。息区間マスク指標（無声×低レベル、2-12k LSD/mod）の gate 追加が必要。
  2. 息区間 LSD 実測: world 0.75–0.92 / **pawB 0.72–0.74（pawB が優位）** ＝ 劣化は特徴量ドメイン共通の天井（フレーム振幅包絡+白色 noise では乱流の微細構造を描けない）。E0 の合否には影響しない。
  3. 対策の担当区分: A/S 内=**息区間限定 oracle a_t**（倍音混入なし＝R1 失敗の原因を回避、DSM 処方）／last-mile=Phase B GAN（P3）／データ=golden の whisper/breath カテゴリ追加（coverage gap 継続）。

### 改善ループ第3巡（R11）: 励起 hygiene — ロボ/ざらつき/こもりへの直接打ち手（T1 により無効実験と判明、記録のみ）

- 実装（`ltv_render.py` HarmonicSource）: **jitter**（周期毎 F0 揺らぎ、frame 毎 gaussian σ=1%）、**shimmer**（振幅揺らぎ σ=6%）、**位相分散 disp**（rand=倍音ランダム固定位相 / quad=−c·k² 2次群遅延）、**短解析窓**（spec の H を win 1024=23ms で抽出）。
- 客観（`results/e0_modhunt.json`）: jitter+shimmer が mod8 0.848→0.740。disp/quad/短窓は客観ほぼ不感 — **ロボ/ざらつきは現行客観バッテリーの盲点**（crest/roughness も鈍い）。判定は耳のみ。
- 耳AB待ち: gt vs spec / specjs(ジッタ+シマー) / specdisp(ランダム位相) / specquad(2次分散) / specjsq(複合) / specw(短窓) / specjsqw(全部)。
  - ロボが js/disp/quad で消える → 励起のゼロ位相コヒーレンス+無揺らぎが犯人 ＝ 設計 v1.5（励起衛生に jitter/shimmer/分散を正式追加、学習不要）。
  - こもりが specw で消える → 特徴窓 46ms の時間スミア ＝ E1 の特徴抽出窓設計に直結。
  - どれも消えない → min-phase 位相そのものが犯人 ＝ mixed-phase 化 or AA-refiner or 枝B の設計判断。

### NSF-HN3 死亡診断（2026-07-11, 確定）
- **固定1サンプル overfit**（`overfit_one.py`, GAN/DataLoader/乱数を排除, warm mc_mel_gan, sample=female_real_feat/209c94d37412922a/00038900, off200 seg128）:
  - mel-L1 **0.26** / mrs **2.44** 床打ち、出力振幅 GT の**半分で stuck**、耳=**「こもりでなくガビ（デジタル歪み）」**。
  - → step 不足でも容量でもなく、**signal path の表現限界**。健全なら1サンプルは即 fit。
- **鋭さ contrast**（1-5kHz log-spec peak−median、GT 比。WORLD 再合成=**1.02** が上限）:

  | arm | contrast ratio |
  |---|---|
  | HN3 warm(GELU) | 0.823 |
  | HN3 no-norm(ChanLayerNorm除去) | 0.813 |
  | HN3 AA-GELU | 0.814 |
  | HN3 nonoise(base_noise=0) | 0.783 |
  | AFHN(周波数軸 scratch, 不安定) | peak 0.70→0.57 collapse |

- **反証実験の結論（1実験1仮説, 同一 overfit gate）**:
  - Exp1 no-norm: 不変 → ChanLayerNorm 瞬時 AGC は主因でない。
  - Exp3 AA-GELU: 不変 → GELU aliasing は主因でない。
  - Exp4級 noise 有無: 同ガビ → base_noise 床は主因でない。
  - **核心（コード確認 `nsf_hn3.py:146-147,205-209`）**: 条件が信号に触れるのは FiLM ch毎スカラー gain/bias + control-net 加算 + 初段 concat のみ。時間混合 conv は全て条件非依存の固定重み ＝「固定基底×スカラー混合」＝鋭い共鳴を張れない。**情報量でなく関数形の問題**（FiLM+ctrl ~1900 スカラー/frame > mel 128）。「ch 増・深く」は無効。
- 詳細: [[nsf3-muffled-fork]]、`results/overfit_one_*`。

### レビューで閉じた判断（2026-07-11）と残る fork
- **NSF-LTV v1.1 = 本命確定**（`current/vocoder.md` §3）。系譜 ≒ NHV (Interspeech 2020) の min-phase・causal・44.1k 版＝実証済み物理。「causal×44.1k×<50ms×CPU」は文献・OSS 未占有（公表実測最速 RT-VC 61.4ms/16k）。
- **対抗案の棄却根拠**: 時変 all-pole/LPC → torchlpc が Numba/CUDA のみで **XPU 学習不可**、周波数サンプリング IIR 学習は r→1 で数学的破綻（狭帯域こそ学べない）。位相明示回帰系（Vocos/APNet2）→ APNet2 自身が「フレームシフトに敏感」と明記＝AFHN 経験と整合、pitch-shift 外挿でも不利。
- **枝B（保険、先に温める）**: B1=BigVGAN-v2 44k(MIT) teacher→causal 小型蒸留（レシピ arXiv:2408.11842）、B2=causal AFHN 再訪（MS-Wavehax 2506.03554 レシピ）。E0/E2 死亡時のみ pivot。
- **残る fork は実測で閉じる**: K_v(1024/2048)・Nb(1025/2049)・±ピッチ同期変調の可聴性 → E0。gate 閾値は E0 レンダの noise floor で較正。

## 計測系（gate）
- **E0 oracle gate（実装済・客観 PASS）**: `training/e0_oracle_ltv.py` + `training/ltv_render.py`。真包絡(TE 自前実装/CheapTrick)+D4C → NSF-LTV レンダラで学習ゼロ再合成。K×Nb×±変調×位相×教師 sweep + 診断アーム（--diag: harm/noise 単離・時間平滑・hop 分割・WORLD 励起注入）。結果=`results/e0_oracle_ltv.json`、試聴 wav=`results/e0_oracle_ltv/`。E1 の gate 閾値較正はこのレンダの noise floor を使う。
- **overfit gate**: `overfit_one.py`（`--arch hn3/afhn` `--aa` `--no-norm` `--base-noise` `--scratch`）。固定1サンプルを丸暗記 → contrast/耳で arch の素性を単離。
- 鋭さ contrast: 1-5kHz log-spec peak−median、active frames、GT 比（WORLD=1.02 上限）。**注意: mel-L1/mrs は鋭さを測れない**（WORLD でも mel-L1 0.77）。**proxy は耳と乖離した前科あり → 最終判定は耳**。
- 試聴: `listen_gui.py`（nsf3=生成 / gt=正解 の対）。既存: `kansei_proxies.py`/`gate0_*`/`annot_gui.py`/`h8_retrieval.py`。

## 規律（過去の失敗から）
- **1実験1仮説**。アーキ+loss 同時変更で判定しない（別セッションが8run で悪化させた前科）。
- **P-C 早期停止禁止**（plateau→breakthrough 型）。
- **判定指標をディスクに永続化、GT 参照を併記**。セッション内 proxy で pivot しない。
- 「HF share 低→HF 不足」の即断は誤診（実は低域過剰だった）。share は帯域細分してから薬を選ぶ。評価 GT が TTS でないか確認（何を正解として測っているか）。

## その他の進行中/保留
- base_noise 床（0.05, -6dB）下げの fine-tune: 設計済み・未完遂（ざらつき低減、製品ライン限定、比較には混ぜない）。
- 実データ主体 mix + 評価も実 GT に（TTS は timbre/style 条件用に限定）。
