# Y-S1条件付きlatent生成器

> status: PROPOSED（2026-09-08、実コードと旧S1-5記録の整理。2026-09-19 段階1接続監査を実施）
> 学習・レンダ実装は存在する。2026-09-19監査で学習データのf0破損と目的関数の平均回帰退化、レンダfps不整合を確認。製品採用ではない。
> codec契約は [causal_codec.md](causal_codec.md)、問いは [RESEARCH.md](RESEARCH.md)。

## 1. 役割

content / 明示prosody / target条件から、Y-S1の32次元latentを100fpsで生成する。source codec latentをcontentとして使わない。製品候補は1-stepで、出荷推論に外部SSL・話者モデルやPythonを残さない。

## 2. 現実装の地図

| 実装 | 役割 |
|---|---|
| [train_cfmys.py](../training/train_cfmys.py) | CFMYS、学習・評価、AR(1)noise |
| [s5_encode.py](../training/s5_encode.py) | codecでlatentキャッシュとABI統計を作成 |
| [s5_render.py](../training/s5_render.py) | 旧E資産・固定+17半音・固定targetによる診断レンダ |
| [train_vc_g.py](../training/train_vc_g.py) | CausalBlockの実装参照。旧G学習経路の再開を意味しない |

CFMYSの既定: latent 32、content 768 + log-F0 + log-energy = 770、speaker embedding 192、hidden 384、8 blocks、time embedding 64。

入力は条件とlatentをconcatし、左pad2のConv1d(k3)。各CausalBlockのdilationは1,1,2,2,4,4,8,8。各段でtimeとspeakerのscale/biasを加え、Linearで32次元を出力する。ctxプロパティは62 frame。継承するCausalBlock内部を含む因果性は試験で確認する。

重みの上位キーはinp、blocks、temb、tfilm、sfilm、out。Pythonのstate_dictが実装確認元であり、Rust生成器のexport/parityは未確認。codecのRust対応を生成器へ拡大解釈しない。

## 3. 条件・統計・noiseの契約

- Y-S1は48kHz、hop480、100fps。contentキャッシュはコード上50fps、f0/energyは44100/512 fpsとして扱われる。
- 条件添字は終端timestampから過去側へfloorする実装。各encoderの実際のtimestampと一致するか未確認。
- lf0 = log(max(f0,50)/200)、energy = log(max(energy,1e-4))。明示vuv入力は現在ない。
- ABIのmu/sdはキャッシュ由来の固定値。発話単位正規化を使わない。意図した全training split統計と、現コードのsample統計は一致していない。
- noiseはAR(1)、既定rho=0.9。chunkをまたぐstateと再現可能なRNGの製品実装は別途必要。
- 現evaluate/renderはt=1でvを一度評価し、z_hat = clamp(z0+v,-8,8)、逆正規化してdecodeする。これは現在の実装であり、正しいsolver契約と認定したものではない。

## 4. 学習意図と現在の差

通常の直線補間CFMを意図する場合、z_t=(1−t)z0+t z1、教師速度z1−z0に対しv(z_t,t,conditions)を学習する、という契約になる。

現コードはrandom tを与えつつv(z0,t,conditions)を学習しており、補間したz_tを渡していない。**2026-09-19監査でこの書き方の帰結を確定**: z0はz1と独立なのでL2最適解は E[z1|c]−z0、1-step出力はz0に依存しないE[z1|c]＝条件付き平均への退化。[audit.json](../results/diag_cfm_audit/audit.json) 実測: seed 8種の送り出しlatentのばらつき0.010–0.012に対し残差0.256–0.346（ratio 0.029–0.045）、v(z0,0)=v(z0,1)（平均差0.003）。blurry latentとeval-L1 0.2316の床はこの退化で説明がつく。多様性が必要なら補間z_tと複数step、決定論的1-stepで良ければCFM形式を捨てて条件回帰として設計し直す、どちらかを明示的に選ぶ。

既定学習値は40k steps、batch8、crop200 frame、lr3e-4、AdamW、OneCycleLR。これは既存コードの記述であり、再学習を指示するレシピではない。

## 5. 段階1監査の結果（2026-09-19）

証跡: `results/diag_cfm_audit/`（audit.json / gt_control.json / threeway.json / f0_cache_scan.json）、診断スクリプト `training/diag_cfm_audit.py`。

**学習データのf0破損（最重要）**: `female_real_feat` の保存f0が約85%のファイルで元音声と無相関（輪郭相関中央値0.166、corr>0.9は15%）。同キャッシュのcontent（ContentVec再計算とcos=1.0000）とenergy（corr=1.0000）は健全。`female_tts_feat`（corr中央値1.000）・`male_feat`（0.995）は同規約で健全。ステム重複は0で、train_cfmysの全学習ペア（20,322）がこの破損コーパス由来＝**学習時のlf0条件は実際の音声とほぼ無関係だった**。F0不追従の一次原因はここ。修正はf0のみの再計算オーバーレイ `data/female_real_f0fix/`（`training/f0fix_real.py`、元キャッシュ不変）。

**レンダ経路のfps不整合**: s5_renderはE出力（実測172.27fps、R.HOP=256）を50fps扱い → content時間軸3.44倍伸長。SF.causal_f0出力（172.27fps）を86.13fps扱い → f0時間軸2倍伸長。T100=content数×2（上限6000）→ 元205.1sに対し60.0sの出力（contentは最初の17.4s分、f0は最初の30s分を引き伸ばし、energyのみ60s整合）。旧S1-5記録の「狙い292Hzに出力99.7Hz」はこの条件 streamで説明可能。修正は添字比率をE1実fps(172.27)・f0実fps(172.27)に直し、T100を音声長から直接算出すること。

**E1の代理content**: renderのcontentはdiag_e2（eval-cos 0.7394）で、E1自身の凍結ゲートcos≥0.90に未達。fps修正後も真ContentVecから0.74-cos乖離したstreamがCFMに入る。学習条件と出荷frontの差として残る構造的項。

**ABI統計の混入**: s5_encodeのmu/sdは全corpus（female_real 41,554 + male_tts 8,695）から一様サンプル → 約17%が男声latent。学習はfemale_real latentのみ使用。コードコメントの「層化抽出」は実装と不一致（一様random）。再実行時のabi.pt混入はtry/exceptで偶然除外されている。

**健全と確認したもの**: codecのstream逐次decodeと一括decodeのwaveform L1≈3e-5（パリティ良好）。GT latentのdecodeと元音声のF0測定は一致（測定規約: 44.1k resample + harvest + stonemask）。latentとfeatのbasename衝突は現データで0件。学習側cond_ofのcontent/f0/energy添字は時間軸1:1で正しい。

**測定の教訓**: pyworld harvestを48k・10ms・stonemaskなしで使うとデコード音声で約3倍のF0誤読が出る。audio診断はencode_featと同一規約（44.1k・HOP512・stonemask）で測ること。

**f0fix再学習(s6_cfm_f0fix)と3対照（2026-09-19、[controls_s6.json](../results/diag_cfm_audit/controls_s6.json)）**: f0オーバーレイで同一レシピ再学習（best eval-L1 0.2284）。latent上のlf0効力は回復（+7st/+12stでΔ=12.6×/15.5× seed spread、旧2.8×/4.8×）したが、**デコード音声のF0は依然ゼロ追従**（100.6→99.9Hz、有声率0.27）。対照で原因を確定: (A) GT latentにlf0方向の差分δを加算しても中域発話ではF0不動（±0.1半音）・端域では破壊的、(B) GT latentからのlf0線形プローブはR²=0.088。**codec z内のF0は非線形に符号化され、決定論的条件回帰（=条件付き平均）はF0微細構造を置けない**。

**s11 mel_in条件（2026-09-19・包絡解決/F0制御喪失）**: causal mel80(/8・clamp±6)を条件末尾へ追加（cin 770→850・他s7同一・40k）。**包絡はcodec天井まで解決**: fecf 6.92→3.70・fe8c 5.19→3.16（codec自体の包絡誤差3.1）・eval-L1 0.2543。「ContentVecが包絡を運ばない=ロボット声の根因」を確認。**一方F0制御が消滅**: 中域掃引で+7st→+0.05st・+12st→+0.26st——mel内の元F0倍音構造がlf0スカラーを上書き。粗いmel周波数ワープでは+18/−16stと乱雑（[s11_env.json](../results/diag_cfm_audit/s11_env.json)・[s11_sweep.json](../results/diag_cfm_audit/s11_sweep.json)・[s11_sweep_warp.json](../results/diag_cfm_audit/s11_sweep_warp.json)）。次設計(推奨): **F0シフト増強**——WORLD再合成のピッチシフト音声をcodec再エンコードしたz'をtargetに、条件はmel/content/energy=元音声・lf0=シフト後、で「lf0が権威・melは包絡」を学習させる（信号処理由来・推論側変更なし・規則適合）。

**耳ゲート第1報とロボット声診断（2026-09-19）**: オーナーの耳判定「**ロボットボイスみたい**」（生成サンプル）。信号レベルで原因を切り分け: (1) 過周期性説は否決——生成のaperiodicity 0.74–0.76は元音声0.67/codec往復0.70より高い、jitter正常。(2) **スペクトル包絡スメアが主因**: 同話者・同内容で、codec自体の包絡誤差3.0–3.1dB（歴史的耳PASS水準）に対し生成は5.2–6.9dB。(3) 因子掃引は全て否決: K=16は−0.13dBのみ、真ContentVec contentは−0.6dBのみ、AR(1) rhoは学習値0.9が最適（0.0→9.32・0.95→7.49・0.99→8.10）、seed間包絡距離5.0 vs GT距離7.0で単一テンプレート収縮ではない、**s9(dim512・約2倍param)もfecf 6.92→7.01・fe8c 5.19→6.99と無効/悪化**。結論: **現在の条件セット(content+lf0+energy+speaker)とlatent L2 velocity目的では包絡微細構造が決定づけられない**。次設計: decode領域補助損失（frozen codec decoder通しのmel損失・step単価3–5倍・要承認）、条件追加、factorization階層化。GANは規則上不可。聴き分けA/B: `fem_codec_roundtrip.wav` vs `fem_e1.wav`（同発話・同話者・GT latent decode vs 生成）。

**s8話者条件強化と統計スワップオラクル（2026-09-19・両方負例、[spkin_eval.json](../results/diag_cfm_audit/spkin_eval.json)・[statswap_oracle.json](../results/diag_cfm_audit/statswap_oracle.json)）**: spk_in=speaker embedding 192dを入力concatへ追加（sfilm維持・他s7同一・40k steps）。**事前固定の主成功条件（女声source対照3seed cos_vs_target≥0.40）にFAIL**: 0.313（s7 0.299）。CER 0.407（s7 0.333より悪化）。一方で学習規約内F0追従はほぼ完全化（+7st→+6.99st・+12st→+11.41st、3 seed安定）。統計スワップオラクル（GT latentへ話者別mu/sdのaffine変換→decode）: cos_target gt 0.204 / mean_swap 0.189 / affine_swap 0.236 —— **話者一次latent統計はidentityを運ばない**。結論: 条件経路容量・一次統計経路のどちらでもなく、学習時の(content, speaker)一致により模型がidentityをcontent側（ContentVecの話者リーク）で説明する**content leakが主仮説**。対策候補は話者敵対content・source-filter分離の再評価・decoder側話者条件（codec契約変更）で、耳ゲート後に1変数ずつ。

**OOD・content差し替え対照（2026-09-19、[ood_content_swap.json](../results/diag_cfm_audit/ood_content_swap.json)・[ood_content_swap_seeds.json](../results/diag_cfm_audit/ood_content_swap_seeds.json)）**: 同一レンダ経路・同一targetに女声HELD明瞭発話（fecf5112354be881_00006519・377Hz）を投入。cos_vs_target 0.29–0.35（namikawa男声0.166から回復）、CER 0.33–0.35 vs codec往復天井0.049（namikawa 0.836から回復・文字起こしは文として成立）、st0のF0はソース追従（379–380Hz）。**E1代理contentと真ContentVecの3 seed比較は区別不能**（cos 0.299/0.308・CER 0.333/0.350）——E1のcos 0.74は現状の品質を制限していない。**男声content OODがフル経路失敗の主要因と確定**。

**他軸交絡の測定（2026-09-19、[spk_asr_s7.json](../results/diag_cfm_audit/spk_asr_s7.json)・[cer_floor_namikawa.json](../results/diag_cfm_audit/cer_floor_namikawa.json)）**: 学習規約内サンプルの話者cos=0.556(base)/0.574(+12st)で、codec天井0.607の92–95%を保持（F0シフトで崩れず）。**フルレンダは話者ターゲット失敗**（target cos 0.166/0.232 ≈ クロス話者floor 0.10–0.21）かつ**内容保持失敗**（whisper CER 0.836/0.759 vs codec往復天井0.23、文字起こしはワードサラダ）。whisperは女性ASMRコーパス（唸り中心）に使えない: GT vs .labでCER 1.38、耳PASS済みcodec往復でも0.93——内容保持の判定は学習コーパスでは耳ゲートが必須。

**補間CFM(s7_cfm_itp)でF0追従が成立（2026-09-19、[audit_itp_k8.json](../results/diag_cfm_audit/audit_itp_k8.json)・[render_s7.json](../results/diag_cfm_audit/render_s7.json)）**: 補間z_t学習＋K=8 Eulerサンプリング（`--interp --sample-k 8`、他はs6と同一）。学習規約内の中域発話（真値253.8Hz）で、base 264.2Hz / +7st→361.5Hz（**+5.43st**、要求−1.57）/ +12st→488.0Hz（**+10.62st**、要求−1.38）——判断基準「要求±3半音以内」を両方PASS。seed 3種のF0ばらつき0.9–4.8Hz（0.2半音未満）で用量反応は単調。サンプラーの多様性も復活（seed spread 0.18–0.22 vs 残差0.27–0.37、v場のt依存あり）。**フルレンダ経路（namikawa・205.1s・修正fps・E1代理content）では**ソース119Hzに対しst0=205.4Hz（女声事例分布への引張り+9.4st）・st17=322.7Hz（シフト差+7.8st/要求17、有声率0.39→0.60）と方向は正しいが圧縮・上方バイアスあり。energy包絡はlog-corr 0.94で追従（[energy_s7.json](../results/diag_cfm_audit/energy_s7.json)）。残る差の候補: 男声contentのOOD性（女声のみ学習）、E1代理content（cos 0.74）、lf0条件対事例priorの相対的な重み。decoder側明示f0条件付け（NSF方式）は圧縮が残る場合の次案として維持。

残る未監査: 学習crop左文脈なしと全発話評価の起動状態差、speaker欠損時のtrainゼロベクトル／eval省略の差、AR(1) noiseのchunk横断state、F0掃引がenergy・話者性へ与える交絡（未測定）。

**信号路NSF decoder v1/v2（2026-09-19〜20・F0権威負例で計7例目）**: v1=学習マスク乗算変調は周波数変換チートでz側pitchを合成(23kで掃引+0.1/+0.25st)。v2=マスク帯域≲50Hzでチートを原理封止したが品質停滞(最終held-mel 0.6586)・掃引不成立(+0.14/+0.23・−0.54/+0.09st)（[s2b_g2_mid.json](../results/diag_cfm_audit/s2b_g2_mid.json)・[s2b_gates_final.json](../results/diag_cfm_audit/s2b_gates_final.json)）。結論: **zがpitchを含む限り、どの経路でもF0権威は奪えない**——次は符号器でzをpitch-blind化(f0摂動不変な符号化。例: 同一発話のf0シフト音声ペアでz一致を強制する符号器正則化)してからNSF decoderを組むcodec契約変更が本命。

## 6. 次のゲート

包絡はmel条件(s11=3.70・s13=3.90)で解決、F0H再構成ヘッドもS1-3超え(0.4358)。F0権威は7負例(s10/s12/s13/f0h-concat/f0h-FiLM/NSF-v1/NSF-v2)で確定——**pitch-rich zを条件とする限りF0権威は成立しない**。次: 符号器のpitch-blind化(f0摂動不変正則化)+NSF decoderのcodec契約変更 / または高品質シフタ増強 / またはs11でF0保留しcontent leak・男声被覆・一軸(息)を先に。

診断レンダの固定target・固定シフトを、学習済みprosody policyやzero-shot成功と呼ばない。K=8サンプリングの生成器step予算はcodec台帳の6ms枠に対して未実測——実機実測をRust化前に必ず行う。
