# 研究中

> 更新: 2026-09-23（第12次・bug7でD1ラダーの定量読みを撤回し耳ゲート待ちへ）。現在地・未解決・次の実験のみを記載。
> 2026-09-19〜20: 信号路NSF v1/v2もF0権威負例(計7負例)——v1はマスク周波数変換チート、v2はチート封じ後に品質停滞+掃引不成立。**確定: pitch-rich zを条件とする限りF0権威は成立しない。次設計=符号器でzをpitch-blind化してからNSF。**現行best=s11+F0H再構成ヘッド(held-mel 0.4358<S1-3基準)。Rust化未実施。

## 1. 現在地

| 対象 | 確認できた内容 | 留保 |
|---|---|---|
| Y-S1女声再構成 | 旧記録S1-3に未知女声held0–2のオーナー耳PASS | 未知女声全体や囁きの網羅的合格ではない。男声再構成FAILも同記録 |
| Rust decoder | [s1_4_probe.json](../results/ys1/s1_4_probe.json): p95 3.0879ms、mean RTF 0.2871、10,000 step、warm-up 2,000 | JSONのcpu_nameは空、state_bytesは0。機材・stateの完全な証跡には不足 |
| S1-4承認 | 旧記録にdecoder上限を≤3.5msへ変更したオーナー承認、parity等のPASS記載 | 承認記録の継承であり今回の再測定ではない。製品E2E未達成 |
| CFM接続（段階1監査） | [diag_cfm_audit](../results/diag_cfm_audit/): codec stream decodeは一括とL1≈3e-5で一致、GT latent decodeのF0測定は元音声と一致 | 測定はencode_feat同一規約（44.1k+harvest+stonemask）でなければデコード音声で約3倍誤読する |
| 学習データf0 | female_real_featのf0が約85%のファイルで元音声と無相関。全20,322学習ペアが対象。**再計算オーバーレイ `data/female_real_f0fix/` 完了**（20,322件・err 0・サンプルcorr=1.0000） | 破損の生成元スクリプトは未特定。旧f0の再利用禁止 |
| CFM目的関数 | z0入力のままのL2学習は条件付き平均へ退化。f0fix再学習(s6)でもdecode F0不追従 — F0の非線形符号化＋平均化のため（[controls_s6.json](../results/diag_cfm_audit/controls_s6.json)、線形プローブR²=0.088） | 決定論的条件回帰はF0軸に使えないと確定 |
| **補間CFM F0追従（s7_cfm_itp）** | **学習規約内で要求±3半音以内をPASS**: 真値253.8Hz発話で+7st→+5.43st・+12st→+10.62st、seed間F0 sd<0.2半音、単調な用量反応（[audit_itp_k8.json](../results/diag_cfm_audit/audit_itp_k8.json)） | 圧縮あり（約80–90%）。F0掃引の他軸交絡は未測定。耳ゲート未実施 |
| フルレンダ経路（修正済み） | namikawa 205.1s=元長・修正fps・s7モデル・K=8: ソース119Hz→st0 205.4Hz/st17 322.7Hz（シフト差+7.8st/要求17）、有声率0.39→0.60、energy包絡log-corr 0.94（[render_s7.json](../results/diag_cfm_audit/render_s7.json)・[energy_s7.json](../results/diag_cfm_audit/energy_s7.json)） | 方向は正しいが圧縮・上方バイアス |
| **他軸交絡（2026-09-19測定）** | 学習規約内: 話者cos 0.556/0.574（codec天井0.607の92–95%、+12stでも維持）。フルレンダ: target話者cos 0.166/0.232≈クロス話者floor(0.10–0.21)=**話者ターゲット失敗**、CER 0.76–0.84 vs codec往復天井0.23=**内容保持失敗**（文字起こしはワードサラダ）（[spk_asr_s7.json](../results/diag_cfm_audit/spk_asr_s7.json)・[cer_floor_namikawa.json](../results/diag_cfm_audit/cer_floor_namikawa.json)） | whisper CERは女性ASMRコーパスには不使用（GT vs lab 1.38・codec天井0.93）。フル経路の失敗は男声OOD content・E1代理(cos 0.74)・条件強度の候補と整合 |
| **OOD・content差し替え対照（2026-09-19）** | 同一経路・同一targetに女声HELD明瞭発話（fecf5112354be881_00006519・377Hz）を投入: cos_vs_target 0.29–0.35（namikawa男声は0.166）、CER 0.33–0.35 vs codec往復天井0.049（namikawa 0.836）、st0のF0はソース追従（379–380Hz）、文字起こしは文として成立。**E1代理contentと真ContentVecは3 seedで区別不能**（cos 0.299/0.308・CER 0.333/0.350）（[ood_content_swap.json](../results/diag_cfm_audit/ood_content_swap.json)・[ood_content_swap_seeds.json](../results/diag_cfm_audit/ood_content_swap_seeds.json)） | **男声OODは主要因と確定、E1代理は現状ボトルネックでない**。残る主要ギャップ=cross-speaker話者条件の弱さ（女声sourceでもcos_vs_target≈0.30 vs 天井0.6+でsource話者寄り0.37）と男声content被覆 |
| **s8話者条件強化＋統計スワップオラクル（2026-09-19・両方負例）** | spk_in(speaker 192dを入力concatへ追加、他s7同一・40k): **主成功条件FAIL** — 女声source対照cos_vs_target 3seed平均0.313(s7 0.299、要求≥0.40)、CER 0.407悪化。学習規約内F0は改善(+7st→+6.99st・+12st→+11.41st・ほぼ完全追従)。統計スワップオラクル(GT latentへ話者別mu/sd変換→decode): mean_swap cos_target 0.189/affine 0.236 vs gt 0.204 = **一次統計はidentityを運ばない**（[spkin_eval.json](../results/diag_cfm_audit/spkin_eval.json)・[statswap_oracle.json](../results/diag_cfm_audit/statswap_oracle.json)） | 条件経路の容量はボトルネックでない。原因は**content leak**(ContentVecがsource話者性を運び、学習時はcontentとspeakerが一致するため模型がcontent側でidentityを説明)とidentityの非線形性に絞り込まれた |
| **耳ゲート第1報＋ロボット声診断（2026-09-19）** | オーナー耳: 生成音声は「**ロボットボイスみたい**」。信号診断: 過周期性でない（aperiod 0.74–0.76 ≥ 元0.67/codec往復0.70、jitter正常）。**スペクトル包絡のスメアが主因**: codec自体の包絡誤差3.0–3.1dB(耳PASS水準)に対し生成は同話者・同内容で5.2–6.9dB。因子掃引で全否決: K=16無効(−0.13dB)・真ContentVecは−0.6dBのみ・rho学習値0.9が最適・seed間包絡距離5.0 vs GT距離7.0=収縮でない・**s9(dim512約2倍)も無効**（fecf 6.92→7.01・fe8c 5.19→6.99悪化）（[robot_voice_diag.json](../results/diag_cfm_audit/robot_voice_diag.json)・[_diag2.json](../results/diag_cfm_audit/robot_voice_diag2.json)・[envprobe.json](../results/diag_cfm_audit/envprobe.json)・[envprobe_rho.json](../results/diag_cfm_audit/envprobe_rho.json)・[envseed_collapse.json](../results/diag_cfm_audit/envseed_collapse.json)・[s9_env.json](../results/diag_cfm_audit/s9_env.json)） | 聴き分け用A/B: `fem_codec_roundtrip.wav`(GT latent decode・同発話同話者) vs `fem_e1.wav`(s7生成)。codec側もHF減半(0.059→0.027)あり無罪ではないが主因は生成器 |
| **s10 decode領域補助損失＋包絡分解（2026-09-19）** | K=2ロールアウト→frozen decoder→log-mel L1(aux_w0.1・40k・他s7同一): **包絡FAIL**（fecf 7.0 vs s7 6.92・目標≤5.5、fe8c 6.78悪化）、eval-L1 0.2896。**F0掃引はほぼ完全化**(+7→+7.36st・+12→+11.97st)。包絡分解: seed平均-GT間6.36 vs seed個別バラつき2.64 = **全seedが同じ誤った平滑テンプレートを共有**（[s10_env.json](../results/diag_cfm_audit/s10_env.json)・[s10_record.json](../results/diag_cfm_audit/s10_record.json)） | 失敗の構造的原因の仮説: **ContentVecはASR用で包絡を設計上保持しない**——条件セットに包絡情報が無く、(音素+話者)からの包絡合成は回帰的に平均へ潰れる |
| **s11 mel_in条件（2026-09-19・包絡解決/F0制御喪失のトレードオフ確定）** | causal mel80(/8)を条件へ追加(他s7同一・40k): **包絡はcodec天井まで解決**——fecf 6.92→**3.70**・fe8c 5.19→**3.16**(codec自体の誤差3.1)・eval-L1 0.2543(歴代最佳)。「ContentVecが包絡を運ばない=ロボット声の根因」仮説を**確認**。**一方F0制御が消滅**(+7st→+0.05・+12st→+0.26): mel内の元F0倍音構造がlf0スカラーを上書き。mel周波数ワープ(粗い近似)では+18/−16stと乱雑で不可、リフタリングの線形プローブは判定不能(R²≈0.14でrawと不変——倍音読みは非線形)（[s11_env.json](../results/diag_cfm_audit/s11_env.json)・[s11_sweep.json](../results/diag_cfm_audit/s11_sweep.json)・[s11_sweep_warp.json](../results/diag_cfm_audit/s11_sweep_warp.json)・[lifter_probe.json](../results/diag_cfm_audit/lifter_probe.json)） | 次設計(推奨): **F0シフト増強**——WORLD等の信号処理ピッチシフト音声をcodecで再エンコードしたz'をtargetに、条件はmel/content/energy=元音声・lf0=シフト後、のペアで学習し「lf0が権威・melは包絡」を教える(推論時の条件変更なし・信号処理由来で規則適合) |
| S1-5出力の原因 | fps不整合＋f0破損＋目的関数退化の3因で「狙い292Hzに99.7Hz」を説明し、修正で追従が出现 — 因果確認済み | — |
| **decoder f0ヘッド2種（2026-09-19・両方F0負例・再構成は基準超え）** | 凍結S1-3 trunk上のNSFヘッド30k: concat型 held-mel 0.4377・FiLMキャリア型 0.4358（S1-3基準0.4418超え）。**F0掃引は両者とも不動**（concat: mel差0.1–0.26=励起未寄与、FiLM: 出力がf0にビット不変=ゲートで励起遮断）（[f0h_gates.json](../results/diag_cfm_audit/f0h_gates.json)・[f0h2_gates.json](../results/diag_cfm_audit/f0h2_gates.json)） | 5負例で統一法則確定: pitch含有特徴への事後f0チャネルは中和される。解は信号路NSF化フル再設計 |
| 官能・操作UX | [kansei_control.md](kansei_control.md) に2026-09-08の相談を保存 | PROPOSED。制御ノブの成立・GUI実装・学習効果は未検証 |

S1-3/4/5の履歴は [整理前RESEARCH](../.archive/docs_2026-09-08/current/RESEARCH.md) の同名見出し。元記録に日付の前後関係の混乱があるため、日付順だけで最新runを判定しない。

元記録の証拠パス: `results/s1_3_c32/held{0..3}_{gt,ema}_norm.wav`、`results/s5_render/namikawa_s5.wav`。現存を確認できない音声や重みは「記録上の所在」とし、再利用前に実ファイルと識別子を確認する。

| **s18 ACF調和対比損失(2026-09-21・tripwire中止)とfork(b) s4 NSFShip(2026-09-21・両ゲートFAIL)** | **【2026-09-23 s18は無効(bug8): harm損失のf0目標が`cb[:, -3]`=mel bin77(≈150Hz定数)でlf0ではなかった。下記s18の「バズ化チート」「L側反証」は実装不備の産物・修正済み未再走。s4 NSFShipの結果はbug8と無関係】** s18: 検査L機械化は機能(v1 STFT型を棄却→v2 ACF型が合成信号±0.5st=608倍で機構PASS)したが、学習統合でbase 92Hz崩落+掃引+1st@6k→**tripwire発動・abort**(harm-w0.5のバズ化チート)。fork(b)起動: NSFShip v3(0.29M・**RTF実測0.62ms/frame=予算内**・学習前プローブ合格)40k学習→**held-mel 2.18(目標0.60不可)・F0掃引完全不動(140.3Hz固定・fe8c無声0%)・包絡22.5**([s4_gates.json](../results/diag_cfm_audit/s4_gates.json)・prereg/台帳は`results/s4_nsfship/`) | NSFShipの軽量信号路(6000fps+最終×8)は容量不足でpitch形成自体不能——NSF系は「容量↔RTF予算」のトレードオフにあり、0.3Mでは再構成すら成立しない。RTF 0.62msに容量の余地は残るが、何倍積んでもNSF系列の品質天井(v2=0.66)を割る見込みは薄い。**fork(b)の初期判定はFAIL——出荷NSFは系列として再検討が必要** |

| **コーラス感の更なる切り分け(2026-09-21・耳第4報)** | オーナー耳: **K=16でも消えない・codec天井には無い**→サンプラー誤差・decoder両方否決、**生成latentそのものが原因と確定**。定量: F0ジッター(2.5ms)はe1_K16=38.9c≈source 38.2cで解消済み=ピッチ変調ではない。周期間コヒーレンス: gtdecode 0.715 vs 生成 0.64-0.65(軽度低下)・seed平均は0.41に崩壊。帯域別aperiodicity: 全帯域で+0.02-0.03の一様上昇(HF特化なし)。content源(e1/cv)は無差別（`results/earbattery/ab_chorus/`） | **残る仮説: 100fps格子latentのサブフレーム位相構造——encoder zは波形位相を暗黙エンコードし decoderはフレーム間位相一貫性から鋭い倍音を再構成する。生成サンプルはこの微細構造の分布は近似するが位相一貫性が不完全で、decoderが整列不全の高次倍音を重ねる=HFコーラス**。検証には生成zとGT zの微細時間構造の直接比較(畳み込み応答・位相スペクトル)が必要 |
| **コーラス/フェーザー感の帰属(2026-09-21・耳第3報)** | オーナー耳: 「高周波のコーラス/フェーザーが掛かった声」。**F0マイクロジッター(5ms解像度・cents RMS)**: source 62→生成K8 87–94(+25〜32cents過剰=知覚的ディチューン)→**K16で61.9–66.5(source水準に復帰)**。f0条件源(causal_f0 vs world)は二次効果。8–50Hz変調シェアもK16で改善。ハーモニック分裂計測は自然ビブラートと分離できず帰属不能。**A/B素材: `results/earbattery/ab_f0src/`(f0src×Kの2×2)+`s11_K16/`(6clip)** | 残る疑い: codec decoder自体のtransposed-conv由来の100Hz周期アーチファクト(codec_ceilingで聴取確認が必要——天井側にも同じ効果があれば生成器ではなくc32契約)。K=16はRTF 3.8ms枠内・重み不変で即適用可 |
| **高域ノイズの切り分け(2026-09-21)** | K依存: K=4→hi 0.052/K=8→0.040/K=16→0.035/K=32→0.034(**K増で単調減少=サンプラー打ち切り誤差が主因**)・GT decode 0.024。latent自体の時間粗糙さは生成<GT(0.30 vs 0.37)でlatentは原因でない。**K=8の1-step分の積分誤差が2-12kHzノイズを盛る**。RTF: K=16=3.8ms/frame・K=32=7.6ms(生成器単体・c32 0.68と合わせK16で約4.5ms/frame=枠内) | K=16への変更は推論時のみ・重み不変・再学習不要で試聴可能。K=32は生成器単体で予算超 |
| **耳ゲート第2報+s11・codec天井の3way帯域帰属(2026-09-21)** | オーナー耳: s16バッテリーは「低品質」。定量帰属: **ザラつき主因**——生成(s11/s16とも)は2–12kHz帯を過剰に盛る(hi_mid src 0.077→s11 0.081・codec 0.053、hi 0.061→0.096・0.050)、aperiodicity過大(0.74 vs 0.67)と整合。**codec天井はvhi(12k+)を1/3に減衰**(0.0126→0.0045)し、生成器はhi帯ノイズ+0.03盛る。s16はs11比で包絡+1.7dB劣化(摂動コスト)（[band_analysis_3way.json](../results/earbattery/band_analysis_3way.json)・3way聴き比べ素材=`results/earbattery/{s11,s16,codec_ceiling}/`） | 品質劣化の内訳: ①2–12kHzノイズ過剰=生成器(K=8サンプラーのSDE項または速度場の高域発散) ②vhi減衰=codec(c32契約) ③s16固有=摂動学習コスト。①が最大で改善余地、②はcodec再設計、③はs11に戻せば無料で解消 |

| **理論統合+0学習検証(2026-09-21)** | 20本の失敗を1つの理論に統合([rep_theory.md](rep_theory.md): 損失可視性公理+位相積分定理)。**検証プローブPASS**: GT潜在に生成器級フレーム独立ノイズ→decode音声が壊滅(L1許容球内に破壊が存在=公理の直接証明)。時間平滑は無傷=コーラスは位相フィールドのフレームレート誤差。理論が導く設計: D1=AR潜在生成 / D2=decode領域GAN FT / D3=因果直接波形mel vocoder | 実験優先から理論優先への転換。次腕は理論の予測(§5反証条件)を検証するものから選ぶ |

| **NAM理論統合+D4設計+AR vs CFM 対比素材(2026-09-21)** | NAM(因果的非線形記憶系の同定)理論を[rep_theory.md](rep_theory.md) §8に統合。NAMの存在は定理P''の産業規模実証。**新設計D4=条件付きNAM vocoder**: 条件(100fps)→48kHz波形を単一因果AR系で直接学習、codec潜在を完全スキップ(位相問題が存在しない平面)。同容量AR overfit free-run(L1 0.89!)がCFM(K8・L1 0.9)より自然(hi_mid 0.0116 vs 0.0133・GT 0.0118)——**損失同値でも逐次性だけで知覚品質が決まる直接実証**(`results/earbattery/nam_test/`) | D4は上流生成器不要(条件=ギター入力に相当)・RTF c32同級見込み。反証条件§8-5に固定 |

| **理論第3版+定理E実測(2026-09-21)** | [rep_theory.md](rep_theory.md) §9: 定理M(状態可観測性——D4波形ARがD1潜在ARより位相課題で厳密に易しい)・定理E(包絡-話者絡み——**実測: mel80から60話者中19.8%特定**=D4がsource話者を再現する予測)・定理D(残余崩壊+**WaveNetカテゴリカル尤度=GAN不使用のtexture回復道**)・訓練双対性(因果convは訓練並列・推論逐次)・予算算術(D4=0.5MMAC/frame=予算2%) | D4設計更新: ①話者条件付きmel正規化必須(E1) ②deterministicで開始しtextureはカテゴリカルヘッドで回復(D) ③コーラスは出現しないはず(M1)——いずれもpreregに反証条件として固定可能 |

| **D4系実装とbug5(2026-09-21・d4a崩壊→d4b移行)** | **d4a(決定論L1波形AR)**: free-run 100Hzロック+DC崩壊=**定理Dの直接確認**(ゼロ文脈ブートストラップで決定論的回帰は条件付き平均に潰れる)。**d4b(カテゴリカルμ-law CE)**実装中に**bug5(学習CEのreshapeスクランブル)**を発見・修正: `[B,256,n]`を`reshape(-1,256)`で切るとクラス軸が時間軸になる=損失自体が壊れ全CE値が無意味(lr/レジーム診断を全部無効化)。**エントロピー床実測**(限界5.301/1サンプル条件付き3.369)との照合が検出器になった→design_laws検査L-4(損失床の数値照合)に制度化。検証: 固定batch CE 4.52→2.44@300step。 | 損失実装は理論床と照合して独立検証する(数値の正常/異常を先に定義する) |

| **D4 pitch権威ラダー(2026-09-21〜22・確定)** | 4腕40k(d4b/b2/c/d)で**定理S(継続-変換緊張)**確定: sample-ARは継続機械に特化しpitch権威は不成立。構造生成(有声化・位相連続)はカテゴリカル尤度で解決済み・幹は健全。bug5(CE reshapeスクランブル)もこの期間に発見・L-4制度化。詳細=rep_theory §10・台帳 | sample-ARのpitchはネット外へ(DSPプライム/フロントエンド)が定理的帰結 |

| **D1潜在AR実行ラダー(2026-09-22〜23・bug6含む・prereg: results/d1_g0/)** | フォレンジック5事実→設計rev2(オーナーレビュー反映・lf0権威主張降ろし/mel80主従逆転)→起動前プローブ3種(非線形pitch R²=0.074/路線(c)先読み10ms生存/RTF 4.3ms)→**bug6(shift_right恒等関数)をオーナーが発見・G0初回無効化**→修正再走(G0: 主23.6/m80 10.9/par 6.5・プライム切替即崩壊)→fallback(A)履歴ノイズ(12.1=定常半減・崩壊不変)→30k前置基準判定(**loop_gain 1.35→~1.0=増幅消失・スケール律速第3条**)→G1フルデータ(**13.0x(seen)/32.9x(held)=FAIL・定常誤差0.7-1.1が律速**)→z0相関フォールバック(15.6=効果なし・権利行使済み)。**確定読み: ループ安定性は解決・残る律速は定常サンプリング誤差(GT履歴条件付きですら3x)・かつ3.85M並列対照自体がs11(1.8x)未達=品質域未到達のままAR/並列比較が不可能**。S1(条件付きAR>並列2.88 vs 6.5)はbug6修正後も成立 | design_laws L-4へbug6(シフト/整列の数値単体テスト・チート利得は正当利得と損失上不可分)追加。**次は判断待ち**: (ii)DAgger+容量増 / (iii)路線(c) / (iv)s11系容量再攻 — いずれも天井3xの定常誤差問題と基準品域未達を抱える |

| **bug7=D1評価器のスケール取り違え・ラダー定量読みの撤回(2026-09-23)** | `eval_d1_g0.py`がGT参照を**正規化潜在のまま**decode(生成側は逆正規化)→参照音声RMS約2倍・hi_mid 1/2〜1/9(発話依存)。旧物差しで報告値を再現し同窓・正参照で訂正: **GT履歴「天井3×」3.25→1.00・切替後「即崩壊」55.7→1.17**。v2評価器(6話者×3seed・last ckpt)で: overfit並列1.06/AR 1.28(hi_mid中央値)=並列≥AR、フルデータAR(g1full)2.55・held 2.09(旧「held 32.9×」は偽)、s7 3.32、s11 K8 1.72/K16 1.61=s11が全数値で最良。フルデータARは自己履歴で実ドリフト(1.00→1.8〜2.2・誤差自己相関0.01→0.42)。**hi_midは耳ラベルのコーラスを分離できない**(e1_K16 1.43 vs 元音声1.32)=D1の主張軸(軌道整合)は未計測だった。設計§2の話者条件が未実装だったことも判明 ([_ledger.md](../results/d4a_namvoc/_ledger.md) bug7節・[reeval_bug7.json](../results/d1_g0/reeval_bug7.json)・各腕`g0_report_v2_*.json`・[chorus_proxy_validation.json](../results/earbattery/chorus_proxy_validation.json)) | **撤回**: 上行の「定常誤差律速・天井3×」「即崩壊」「並列がs11未達(=(iv)の根拠)」「S1」。**残る**: loop_gain・フルデータの暴露ドリフト・overfitで並列≥AR。design_laws L-6(評価参照の元wav照合assert・代理指標の耳ラベル検証)追加。**次=耳**: 盲検A/B `results/earbattery/d1_ab/`(代理予測は事前登録済み)+同条件フル並列対照`diag_d1_g1par` |

| **AR vs 並列の対照整備と誤差帯域帰属(2026-09-23・diag_・0〜短時間学習)** | 同条件フル並列`diag_d1_g1par`(D1ARのno_history=**フレーム独立サンプラー**)は全指標でARより大幅に悪い(seen hi_mid 5.56 vs 2.55・env 2.96 vs 2.06)。だがs7/s11(CFMYS)は速度場畳み込みの**系列同時サンプラー**でありこの対照の相手ではない→同予算CFMYS`diag_cfm_small_nospk`(実効3.855M・batch4・話者/mel80なし)を追加: **ARと包絡同等(env 2.06 vs 2.05)・ARは高域ノイズのみ低い(hi_mid 2.55 vs 3.65)・コーラス代理は混在・ただしARは自由走行でピッチ誤差|Δf0|中央値2.2半音(同予算CFMYS 0.5〜1.4・s11 0.3〜0.5)・有声率−21pt(同 −11〜−14)=lf0不追従+自己履歴ドリフトの帰結**。**衝突掃引(+7st要求): AR 0.06〜0.64st vs 同条件フレーム独立並列 +4.3〜+5.2st=履歴がpitch権威をlf0から奪う(定理Sフレーム版)の直接証拠**(評価器の掃引ガード修正後)。潜在軌道: 時間結合のある生成器は全てGTより平滑(jit s7 0.38/小CFMYS 0.38/AR 0.54/s11 0.61、フレーム独立のみ2.38)=「並列はジッターする」型のF6はs11に当てはまらない。**誤差帯域帰属: 潜在の16–50Hz帯はピッチ/位相を運ぶ(GT自身の同帯域除去でf0 −12.7半音崩落=時間平滑・低域通過は不可)。s11の誤差の52%がこの帯域にあり、その帯域だけGTに差し替えると f0 を保ったまま調波スメアが −1.16→−0.49**=コーラス候補はピッチ/位相を運ぶフレームレート微細構造の誤差(弱検証代理・「欠落か誤りか」は切り分け不能) ([_ledger.md](../results/d4a_namvoc/_ledger.md)・[latent_traj_stats.json](../results/d1_g0/latent_traj_stats.json)・[err_bands.json](../results/earbattery/err_bands/err_bands.json)) | 耳素材: `results/earbattery/d1_ab/`(12試行・優先 fullc→full→ovf→fullm)・`results/earbattery/err_bands/`(README付)。数値だけでは(ii)/(iii)/(iv)を選ばない |

| **耳の盲検A/B(2026-09-23・オーナー・12試行)** | 錨(GT decode)が12/12で1位(聴取信頼・全生成系が天井より劣る)。**fullc(AR vs 同予算CFMYS) AR 2-1**(n=3・明確でない)・fullm(AR vs フレーム独立) 3-0・full: AR対s11 1-2/対s7 1-2(平均順位 s11 2.67・s7 3.0・AR 3.33)・ovf AR 1-2。**誤差帯域: s11から16–50Hz誤差を除去してもコーラスは消えず**=代理による高帯域帰属は耳で否定。代理(comb)の事前登録予測と耳の一致は偶然水準(7/18・11/18)=comb/nccもコーラス代理として不適格 ([ear_result.json](../results/earbattery/d1_ab/ear_result.json)) | **F6(ARがコーラスを減らす)は耳で支持されず**。事前登録forkの「差がない」側→D1の設計根拠は消えたと読む(D1のクローズはオーナー判断)。コーラスの所在は未解明に戻った: 16–50Hz帯ではない(耳)。「フェーザー」=遅い帯域の誤差仮説は未検証 |

| **コーラスの正体=codec decoderの一般応答(2026-09-23・盲検8本・オーナー)** | 錨=無・s11対照=有(正答)。s11の0–4Hz/4–16Hz誤差だけでもコーラス、16–50Hz除去でも残る。**エネルギー整合の合成ガウス雑音を潜在に足すだけでコーラス**(全帯域/16–50Hz/4–16Hz=有・0–4Hz=微)、誤差×0.5(rms 0.08〜0.11)でも微〜有。f0は全て保持 ([chorus_probe/ear_result.json](../results/earbattery/chorus_probe/ear_result.json)) | **Y-S1 decoderは潜在のずれを帯域によらずコーラス化し閾値が低い**=生成器の作り方(AR/並列/K/容量)では消えない理由を一括説明。旧「生成latentそのものが原因と確定」は「生成latentがGTからずれること×decoderの過敏さ」と精密化。次の候補: decoderの潜在摂動頑健化(潜在ノイズ増強FT・教師=実音声)(オーナー判断)。留保: 1発話・1聴取者 |

## 2. 現在の問い

**D1の主張軸(軌道整合=コーラス)を耳で測った(2026-09-23): F6は支持されず(同予算でAR 2-1・s11/s7に1-2・コーラスは16–50Hz誤差の除去でも残る)。次の幹の選択はオーナー判断待ち。**

**追加(同日)**: コーラスは codec decoder の一般応答(合成雑音でも鳴る)と判明 → 生成器側(D1含む)では解けない。decoder の潜在摂動頑健化が最有力候補(オーナー判断待ち)。

**2026-09-24 decoder頑健化FT(ys1_nrft)の耳**: s11のコーラスは形式上減った(2/3)が、**元音声を横に置くと codec往復(GT decode)自体が2/3発話で失格**・s11は全失格。**律速は Y-S1 c32 codec の再構成品質**(旧「S1-3耳PASS」は元音声非併置)。失格理由=コーラス・ざらつき・声質・こもりの全部。

**2026-09-24 天井の盲検**: 3/3で 元音声>BigVGAN v2(mel再構成・非因果・参照専用)>Y-S1往復。**オーナー: BigVGANは合格**。→目標品質は神経ボコーダで到達可能・Y-S1の失格はその設計/学習に固有(活性のアンチエイリアス無し・設計書必須のCQT判別器を学習で省略・幅/規模)。次の論点: Y-S1 decoderをBigVGAN級の要素(因果AA・CQT判別器・規模)で作り直すか(潜在ABI維持)、表現をmel等に替えるか([ceil_ab/ear_result.json](../results/earbattery/ceil_ab/ear_result.json))。

**2026-09-26 設計の転換案 [artic_a2.md](artic_a2.md)(PROPOSED・学習未起動)**: decoder 作り直し 2 腕は合格錨 BigVGAN(held24 logmel 0.195)から数値で遠く(A 0.429・B 0.672)耳の盲検を撤回。オーナー指摘「NAM A2 を継承すると言いながら構音学の示唆が無い」を受け、codec 潜在を捨てて声を物理の境界で切る案を設計: 速い構造=入力声の LP 残差(励起・可逆)、遅い構造=構音界面(LAR・ピッチ盲・~15Hz 以下)、声道=物理骨格(時変全極)+A2 回路。学習なしの Step 0(S0-1〜7・事前登録)をオーナー承認後に実施。README §3「content は SSL」と食い違うため方針変更はオーナー判断。

**2026-09-24 decoder v2 対照腕(オーナー承認)**: 本命 B=NAM A2思想(全層48kHz・上げ層なし・LeakyReLU+1x1・明示f0励起・frame-rate control net 注入)/対照 A=c32+因果AA(BigVGAN系)。0学習プローブ: 両者ストリーミング一致・未来不変PASS・A追加遅延0.44ms・B追加遅延0。RTFはeagerで予算超(A 4.44/B 3.59ms)・MACはBがc32の77%→Rust実測待ちで両腕diag_。学習中(A→B・`results/diag_dec2_{aa,a2}/`)。判定=元音声・BigVGAN併置の盲検でGT decodeが『合格』か。

- **最初に耳**: `results/earbattery/d1_ab/`(README・answers.mdあり・12試行)。問い=「ARは並列よりコーラス感が少ないか」。**本命は fullc(AR vs 同予算CFMYS=同時結合)**。full(AR vs s7 vs s11 K16)・ovf(overfit対)・fullm(AR vs フレーム独立)は補助。
- 併せて `results/earbattery/err_bands/`: s11の出力から16–50Hz誤差を抜くとコーラスが消えるか(消えれば、D1に限らず「フレームレート微細構造を正しく生成する」ことが本丸)。
- fork: **ARのコーラスが少ない** → F6支持 → (ii)の課題は「天井3×」ではなく、フルデータの暴露ドリフト(DAggerの正当な標的)・**自由走行のピッチずれ2.2半音と無声化−21pt(lf0不追従=定理Sフレーム版の実害)**・未実装の話者条件(設計§2)・mel80なしの包絡(env 2.1 vs s11 1.2)。**差がない/並列が良い** → F6不支持 → D1の設計根拠が消え、(iii)はpitch権威の観点で、(iv)は別根拠で判断する(旧根拠「並列の容量未飽和」はbug7の産物)。
- (iii) 路線(c)ハイブリッドの根拠(因果pitchシフタ先読み10ms生存)はbug7と無関係で生存。
- **F0権威の(a)路線(pitch鋭敏損失)が未検証に戻った**: s18(ACF調和対比損失)はbug8(f0目標が`cb[:, -3]`=mel bin≈150Hz定数)で無効。「L側単独の権威付与は反証→fork(b)」は撤回。正しいf0目標(`LF0_ROW`)での再走はpreregの再記入が要る(オーナー判断・未実行)。
- 代理指標: hi_midは高域ノイズ代理としてのみ使用。comb_db/period_nccは弱検証の補助(過周期でも上がる)。昇格・分岐は耳。

旧課題(話者ターゲット・男声被覆・CER等)は保留継続。

## 3. 次の一実験（提案・未実行）

**F0権威の5負例で統一法則を確定**（s10補助損失・s12 WORLD増強・s13倍音除去mel・s1_f0h concat・s1_f0h2 FiLMキャリア）: pitchを既に含む特徴を持つ系（z・trunk features・ContentVec+mel）では、学習時に一貫する追加入力は常に冗長で、模型は特徴側からpitchを再構成して追加入力を中和する（FiLMはゲートで励起を完全遮断、出力はf0にビット不変）。ただし**両ヘッドの再構成はS1-3基準を上回る**（concat 0.4377・FiLM 0.4358 vs 0.4418・trunk凍結・30k steps）——ヘッド品質は問題ではない。

1. **信号路NSF化フルdecoder再設計（完了・v1/v2ともF0権威負例）**: v1(s2_nsf・学習マスク): 乗算マスクでも**周波数変換チート**を学習(23k: st0真値375Hz精確再現・掃引+0.1/+0.25st——位相含有マスク×キャリアの積でz側pitchを合成)。v2(s2b_nsf_blm・マスク帯域≲50Hzでチート原理封じ): チートは封じたが**品質停滞(最終best held-mel 0.6586・G1不可)**しF0掃引も不成立(60k最終: +0.14/+0.23・−0.54/+0.09st、有声率0.15–0.38)——proj線形混合が調和チャネルを減衰させpitch薄い局所解に落ちる（[s2b_g2_mid.json](../results/diag_cfm_audit/s2b_g2_mid.json)・[s2b_gates_final.json](../results/diag_cfm_audit/s2b_gates_final.json)・`results/s2_nsf_train.log`・`results/s2b_nsf_blm_train.log`）。**結論: pitch-richなzを条件とする限り、信号路NSF化でもF0権威は成立しない。次設計は符号器側でzをpitch-blind化(f0無関係化)してからNSF decoderを組む**——codec契約変更の本命。**s3_codec_f0実行中(診断位置づけ・台帳はresults/s3_codec_f0/_ledger.md)**: 符号器スクラッチ+NSF decoder+不変性損失(片側stop-grad — 対称化が原理に忠実、契約逸脱[CTX48・gain aug/quiet欠落]は台帳に記録)。RTFプローブ: **NSF decoder 14.29ms/frame=予算4倍→出荷不可、出荷NSFは別設計**。c32出荷decoder 0.68ms/frame、**CFMYS K=8は1.90ms/frame=6ms枠内**。s3の結果はcodec契約変更要否の判定材料。

**s14_cfm_perturb(完了・F0/話者FAIL・包絡PASS)**: 入力側一括摂動(実効矛盾率=プール被覆2割×p0.5=**1割に低下**)のため失敗——F0掃引−0.14/−0.47st・cos_vs_target 0.23(CFG w=2でも0.233=条件が実質不使用)・包絡3.46 PASS・eval-L1 0.2553([s14_gates.json](../results/diag_cfm_audit/s14_gates.json))。**s15_cfm_perturb100(完了・FAIL)**: 矛盾率100%(プール3,996件のみ・p1.0)でもF0掃引+0.23/−0.03st・包絡4.63・cos 0.226([s15_gates.json](../results/diag_cfm_audit/s15_gates.json))。**出力pitchは真値に張り付きlf0を無視——WORLDアーチファクトから摂動量rを検出して元に戻す「摂動の可逆性」が失敗原因と判定**(理論の反証ではなくexecution)。**s16_cfm_pvshift(完了・F0 FAIL・話者軸初の改善)**: 位相ボコーダ摂動(フォルマント同時移動)でもF0掃引+0.37/+0.60st・包絡5.18——だが**cos_vs_target 0.226→0.307に改善**(フォルマント摂動が話者条件の使用を初めて生成)([s16_gates.json](../results/diag_cfm_audit/s16_gates.json))。**3連否決で根因確定: 潜在L1目的はpitchに不感**(R²=0.088と整合)——話者中央値+摂動下でも残るprosody輪郭で±3stは許容され、lf0を使う動機が損失に存在しない。**s17_cfm_pv_aux(完了・F0 FAIL・摂動系4連続非達成で系列打ち切り)**: aux decode-mel損失(0.3)併用でもF0掃引−0.41/−0.77st・cos 0.297維持・包絡4.99([s17_gates.json](../results/diag_cfm_audit/s17_gates.json))。**最終帰結: 潜在L1も80-mel L1も絶対pitchに鈍感(±3stは許容)で、話者中央値(speaker条件=真)+摂動下でも残るprosody輪郭(content)だけで損失を満たせる——lf0を使う動機は条件設計でも損失設計でも生えない。** 次の選択: (a) pitch鋭敏な補助損失(デコード音声の微分可能F0/調和ピーク損失——重い) (b) s3系(pitch-blind z+NSF・構造的権威。診断では陽性:+7.9/+10.45st)の本格化——出荷サイズNSFの再設計を含む (c) 現行資産(s11+f0h)で耳ゲート・話者軸(s16のformant摂動知見: cos 0.307)を先に。

**s14系列(統一処方)**: 15負例の統一原理「漏洩しうる条件は摂動で嘘にし、権威を持たせたい条件だけ真実にする」を1腕で実装——**入力側一括摂動**(content=シフト音声ContentVec・mel=シフト音声mel[既存f0shift_wav/contentキャッシュ3,996件]、lf0/energy/speaker/target=元音声)+ **CFG**(speaker条件ドロップ0.2、推論時はcond/uncond外挿)。ベースはs11レシピ(mel_in・interp・40k)。**成功条件走行前固定: F0掃引±0.5半音(中域3seed)・女声source cos_vs_target≥0.40(CFG)・包絡mel-L1≤4.0・男声source CER記録**。同腕でF0権威・content leak遮断・包絡維持を同時検証。
2. **高品質シフタによる一貫増強**(s12のexecution改善)は代替のまま。
3. **s11+F0HでF0制御を保留し前進**（content leak・男声被覆・一軸(息)）も並行可能。

成功条件・計算時間枠・評価splitは走行前に固定する。**2026-09-20制定: 以降の全腕は [design_laws.md](design_laws.md) の検査L/I/C記入（`results/<tag>/prereg.yaml`）を起動条件に追加**。

## 4. 制度(2026-09-20外部評価により追加)

- **feat生成検証ゲート恒久化**: `encode_feat.py --verify <dir>`(f0輪郭相関≥0.9・energy≥0.99)。実証: female_tts_feat PASS(1.000)・female_real_feat FAIL(−0.031)=過去の破損を正しく検出。新規feat生成後に必須。
- **固定耳バッテリー**: `results/earbattery/manifest.json`(会話2・喘ぎ1・囁き3)。腕ごとに1回、最初の耳判定に使う。
- **文献1チェック**: 腕起動前に「この失敗様式は既知か」を1回調べる。
- **f0破損の生成元**: git履歴に書き込みスクリプトなし(encode_feat自体は後追い整備・docstring「format verified 2026-07-20」)。未追跡スクリプトと推定・復元不能。検証ゲートが再発対策。
- **台帳規則の運用**: >1h学習は台帳を先に書く(s3は違反して後追い・s14は起動直後)。

## 5. 進行状態の扱い

旧文書の「学習中」「全停止」は時点の記録であり、現在のプロセス状態ではない。2026-09-19第7次時点で実行中のジョブなし（f0fix・s6〜s11学習・監査・レンダすべて完了）。学習再開時はプロセス、checkpoint、データ、出力先を読み取りで確認する。

失敗・撤回の最小索引は [FAILURE_NOTES.md](FAILURE_NOTES.md)、段階計画は [ROADMAP.md](ROADMAP.md)。

- 2026-09-27 **Artic-A2 打ち切り(S0-4 耳 FAIL)**: LAR 誤差の盲検で 4–15Hz×50% と 15–50Hz が「有」→ 中心仮説反証。コーラス/フェーザー感は時変包絡の 4Hz 以上の揺れで一般に鳴る。
  持ち越し(RRPS・残差とフィルタの整列則・LPC 界面のトレードオフ・声道長比 1.139)は [artic_a2.md](artic_a2.md) 末尾。
- 2026-09-27 **学習ありの DDSP による男→女ゼロショット VC [ddsp_vc.md](ddsp_vc.md)(2026-09-28 耳 FAIL・変換は全て聴取不可・再合成の段階で錨の 3 倍)**: 因果・先読み 0・遅延 10ms・Rust 推論 parity 8.6e-7・RTF 0.36 まで完成。
  選択 = v2 22.5k(目標 ECAPA 0.242・元話者 0.187・英 CER 増分 0.10)。目標らしさは別の実女性(0.42)に届かず未達。段 R 延長(v2r)は再構成が良くなるほど元話者へ寄る。物理パラメータ変換の天井は [physvc_zs.md](physvc_zs.md)。
- 2026-09-29 **因果ボコーダ [nvoc.md](nvoc.md)(PROPOSED・耳に出す数値条件に未達・耳はオーナー判断待ち)**: 最良 = 調波源なし・再構成のみ 160k(held21 logmel 0.286・PESQ 3.09・hf −1.61dB・倍音間コントラスト 3.18)。耳の較正は PESQ 2.96(Y-S1 不合格)〜4.10(BigVGAN 合格)。GAN 段はどの設定でも PESQ を下げた。因果・容量・調波源・16k/48k の折り返しは律速でない。Rust RTF 0.31。
- 2026-10-01 **ZS-VC [zsvc.md](zsvc.md)(REJECTED・耳 2 回「ガビガビ」)**: 声質(目標 ECAPA 0.50)・ピッチ(ずれ −0.3 半音)・漏れは数値で解決したが、出力部のフレーム周期 200Hz の包絡変調(+15dB・PESQ は盲目)が「ガビガビ」の正体。NAM A2 の音声レート原則から外れた構造の人工音。
