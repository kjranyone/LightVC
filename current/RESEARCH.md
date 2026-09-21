# 研究中

> 更新: 2026-09-20（第11次）。現在地・未解決・次の実験のみを記載。
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

## 2. 現在の問い

**F0権威をlf0条件に戻しつつmel条件の包絡利得を維持できるか（F0シフト増強で「lf0が権威・melは包絡」を学習させる）。**

2026-09-19の流れ: 耳ゲート「ロボットボイス」→包絡スメアと診断→K/rho/content源/容量(s9)/decode補助損失(s10)を全否決→**s11のmel_in条件で包絡解決**(根因=ContentVecの包絡欠落を確認)→**F0制御が消滅**(mel倍音がlf0を上書き、粗いmelワープは不可)。クロススペーカー話者ターゲット(content leak)は未解決のまま。詳細と証跡は [cfm_ys1.md](cfm_ys1.md) §4–6。

残る未解決: F0権威と包絡の両立、話者ターゲット(content leak)、男声content被覆、CER 0.33 vs 天井0.049、ABI統計の男声17%混入、crop左文脈なしと起動状態の差、AR(1) noiseのchunk横断state、K=8の生成器レイテンシ実測。

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
