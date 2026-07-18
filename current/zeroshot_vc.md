# Zero-Shot Kansei VC フル詳細設計

> status: **PROPOSED**（未検証。昇格は overfit gate → 人間の耳ゲート通過時のみ）
> 最終更新: 2026-07-17
>
> 位置づけ: `current/README.md` §「B4 Dual-Path Kansei VC」/「SFRV」の**VC 前段**（content/prosody/target/style/texture → target-manifold 音響表現）を、**vocoder 解決後（freebig）の現実**に合わせて具体化したもの。合成段（vocoder）は `current/vocoder.md`（freebig, ほぼ ADOPTED 級）に委譲し、本書は**その上流だけ**を扱う。1ネットワーク集合=1ファイル。
>
> ルール継承（`CLAUDE.md` / README §1）: causal・E2E<50ms・CPU・Rust(Candle)・MIT・キメラ禁止（推論は100%自作 weights）・VC teacher 蒸留禁止・補助モデルは loss/eval/蒸留のみ・GAN は最後の texture のみ・判定は最終的に人間の耳・proxy 単独昇格禁止。

---

## 0. 本書の起点（このセッションで確定した3つの前提）

1. **合成段は解決した。** freebig（自作 Vocos 型 = ConvNeXt + 自由位相 ISTFT ヘッド、**F0 非依存・mel 入力**、28.8M、groups=1）を女声大コーパス（2776話者/41k 発話）で本気訓練し、**耳で BigVGAN 同等**に到達（`checkpoints/freebig/foundation_bigvgan_parity.pt`）。低遅延 config C（256/128, causal, **5.8ms**）は品質確定後に fresh 訓練予定。→ **本体差別化は VC 前段に移った**（[[freeuniv-universal-vocoder]]、`current/vocoder.md` §0）。
2. **インターフェースは mel。** freebig は mel → 波形の F0 非依存ボコーダ。したがって VC 前段の最終出力は **target 話者の mel スペクトログラム**であり、ここが前段と vocoder の ABI（sr/n_mels/win/hop は freebig の mel 仕様に固定）。**F0/ピッチ制御は全て mel を作る前段の責務**（vocoder は F0 入力を持たない）。
3. **ゼロショットの3分解と責務**（司令の整理を確定前提とする）:

   | 分布外の源 | 誰が吸収するか | 根拠 |
   |---|---|---|
   | **(a) 未知の元話者**（製品利用者＝任意の声） | **content encoder**（話者不変 content）＋ prosody 明示分離 | Phase A gate、C2-2 leakage、m2 perturb/GRL |
   | **(b) 未知/カスタムのターゲット萌え声** | **target 条件経路**（複数参照を Factor 分解 → 因子別ブレンド → zero-shot 埋め込み ＋ few-shot ハイブリッド） | §4, §5 |
   | **(c) mel 軌跡そのものの分布外** | **universal vocoder（freebig）** | 多話者ユニバーサル化で耳大改善＝汎化で吸収を実証 |

   → 前段が (a) を、埋め込み/微調整が (b) を、freebig が (c) を吸収する三層。**前段は「自然な target-manifold mel を作る」ことだけに集中でき、mel の細部不完全は universal vocoder が波形テクスチャで救う**（ただし muffle した mel は muffle して出る＝§8 リスク）。

---

## 1. 不可侵制約と、それが強制する設計

- **キメラ禁止 → 出荷グラフ = 自作 content encoder E + 自作 target encoder T/A + 自作 mel generator G + freebig vocoder V。** SSL(HuBERT)・ECAPA・WavLM-SV・ASR は**学習時の teacher/discriminator/perceptual/leakage 監督のみ**。推論に一切残さない。
- **F0 非依存 vocoder → F0 は前段の生成対象**。弱基音の萌え声で F0 追跡が原理的に不能（[[f0-octave-weak-fundamental]]）でも、**出力側 F0 は我々が生成する**（男声→萌えのレジスタ up）ので追跡不要。入力側（操作者男声＝基音が強い）と参照側のみ F0 を測り、オクターブ補正必須。
- **回帰は muffle（P1） → mel も音響ターゲット**。よって mel-L1 は warm-up の manifold 到達用に限定し、**crisp の最終供給は (i) content の時間微細構造 (ii) AdaIN の target 統計注入 (iii) CIPT の出力側監督 (iv) freebig 自身が GAN 学習済で mel→波形にテクスチャを足す**、で担う。これは vocoder が既に成功させた「Phase A 包絡回帰 warm-up → Phase B GAN」の同型（`current/vocoder.md` §3.5）＝**採用済みパターンの再利用であって P1 違反ではない**。
- **VC teacher 禁止 → target は実音声・同一内容ペア・信号処理由来のみ**。cross-identity 学習は CIPT（出力側 identity 監督、GT 不要）で獲得する（別 VC の変換音声を target にしない）。

---

## 2. システム全景（データフロー）

```text
operator wav (streaming, ~11.6ms hop, causal)
 │
 ├─ 自作 causal content encoder  E   ──► content c_t         (話者不変, retrieval-ready)   ← HuBERT 蒸留 teacher, C2-2/GRL/perturb で leakage 抑制
 └─ 明示 prosody 抽出                ──► F0_t, vuv_t, energy_t (操作者の演技 = 受肉の土台)

target 参照(数秒) or プリセット
 ├─ 自作 timbre encoder  T          ──► timbre code  z_spk    (静的音色 = 誰の声か)          ← ECAPA 蒸留 teacher
 └─ 自作 articulatory-style enc A   ──► s_art_t               (動的構音スタイル = 萌えの核)   ← formant 特徴で監督

prosody policy（§3.3）: 保持(受肉) ⊕ auto-moe shaping ⊕ GUI ツマミ
     F0_t, energy_t ──► F0'_t, energy'_t, register/liveliness

(c_t, F0'_t, energy'_t, z_spk, s_art_t) ─► 自作 causal mel generator  G   ──► target mel  m̂_t
     G の条件付けは AdaIN（単一・γ/β 加算統合, §3.4）。回帰非依存 + 敵対 + CIPT。
     m̂_t は freebig の mel 仕様(ABI)に厳密一致。

m̂_t ─► freebig universal vocoder V (F0非依存, 自作weights, config C causal) ─► 出力 waveform 44.1k
```

全ブロック causal・小型・自作 weights → Candle/CPU 移植可。B4/SFRV の「target-manifold latent generator → streaming decoder」を、**latent=mel / decoder=freebig** として具体化したもの。

---

## 3. 各ネットワークの入出力・設計

### 3.1 Content Encoder E（自作 causal, HuBERT 蒸留）

- **入力**: operator wav（causal フレーム、左寄せ窓）。**出力**: content c_t（話者不変、86fps 級）。
- **teacher**: **HuBERT（Apache-2.0）を第一候補**（WavLM=CC BY-SA の grey 回避、README §Concept 整合性）。蒸留で作る自作 weights は MIT 化可能。
- **leakage 抑制スタック**（採用実績、[[m2-clone-adain-grl]] / C2-2）:
  1. **content 摂動**（`perturb_content.py`, WORLD formant/pitch ±20%）＝ 残留話者情報を削る。単独で target 別分化が始まった実績。
  2. **GRL 話者敵対スクラブ**（不安定 → LayerNorm + grad clip(10) + alpha ramp(10k step) + 重み 0.3 必須）。
  3. **C2-2 gender/leakage 罰**（`train_gender_classifier.py` 資産）＝ content から gender が漏れないことを罰則化。
- **因果性の制約**: lookahead ≤ 1–2 frame（HN3 仮説の再掲）。**Phase A gate（下記）を通るまで generator を学習しない**（README 厳守）。
- **Phase A gate**（README §Phase A）: held-out で ASR/WER 保持・source speaker/gender 予測されにくい・timing/pause 保持・live 崩れない。h8/gate0 系ハーネス再利用。

### 3.2 Prosody 抽出（明示特徴）

- F0（オクターブ補正必須、[[f0-octave-weak-fundamental]]）、vuv、energy、rhythm、pause、attack/release。
- 目的: 操作者の演技を残し、source timbre は残さない（README Prosody Path）。**入力男声は基音が強く F0 追跡容易**＝ここでの F0 は信頼できる。

### 3.3 Prosody Policy（保持 ⊕ auto-moe ⊕ ツマミ）

README の 2 路線（受肉 vs auto-moe）を **M2 の耳確定に沿って統合**:

- **土台 = 操作者 prosody 保持（受肉）**。`self_prosody` ツマミで保持量。
- **上乗せ = auto-moe shaping**（README §M2 採用: 中立に喋っても萌え delivery を生成付与、ただし操作者の生 prosody に**上乗せ**して蚊帳の外にしない）。
- **GUI 連続ツマミ（焼き込まない, M2 要件）**: pitch register / liveliness（F0 変換、`render_m2.py` の register_st / exaggerate 実装済）、breathiness / softness / cuteness 強度 / tension。→ すべて G の条件入力 or mel 後処理として作用（`current/vocoder.md` §3.8 の包絡ツマミは freebig 段では持てない＝F0 非依存 mel 入力のため、**ツマミは前段 G の条件で実装**）。

### 3.4 Mel Generator G（自作 causal, AdaIN 条件付け）

- **入力**: content c_t（時系列, envelope 供給）+ prosody(F0'_t, energy'_t) + timbre z_spk + s_art_t。**出力**: target mel m̂_t（freebig ABI 準拠）。
- **本体**: causal TCN / ConvNeXt 系、**groups=1（XPU 安全）**、AdaIN 条件付け。
- **条件付けの確定則**（[[m2-clone-adain-grl]] / [[moe-articulatory-clone]] の hard-won lessons、踏むと再失敗）:
  - **加算 FiLM は clone 失敗**（source 音色統計が残り converted が target 非依存）。→ **AdaIN（instance_norm(x)·(1+γ)+β）必須**。instance_norm が source channel 統計（音色）を各層で強制除去し、G が z_spk を使わざるを得なくする。
  - **二段 AdaIN は打ち消す**（IN(IN(x)·(1+g)+b)=IN(x)）。→ **timbre と s_art は単一 AdaIN に γ/β を加算統合**（films + films_art、art=0 初期化で timbre 保存・warm-start クリーン）。
  - **s_art（動的構音）を効かせるには content を必要にする**: content が構音を全供給すると G は s_art を無視。→ content から構音統計を GRL で除去（articulatory-scrub、入力 utt 自身の formant 要約を回帰 adversary で）。萌え署名 = F1↑・F2_range↓・artic_dynamic↓・vowel_space↓（データ検証済、ICC で clone 可能な話者クセ）。
- **zero-init 禁止**（P-A 同型）: head bias = データ平均 log-mel、weight std 小。

### 3.5 Target Encoders T / A（zero-shot 埋め込み）

- **T（timbre, 静的音色）**: 数秒参照 → z_spk。ECAPA 蒸留 teacher（推論非依存）。speaker-aux CE（timbre code → 話者分類）で identity を強制注入（spk_ce が落ちる実績）。
- **A（articulatory-style, 動的萌えクセ）**: 参照の formant 軌道 → s_art_t（時変）。**AAI 的表現**。static timbre だけでは萌えの核（動的 gesture クセ）が欠落＝ここが frontier。
- **年齢感/明るさ/距離感/親密さ/ASMR comfort** 等は README Target Style Path 準拠で z_spk とは別に style 条件化（単一 ECAPA ベクトルに潰さない）。

---

## 4. ゼロショット戦略と few-shot fallback（ハイブリッド）

> **ターゲットの与え方の中核は §5「Disentangled 多参照・因子別ブレンド」**。本節はその土台となる zero-shot 抽出 / few-shot 定着 / retrieval fork を述べる。1参照から z_spk/s_art を抽出する「単参照」は、§5 の Factor×Reference 重み行列で参照数=1・全 Factor 同一参照の**特殊ケース**として包含される。

### 4.1 zero-shot（既定・即時）

- ターゲットは**数秒の参照音声から load 時に z_spk / s_art / prosody-style / aperiodicity 埋め込みを1回抽出**、per-target 学習なし。プリセット library は各 Factor 埋め込みをあらかじめ焼いた束（GUI プリセット）＝§5 の重み行列を含む。
- (a) は E が、(c) は freebig が吸収するので、zero-shot 経路で「未知操作者 → 未知ターゲット」が原理上成立する。

### 4.2 few-shot fine-tune（premium・任意、コンポーザ産物に対して）

- **本セッションの知見**: ターゲット声を訓練分布に入れると品質/**定位感**が改善（[[freeuniv-universal-vocoder]]: vocoder では「定位感のブレ＝汎化ギャップ、ターゲット声を学習に含めれば解消」を測定が支持）。freebig はユニバーサル化で天井到達したが、**前段 G の target likeness/定位感は few-shot で更に締まる**と読む。
- **設計**: ユーザが §5 のコンポーザで作った**合成ターゲット（Factor ブレンド後の条件束）を固定 target として** G を数千 step fine-tune（凍結不可＝CIPT A1 の soft-negative が示す通り凍結では identity を頑健に押せない）。**few-shot は「単一実声」でなく「コンポーザ出力＝自分だけの声」を焼く**のが本設計の眼目。必要なら freebig 側に軽量 LoRA を併用（キメラ禁止＝自作 weights の追加学習ゆえ規約内）。
- **UX（コンポーザ前提に拡張）**: (1) 複数参照を投入 → (2) Factor×Reference 重み行列を GUI で調整（§5.1）→ (3) zero-shot で即プレビュー → (4) 気に入ったら「この声を登録」で数分バックグラウンド few-shot、プリセット化。合成声は重み行列＋参照埋め込みのスナップショットとして保存（再現可能）。

### 4.3 retrieval を runtime に置くか（fork, 既定は置かない）

- README §B4 HN1 は soft differentiable retrieval（学習 cross-attention over target pool）を提案。**ただし [[frontier-own-no-existing-algo]]（ユーザ厳命: kNN-VC 等の外部手法に乗らない）と、mel-generator + AdaIN + CIPT が既に耳で前進している事実**から、**既定は非 retrieval（埋め込み条件）**。retrieval は「参照が長い/多様な premium ケース」の研究アーム（HN1）として保留し、採用は耳 AB で hard-kNN 連結の境界 rough を超えた時のみ。

---

## 5. Disentangled 多参照・因子別ブレンド（自分だけの声コンポーザ）

**要件（ユーザ新要件）**: ターゲットを「1参照から選ぶ」でなく、**複数参照音声を萌え Factor に分解し、Factor ごとに重み配分して再合成**する。UI メンタルモデル ＝「参照 A の息70＋B の息30、音色は A50＋C50、構音は B100…」の **Factor × Reference 重み行列**。これは §3.5 の target 経路（T/A/style）が既に**因子分離された埋め込み**を出す設計だからこそ成立する（単一 ECAPA に潰さない、を守った直接の配当）。

### 5.1 Factor 集合の確定と採用資産へのマッピング

各 Factor は「**参照から独立抽出可能**」かつ「**独立に再合成し G の別々の条件点に注入可能**」でなければならない（さもなくば §5.3 の相互漏れで破綻）。

| Factor | 抽出器（参照→埋め込み） | G への注入点 | 対応採用資産 |
|---|---|---|---|
| **timbre（音色・誰の声か）** | T → z_spk（静的） | 単一 AdaIN の γ/β（音色統計） | [[m2-clone-adain-grl]]（AdaIN 決定打） |
| **articulation（構音クセ＝萌えの核）** | A → s_art_t（時変） | 同一 AdaIN に γ_art/β_art を**加算統合** | [[moe-articulatory-clone]] |
| **breath / aperiodicity（息・非周期・ASMR）** | Br → a_emb（帯域 aperiodicity/HNR 要約） | prosody/energy 経路 + G の noise/breath 条件 | 明示 prosody path、ROADMAP 低レベル一級市民化 |
| **register / 抑揚（prosody delivery）** | Pr → f0-style（F0 range/contour 統計） | prosody policy（受肉土台に補正） | `render_m2.py` register_st/exaggerate、§3.3 |
| **texture（質感・last-mile）** | （参照からは弱条件、主に GAN が担う） | Phase E GAN + freebig | P3、§8.1 |

- **timbre と articulation は同一 AdaIN に γ/β 加算統合**（§3.4 の「二段 AdaIN は打ち消す」則を厳守。Factor が増えても注入は単一 AdaIN への加算で束ねる）。
- **breath/register は prosody 経路**＝AdaIN とは別注入点なので、そもそも AdaIN 系（timbre/art）と直交して混ざりにくい（構造的 disentangle の下地）。
- **texture は参照ブレンド対象にしない**（GAN 最終段の責務、参照から安定抽出できない）。UI では固定 or asmr_comfort ツマミに委譲。

### 5.2 ブレンド数理（Factor 内加重・跨 Factor 独立）

- **重み行列 W**: `W[factor, ref] ≥ 0`、**各 Factor 列で正規化 `Σ_ref W[f,·] = 1`（UI 表示は和=100）**。「トータル100を参照間に配分」の意味論＝**各 Factor は参照集合の凸結合**。
- **Factor 内ブレンド（表現に応じた合成）**:
  - **timbre**: AdaIN 統計（γ,β = 各参照の channel mean/std）の**加重平均**（`γ_blend = Σ w_r γ_r`）。AdaIN は統計注入なので統計の凸結合が自然（[[m2-clone-adain-grl]] の PoC は AdaIN 版が F1 を 8倍強く制御＝統計空間が意味的に線形に近い傍証）。
  - **articulation**: s_art 埋め込みの加重平均（時変ゆえ話速アライン後）。
  - **aperiodicity/breath**: a_emb の加重平均（帯域別 HNR/ap は加算的で平均が物理的に妥当）。
  - **register**: F0 統計（log-F0 の平均/レンジ）を加重平均（log 領域で線形補間＝ピッチは対数知覚）。
- **正規化の意味論と外挿**: 凸結合（和=1・非負）が既定＝**参照の内挿のみ**＝知覚的に安全側。**負値/和≠1 の外挿は既定禁止**（分布外 mel → §8 muffle/破綻リスク）。外挿は「誇張ツマミ」として別途 clamp 付きで研究アーム化（例 breathiness を参照最大より上へ、は §3.3 の GUI ツマミ側で範囲制限して実装。参照ブレンド行列では持たない）。
- **統合点**: 各 Factor をブレンド → §3.4 の単一 AdaIN（γ/β 加算）＋ §3.3 の prosody 注入点へ。**G から見れば「1つの合成ターゲット条件束」**であり、参照が何本かは不可視（few-shot でこの束を焼ける、§4.2）。

### 5.3 Disentanglement 要件（ブレンド独立性の担保）

跨 Factor 独立が**本丸**。漏れがあると「息を変えたら音色も動く」＝コンポーザが破綻する。既存の leakage 抑制資産を**Factor 間漏れ抑制**として再目的化する:

- **C2-2 gender/leakage 罰・GRL**（[[m2-clone-adain-grl]] / `train_gender_classifier.py`）: content から話者/gender を抜くだけでなく、**各 Factor 埋め込みが「担当外の情報」を持たないことを罰する**。例: a_emb（breath）から timbre が予測できたら罰（相互情報最小化 / cross-adversary）。
- **s_art scrub（articulatory-scrub）**: content から構音統計を GRL 除去（[[moe-articulatory-clone]]）＝ articulation Factor を content・timbre から独立させる。
- **訓練時 Factor swap 正則化**: 同一発話内で Factor を別参照のものに差し替えても他 Factor 出力が動かないことを consistency loss で強制（§8 の disentanglement gate を学習信号化）。
- **AdaIN の構造的分離**: instance_norm が各層で source 統計を除去するので timbre/articulation は統計空間で加算分離。breath/register は別注入点で構造的に直交。

### 5.4 コンポーザと責務3分解の整合

- コンポーザは **(b) 未知/カスタムのターゲット萌え声**の与え方の一般形。(a) 元話者は E が、(c) mel OOD は freebig が引き続き吸収。
- zero-shot（§4.1）= 重み行列を即時反映しプレビュー。few-shot（§4.2）= 行列確定後の合成束を焼く。retrieval fork（§4.3）とは独立。

---

## 6. 学習方式（実音声のみ・VC teacher 不使用）

段階は README の Gate 0 → Gate 1 → Phase A → B → C → D → E に従属。以下は**本前段固有**の loss 設計。

### 6.1 自己再構成（Phase B: target manifold autoencoding）

- X について c=E(X), p=prosody(X), z=T(Xref), s_art=A(Xref) → G → m̂ が X の mel を再構成 → freebig → 波形。
- loss = mel-L1（**warm-up 限定**, manifold 到達）+ mel-domain 敵対（Phase E texture）。**mel-L1 は anneal**（vocoder Phase A→B の用量反応が「純回帰 55% > floor > anneal→0」だった教訓の裏返し＝**GAN 併走時は mel アンカーを lc=False で残す**、`current/vocoder.md` §3.5 の ganlc0 教訓）。

### 6.2 Cross-identity（Phase C: CIPT、SECS 天井の突破機構）

- **CIPT = Cross-Identity Perceptual Training**（[[cipt-cross-identity-plan]]、耳で確定・`train_cipt2.py` / `cipt_w5_final.pt`）:
  - male content → female z_spk の**実推論分布**を生成し、**生成波形の ECAPA 類似度を z_spk 側へ微分逆伝播**（GT 不要ゆえ同一内容ペア不要）。
  - **構音固定**: ContentVec/ASR 一致（何を喋るか）。
  - **cross 出力を GAN の fake に入れて実音声多様体に留める**（ECAPA 敵対的騙し回避）。
  - **女性 self-recon で品質担保**。
  - **G は fine-tune（凍結不可）**。凍結 front-end（A1）は soft-negative、decoder fine-tune（A2）で SECS 0.5 天井を +0.13 突破・耳「本人ぽい」。
- **評価は別 SV（WavLM-SV）held-out ＋ 耳**（SECS は動かす proxy 限定、高 SECS でも悪く聴こえた前例）。

### 6.3 補助監督（推論非依存）

- HuBERT（content 蒸留）、ECAPA（timbre 蒸留 + CIPT identity）、formant（s_art 監督）、gender classifier（C2-2 leakage 罰）、BigVGAN mel / MPD・MRD（texture GAN）。全て loss/eval/蒸留のみ。

### 6.3b Factor disentanglement loss（§5.3 のブレンド独立性を学習信号化）

- **相互情報最小化 / cross-adversary**: 各 Factor 埋め込み（z_spk / s_art / a_emb / f0-style）から「担当外 Factor」が予測できないことを罰する（例 a_emb→timbre 予測 adversary）。
- **Factor swap consistency**: 同一発話で1 Factor だけ別参照に差し替え → 他 Factor の G 出力寄与が不変であることを consistency loss で強制（§8 disentanglement gate の学習版）。
- これらは §5.3 の C2-2/GRL/s_art scrub と同族で、**「息を変えたら音色も動く」破綻を訓練時に潰す**。単独実験で導入（1実験1仮説）。

### 6.4 augmentation（ROADMAP §2.3 準拠）

- 入力/条件側のみ摂動（F0/formant/SR で speaker 落とし、srcshift で cross-gender F0 レジスタ）。**target mel（G が描くもの）は汚さない**（SpecAugment/time-mask 禁止＝P1 平滑化圧力）。TTS で実音声を水増ししない。

---

## 7. レイテンシ内訳（E2E <50ms, causal, CPU）

| 要素 | lookahead / 遅延 | 備考 |
|---|---|---|
| 入力デバイスブロック | 2.9–11.6ms | |
| content encoder E（causal 蒸留） | 0–23ms（B4 予算） | 最大の可変。HuBERT 蒸留の LA を実測で詰める（最難、G-enc） |
| prosody 抽出（F0 左窓, causal 補間） | 0 先読み | 応答遅れのみ（出力遅延に非加算） |
| mel generator G（causal, hop 集約） | ~11.6ms | groups=1・AdaIN、計算は軽い |
| **freebig vocoder V（config C, causal）** | **5.8ms** | 実測（256/128）。品質 config は 46.4ms なので config 選択がレバー |
| 計算 wall-clock | 2–5ms | CPU RTF 実測 0.046（vocoder 単スレ 22倍速）＝計算は余裕 |
| 出力デバイスブロック | 2.9–11.6ms | |
| **合計** | **~25–45ms（タイト）** | 支配項 = content encoder LA と I/O ブロック。**壁は flops でなく因果性**（E7） |

- **causality CI（必須ゲート）**: 時刻 T 以降入力ランダム化 → T 以前出力ビット一致（center=True 混入の前科を構造検出）。streaming ≡ offline パリティ。torch ≡ Candle ≤1e-4。
- ボトルネック順: **content encoder LA（最難・G-enc で蒸留）> I/O ブロック > mel hop**。vocoder は解決済（config C=5.8ms）。計算量は全経路で余裕。

---

## 8. リスク・未解決・反証すべき仮説（RESEARCH 作法）

0. **跨 Factor 独立の未達（コンポーザの make-or-break）**: ブレンド独立が本丸。漏れがあると「A の息＋B の音色」を作ったつもりが息を上げると音色も動く＝コンポーザが商品として破綻。→ §8 disentanglement gate（単一 Factor swap で他 Factor が動かないか）が最優先ゲート。未達なら Factor 数を絞る（timbre+breath の2軸から）か、注入点の直交性を上げる。**任意組合せの知覚的破綻（A の息＋B の音色が生理的に不自然）は disentangle が完璧でも起こりうる → Kansei 耳ゲートで弾く前提**（機械は独立性を保証、自然さは人間が判定）。凸結合（内挿のみ）に限る設計もこの破綻確率を下げるための安全策。
1. **mel 中継の muffle（最重要リスク）**: universal vocoder は (c) を吸収するが、**muffle した mel は muffle して出る**（vocoder は入力 mel に忠実）。→ G が crisp な mel を作れるかが天王山。反証実験: G の mel を freebig に通した出力 vs GT を耳 AB。mel-L1 単独で muffle するなら CIPT/GAN の用量を上げる（vocoder Phase A→B の用量反応と同型で監視）。
2. **AdaIN と CIPT の勾配衝突**: CIPT は decoder(=G) fine-tune 前提だが、AdaIN の instance_norm が identity 勾配を殺す可能性。→ 単独実験で ECAPA-through-vocoder 勾配が G に届くか計測（届かなければ multi-scale/重み強化、CIPT リスク3）。
3. **ECAPA-through-freebig の微分経路**: CIPT は「生成波形の ECAPA」を逆伝播＝freebig も微分経路に入る（凍結でも forward-diff）。XPU/メモリで重い可能性 → mel 段 identity proxy（mel→ECAPA 近似 head）で近似する fallback を用意。
4. **s_art の zero-shot 汎化**: 動的構音クセを数秒参照から安定抽出できるか未検証（PoC は PCA24 content 圧縮下で成立、フル content では GRL scrub が必須）。反証: articulatory-scrub なしで s_art ablation が効くか。
5. **content encoder の因果蒸留（G-enc, 最難・未着手）**: HuBERT-L6 級の非因果 content を lookahead ≤2frame の causal student に蒸留して retrieval/leakage を保てるか。ここが崩れると全体が崩れる。**先に G-enc gate を単独で通す**（ROADMAP クリティカルパス）。
6. **few-shot fine-tune の破壊リスク**: G の数千 step fine-tune が汎用性/他プリセットを壊さないか（過学習）。→ held-out プリセットで cross-axis drift 監視、LoRA 化で本体保護。
7. **auto-moe が「pitch-shift した男声」に退行**: M2 の耳確定＝音色変換+pitch では萌えにならない。s_art（構音）が本当に効くかが分岐点。効かなければ受肉（self_prosody 高）に倒す安全策。
8. **proxy と耳の乖離（前科多数）**: SECS/mel-L1/band-MAE は本プロジェクトで何度も耳と乖離。**昇格は耳のみ**、proxy は機構特定用。

---

## 9. 次の検証実験（安い順・1実験1仮説・各ゲート永続化）

前提: **vocoder が耳ゲートを通っている今、初めて上流に投資してよい**（ROADMAP クリティカルパス: vocoder → 多話者+timbre → G-enc → 結線 → realtime）。

1. **Z0 mel-recon 疎通（最安）**: 既存 target 話者の実 mel を freebig に通し耳で透明を再確認（ABI 固定）。→ G の出力 mel が freebig で crisp に出る上限を係留。
2. **Z1 G 単体自己再構成（Gate 1/Phase B）**: c=E(X) 暫定（既存 content 特徴で可）→ G(AdaIN) → freebig。**聴く**: AdaIN clone が target 音色を追随し muffle しないか。m2 の centroid proxy で分化確認 → 耳。
3. **Z2 CIPT cross（Phase C）**: `train_cipt2.py` 資産を freebig-mel 経路に接続、male→female の出力 ECAPA を逆伝播。**聴く**: 「本人ぽい」再現＋SECS 天井突破（別 SV held-out）。**リスク2/3 を先に単独計測**。
4. **Z3 s_art（moe 構音, frontier）**: articulatory-scrub + s_art AdaIN 統合。**聴く**: 中立男声が pitch-shift 男声でなく萌えになるか（M2 の分岐点）。
5. **Z4 disentanglement gate（コンポーザの前提・最優先, 客観+耳）**: 単一 Factor swap（breath だけ参照 A→B、他固定）→ **他 Factor の出力寄与が動かないか**を計測（timbre 統計 centroid・s_art formant 署名・a_emb HNR を proxy、最終は耳）。動くなら §6.3b disentanglement loss を単独投入して再測。**跨 Factor 独立が未達なら Z6 に進まない**。
6. **Z5 2参照補間の知覚連続性**: 1 Factor を A100→A50B50→B100 と掃引し、出力が知覚的に連続に遷移するか（凸結合の妥当性）。飛び/破綻があれば埋め込み空間の非線形性 → 補間法（log/球面）見直し。
7. **Z6 Factor 重み行列の耳 AB**: 複数 Factor を同時ブレンドした「合成声」を golden 操作者で生成、プリセット間で officer が知覚差を得るか・不快方向に倒れないか（README GUI 採用条件準拠）。任意組合せの生理的不自然さを Kansei 耳ゲートで検出。
8. **Z7 zero-shot 未知ターゲット & few-shot**: 品質 vs 参照長、未知話者。コンポーザ合成束の few-shot 定着で定位感改善を耳 AB（zero-shot vs +数千 step）。
9. **Z8 G-enc（content 因果蒸留）**: HuBERT → causal student、retrieval-match + leakage を lookahead 依存で gate（最難、Phase A）。
10. **Z9 realtime**: causal 化・streaming≡offline・Candle パリティ・RTF・block jitter 実測（E4/M3）。

各段は既存計測系（`kansei_proxies.py` / `gate0_*` / `annot_gui.py` / `listen_gui.py` / `h8_retrieval.py`）を流用。**機械=アーティファクト検出（disentanglement 独立性含む）、人間=官能評価のみ**（ROADMAP §3 判定者分担）。

---

## 10. 既存設計との索引整合

- **合成段**: `current/vocoder.md`（freebig）。本書は vocoder を凍結資産（+任意 few-shot LoRA）として consume するのみ。旧 NSF-LTV 系譜・F0 駆動 harmonic は**再開しない**（A/S 天井死亡、[[as-source-filter-ceiling]]）。
- **B4/SFRV**（README §B4）: 本書は SFRV の HN2 スロット（decoder）を freebig に、HN3（content）を §3.1 に、HN1（retrieval）を §4.3 fork に写像。矛盾なし。
- **採用資産の再利用**: AdaIN clone（[[m2-clone-adain-grl]]）、articulatory s_art（[[moe-articulatory-clone]]）、CIPT（[[cipt-cross-identity-plan]]）、C2-2 gender leakage、`render_m2.py` ツマミ。
- **多参照コンポーザ（§5）**: §3.5 の因子分離 target 経路（timbre/articulation/breath/register を単一 ECAPA に潰さない）を前提に成立。leakage 抑制資産（C2-2/GRL/s_art scrub）を Factor 間漏れ抑制に再目的化。texture は参照ブレンド外（GAN 最終）。
- **踏まない negative results**: 加算 FiLM / 二段 AdaIN / code 監督のみ / kNN-VC 外部モデル runtime / DAC latent 線形 delta / mel への純回帰 / F0 駆動 vocoder / target 側 SpecAugment。
