# GUI 設計 — 萌バ美声リアルタイム VC

> status: PROPOSED
> network: n/a（アプリUI設計）
> 最終更新: 2026-07-17（Voice Compositor 要件を追加）
> 対象クレート: `crates/lightvc-app`（egui / eframe 0.34）, `crates/lightvc-audio`
> 前提: `current/README.md`（kNN-VC 型・受肉路線・Kansei 6 軸）を読んだ上での UI 設計。

本書は「実装済み egui GUI」を土台に、ASMR・官能バ美肉リアルタイム VC プロダクトとして
Beatrice / paravo と戦える操作系へ発展させる設計を提案する。**コードは未変更**。既存を
残す/変える/足すを明示し、実装優先順位まで示す。

---

## 0. 現状の把握（実コード基準）

### 0.1 タブ構成と描画フロー
`app.rs::render()` が ctx レベルで以下を積む。

- **Top panel** `tabs`: ロゴ画像 + タブ3つ `Offline / Realtime / Voices` + 右端に設定ギア `=`。
- **Bottom panel** `status`: ステータスドット + メッセージ + 右端に Output レベルメータ。
- **Settings `egui::Window`**: 入力/出力デバイス選択のみ（`selected_input/output: Option<usize>`）。
- **Central panel**: `current_tab` により Offline / Realtime / Catalog を描画。
- 起動時スプラッシュ（~0.5s）、kawaii ネオンテーマ（`theme.rs`）。

タブは `enum Tab { Offline, Realtime, Catalog }`、初期タブは `Offline`。

### 0.2 状態管理
- `AppState`（`Arc<Mutex<>>` 共有）: dac/converter パス, `pipeline: Option<Arc<Mutex<Backend>>>`,
  hot-swap 用 `pipeline_slot: Arc<Mutex<Option<Arc<Mutex<Backend>>>>>`, `voices: Vec<VoiceEntry>`,
  `selected_voice`, `status/error`, `offline_result`, RT チャネル, デバイス選択。
- **RT スレッド**: `ensure_rt_thread_static` が `inference_loop` を spawn。UI とは
  `crossbeam_channel`（`RtControl` 送信 / `RtMetrics` 受信）で通信。converter 無しでも
  bypass で常駐（audio path 単体テスト可）。
- `RtControl`: `StartWithDevices / Stop / SetMode / SetProsody / SetVelocityScale / Bypass /
  LoadReference / SetB1Timbre / SetB1Tau / SetWetDry`。
- `RtMetrics`: `input_rms / output_rms / latency_ms / rtf / disconnected / overrun / underrun /
  current_mode / auto_degraded`。
- `Backend`: `Legacy(VcPipeline) | B1(B1Streaming)`。

### 0.3 各タブの現状 UI
- **Realtime** (`realtime_tab.rs`):
  - Row1 2カラム: Model カード（drop zone + config DragValue 群 + Load）｜ Status カード
    （status badge / latency ms / RTF / overrun:underrun）。
  - Row2 Input レベルメータ（auto-degrade 警告付き）。
  - Row3 操作バー: `Bypass` トグル / モードピル `Strict・Balanced・Quality` / `Start・Stop`。
  - Row4 Prosody: `ProsodyMode` combo（Imitate/Preserve/Blend/Flatten） + blend slider + velocity slider。
  - 末尾 collapsing「B1 Adapter (UTTE)」: adapter/quantizer/timbre パス, Load, Tau, Wet/Dry。
- **Offline** (`offline_tab.rs`): Source / Reference ファイル選択＋Play、Voice catalog quick-pick
  ピル、Prosody（mode+blend）、Conversion Strength（velocity）、Convert CTA、Output（Play/Save As）。
  `process_full` で Python parity 変換。
- **Voices** (`voice_catalog.rs`): Add Voice（Name + Browse）、リスト（index/name/path,
  Play・Use・Remove）、Import/Export（JSON: name+path）。

### 0.4 オーディオ I/O 能力（`lightvc-audio`）
- `AudioEngine::start(in_dev, out_dev)` と `start_with(..., buffer_size: cpal::BufferSize)`。
  **buffer_size は UI 未露出**（常に既定）。capture/playback SR を個別保持し個別リサンプル。
- `overrun_count / underrun_count / is_disconnected`。
- `supported_input_configs / supported_output_configs` あり（未使用）。
- **ASIO は feature `asio = ["cpal/asio"]` のみ存在**、ホスト選択 UI・feature 導線なし。

### 0.5 テーマ資産（`theme.rs`, 流用可能な既製ウィジェット）
`knob / knob_labeled`（縦ドラッグ 0..1, ダブルクリックで 0.5 リセット、未使用）、`level_meter_kind(_compact)`、
`stat_card`、`pill_button`、`operation_button`、`status_badge`、`drop_zone`、`info_card / glow_card / cyan_card`、
`primary_button / tab_button / icon_button`。**Kansei ノブに必要な knob 資産は既にある**。

### 0.6 現状の欠落（プロダクト要件とのギャップ）
| 要件 | 現状 | ギャップ |
|---|---|---|
| バッファサイズ (256等) | 露出なし | **無し** — Beatrice/paravo の中核 UX |
| ASIO 選択 | feature のみ | **ホスト/デバイス UI 無し** |
| 実測レイテンシ内訳 | `10+3+algo+3+10` 定数 | **推定値のみ・内訳非表示** |
| CPU 負荷 | RTF のみ | CPU% 無し（RTF で代用可） |
| Mute / Passthrough | Bypass のみ | Mute（無音出力）と Passthrough を分離してない |
| プリセット萌え声 | file+name のみ | **プリセット library / few-shot 登録フロー / お気に入り 無し** |
| Kansei ノブ | prosody+velocity | **pitch/formant/register/breath/style/texture 無し** |
| 官能 AB 評価導線 | Offline に散在 | **AB 比較・評価記録 無し** |
| 初心者/上級者両立 | フラット3タブ | ワンクリック開始導線が弱い |

---

## 1. 設計原則

1. **既存資産を壊さない**: `AppState`・`RtControl`・`inference_loop` の3層構造と `pipeline_slot`
   hot-swap を土台に、コントロールは `RtControl` バリアント追加で拡張（スレッド構造は不変）。
2. **受肉が主・自動萌えが従**（README 準拠）: 操作者が萌えを演じ、VC は timbre/style を変える。
   UI は「演技を保つ（self prosody）」を既定に、Kansei ノブは *補正* として置く。
3. **耳が最終審**（CLAUDE.md Kansei ゲート）: すべての品質操作の隣に「聴く・比べる・記録する」導線。
   proxy 数値（RTF・レベル）は副次表示。
4. **1画面2モード**: 初心者は Live 画面の CTA 一つで配信開始、上級者は同画面の展開で
   バッファ・レイテンシ内訳・Kansei を触れる（モード分割ダイアログにしない）。
5. **低遅延を可視化して交渉可能に**: 遅延は「隠す数値」でなく「操作者が品質と取引する軸」。

---

## 2. 情報設計（タブ再編）

現行3タブを **4タブ + グローバル Transport バー** に再編。`Tab` enum を拡張する。

```
enum Tab { Live, Voice, Kansei, Studio }   // 旧 Realtime→Live, Catalog→Voice, Offline→Studio
```

- **Live**（旧 Realtime）: リアルタイム変換の運転席。初心者はここだけで完結。
- **Voice**（旧 Voices を発展）: プリセット + ユーザ few-shot 登録 + お気に入り + 検索。
- **Kansei**（新規）: 官能ノブ + プリセット保存 + AB 官能評価。Live からノブだけ抜粋表示も可。
- **Studio**（旧 Offline を発展）: オフライン変換 + gt/変換/target 3系統 AB + 評価記録。

**グローバル Transport バー**（Top panel を拡張）: どのタブでも常時見える運転状態。
```
┌────────────────────────────────────────────────────────────────────┐
│ [LOGO]  Live  Voice  Kansei  Studio        ● CONVERTING  46ms  =    │
│                                            ▶Start ⏸Bypass 🔇Mute     │
└────────────────────────────────────────────────────────────────────┘
```
Start/Bypass/Mute とレイテンシ・状態ランプを常時トップに出し、タブ移動しても運転が切れない。
（現行は Realtime タブ内にのみ Start/Stop があり、他タブに行くと操作不能。これを解消。）

---

## 3. Live タブ（リアルタイム運転席）

### 3.1 ワイヤフレーム
```
┌─ LIVE ─────────────────────────────────────────────────────────────┐
│ ┌─ Voice ───────────────┐ ┌─ Signal ───────────────────────────┐   │
│ │ ★ Venus (warm)      ▼ │ │ IN  ▁▃▅▇▅▃  -12dB                   │   │
│ │ [preset]  Use…        │ │ OUT ▁▃▅▇█▅  -9dB                    │   │
│ └───────────────────────┘ │ CPU ███░░ 34% (RTF 0.34)           │   │
│                           └────────────────────────────────────┘   │
│ ┌─ Latency ──────────────────────────────────────────────────────┐ │
│ │  ⟨ 46 ms ⟩ E2E        Quality ◀──●──▶ Speed                    │ │
│ │  buf 10 │ resamp 3 │ algo 20 │ resamp 3 │ buf 10   (ms)         │ │
│ │  [ Buffer 256 ▼ ]  Strict · (Balanced) · Quality               │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ ┌─ Quick Kansei ─────────────────────────────────────────────────┐ │
│ │   (Pitch)   (Formant)  (Breath)   (Air/ASMR)   [more → Kansei]  │ │
│ │     ◔          ◑          ◔           ◕                          │ │
│ └────────────────────────────────────────────────────────────────┘ │
│              ▶  START STREAMING          ⏸ Bypass   🔇 Mute          │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 コンポーネント
- **Voice セレクタ**（左上）: 現在の target 声。ドロップダウンでお気に入り即切替
  （`RtControl::LoadReference` を送信、既存導線を流用）。★でお気に入り。
- **Signal カード**: IN/OUT レベル（既存 `level_meter_kind_compact` 流用）+ **CPU バー**。
  CPU は当面 RTF から算出（`cpu_pct ≈ rtf * 100`、`RtMetrics.rtf` 既存）。将来 process 実測に置換。
- **Latency カード**（新規・要件⑤の核）:
  - E2E 実値（`RtMetrics.latency_ms`）を大きく表示。
  - **内訳バー**: `buf | resamp | algo | resamp | buf`（現行 `latency_ms_last = 10+3+algo+3+10` を
    分解して送る）。algo は `algorithmic_latency_ms()`。
  - **Quality↔Speed スライダ**: LatencyMode `Strict/Balanced/Quality` をスライダ化（内部は既存3値。
    将来連続 lookahead に拡張余地）。ピルも残し二重操作可。
  - **Buffer サイズ combo**: 128/256/512/1024。`RtControl::SetBufferSize(u32)` 新設 →
    `inference_loop` で `AudioEngine::start_with(cpal::BufferSize::Fixed(n))`。**再起動が要るので
    Start 前に設定 or 変更時に自動 re-arm**。auto-degrade（既存 `[07-5]`）が発火したら Speed 寄りへ
    自動移動しトースト表示。
- **Quick Kansei**（要件⑥初心者導線）: Kansei タブの上位4ノブ（Pitch/Formant/Breath/Air）を
  抜粋。`theme::knob` 流用。`[more → Kansei]` で全ノブへ。
- **Transport**: START（大 CTA）/ Bypass / **Mute**（新規: 出力無音、`RtControl::Mute(bool)`。
  Bypass=素通し と Mute=無音 を分離。配信中の「ちょっと黙る」に必須）。

### 3.3 既存からの差分
| 項目 | 対応 |
|---|---|
| Model カード（drop/config） | **Live から撤去 → Studio/設定送り**。運転席に生モデル設定は出さない |
| B1 Adapter collapsing | **設定 or 開発者パネル送り**。運転席から隠す |
| Status カード | Transport バー + Signal カードへ分解 |
| latency 定数表示 | 内訳分解して送る（RtMetrics 拡張、下記5章） |
| Buffer/Mute | **新規 RtControl 2つ追加** |

---

## 4. Voice タブ（萌え声の管理・要件②）

### 4.1 ワイヤフレーム
```
┌─ VOICE ────────────────────────────────────────────────────────────┐
│  🔎[ search…        ]   Filter: ★Fav  Preset  Mine  ZeroShot        │
│ ┌─ Presets ──────────────────────────────────────────────────────┐ │
│ │ [Venus♀warm] [Mars♀bright] [Lyra♀soft] [Whisper] [ASMR-close]  │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ ┌─ My Voices ────────────────────────────────────────────────────┐ │
│ │ ★ 1  MyTarget-A     ▶Play  Use✔   …3 refs   tag: sweet   🗑     │ │
│ │   2  MyTarget-B     ▶Play  Use    …1 ref                🗑     │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ ┌─ Register (few-shot) ──────────────────────────────────────────┐ │
│ │ Name[__________]  + Drop 3–10s refs ▏[wav][wav]  ▶preview       │ │
│ │ ○ Record from mic 8s  ●REC                                      │ │
│ │ [ Analyze & Register ]   ✓ quality: good (SNR ok, 6.2s)         │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### 4.2 発展点（`voice_catalog.rs` を土台）
- **プリセット library**: 同梱の萌え声プリセットを別セクション表示。`VoiceEntry` に
  `kind: {Preset, User, ZeroShot}` と `favorite: bool`、`tags: Vec<String>` を追加。プリセットは
  読み取り専用（Remove 不可）。
- **few-shot 登録フロー**: 現行「1ファイル+名前」を **複数参照 drop（3–10s ×数本）+ 品質チェック
  （長さ/SNR/無音率を簡易判定して緑/黄表示）+ Analyze & Register** に発展。zero-shot 設計が
  参照プール（README kNN プール precompute）を要求するため、登録時に埋め込み/プールを事前計算し
  `VoiceEntry` にキャッシュパスを持たせる（設計フック。現段階は参照 wav 束の保存で足りる）。
- **マイク録音登録**: `AudioEngine` capture を流用し 8s 録音 → その場で参照化。
- **お気に入り/検索/タグ**: ★トグル、検索ボックス、フィルタピル。Live のドロップダウンは★のみ表示。
- **切替**: 各行 `Use` は既存 `on_select`→`LoadReference` を流用（変更不要）。

### 4.3 既存からの差分
残す: リスト構造・Play/Use/Remove・Import/Export。
足す: プリセット節 / 複数 ref 登録 / 録音 / お気に入り / タグ / 検索 / **Voice Compositor（§4.4）**。
変える: `VoiceEntry` フィールド拡張、Import/Export JSON にフィールド追加（後方互換で読む）。

---

## 4.4 Voice Compositor（多次元・多参照 声ブレンダ・新要件）

### 4.4.1 コンセプト
Beatrice / paravo は「単一ターゲットを選ぶ」だけ。LightVC は **複数の参照音声から Factor 単位で
成分を抽出し、Factor ごとに重みを配分して「自分だけの声」を合成**する。操作者のメンタルモデルは
「トータル 100 として、参照 A のブレス 70 + B のブレス 30、声質 A50 + C50、構音 B100…」。
**声を成分から作れること自体が競合に対する明確な差別化**（§8 に反映）。

受肉路線との整合: Compositor が合成するのは **timbre / texture / breath など声の"素材"**であり、
萌えの本体である delivery（抑揚・語尾・間）は引き続き操作者の演技（`ProsodyMode::SelfPreserve`）と
Kansei ノブ側で扱う。Compositor = 「どんな声か」を作る、Kansei = 「その声をどう鳴らすか」を整える。

### 4.4.2 ワイヤフレーム（Factor × Reference 重みマトリクス）
```
┌─ VOICE ▸ Compositor ───────────────────────────────────────────────┐
│  Composing: [ MyBlend-01 ]  ▶Preview(live)  [Save preset]  ★        │
│  References: (A) Venus  (B) MyRec-2  (C) Lyra   [+ Add reference]    │
│ ┌───────────────┬────────┬────────┬────────┬─────────────────────┐ │
│ │ Factor \ Ref  │  A     │  B     │  C     │  Σ (auto = 100)     │ │
│ ├───────────────┼────────┼────────┼────────┼─────────────────────┤ │
│ │ Breath / Air  │ �◨70    │ ▨30    │ ·0     │ ▓▓▓▓▓▓▓░░░ 100 ✓    │ │
│ │ Timbre 声質   │ ▨50    │ ·0     │ ▨50    │ ▓▓▓▓▓▓▓░░░ 100 ✓    │ │
│ │ Articulation  │ ·0     │ ◨100   │ ·0     │ ▓▓▓▓▓▓▓░░░ 100 ✓    │ │
│ │ Register 抑揚 │ ▨40    │ ▨40    │ ▨20    │ ▓▓▓▓▓▓▓░░░ 100 ✓    │ │
│ │ Texture       │ ▨60    │ ·0     │ ▨40    │ ▓▓▓▓▓▓▓░░░ 100 ✓    │ │
│ └───────────────┴────────┴────────┴────────┴─────────────────────┘ │
│  [Reset row]  [Solo A]  [Even split]     → Studio で AB 官能評価 ▷  │
└────────────────────────────────────────────────────────────────────┘
```

### 4.4.3 コンポーネント
- **Factor 行**（既定5）: `Breath/Air`・`Timbre 声質`・`Articulation 構音`・`Register 抑揚`・`Texture`。
  Kansei ノブ（§5.2）と名前空間を共有するが、Compositor は「参照からの成分配合」、Kansei は
  「合成後の連続補正」で役割が分かれる。
- **Reference 列**: 登録済み参照音声 A,B,C…（§4.2 の few-shot 登録 / 録音で追加。`[+ Add reference]`）。
- **セル = 重みスライダ**: 0–100。**各 Factor 行の和 = 100 に自動正規化**（1 セルを動かすと同行の
  他セルが比例再配分）。行末に合計バーと ✓（=100 になっていれば緑）。
- **行ユーティリティ**: `Reset row`（均等）/ `Solo`（1参照 100）/ `Even split`。
- **ライブプレビュー**: `▶Preview(live)` で現ブレンド行列を実時間経路へ送り即試聴、または
  Studio へ渡して gt/変換/target と並べる。
- **保存**: ブレンド結果を `VoiceEntry`（kind=`Composite`）として命名保存・お気に入り。以後は
  Live のドロップダウンから単一声と同様に即選択できる（Compositor の内部行列も保持）。

### 4.4.4 段階 UX
- **初心者**: Compositor に触れず、Preset / Composite 済みの声を Live で 1 クリック選択。
- **上級**: Voice ▸ Compositor で行列を作り込み、Studio で 6 軸官能評価しながら詰める。

### 4.4.5 最小先行版（P1 で最初に切る PR）
**2 参照 × 3 Factor**（Breath / Timbre / Articulation）の重みスライダ + 行正規化のみ。
- 参照は既存 `voices` から 2 つ選ぶだけ（新規登録フロー不要）。
- `RtControl::SetBlendMatrix`（下記 7.5）を送るが、DSP 未接続なら no-op + 淡色 +「配線待ち」。
- これで「行正規化 UX」「行列 → RtControl 契約」「Composite 保存」を早期検証し、
  N 参照 × 5 Factor + 登録フロー + ライブプレビューは段階拡張する。

### 4.4.6 Kansei 連携（Human-in-the-Loop）
Compositor で作った Composite 声を Studio（§6）の gt / 変換 / target 3 系統 AB に流し、
6 軸官能評価を `results/kansei_evals.jsonl` に記録。記録行に **blend 行列（Factor×Ref 重み）と
参照 ID 群**を含め、「どの配合が耳で勝ったか」を後から追える（耳ゲート一次証拠）。

---

## 5. Kansei タブ（官能ノブ・要件③）

### 5.1 ワイヤフレーム
```
┌─ KANSEI ───────────────────────────────────────────────────────────┐
│  Preset: [ ASMR-whisper ▼ ]   [Save]  [Save As…]   [Reset]          │
│ ┌─ Timbre / Register ─────────┐ ┌─ Breath / Texture ──────────────┐ │
│ │  (Pitch)   (Formant)         │ │  (Breath)   (Air/ASMR)          │ │
│ │    ◑         ◑               │ │    ◔          ◕                 │ │
│ │  (Register: chest◀─▶head)   │ │  (Texture: smooth◀─▶airy)       │ │
│ └─────────────────────────────┘ └─────────────────────────────────┘ │
│ ┌─ Delivery (受肉) ──────────────────────────────────────────────┐ │
│ │  Prosody: (Self●)  Imitate  Blend  Flatten   Blend◀──●──▶       │ │
│ │  Conversion strength ◀────●────▶ 1.0                            │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ ┌─ AB 官能チェック ──────────────────────────────────────────────┐ │
│ │  ▶A(before)  ▶B(after)   A/B ⇄   ★★★☆☆  [記録]                 │ │
│ └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 ノブと制御のマッピング
`theme::knob`（0..1, ダブルクリック 0.5）を全面採用。各ノブは新 `RtControl` で
リアルタイム反映（Live と共有状態）。

| ノブ | 意味 | 制御先 | RtControl |
|---|---|---|---|
| Pitch | ピッチシフト（±semitone） | prosody/F0 レジスタ | `SetPitchShift(f32)` 新設 |
| Formant | 口腔サイズ（幼さ/大人） | timbre 変形 | `SetFormant(f32)` 新設 |
| Register | 胸声↔頭声 | F0 レジスタ配分 | `SetRegister(f32)` 新設 |
| Breath | 息成分の量 | texture/低レベル音量 | `SetBreath(f32)` 新設 |
| Air/ASMR | 近接・空気感（ASMR 近さ） | texture path | `SetAir(f32)` 新設 |
| Texture | smooth↔airy | vocoder texture | `SetTexture(f32)` 新設 |
| Prosody / Blend / Strength | **既存** `SetProsody / SetVelocityScale` を流用 | — | 既存 |

> 注: これらノブの DSP/モデル実接続は本 UI 設計の範囲外（`current/vocoder.md` 等の別設計依存）。
> **UI 側は `RtControl` バリアントと状態を先に定義**し、未接続ノブは「配線待ち」で no-op か
> 既存 prosody へマップ。受肉路線に沿い **Prosody 既定は `Self`（操作者の演技保持）** を新モード
> として `ProsodyMode` に追加提案（現行4値に `SelfPreserve` を足す）。

### 5.3 プリセット保存
Kansei 全ノブ + prosody + strength を1つの `KanseiPreset` として名前保存（JSON、Voice の
お気に入りと別管理）。Live の Quick Kansei と同じ状態を編集。

---

## 6. Studio タブ（オフライン + 官能評価・要件④）

`offline_tab.rs` を土台に **3系統 AB 評価** を中核化。

### 6.1 ワイヤフレーム
```
┌─ STUDIO ───────────────────────────────────────────────────────────┐
│  Source[____ .wav]▶  Reference[____ .wav]▶  or ★Voice[Venus ▼]      │
│  Prosody[Self ▼] Blend◀●▶  Strength◀●▶     [ CONVERT ]  ⟳           │
│ ┌─ Compare (Kansei gate) ────────────────────────────────────────┐ │
│ │  ▶ Source(gt)    ▶ Converted    ▶ Target(ref)                   │ │
│ │  A/B loop: [Source ⇄ Converted]  loop 2s ▷                      │ │
│ │  Rate: Smooth ★★★★☆  Tender ★★★☆☆  Clarity ★★★★☆              │ │
│ │        Embodiment ★★★☆☆  Fatigue ★★★★☆   [ Save eval → jsonl ] │ │
│ └────────────────────────────────────────────────────────────────┘ │
│  Output: 44100Hz 3.2s   [▶Play] [Save As…]                          │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 発展点
- **3系統聴き比べ**: Source(gt) / Converted / Target(ref) を並べて即再生（既存 AudioPlayer 流用）。
- **AB ループ**: 2 系統を短ループで交互再生（`AudioPlayer` に区間ループ追加）。
- **Kansei 6 軸評価**（README の Smoothness/Tenderness/Clarity/Embodiment/Control/Fatigue）を
  星 or スライダで記録し **`results/kansei_evals.jsonl`** へ追記（source/ref/model ckpt/日時/スコア）。
  これが CLAUDE.md「耳ゲート」の一次証拠フォーマットになる。
- 既存 `process_full`（Python parity）変換・Save As は維持。

---

## 7. 状態管理・データフローの差分

### 7.1 `RtControl` 追加（スレッド構造は不変、バリアント追加のみ）
```
Mute(bool)
SetBufferSize(u32)          // inference_loop で next Start に反映 / 自動 re-arm
SetPitchShift(f32) SetFormant(f32) SetRegister(f32)
SetBreath(f32) SetAir(f32) SetTexture(f32)
```
`inference_loop` の `match msg` に腕を追加。未配線ノブは当面 pipeline の該当 setter が無ければ
warn ログ＋no-op（UI 先行、モデル側は別設計で接続）。

### 7.2 `RtMetrics` 拡張（レイテンシ内訳・要件⑤）
```
latency_ms（既存, E2E）に加え:
  buf_in_ms / resamp_in_ms / algo_ms / resamp_out_ms / buf_out_ms
cpu_pct（当面 rtf*100、将来 process 実測）
```
現行 `latency_ms_last = 10.0 + 3.0 + algo_ms + 3.0 + 10.0` を分解して各フィールドに入れるだけ。

### 7.3 `AppState` 拡張
- `buffer_frames: u32`（既定 256）, `muted: bool`。
- `voices: Vec<VoiceEntry>` の `VoiceEntry` に `kind({Preset,User,ZeroShot,Composite}) / favorite /
  tags / ref_paths: Vec<PathBuf>` と、Composite 用 `blend: Option<BlendMatrix>`。
- `kansei: KanseiParams`（各ノブ f32）と `kansei_presets: Vec<KanseiPreset>`。
- `compositor: BlendMatrix`（編集中の行列, §7.5）。
- `Tab` enum を4値へ、初期タブは `Live`。

### 7.5 Voice Compositor データフロー（新要件）
```
struct BlendMatrix {
    references: Vec<VoiceRef>,        // 列: 参照 ID + 表示名
    weights: Vec<[f32; N_REF]>,      // 行 = Factor、各行の和 = 1.0（正規化済み）
}                                     // Factor 順: Breath/Timbre/Articulation/Register/Texture
```
- **正規化は UI 側**: 1 セル変更時に同行を比例再配分し和 = 1.0 を保証してから状態へ格納。
  RtControl には常に正規化済み行列を送る（DSP 側は生値を受けない）。
- 新 `RtControl::SetBlendMatrix(BlendMatrix)`（行列全体を送る。頻度は低いので clone で十分）。
  `inference_loop` は該当 setter があれば pipeline へ、無ければ warn + no-op（UI 先行方針通り）。
- `RtMetrics` は **不変**（合成はレイテンシ内訳に段を足さない前提。将来 blend コストが乗るなら
  `algo_ms` に内包）。
- Composite 声の選択は既存 `LoadReference` 経路と両立: Composite 保存時に代表 target を
  `LoadReference` で送りつつ `SetBlendMatrix` で配合を上書きする2段構成。

### 7.4 グローバル Transport
Start/Bypass/Mute/status/latency を Top panel へ移動。Live タブの Row3 操作バーは
Transport に統合し、Live 内は Latency/Kansei に集中。他タブ滞在中も運転継続。

---

## 8. Beatrice / paravo との UX 比較

| 観点 | Beatrice / paravo | 現 LightVC | 本提案 |
|---|---|---|---|
| バッファ | 256 等を明示選択・超低遅延前面 | 露出なし | **Live Latency カードで 128–1024 選択 + 内訳可視化** |
| レイテンシ表示 | 実測 ms を前面 | 定数推定・1数値 | **E2E + 段別内訳バー + Quality↔Speed** |
| ASIO | 標準対応・ホスト選択 | feature のみ | **設定でホスト/デバイス/ASIO 選択（feature 有効時）** |
| 声切替 | プリセット/モデル即切替 | file+name | **Preset library + お気に入り即切替 + few-shot 登録** |
| 声の作り方 | **単一ターゲット選択のみ** | 単一選択 | **Voice Compositor: 複数参照×Factor 配合で声を成分合成（明確な差別化）** |
| 音色微調整 | pitch/formant 等スライダ | prosody+velocity のみ | **Kansei 6 ノブ + プリセット保存** |
| 官能評価 | 基本なし | 分離・記録なし | **AB 3系統 + 6軸記録（差別化の核）** |
| 思想 | 汎用ボイチェン | — | **ASMR/官能特化 + 受肉 + 耳ゲート内蔵** |

差別化: (a) レイテンシを「隠す数値」でなく操作軸に、(b) ASMR 特化 Kansei ノブ（Breath/Air）、
(c) Human-in-the-Loop 官能評価を製品に内蔵（競合に無い）、(d) **Voice Compositor で声を
成分合成（単一ターゲット選択しかない競合に対する最大の差別化）**。低遅延・ASIO・バッファ選択は
「追いつくべき土俵」として必達。

---

## 9. 実装優先順位

**P0（競合に追いつく必達・低リスク）**
1. `RtControl::SetBufferSize` + Live Latency カードでバッファ選択（`start_with` 既存活用）。
2. `RtMetrics` レイテンシ内訳分解 + 内訳バー表示（定数を分解するだけ）。
3. グローバル Transport バー（Start/Bypass/**Mute**）+ Mute 実装。
4. 設定ダイアログに ASIO/ホスト選択（`supported_*_configs` 活用、feature gate）。

**P1（差別化の核）**
5. Kansei タブ + `theme::knob` 全面採用 + `SetPitch/Formant/Register/Breath/Air/Texture`
   バリアント定義（未配線は no-op、UI 先行）。`ProsodyMode::SelfPreserve` 追加。
6. Voice タブ発展: プリセット節 + お気に入り + few-shot 複数 ref 登録 + タグ/検索。
7. **Voice Compositor**（§4.4）: **最小先行版 = 2 参照 × 3 Factor スライダ + 行正規化 +
   `SetBlendMatrix` 契約 + Composite 保存**（P1 の中で 6 の直後、7 と並行可）。以降
   N 参照 × 5 Factor + 登録フロー + ライブプレビューへ段階拡張。
8. Studio 3系統 AB + Kansei 6軸評価 → `results/kansei_evals.jsonl`（blend 行列も記録）。

**P2（磨き込み）**
9. マイク録音からの声登録、KanseiPreset 保存/呼び出し、AB ループ再生、CPU 実測化、
   Compositor の N 参照 × 5 Factor フル化・ライブプレビュー・自動 Factor 抽出サムネ。

**先に切り出すべき最小差分（P0 の 1 本目）**: `Tab` を4値化せず現行3タブのまま、
Realtime タブに「Buffer combo + レイテンシ内訳 + Mute」だけ足す PR。これで既存構造を壊さず
低遅延 UX を先行検証でき、以降のタブ再編を段階導入できる。

---

## 10. リスク・留意

- **バッファ変更は stream 再 arm 必須**（`AudioEngine` を作り直す）。変更時に自動 Stop→Start する
  UX にし、配信中の切替は警告。
- **未配線 Kansei ノブ**を「効いてるフリ」にしない。接続前は淡色 + 「配線待ち」ツールチップ。
- **egui 0.34 の deprecated panel show**（`app.rs` コメント既知）: タブ再編時に Ui ツリー移行を
  一緒にやると事故る。**Transport/カード追加は現構造のまま**進め、移行は別 PR。
- **Kansei ノブの DSP 実体**は `current/vocoder.md` / texture path 設計に依存。UI は契約
  （RtControl + 状態）を先に固め、モデル側 setter が来たら結線するだけにする。
- ASIO SDK は再配布禁止（CLAUDE.md）。feature gate 前提、同梱しない。

---

## 付録: 変更ファイル見取り図
- `app.rs`: `Tab` enum, Transport バー, `AppState` フィールド, 新 RtControl 送信。
- `realtime_tab.rs`→`live_tab.rs`: Latency カード, Quick Kansei, Model/B1 撤去。
- `voice_catalog.rs`→`voice_tab.rs`: プリセット/お気に入り/few-shot/検索 + **Voice Compositor
  （Factor×Ref 行列, 行正規化, Composite 保存）**。
- `kansei_tab.rs`（新規）: ノブ群 + プリセット。
- `offline_tab.rs`→`studio_tab.rs`: 3系統 AB + 6軸評価記録。
- `app.rs`(RtControl)/`realtime_tab.rs`(inference_loop): バリアント追加（`SetBlendMatrix` 含む）,
  RtMetrics 拡張。`BlendMatrix` 型定義（正規化ロジックは UI 側）。
- `lightvc-audio`: 変更ほぼ不要（`start_with` 既存, buffer_size を渡すだけ）。
```
