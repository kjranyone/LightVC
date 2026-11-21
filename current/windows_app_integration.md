# Windows アプリ組み込みの指示書(2026-11-21)

status: 作業指示 + 作業結果(§10・2026-11-21 の後のセッション)。前提と手順の正本は `current/windows_handoff.md`。この文書は、Windows 実機(i5-12600KF)での初回の確認の結果(§1)と、そこから組み込みで先に進める作業(§2〜)をまとめる。

## 1. 実機の確認結果(2026-11-21・コミット 8b2b0cc)

機械: Intel Core i5-12600KF(P コア 6 + E コア 4・論理 16)。WASAPI・ASIO は未使用(ASIO SDK なし)。

| 段 | 結果 |
|---|---|
| 重みの取得・zip の SHA256 | 一致(`FA4C80A5781EBB8BBF46B953C28F7CFBC292112EB736D3E40D9C27FA24BEE538`) |
| ビルド `cargo build --release -p lightvc-app` | 成功(2 分 35 秒) |
| `cargo test --release -p lightvc-core` | 49 件中 46 通過・3 件失敗(§4)。K3G の一致と `ship_gate_k3g16kv` は通過 |
| 関門 S(既定のピン止め・6 回の最悪 p99) | **不合格**: 音楽つき 8.41 ms・音楽なし 8.39 ms・対照(p30 + 門)8.40 ms(線 < 5 ms) |
| 関門 S の RTF | K3G 0.94〜1.04・p50 4.4〜5.0 ms |
| `LVC_NO_PIN=1`(2 回) | p99 8.36 / 8.75 ms・RTF 1.07 / 1.10 |
| `LVC_PIN_SMT=1`(2 回) | p99 9.77 / 9.64 ms・RTF 1.26 / 1.23(最も悪い) |
| `realtime-snv` 5 分・GUI のリアルタイム・耳の途切れ | **未実施**(操作者の録音が無い) |
| `profile_k3g` の内訳 | 未取得 |

Linux の開発機(RTF 0.80〜0.84・6 回の最悪 4.91 ms)に対して、この機は約 2 倍遅いブロックの尾を持つ。対照の旧経路(p30 + 門)も同じ尾なので、K3G 固有ではなく、この機のスレッドの置き方・待ちの起床(Windows のタイマ・スケジューラ)の可能性が高い。ただし `win::pin_perf` が P コアに載せているかは未確認。

## 2. 先にやること(この順)

1. **スレッドの置き方の確認**: `affinity.rs` の `win::pin_perf` が効いているか。タスク マネージャーの論理プロセッサ別の表示か ETW で、重い 4 スレッドが P コアの論理 CPU(コアごとに 1 つ)に載ることを見る。載っていなければ、`GetSystemCpuSetInformation` の `EfficiencyClass` の読みと `SetThreadSelectedCpuSets` の戻り値(失敗を握りつぶしていないか)を直す。
2. **タイマ・起床**: Windows の既定のタイマ分解能(約 15.6 ms)と、`recv_spin` の待ち(`LVC_SPIN_US`・既定 1000 µs)が尾を作っていないか。`timeBeginPeriod(1)` 相当と `LVC_SPIN_US` を振って関門 S を比べる(0 / 500 / 1000 / 2000)。
3. **電源プラン**: 「高パフォーマンス」にして関門 S を再測定する(省電力のままだと P コアの周波数が落ちる)。測定のときは他の重いプロセスを止める。
4. **`profile_k3g`** を取り、`windows_handoff.md` §4 の Linux の内訳(門 0.55・前段 0.53・模型 約 1.4・描画器 1.2 ms)と比べ、どの段が伸びているかを特定する。
5. **`realtime-snv`** を 5 分(`--threads 4 --seconds 300`)。操作者の登録は、この機のマイクで録った自分の普通の発話(48 kHz・計 30 秒ほど)。目標は `pf000_0/1/2.wav`。5 秒ごとの compute ms p99 < 5.0、overrun / underrun が増えない、耳で途切れが無い。

数値が通るまでは、GUI のリアルタイムの合否判断をしない。

## 3. アプリ組み込みの作業

組み込みの面(これだけを使う):

- `lightvc_core::snv_stream::SnvStream`: `load(front, model, voc)` → `enable_parallel(4)` → `set_semitones(k)`(運用 = 0 / 5 / 7 / 12)→ `enroll_operator(&[&[f32]])`(操作者の発話・48 kHz)→ `enroll_target(&[&[f32]])`(目標の参照)→ 240 サンプルごとに `process_block(x, out)`。`gate_open()` で門の状態を取れる。
- 既定の重みの場所(リポジトリの根からの相対):
  - 前段 `results/snv_p45_gate_t80_export`
  - K3G `results/snv_k3g16kv_export`
  - 出力部 `results/rvoc3hi/export_330k`
  - 配布物の形では `models/front_gate`・`models/k3g`・`models/rvoc3hi`。
- アプリの該当コード: `crates/lightvc-app/src/realtime_tab.rs`(GUI のリアルタイム)・`cli.rs`(`realtime-snv`)・`crates/lightvc-audio`(入出力)。

やること:

1. 音声スレッドでは `process_block` だけを呼ぶ。登録(`enroll_*`)・重みの読み込みは別スレッドで済ませ、ロックと確保を音声の経路に置かない。
2. 入出力は 48 kHz・240 サンプル(5 ms)固定。WASAPI の共有モードは `BufferSize::Fixed` が効かず 30 ms を切れない。30 ms 級の遅延には `--features asio` と ASIO SDK(リポジトリにコミットしない)。
3. 不足(overrun / underrun)の計数を GUI に出し、5 秒ごとの compute ms の p50 / p95 / p99 を同じ値で見られるようにする(`realtime-snv` の表示と同じ定義)。
4. 操作者の登録は GUI からマイク録音(30 秒ほど)または wav の選択。目標の声の参照は wav の選択。
5. CLAP / VST3(`lightvc-clap`)は Windows の MSVC で `cargo run -p lightvc-xtask -- bundle` を直に作る(Linux からの検査はできていない)。

## 4. 既知の試験の失敗(K3G の不具合ではない)→ 解決(§10.8)

`parity_end_to_end_gate`・`ship_gate_gate`・`parallel_bit_identical` は、旧経路の書き出し(`results/snv_p30_export`・`results/snv_stream_export`・`results/pfront_export`)が無いと `snv_stream.rs:327` で「指定されたパスが見つかりません」になる。この機には後から `ch160` から取得した(`scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/<名前> results/`)。取得後の再実行は §10.8(全て通過。P11 の組には `results/p_model_export` の取得も要った)。

`Get-FileHash` がこの機の PowerShell に無い。ハッシュは `certutil -hashfile <file> SHA256` で測る。

## 5. 採らない手

系の数を減らす・フレームを間引くなど、K3G の形を変える手は採らない(数値と耳の関門のやり直しになる)。削る場合は `windows_handoff.md` §4 の「次に削れる所」から、PyTorch の側で品質を測ってから。

## 6. ASIO 対応の手順

現状(コードを読んで確認した事実): `--features asio`(`lightvc-app/Cargo.toml` の `asio = ["cpal/asio"]`)は cpal 側の ASIO を有効にするだけで、**ホストを選ぶ所が無い**。`lightvc-audio/src/engine.rs`・`stream.rs`、`lightvc-app/src/audio_playback.rs`・`realtime_tab.rs:341` は全て `cpal::default_host()` を呼ぶ。Windows の既定のホストは WASAPI なので、`asio` を付けてビルドしても ASIO は使われない。`lightvc-audio` には `asio` の feature 自体が無い。

手順:

1. **ASIO SDK**: Steinberg の ASIO SDK を各自で入手する(再配布禁止・リポジトリにコミットしない)。環境変数 `CPAL_ASIO_DIR` に SDK の根を指す(cpal の asio ビルドが読む)。ビルドには LLVM / clang(bindgen)と MSVC が要る。
2. **feature の貫通**: `lightvc-audio` に `asio = ["cpal/asio"]` を足し、`lightvc-app` の `asio` から `lightvc-audio/asio` も有効にする。
3. **ホスト選択の関数**を `lightvc-audio` に 1 つ作り、全ての `cpal::default_host()` をこれに置き換える。
   - `#[cfg(all(windows, feature = "asio"))]` では `cpal::host_from_id(cpal::HostId::Asio)` を先に試し、失敗(ドライバ無し)なら `default_host()` に落とす。
   - 環境変数 `LVC_HOST=wasapi|asio` と CLI の `--host` で上書きできるようにする。
   - GUI の設定にホストの選択(WASAPI / ASIO)を出す。入力と出力は同じホストの装置から選ぶ。
4. **バッファ**: ASIO はドライバのバッファに従う。`BufferSize::Fixed(240)`(5 ms)が通るかを装置ごとに確かめる。通らなければ装置の最小値を取り、240 サンプルへの組み直し(リングバッファ)を `SnvStream::BLOCK` の側で行う。48 kHz のドライバでない装置は rubato で変換する(`lightvc-audio/src/resample.rs`)。
5. **確かめ方**: `realtime-snv --host asio --buffer 240` で overrun / underrun が増えないこと。報告する遅延は「入力バッファ + 5 ms(1 ブロック)+ 出力バッファ + 前段の予備」を足した値。目標は p95 < 30 ms(`current/README.md`・50 ms は失格上限)。
6. **ASIO の注意**: ASIO は 1 つのドライバを 1 プロセスが占有する。DAW のプラグインとアプリの同時の起動はできない。ASIO4ALL などの汎用ドライバは遅延が伸びるので、専用ドライバ(または Voicemeeter 等の仮想)を勧める。

## 7. GUI(GPUI を含む)の実装手順

現状: アプリの GUI は **egui(eframe 0.34)**。プラグインの画面も egui(`nice-plug-egui`・egui 0.34)で、アプリとプラグインで同じ見た目の部品を使える。`crates/lightvc-app/src/` の `app.rs`・`realtime_tab.rs`・`offline_tab.rs`・`widgets.rs`・`theme.rs` が実装。

方針: **まず egui のまま、音声経路と画面を切り離す**。GUI の枠組みの置き換え(GPUI など)は、遅延の合否が通ってから別の課題として扱う。理由は、今の止まっている所が GUI ではなく 1 ブロックの時間(§1・§2)だから。

egui のまま優れた実装にする手順:

1. **スレッドの分離**: 音声スレッド(`process_block` 専用・4 スレッドの並列)・制御スレッド(登録・読み込み)・GUI スレッドを分ける。GUI から音声へはロックを使わず、`rtrb`(SPSC)かアトミックで渡す。メーターと計数(overrun / underrun・p50 / p95 / p99)は音声側がアトミックに書き、GUI が読む。
2. **再描画**: 音声の動作中だけ `request_repaint_after(33ms)` 程度で回す。止まっているときは再描画しない(GPU・CPU を食わない)。音声スレッドの P コアを GUI が奪わないよう、GUI スレッドは E コア側へ寄せる(`affinity.rs` に GUI 用の関数を足す)。
3. **画面の構成**: 登録(操作者の録音・目標の参照)→ 運用(半音 0 / 5 / 7 / 12・門の状態表示・入出力メーター)→ 診断(ホスト・バッファ・遅延・compute ms の p99・overrun / underrun)。診断の数値は `realtime-snv` の定義と同じにする。
4. **検証**: `lightvc-app gui --demo`(模擬の値)と `tools\snap.ps1` でスクリーンショットを撮り、実装と見た目を合わせる(`AGENTS.md` の GUI Review Protocol)。
5. **egui のバージョン**: アプリ(eframe 0.34)とプラグイン(egui 0.34)をそろえたままにする。片方だけ上げない。

GPUI(Zed の GUI フレームワーク)へ移すかどうかの検討項目(今は着手しない):

- **ライセンス**: 採用前に Apache-2.0 / GPL の別を `cargo tree --workspace -e normal --format '{p} {l}'` で確かめる(プロジェクトは MIT・GPLv3 依存は禁止)。依存の網の中に GPL が入るなら採らない。
- **Windows の描画**: GPUI の Windows 対応の成熟度と、ビルドの重さ(依存の数・ビルド時間)を試作で測る。
- **プラグインの画面**: GPUI は VST3 / CLAP のホストの窓へ埋め込む手段が整っていない可能性が高い(`nice-plug-egui` のような橋が無い)。プラグインの画面は egui のまま残るので、アプリだけ GPUI にすると**画面の部品が 2 系統になる**。この点が移行の最大の費用。
- **判定**: 試作(登録の画面 1 つ)で、ビルド時間・起動時間・メモリ・音声スレッドへの干渉(§2 の関門 S を GUI を出したまま測る)を egui と比べ、勝つ見込みが無ければ egui に留める。

## 8. VST3 / CLAP プラグイン化の手順

現状: `crates/lightvc-clap`(nice-plug + nice-plug-egui + clap-wrapper 0.3)があり、`SnvStream` を内部に持つ(`lib.rs` 3〜20 行・159 行目で `load` と `enable_parallel`)。Linux の検査は済まず、Windows の MSVC でも未ビルド。

手順:

1. **ビルド**: Windows に MSVC(Visual Studio Build Tools)・CMake・Ninja を入れ、`cargo run -p lightvc-xtask -- bundle` を実行する。clap-wrapper が VST3 SDK 3.8(MIT)を同梱する。`bundle` の出力(`.clap`・`.vst3`)の置き場は xtask の出力に従う。
2. **インストール**: `cargo run -p lightvc-xtask -- install`。`.vst3` は `C:\Program Files\Common Files\VST3\`、`.clap` は `C:\Program Files\Common Files\CLAP\` に置かれるか確かめる(権限が足りなければ管理者権限が要る)。
3. **重みの探し方**: プラグインは DAW のプロセス内で動くので、作業ツリーの `results/...` の相対パスが使えない。`LightVcParams` の `front` / `model` / `voc` の既定を、プラグイン自身の位置(`.vst3` バンドルの隣の `models/`)か、`%APPDATA%` 下の既定の場所から探す形にする。見つからなければ画面にエラーを出し、音声は素通し(バイパス)にする。重み 約 170 MB の読み込みは音声スレッドでやらない。
4. **スレッド**: 4 スレッドの並列(`enable_parallel(4)`)は DAW のプロセス内に常駐する。DAW の他のプラグインと P コアを奪い合うので、DAW の音声バッファと負荷の余裕が小さいときは動かない。5 ms ごとの `process_block` の期限を守れないときは、出力を無音にせず直前の出力を続けるか、バイパスへ落ちる設計にする。
5. **遅延の報告**: プラグインの `latency_ms`(`lib.rs` 131 行)を、DAW に正しく伝える(CLAP の latency 拡張・VST3 の `getLatencySamples`)。乾いた信号を遅延させて混ぜる仕組みは実装済み(`lib.rs` 191 行)。DAW のバッファが 240 の倍数でないと、内部で 240 サンプルへ組み直す分の遅延が増える。報告値にこれを足す。
6. **サンプルレート**: 内部は 48 kHz 固定。DAW が 44.1 kHz などのときは rubato で変換する(実装済み)。変換の遅延も報告値に足す。
7. **確かめ方**: Reaper など VST3 / CLAP を読める DAW で、(a) 読み込みの成否と画面の表示、(b) 5 分の連続動作で DAW のバッファ(256 / 512)ごとの途切れ、(c) 遅延補正の合い方(乾いた信号との位相)、(d) 保存と再読み込みの後のパラメータ(半音・登録済みの声)の復元、を見る。実行時間の p99 は §2 の 5 ms の線と同じ。
8. **ライセンス**: `cargo tree --workspace -e normal --format '{p} {l}'` で GPL が入っていないことを確かめ、`NOTICE` に clap-wrapper・VST3 SDK・nice-plug を書く。VST3 の商標(VST は Steinberg の登録商標)の表記を配布物に入れる。

## 9. 優先順(まとめ)

1. §2 の速度の尾(P コアの置き方・タイマ・電源プラン・`profile_k3g`)と `realtime-snv` の 5 分。**これが通らない限り、以下は音が途切れる**。
2. §6 ASIO のホスト選択(WASAPI の共有モードでは 30 ms を切れない・コードの欠けは小さい)。
3. §3 の音声スレッドと GUI の分離、診断の表示(§7 の egui のままの手順)。
4. §8 VST3 / CLAP のビルドと DAW での確認。
5. GPUI の試作(§7 の判定)は最後。

## 10. 作業結果(2026-11-21 の後のセッション)

### 10.1 スレッドの置き方(§2.1)= 実装は正しく・欠けは rank 0

- CPU 集合の読みは正しい(公式の構造体の並び +8 Id / +12 Group / +15 CoreIndex / +18 EfficiencyClass と一致。P = lp 0-11・E = lp 12-15)。`SetThreadSelectedCpuSets` は成功・読み戻しも `GetThreadSelectedCpuSets` で一致(lp 0,2,4,6,8,10 = P コア 6 個をコアごとに 1 論理 CPU)。ピン止めしたスレッドの走行先のヒストグラムも P コアのみ。
- **欠けは rank 0(`process_block` を呼ぶスレッド)**: `Team::new` はワーカー(1..n)しか置き換えない。置いていないスレッドは時々 E コアに載る(実測・ヒストグラム)。関門 S の試験スレッドも該当していた。
- 直した所: `affinity.rs` に `pin_audio_driver`(process_block を呼ぶスレッド用)を足し、関門 S の `run()`・CLI の処理スレッド・GUI の推論スレッド・プラグインの推論スレッドで呼ぶ。CPU 集合の ID はプロセスで 1 回だけ計算し、`SetThreadSelectedCpuSets` の失敗は握りつぶさず 1 回だけ記録する。

### 10.2 タイマ・起床とスケジューラ(§2.2)= MMCSS が主効

関門 S(6 回の最悪 p99・`LVC_K3G=results/snv_k3g16kv_export`・スレッド 4):

| 条件 | 音楽つき | 音楽なし | 対照(p30 + 門) |
|---|---|---|---|
| §1 の初回(8b2b0cc) | 8.41 | 8.39 | 8.40 |
| rank 0 ピン + `timeBeginPeriod(1)` | 6.79 | 7.67 | 7.52 |
| + MMCSS(Pro Audio) | 6.46 | 5.93 | 5.11 |
| + `LVC_SPIN_US=500`(採用した既定に近い形) | 5.81 | 6.60 | 5.95 |

- 試して採らなかった物: `LVC_SPIN_US` 0 / 2000(同程度か悪い)・`LVC_PIN_SMT=1`(同程度)・`LVC_DRIVER_CORE=1`(駆動系の最初の P コアへの固定は 6.32/5.77/6.83 と逆効果)。
- 最終の 6 回の内訳: p50 4.22〜4.45・p95 4.37〜5.52・RTF 0.85〜0.91(対照は p50 3.7〜3.9)。対照の回 1 は p99 4.94 まで下がる = 静けさで ±1 ms 程度動く。
- **測定環境**: 電源プランは「高パフォーマンス」・AC の最小プロセッサ状態 100 %。測定中も Zed 約 0.8 コア・dwm・Taskmgr・remoting(遠隔セッション)が稼働(§2.3 の「他の重いプロセスを止める」はオーナーの判断が要る)。残る 1 ms 級の尾はこの環境負荷と機械の差が主。
- 実装(全て既定で有効・`LVC_NO_TIMER` / `LVC_NO_MMCSS` / `LVC_NO_PIN` で切れる): `timeBeginPeriod(1)`(Team と音声ループの開始時)・MMCSS の Pro Audio クラス(`AvSetMmThreadCharacteristicsW`・`pin_current_to_perf` の中)・GUI の描画スレッドは E コアへ(`pin_current_to_efficiency`)。

### 10.3 内訳(§2.4)

`profile_k3g`(音楽なし・並列 4): 全体 4.30 ms = 門 0.64(網 0.59)+ 前段 0.48(CMNDF 0.21)+ 固定の窓 0.04 + 模型 1.79(組ごと 1.75)+ 励起 0.06 + 描画器 1.29(残差 0.78・ワーカー待ち 0.14)。Linux の内訳(0.55/0.53/1.4/1.2)に対し**特定の段の突出は無く一律 1.1〜1.3 倍** = 経路の劣化ではなく機械と環境の差。この走行の p50 4.25・p95 4.91。

### 10.4 realtime-snv 5 分(§2.5)= 数値は概ね届く・耳は未

測定: `realtime-snv --operator pf000_0/1/2 --target pf000_1 --threads 4 --seconds 300 --host wasapi --input 0 --output 4`(入力 fifine マイク 48 kHz・出力 E2x2 の Playback 1/2 44.1 kHz・up/down 変換あり)。登録の音声は pf000 の代用(この機のマイクで録った本人の発話ではない・計測のために置き換え)。

- 最終の走行(修正後・5 分): 60 窓のうち 40 窓が **overrun 0・underrun 0**。underrun の総計は 3035(約 69 ms ぶん・t≈200 秒に一瞬の負荷の山で 1 回だけ・あとは増えない)・**overrun は最後まで 0**。compute ms p50 3.96〜4.18・p95 4.63〜5.18・p99 4.97〜5.86(予算 5.0 に対して概ね 5.0〜5.5・静かな窓では 5.0 を切る)。
- 最初の走行(他セッションの java/rustc のビルドと重なった): 本体は安定し、負荷の山の間だけ p50 が 5 を超え出力が薄まる窓が 3 つ(overrun 総計 3328・underrun 総計 153589 はその 3 窓に集中)。**負荷が去れば完全に回復する**(以前は回復しない崩壊だった・下の直し)。
- **直した不具合 2 つ**:
  1. CLI のループにバックログの上限が無く、一瞬の遅れから `in48` が無限に溜まり `drain` の O(n) で処理量が崩壊する死亡スパイラル(15 秒ほどで全崩壊するのを実測)。GUI と同じ「8 ブロックを超えたら古い入力を捨てる」を入れた。
  2. 開始直後の約 2.2 秒の無音(重み 170 MB の初回ページイン)。読み込みの後に無音 1 ブロックを流すウォームアップを CLI・GUI・プラグインの読み込みに追加し、出力リングに 1024 標本の先送り。開始バーストは消えた(60 秒の検証走行で underrun 0)。
- 入出力のレートが違っても開けるように CLI の交渉を直した(48 kHz で開けなければ各装置の既定レートで開き up/down で繋ぐ。修正前は不一致で即エラー)。
- **耳の確認(途切れが無いか)は未実施** = オーナーの作業。上の数字は無音の窓が無いことを示すが、最終判断は耳。

### 10.5 アプリ組み込み(§3・§7)

- 音声スレッドの分離: `EngineSlot`(Arc<Mutex<Option<SnvStream>>>)を廃止。読み込みと登録は別スレッドで済ませ `RtControl::Stream` で渡し、**推論スレッドが変換器を単独で持つ**(process_block の経路にロック無し)。半音も制御の列で渡す。
- 診断(§3.3): 5 秒ごとの compute ms の p50/p95/p99 + ブロック数 + overrun/underrun + ホスト名 + 門の状態を `realtime-snv` と同じ定義で GUI に出す。p99 ≥ 5.0 は赤。
- 登録(§3.4): 操作者の登録は「Record operator (30 s)」ボタン(既定のマイクから 48 kHz の WAV に保存し登録の列に加える)または WAV の選択。目標の参照は Voices タブの WAV 選択のまま。
- 再描画(§7.2): リアルタイム動作中は `request_repaint_after(33ms)`・止まっているときは再描画しない。GUI スレッドは E コアへ。
- スクリーンショット(§7.4): docs/screenshots/live.png を差し替えた(計断行・ホスト・REC ボタンの表示を画像で確認)。

### 10.6 ASIO のホスト選択(§6)

- `lightvc-audio` に `asio = ["cpal/asio"]` feature を足し、`lightvc-app` の `asio` から貫通。
- ホスト選択の関数 `lightvc_audio::host::pick(HostPref)` を作り、`cpal::default_host()` の呼び出しを全て置き換えた(engine.rs・stream.rs・realtime_tab.rs・audio_playback.rs)。順位: 明示(設定 / `--host`)> `LVC_HOST=wasapi|asio` > 既定(asio ビルドでは ASIO を先に試し失敗なら WASAPI)。
- GUI の設定にホストの選択(default / wasapi / asio)を足した。入出力は同じホストの装置から選ぶ。CLI は `realtime-snv --host asio --buffer 240`・装置の一覧は `lightvc-app devices`。
- ASIO SDK がこの機に無い(CPAL_ASIO_DIR 未設定)ため `--features asio` のビルドと実機確認は未実施。装置側の実装は済んでいる。

### 10.7 VST3 / CLAP(§8)= ビルドと配置は済み・DAW での確認は未

- `cargo run -p lightvc-xtask -- bundle` が通った(MSVC・CMake/Ninja 不要・6 分 49 秒)。`target/bundled/` に LightVC.vst3 と LightVC.clap。
- `cargo run -p lightvc-xtask -- install`: システムの VST3 置き場は権限不足(管理者が要る)なので、**Steinberg のユーザー置き場** `%LOCALAPPDATA%\Programs\Common\VST3` と CLAP の `%LOCALAPPDATA%\Programs\Common\CLAP` へ入るように xtask を直した(システム置き場への上書きはオーナーが管理者権限で)。
- 煙の確認: 両バンドルの DLL を `LoadLibrary` し、CLAP は `clap_entry`・VST3 は `GetPluginFactory` / `InitDll` のエクスポートを確認。
- 重みの探し方(§8.3)を実装: 未指定なら実行形式から 3 階層上まで `models/<部品>` を探し、`%APPDATA%\LightVC`・作業ツリーの `results/` も候補にする。
- §8.4 の置き方: プラグインの推論スレッドも `pin_audio_driver` + MMCSS・重みのページインのウォームアップを足した。§8.5〜8.6 の遅延の報告と 48 kHz 固定 + rubato は元から実装済み。
- **Studio One 3 がこの機にある**(VST3 を読める)。読み込み・5 分の連続動作・遅延補正・保存の再読み込み(§8.7 の a〜d)は GUI の操作が要るため未実施 = オーナーの作業。
- NOTICE に VST 商標の表記を足した。`cargo tree` で GPL は self_cell(Apache-2.0 で使用)のみ。

### 10.8 試験(§4)= 全て通過

- `cargo test --release -p lightvc-core`: 49 件中 49 通過(初回の走行は 47 通過・2 失敗 = P11 の組が `results/p_model_export` を必要としていた。ch160 から取得して再実行し通過)。
- 合わせて取得した書き出し: `results/p_model_export`(今回)・`results/snv_p30_export`・`results/snv_stream_export`・`results/pfront_export`(前回)。


