# Windows 実機への引き継ぎ(段 P126 K3G-v4 の製品経路・2026-11-28 更新)

status: 現役引き継ぎ資料(旧 `windows_app_integration.md` の確定事項を統合・全文履歴は git `b749b59`)。

## 1. 製品の経路

前段(`v4_dual` + 門と瞬時マスク網 `inst_mask = true`)→ K3G-v4(符号と回路の 3 組の平均・錨)→ 出力部 `snv_p123_s`(`delay = 480`・因果 3 点包絡平滑 `env_smooth = 0.25`) → 出力の番(`out_guard.rs`)。`lightvc_core::snv_stream::SnvStream` が 48 kHz・240 サンプル(5 ms)単位で流し、アルゴリズム遅延は `HOP + DELAY = 720` サンプル(`15.0 ms`)。

- **実時間並列**: **4 スレッド**が既定(`enable_parallel(4)`・Linux 機での関門 S 6 回反復最悪は音楽つき `p99 = 4.89 ms < 5.0 ms`・音楽なし `4.67 ms`)。
- **f16 重みと注意 KV**: `manifest.json` の `cfg.half` と `cfg.kv16` によりフレーム重み・注意 KV ともに f16。
- **GUI・音声 I/O・プラグイン**:
  - アプリ GUI は `gpui 0.2.2`(要求駆動描画・静止時 CPU ほぼ 0・メモリ約 60 MB、`DESIGN.md` 準拠)。プラグイン画面のみ `nice-plug-egui`。
  - ASIO は `crates/lightvc-audio/src/asio_win.rs` のネイティブ COM(`IASIO`)実装により **Steinberg ASIO SDK 不要**で標準ビルドで動作し、WASAPI/ASIO 混在入出力にも対応する。
  - `cargo run -p lightvc-xtask -- install` は管理者権限不要のユーザーフォルダ `%LOCALAPPDATA%\Programs\Common\{VST3,CLAP}` へ導入する。

## 2. 重み(git には無い — `ch160` から `scp` で取得)

アプリ・プラグイン・`xtask package` が読むのは次の 3 つで、各フォルダの `manifest.json` と `weights.bin` が本体(`fixture*/` は一致試験専用)。取得コマンドはルートの `handover.md` を参照。

| 部品 | 作業ツリーの場所 | 配布物の中 | weights.bin | sha256 の先頭 |
|---|---|---|---|---|
| 前段 | `results/snv_p126_gate_export/` | `models/front_gate/` | 20,888,584 B | 4efdc07af3a2391a |
| K3G | `results/snv_p126_k3g16kv_export/` | `models/k3g/` | 135,379,140 B | 5afd51c2fdc5a487 |
| 出力部 | `results/snv_p123_s/export_20k/` | `models/rvoc3hi/` | 13,020,228 B | 7984a4226b8af238 |

GUI と CLI の既定はリポジトリの根からの `results/...`(プラグイン `resolve_export` は実行形式周辺の `models/*` → `%APPDATA%\LightVC\models` → `results/...` の順に探索)。

## 3. Windows 実機で確かめること(この順)

1. **ビルド**: `.\dev.ps1 -BuildOnly -Release`(`fxc.exe` を自動で `PATH` に追加・標準ビルドで WASAPI + ネイティブ COM ASIO 対応)。
2. **単体とボイスの試験**: `cargo test --workspace --release`(速い・普段はこれだけ)。**モデルの一致と出荷の関門**は推論コードか書き出しを替えたときだけ `cargo test --release -p lightvc-core --features model-tests -- --test-threads=1`(`results/` の製品の 3 つの書き出しと `pfront_export_p124` の `fixture*/` が要る・無い試験は読み飛ばす)。
3. **1 ブロックの時間(関門 S)**: 電源プラン「高パフォーマンス」・他プロセス停止の上で 4 スレッド `p99 < 5.0 ms`(6 回最悪):
   ```powershell
   cargo test --release -p lightvc-core gate_s_k3g16 -- --ignored --nocapture --test-threads=1
   ```
   - `affinity.rs` の `win::pin_perf`(`EfficiencyClass` 最大の P コア集合へ配置 + MMCSS + タイマ分解能引き上げ)が効いているか確認する(比較用環境変数: `LVC_NO_PIN=1`・`LVC_PIN_SMT=1`・`LVC_SPIN_US`)。
4. **配布パッケージと導入**: `cargo run -p lightvc-xtask -- package` および `cargo run -p lightvc-xtask -- install`。
5. **マイク → スピーカー / DAW 検証**:
   - `lightvc-app realtime-snv --host asio --buffer 240 --operator <自分の声.wav> --target voices/imouto.flac --threads 4 --seconds 300` および `lightvc-app gui` で、`compute ms p99 < 5.0`・`overrun/underrun` 非増加・耳で途切れ無しを確認。
   - DAW(Studio One 等)で VST3 / CLAP を読み込み、PDC(`SnvStream::LATENCY_SAMPLES = 720` = `15.0 ms` + ホストバッファ)による dry/wet 位相整合を確認。

## 4. 間に合わないときの診断

- **段ごとの内訳計測**: `cargo test --release -p lightvc-core profile_k3g -- --ignored --nocapture --test-threads=1`。
- **採らない手**: 系の数を減らす・フレームを間引くなど、耳の関門を通った形そのものを変える手は採らない。出力値を変える高速化(描画器の f16 化等)は必ず先に PyTorch 側で品質関門を通す。
