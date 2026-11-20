# Windows 実機への引き継ぎ(K3G の製品の経路・2026-11-20)

status: 引き継ぎ資料。オーナー 2026-11-20「いやWindows向けアプリ実装は実機のエージェントにやらせるわ。」

Linux の開発機で済んでいること(コミット `04f59a6`):

- 数値の関門・耳の関門(オーナーの判定)・Rust の一致・出荷の関門(先読み 0・台帳 20 ms)。
- 4 スレッドの実時間: RTF 0.80〜0.84、1 ブロック(5 ms)の p99 は典型 4.2〜4.4 ms、6 回の最悪 4.91 ms(線 < 5 ms)。

Windows では、まだ何も動かしていない。

## 1. 製品の経路

前段(門と重みの網つき)→ K3G(符号と回路の 3 組の平均・錨)→ 出力部 rvoc3hi。`lightvc_core::snv_stream::SnvStream` が 48 kHz・240 サンプル単位で流す。

- 実時間には **4 スレッド**が要る(`enable_parallel(4)`・GUI と CLI の既定は 4・プラグインの既定も 4)。2 スレッドでは p95 5.05 ms で予算を超える。
- 重みは f16 で持つ(manifest の `cfg.half` と `cfg.kv16`)。F16C の命令が無い CPU では遅い経路に落ちる(出力は同じ)。

## 2. 重み(git には無い)

アプリが読むのは次の 3 つで、各フォルダの `manifest.json` と `weights.bin` が本体。`fixture*/` は一致の試験だけが使う。

| 部品 | 作業ツリーの場所 | 配布物の中 | weights.bin | sha256 の先頭 |
|---|---|---|---|---|
| 前段 | `results/snv_p45_gate_t80_export/` | `models/front_gate/` | 20,888,584 B | 4efdc07af3a2391a |
| K3G | `results/snv_k3g16kv_export/` | `models/k3g/` | 135,379,140 B | 3110a86b029f8e05 |
| 出力部 | `results/rvoc3hi/export_330k/` | `models/rvoc3hi/` | 13,020,228 B | 5ffbc09b362da00d |

Linux の配布物 `target/LightVC-0.1.0-linux.zip` の `models/` は OS に依らない。GUI と CLI の既定はリポジトリの根からの `results/...`。配布物の形で動かすときは `--front models/front_gate --model models/k3g --voc models/rvoc3hi`。

## 3. Windows で確かめること(この順)

Linux から Windows 向けに型の検査を済ませた(2026-11-20・`cargo check --target x86_64-pc-windows-gnu`)。

- `lightvc-app`(GUI・CLI)・`lightvc-audio`・`lightvc-core`: 通る。
- `lightvc-clap`(CLAP / VST3): ビルドの手順の中で C++ のコンパイラが要り、Linux からの検査はできていない(mingw の g++ が無い)。Windows の MSVC で直に作る。

1. **ビルド**: `cargo build --release -p lightvc-app`。30 ms 級の遅延には `--features asio` が要る(ASIO SDK は各自で用意する・WASAPI の共有モードは 30 ms を切れない)。
2. **一致**: `cargo test --release -p lightvc-core`。`results/` の 3 つの書き出しと `fixture*/` が要る。`results/snv_k3g_export`・`results/snv_k3g16_export` が無い試験は読み飛ばす。
3. **1 ブロックの時間(関門 S)**: 4 スレッドで p99 < 5 ms。6 回測り、最悪の値で判定する。
   ```
   set LVC_K3G=results/snv_k3g16kv_export
   cargo test --release -p lightvc-core gate_s_k3g16 -- --ignored --nocapture --test-threads=1
   ```
4. **マイク → スピーカー**: `lightvc-app realtime-snv --operator <自分の声.wav> --target <目標の声.wav> --threads 4 --seconds 300`。5 秒ごとの表示で次を確かめる。
   - `compute ms p99` < 5.0
   - `overrun` と `underrun` が増えない
   - 耳で途切れが無い

   GUI のリアルタイムでも同じことを見る。
5. **スレッドの置き方**(`crates/lightvc-core/src/affinity.rs` の `win::pin_perf`): 混成の CPU では、`GetSystemCpuSetInformation` の `EfficiencyClass` が最大の CPU 集合(コアごとに 1 つ)へ `SetThreadSelectedCpuSets` する。型の検査しかしていない。
   - 重いスレッドが P コアに載っていることを確かめる(タスク マネージャーか ETW)。
   - 比べ: `LVC_NO_PIN=1`(置かない)・`LVC_PIN_SMT=1`(P コアの全ての論理 CPU)。
   - 待ちの回し時間は `LVC_SPIN_US`(既定 1000 µs)。

## 4. 間に合わないとき

- **内訳を出す**: `set LVC_K3G=results/snv_k3g16kv_export` の上で `cargo test --release -p lightvc-core profile_k3g -- --ignored --nocapture --test-threads=1`。Linux の 4 スレッドの内訳(ブロックあたり)は次のとおり。

  | 段 | ms |
  |---|---|
  | 門 | 0.55 |
  | 前段 | 0.53 |
  | 模型 | 約 1.4 |
  | 描画器 | 1.2 |
  | 計 | 約 4.0 |

- **次に削れる所**: 描画器の重みの f16 と、前段の一部を門の網と並べること。どちらも数値が変わるなら、先に PyTorch の側で品質を測る(研究の正本 `current/paper_snv.md` 段 P94 の手順)。
- **採らない手**: 系の数を減らす・フレームを間引くなど、通った形そのものを変える手は採らない(数値の関門と耳の関門をやり直すことになる)。
