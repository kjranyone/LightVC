# LightVC

リアルタイム声質変換。変換の経路は paper_snv の段 P93 の K3G
（門と重みつきの前段 → 符号と回路の 3 組の平均 → rvoc3hi・48 kHz・240 サンプル = 5 ms 単位）で、Rust の
`lightvc_core::snv_stream::SnvStream` が流す。実時間には 4 スレッド(`--threads 4`・既定)が要る。研究の現状は `current/README.md` と `current/RESEARCH.md`。

## 使い方（CLI）

```
# オフライン変換（48 kHz mono WAV）
lightvc-app snv --input in.wav --output out.wav --operator my_voice.wav --target target_voice.wav --semitones 5

# リアルタイム変換（マイク → スピーカー）
lightvc-app realtime-snv --operator my_voice.wav --target target_voice.wav --semitones 5

# GUI（オフライン / リアルタイム / 声の一覧）
lightvc-app gui
```

- `--operator`: 操作者（自分）の普通の発話（計 30 秒程度・複数可・48 kHz）
- `--target`: 目標の声の参照（計 20 秒以上・複数可・48 kHz）。架空の声か、本人の同意がある声だけを使う
- `--semitones`: 固定の半音（運用は 0 / 5 / 7 / 12）
- 部品の書き出しは既定で `results/snv_p45_gate_t80_export`・`results/snv_k3g16kv_export`・`results/rvoc3hi/export_330k`（`--front` / `--model` / `--voc` で変える。配布物では `models/front_gate`・`models/k3g`・`models/rvoc3hi`）

CLAP / VST3 プラグインは `cargo run -p lightvc-xtask -- bundle` で作る。エディタの Model / Enrollment に書き出しの場所と登録の WAV を入れて Load する。

重みはリポジトリに含めない。配布物（`cargo run -p lightvc-xtask -- package`）に入れる重みの学習データと第三者のライセンスは [NOTICE](NOTICE) に書いてある。

## 遅延

- 設計目標: E2E p95 < 30 ms（理想 20 ms 級・p95 ≥ 50 ms は失格）。
- Windows の WASAPI 共有モードでは 30 ms を切れない。30 ms 級には ASIO ドライバが要る（ASIO SDK は再配布不可のため同梱しない）。
