# 引き継ぎ(2026-11-28T10:26Z — Windows 実機向け)

最初に `AGENTS.md`・`current/README.md`・`current/windows_handoff.md`・`current/windows_app_integration.md`・`current/paper_snv.md`(rev247)を読むこと。

## 現在の状態(段 P126 製品採用・Linux 側完了 → Windows 実機へ引き継ぎ)

- **オーナーの耳の関門(2026-11-28)**: 試聴パッケージ `results/snv_p126/ear/index.html` に対し **「うーん、ええんちゃう」** の判定で **ADOPTED 昇格完了**。
- **製品の既定書き出し(GUI・CLI・CLAP/VST3・配布物 `xtask package`)を段 P126 構成へ切り替え済み**:
  - 前段 + 瞬時マスク門(`v4_dual` + `inst_mask = true`): `results/snv_p126_gate_export` (`models/front_gate/`・`weights.bin` `20,888,584 B`・sha256 先頭 `4efdc07af3a2391a`)
  - 変換網 `K3G-v4` (`snv_p125{a,b,e}_c`・f16 kv・`target_v4 = true`・`fb = anch`): `results/snv_p126_k3g16kv_export` (`models/k3g/`・`weights.bin` `135,379,140 B`・sha256 先頭 `5afd51c2fdc5a487`)
  - 描画器 `snv_p123_s` (`delay = 480`・`env_smooth = 0.25`・アルゴリズム遅延 720 標本 = `15.0 ms`): `results/snv_p123_s/export_20k` (`models/rvoc3hi/`・`weights.bin` `13,020,228 B`・sha256 先頭 `7984a4226b8af238`)

## Windows 実機で行う手順

1. **リポジトリ同期と書き出しの取得**(`results/` は `.gitignore` 対象のため Linux 開発機 `ch160` から取得する):
   ```powershell
   git pull
   mkdir -Force results\snv_p123_s
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_p126_gate_export results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_p126_k3g16kv_export results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_p123_s/export_20k results/snv_p123_s/
   # cargo test --release -p lightvc-core の全一致試験を通すための更新・追加 fixture:
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/pfront_export_p100 results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/pfront_export_p124 results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_p45_gate_t80_export results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_p45_gate_t80_export_v3 results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_k3g_export results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_k3g16_export results/
   scp -r ch160:/home/kojirotanaka/kjranyone/LightVC/results/snv_k3g16kv_export results/
   ```
2. **ビルド・一致試験・関門 S・配布パッケージ生成**:
   - GPUI アプリのビルド(`fxc.exe` が PATH に必要・`.\dev.ps1 -BuildOnly` が自動で PATH に追加・標準ビルドでネイティブ COM ASIO 対応):
     ```powershell
     .\dev.ps1 -BuildOnly -Release
     ```
   - Rust 全一致・未来不変性・遅延台帳試験:
     ```powershell
     cargo test --workspace --release -- --test-threads=1
     ```
   - 4 スレッド実時間関門 S(既定で `results/snv_p126_k3g16kv_export` を計測):
     ```powershell
     cargo test --release -p lightvc-core gate_s_k3g16 -- --ignored --nocapture --test-threads=1
     ```
   - CLAP / VST3 バンドルと Windows 配布 zip(`target/LightVC-0.1.0-windows.zip`)の生成:
     ```powershell
     cargo run -p lightvc-xtask -- package
     cargo run -p lightvc-xtask -- install
     ```
3. **実機リアルタイム・GUI・DAW 確認**(`current/windows_handoff.md` §3・`current/windows_app_integration.md` 参照):
   - `lightvc-app realtime-snv --operator <自分の声.wav> --target <目標の声.wav> --threads 4 --seconds 300` および `lightvc-app gui` でのリアルタイム変換・耳での確認。
   - DAW(Studio One 等)での VST3 / CLAP 読み込みと PDC(`SnvStream::LATENCY_SAMPLES = 720` = 15.0 ms + バッファ)整合確認。

## ここまでの研究結論(2026-11-27〜28)

| 段 | 問い | 結果 |
| --- | --- | --- |
| P120〜P121 | 「原音→T」劣化と K 上乗せ雑音の分解 | オーナー判定「1-> たぶんL。Tにもある」「原音→Tで大幅劣化してるから、耳に出す意味なしｗ」。① P97/P114 の破損 `f0` 学習・② 前段 v3.1 高さ誤り・③ 因果平滑の 1 フレームずれを特定 |
| P122〜P123 | 教師 `harvest` `f0` + `(t-1)·HOP` 時刻同期での描画器回復 | **`snv_p123_s` 採択**: 教師写し錨 PESQ `3.6499`・36 声 `3.2494`・Rust `rvoc` 一致 `2.04e-5` |
| P124 | 因果高さ追跡 `f0_net`/`f0_voc` 分離(`v4_dual`) + 既存 `K3G` 検証 | Step a(因果写し PESQ `3.5404`・`am_db +0.34 dB`・P108 耳 6 地点 `[0.02,0..0]`)PASS。Step b `R3` は `k=12` の `M`(`−0.0658`)のみ `K3G` の有声分布依存で外れ → P125 へ |
| P125 | 前段 `v4` 全コーパス特徴での `K3G` 3 系(`snv_p125{a,b,e}`)一からの学び直し | `P125_anch` が `devA` `k=0,7`(CER `−0.0040`・`M +0.0373`)・**`k=12`(CER `−0.0025`・`M +0.0219`[`0.4143`])**・`P120`(`K 2.3037`・`Kc0 3.1020`・`d2 1.217`)・`zz95`・`ff216 0.1041` を全通過。伴奏単体 `pad_db` のみ `hyst=[40,100]` 待ち未マスクで外れ → P126 へ |
| P126 | 門の持続ヒステリシス待ち中および単発手拍子(`gate_p ≥ 0.8`)への瞬時 STFT マスク適用 | **`ADOPTED`(全客観関門 + Rust + 耳「うーん、ええんちゃう」PASS)**: `lead0 pad_db = −24.61 dB`・`claplead15 = −33.41 dB`、`devA` `k=0,7`(CER `−0.0050`・`M +0.0351`)・`k=12`(CER `−0.0057`・`M +0.0307`[`0.4238`])・`P120`・`zz95`・`ff216`・Rust 一致・並列 1 bit 一致・4 スレッド関門 S(`p99 = 4.89 / 4.67 ms < 5.0 ms`)全て通過し製品の既定へ昇格 |
