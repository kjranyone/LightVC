//! CLI argument parsing and subcommand dispatch.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use lightvc_audio::Resampler;
use lightvc_core::{
    converter::{AnyConverter, ConverterConfig, LatencyMode},
    pipeline::VcPipeline,
    DacConfig, FreeResynth,
};

#[derive(Parser)]
#[command(
    name = "lightvc",
    version,
    about = "LightVC real-time voice conversion"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Validate DAC encode/decode round-trip on a WAV file.
    Roundtrip(RoundtripCmd),
    /// Apply converter to a WAV file (offline processing).
    Convert(ConvertCmd),
    /// Apply B1 UTTE adapter pipeline to a WAV file (offline).
    ConvertB1(ConvertB1Cmd),
    /// FreeVocoder resynthesis (WAV → Rust mel → freeC vocoder → WAV),
    /// streamed through the deployed `chunk_samples()` realtime path.
    Resynth(ResynthCmd),
    /// v2f (harmonic-prior, causal, 30 ms class) resynthesis: WAV → WAV.
    V2f(V2fCmd),
    /// バ美声変換 (front→E→G→V ストリーム): WAV → WAV。
    Vc(VcCmd),
    /// DDSP-VC のリアルタイム変換 (マイク → 48 kHz・240 サンプル単位 → スピーカー・ゼロショット参照)。
    RealtimeDdsp(RealtimeDdspCmd),
    /// 構音 VC(表の引き・製品経路)のオフライン変換: WAV(48 kHz)→ WAV。ストリーミングの経路(240 サンプル単位)をそのまま通す。
    Artic(ArticCmd),
    /// 構音 VC のリアルタイム変換(マイク → 48 kHz・240 サンプル単位 → スピーカー)。
    RealtimeArtic(RealtimeArticCmd),
    /// 構音 VC の登録だけを作る(目標または操作者の参照 WAV → enroll.bin・enroll.json)。
    ArticEnroll(ArticEnrollCmd),
    /// Launch desktop GUI (3 tabs: offline/realtime/catalog).
    Gui(GuiCmd),
}

/// 構音 VC の部品の書き出しと登録(training/export_{rvoc,lvl,f0est,c1,artic_vc}.py の出力)。
#[derive(Parser, Clone)]
pub struct ArticModel {
    #[arg(long, default_value = "results/rvoc3hi/export")]
    pub rvoc: std::path::PathBuf,
    #[arg(long, default_value = "results/f2_2/export")]
    pub lvl: std::path::PathBuf,
    #[arg(long, default_value = "results/f0est3/export")]
    pub f0est: std::path::PathBuf,
    #[arg(long, default_value = "results/c1_1/export")]
    pub c1: std::path::PathBuf,
    /// 目標話者の登録(enroll.bin・enroll.json)。--reference を渡したときは使わない
    #[arg(long, default_value = "results/artic_vc_export")]
    pub enroll: std::path::PathBuf,
    /// 目標話者の参照音声(48 kHz・20 秒以上)。渡すと Rust で登録する
    #[arg(long)]
    pub reference: Option<std::path::PathBuf>,
    /// 登録の定数(training/export_enroll.py の出力: mel・DCT)
    #[arg(long, default_value = "results/enroll_export")]
    pub enroll_consts: std::path::PathBuf,
    /// 表の柔らかい引きの鋭さ β(Σ p^β T / Σ p^β)。0 = 硬い引き(argmax)
    #[arg(long, default_value_t = 1.0)]
    pub soft: f32,
    /// 表の引きの因果平均の幅を上書き(既定 = 柔らかい引き 1・硬い引き 3)。D115 の診断用
    #[arg(long)]
    pub w_sm: Option<usize>,
    /// 『目標らしさ ↔ 聞き取りやすさ』のつまみ(0〜1): 元の声の包絡(F3・因果)を混ぜる割合
    #[arg(long, default_value_t = 0.0)]
    pub mix: f32,
    /// F3 の書き出し(training/export_env.py の出力)
    #[arg(long, default_value = "results/f3_1/export")]
    pub f3: std::path::PathBuf,
    /// F4 の書き出し(周期性の推定器・training/export_env.py)。渡すと条件の周期性 = F4 + (目標の p75 − 操作者の p75)(F4 で学習した出力部用)
    #[arg(long)]
    pub f4: Option<std::path::PathBuf>,
    /// c0 も『目標の表 + 操作者の残差』に: c0 += Σp (t0_目標 − t0_操作者)[u](D121・両方の登録に t0 が要る)
    #[arg(long)]
    pub c0_table: bool,
    /// c0 を元の声のエネルギーを保ち出力の包絡の形と整合する水準に解き直す(D119・F3 が要る = --warp / --operator)
    #[arg(long)]
    pub c0_energy: bool,
    /// 有声フレームの包絡の較正(f32 LE 24 個・条件の単位)。D114 の診断用
    #[arg(long)]
    pub env_bias: Option<std::path::PathBuf>,
    /// --reference で作った登録を保存するディレクトリ(次回から --enroll で使える)
    #[arg(long)]
    pub save_enroll: Option<std::path::PathBuf>,
    /// 操作者(自分の声)の参照音声(48 kHz・20 秒以上)。渡すと 包絡 = 自分の声の包絡 + (1 − mix)·単位ごとの話者差(目標の表 − 自分の表)
    #[arg(long)]
    pub operator: Option<std::path::PathBuf>,
    /// 操作者の保存済みの登録(--operator の代わり)
    #[arg(long)]
    pub operator_enroll: Option<std::path::PathBuf>,
    /// --operator で作った登録を保存するディレクトリ
    #[arg(long)]
    pub save_operator: Option<std::path::PathBuf>,
    /// 操作者の声の単位の中の動きを周波数伸縮してから足す: 包絡 = Σp T_目標 + W_α(自分の声の包絡 − Σp T_自分)
    #[arg(long)]
    pub warp: bool,
    /// --warp の伸縮率 α を固定する(既定 = 登録の表どうしの形の相関で当てる)
    #[arg(long)]
    pub alpha: Option<f32>,
    /// --warp で足す自分の声の動きの量 γ(0〜1・1 = 全部)
    #[arg(long, default_value_t = 1.0)]
    pub resid: f32,
    /// 遷移で動きの量を変える(D147): γ_min,γ_max,d_lo,d_hi(C1 の事後の全変動の因果 3 平均が d_lo 以下で γ_min・d_hi 以上で γ_max)。--resid の代わり
    #[arg(long, value_delimiter = ',')]
    pub gamma_adapt: Option<Vec<f32>>,
    /// 動きの量の DCT の帯ごとの重み(D153): 例 "1-8:1,9-24:0"(行 1..24 の範囲と重み・書かない行は 1)
    #[arg(long)]
    pub resid_dims: Option<String>,
    /// --warp の操作者の動きに目標の動きの共分散をかぶせる(M = C_目標^½ (W C_操作者 Wᵀ)^−½・登録に res_cov が要る)
    #[arg(long)]
    pub color: bool,
    /// 条件の周期性に目標の登録の 75 パーセンタイルを使う(既定 = 平均・古い登録には値が無く平均のまま)
    #[arg(long)]
    pub per_p75: bool,
    /// 包絡のポストフィルタ β(c2..c24 を β 倍・1 = なし)
    #[arg(long, default_value_t = 1.0)]
    pub postfilter: f32,
    /// c0 の水準のずらし(c0n の単位・D186: 目標の参照の有声の c0n 平均 − 操作者の参照の有声の c0n 平均)
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub c0_shift: f32,
    /// 内容 → 波形の生成器の音量の条件の水準ずらし(c0n の単位・D558・男声 → 女声の構成だけ・無音には足さない)
    #[arg(long, allow_hyphen_values = true, conflicts_with = "c0_ppg_ref")]
    pub c0_ppg_shift: Option<f32>,
    /// --c0-ppg-shift を操作者の参照の WAV から作る: 4.40 − 末尾 12 s(短ければ前を無音で埋める)の発声中の c0n 平均(D558 の登録の定数)
    #[arg(long)]
    pub c0_ppg_ref: Option<std::path::PathBuf>,
    /// 高さの移動の基準を操作者の登録の f0 中央値に固定する(走る中央値は文の途中で基準が動き抑揚を崩す)。--operator / --operator-enroll が要る
    #[arg(long)]
    pub med_from_operator: bool,
    /// 遅い音色の揺らぎを操作者の揺らぎから作る(D451): 時定数 tau 秒,利得(例 4,1.5)。内容 → 波形の生成器と操作者の登録が要る
    #[arg(long, value_delimiter = ',')]
    pub opslow: Option<Vec<f64>>,
    /// 内容 → 波形の条件の周期性を F4 の動き(目標の水準 + 操作者の動き)に(diag_ppgvoc53)。--f4 が要る
    #[arg(long)]
    pub per_dyn: bool,
    /// 連続の内容の特徴 C2 の書き出し(training/export_c1.py・diag_ppgvoc52 の生成器と組で使う。例 results/c2_1/export)
    #[arg(long)]
    pub c2: Option<std::path::PathBuf>,
    /// F1 の有声の判定のヒステリシス(確率: 入る,抜ける。例 0.7,0.3)。既定 = 0.5 の一つの閾値
    #[arg(long, value_delimiter = ',')]
    pub voicing_hyst: Option<Vec<f32>>,
    /// F1 の音高の事後に因果の前向きフィルタ(σ 10 ビン・ε 0.02)をかける(有声区間の中の大きな跳びを減らす・先読み 0)
    #[arg(long)]
    pub f0_track: bool,
    /// F1 の音高の確信度の下限(これ未満のフレームは無声・例 0.3・D108)
    #[arg(long)]
    pub f0_conf: Option<f32>,
    /// c0・包絡を F2・F3 の代わりに因果な CheapTrick(窓は (t+1)H で終わる・因果の 3 点の平滑)で作る(D253・既定 off = 従来とビット一致)
    #[arg(long)]
    pub causal_ct: bool,
    /// 全体写像: 包絡 = 伸縮 W_α(操作者の包絡)+ 長時間平均の差(表の単位ごとの引きを使わない・--warp と --operator-enroll が要る・既定 off)
    #[arg(long)]
    pub global_map: bool,
    /// 全体写像の長時間平均の差 Δ の倍率(既定 1・D256 の用量反応)
    #[arg(long, default_value_t = 1.0)]
    pub gmap_scale: f32,
    /// 診断: 高さを目標の声域へ移さない(D261)
    #[arg(long)]
    pub no_f0_shift: bool,
    /// 診断: --f4 の周期性を目標の水準へ移さない(差 = 0・D261)
    #[arg(long)]
    pub f4_no_offset: bool,
    /// 全体写像の Δ を下限つきの log-mel の領域で足す(無音・小さい帯には乗らない・c0 も整合・D262)
    #[arg(long)]
    pub gmap_floor: bool,
    /// どの変換でも操作者の c0・包絡からの差を下限つきの log-mel の領域で足し直す(--causal-ct と操作者の登録が要る・D263)
    #[arg(long)]
    pub floor_apply: bool,
    /// 参照を読む R の S 形(training/export_rr.py の出力・例 results/d587_rbar/export)。生成器の条件の包絡の行に 利得·ΔS/10 を足す(D579〜d587)
    #[arg(long)]
    pub rr: Option<std::path::PathBuf>,
    /// R の目標の記憶(mem_k{l}.f32 / mem_v{l}.f32 の置き場)
    #[arg(long)]
    pub rr_mem: Option<std::path::PathBuf>,
    /// R の差分の利得(既定 1)
    #[arg(long, default_value_t = 1.0)]
    pub rr_gain: f32,
    /// R の目標の記憶の間引き(既定 2 = 列を 1 つおき・計算量 1/2)
    #[arg(long, default_value_t = 2)]
    pub rr_stride: usize,
    /// R を同期で回す(フレーム t の R を待つ・一致検査用)。既定 = 別のスレッドで 1 フレーム遅れ(音声の遅延は増えない)
    #[arg(long)]
    pub rr_sync: bool,
    /// R の操作者の統計を作る操作者の参照の WAV(48 kHz)
    #[arg(long)]
    pub rr_op_ref: Option<std::path::PathBuf>,
    /// 高さを固定の半音だけ移す(例 7・男声 → 女声の運用 +0 / +5 / +7 / +12)。目標の声域への移動の代わり
    #[arg(long, allow_hyphen_values = true)]
    pub semitones: Option<f32>,
    /// 低い高さの基音の傾き σ dB/oct(有声の枠・目標の声域から f0 まで・D614 は 4)。--causal-ct が要る
    #[arg(long)]
    pub h1fix: Option<f32>,
    /// 生成器の後ろの適応 H1 EQ の目標 H1 − H2 dB(D622 で選んだ値は 3)。--causal-ct が要る
    #[arg(long)]
    pub h1eq: Option<f32>,
    /// 適応 H1 EQ を働かせる出力の f0 の上限 Hz(既定なし = 目標の声域まで)
    #[arg(long)]
    pub h1eq_fmax: Option<f32>,
    /// 構音の描画(生成器なし・current/artic_render.md)の書き出し(training/export_artic_render.py の出力・例 results/artic_render1/export)。--c2 と内容 → 波形の生成器の条件が要る
    #[arg(long)]
    pub render_artic: Option<std::path::PathBuf>,
    /// 構音の描画の声道長のつまみ α(帯の中心を α 倍・1 = 中立)
    #[arg(long, default_value_t = 1.0)]
    pub render_alpha: f64,
    /// 構音の描画の大きさの音節の動きのつまみ γ(1 = 操作者のまま・既定 = 書き出しの level_gamma)
    #[arg(long)]
    pub render_level: Option<f64>,
    /// 構音の描画で目標ごとの曲線(登録の render_curve.f32)を使わず全体の曲線にする
    #[arg(long)]
    pub render_global_curve: bool,
}

/// 実女声の入力の発声中の c0n の平均(D557)。
const C0_FEM: f32 = 4.40;

/// 操作者の参照から作る登録の定数のキャッシュ(<参照>.<種類>.<設定の hash>.json・同じ操作者で毎回の作り直しを省く)。
fn op_cache_path(r: &std::path::Path, kind: &str, key: &str) -> std::path::PathBuf {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    if let Ok(md) = std::fs::metadata(r) {
        md.len().hash(&mut h);
        md.modified().ok().hash(&mut h);
    }
    let mut p = r.as_os_str().to_owned();
    p.push(format!(".{kind}.{:016x}.json", h.finish()));
    std::path::PathBuf::from(p)
}

impl ArticModel {
    fn load(&self) -> Result<lightvc_core::artic_vc::ArticVc> {
        let mut m = self.load_inner()?;
        m.soft = if self.soft > 0.0 { Some(self.soft) } else { None };
        m.w_sm = self.w_sm.unwrap_or(if self.soft > 0.0 { 1 } else { lightvc_core::artic_vc::W_SM });
        anyhow::ensure!((0.0..=1.0).contains(&self.mix), "--mix は 0〜1");
        m.per_p75 = self.per_p75;
        m.per_dyn = self.per_dyn;
        if let Some(d) = &self.c2 {
            m.set_c2(d)?;
        }
        if let Some(d) = &self.render_artic {
            anyhow::ensure!(self.h1eq.is_none(), "--render-artic と --h1eq は併用しない(描画は生成器の後ろの EQ を使わない)");
            anyhow::ensure!((0.7..=1.4).contains(&self.render_alpha), "--render-alpha は 0.7〜1.4");
            m.set_render(d, 0)?;
            m.set_render_alpha(self.render_alpha);
            if let Some(g) = self.render_level {
                anyhow::ensure!((0.0..=1.0).contains(&g), "--render-level は 0〜1");
                m.set_render_level(g);
            }
            let cf = self.enroll.join("render_curve.f32");
            if !self.render_global_curve && cf.exists() {
                let b = std::fs::read(&cf)?;
                let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                anyhow::ensure!(v.len() == 1025, "render_curve.f32 は 1025 個");
                m.set_render_curve(&v);
                println!("構音の描画: 目標ごとの曲線 {}", cf.display());
            } else {
                println!("構音の描画: 全体の曲線(目標ごとの曲線なし)");
            }
        }
        anyhow::ensure!((0.5..=2.0).contains(&self.postfilter), "--postfilter は 0.5〜2");
        m.postfilter = self.postfilter;
        anyhow::ensure!(self.c0_shift.abs() <= 6.0, "--c0-shift は ±6 以内");
        m.c0_shift = self.c0_shift;
        m.no_f0_shift = self.no_f0_shift;
        if let Some(st) = self.semitones {
            anyhow::ensure!((-24.0..=24.0).contains(&st), "--semitones は ±24 以内");
            m.semitones = Some(st);
        }
        if let Some(tg) = self.h1eq {
            anyhow::ensure!((0.0..=12.0).contains(&tg), "--h1eq は 0〜12 dB");
            anyhow::ensure!(self.causal_ct, "--h1eq には --causal-ct が要る(内容 → 波形の生成器)");
            m.set_h1eq(tg, self.h1eq_fmax);
        }
        if let Some(sg) = self.h1fix {
            anyhow::ensure!((0.0..=12.0).contains(&sg), "--h1fix は 0〜12");
            anyhow::ensure!(self.causal_ct, "--h1fix には --causal-ct が要る(内容 → 波形の生成器)");
            m.set_h1fix(sg, lightvc_core::enroll::Analyzer::load(&self.enroll_consts)?.dct_rows());
        }
        if self.floor_apply {
            anyhow::ensure!(self.causal_ct, "--floor-apply には --causal-ct が要る");
            m.floor_apply = Some(lightvc_core::enroll::Analyzer::load(&self.enroll_consts)?.dct25());
        }
        if self.causal_ct {
            m.cct = Some(lightvc_core::enroll::CausalCt::new(&lightvc_core::enroll::Analyzer::load(&self.enroll_consts)?));
            println!("c0・包絡 = 因果な CheapTrick(D253)");
        }
        if self.per_p75 && m.target().per_p75.is_none() {
            println!("注意: 登録に周期性の 75 パーセンタイルが無い(古い登録)ので平均を使う");
        }
        let op = match (&self.operator, &self.operator_enroll) {
            (Some(r), _) => Some(self.enroll_wav(r, self.save_operator.as_deref(), "操作者")?),
            (None, Some(d)) => Some(lightvc_core::artic_vc::Enroll::load(d)?),
            (None, None) => None,
        };
        if self.mix > 0.0 || op.is_some() {
            m.load_f3(&self.f3)?;
            m.mix = self.mix;
        }
        if self.c0_table {
            let o = op.as_ref().context("--c0-table には操作者の登録が要る")?;
            let (tt, to) = (m.target().t0.clone().context("目標の登録に t0 が無い(登録し直す)")?, o.t0.clone().context("操作者の登録に t0 が無い(登録し直す)")?);
            m.c0tab = Some(tt.iter().zip(to.iter()).map(|(a, b)| a - b).collect());
        }
        if self.c0_energy {
            anyhow::ensure!(op.is_some() || self.mix > 0.0, "--c0-energy には F3(--operator か --mix)が要る");
            m.c0e = Some(lightvc_core::enroll::Analyzer::load(&self.enroll_consts)?.dct_rows());
        }
        if let Some(bp) = &self.env_bias {
            let b: Vec<f32> = std::fs::read(bp)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            anyhow::ensure!(b.len() == m.target().d, "--env-bias は {} 個", m.target().d);
            m.env_bias = Some(b);
        }
        if let Some(d4) = &self.f4 {
            let mut off = [0.0f32; 4];
            match (m.target().per_p75, op.as_ref().and_then(|o| o.per_p75)) {
                (Some(t), Some(o)) => {
                    for i in 0..4 {
                        off[i] = t[i] - o[i];
                    }
                }
                _ if op.is_some() => anyhow::bail!("--f4 の差には目標と操作者の両方の登録に per_p75 が要る"),
                _ => println!("注意: --f4 で操作者なし = 元の声の周期性のまま"),
            }
            if self.f4_no_offset {
                off = [0.0; 4];
            }
            m.load_f4(d4, off)?;
        }
        if let Some(v) = &self.opslow {
            anyhow::ensure!(v.len() == 2, "--opslow は tau,gain");
            let o = op.as_ref().context("--opslow には操作者の登録(--operator / --operator-enroll)が要る")?;
            m.set_opslow(o, v[0], v[1])?;
            println!("遅い揺らぎ = 操作者の揺らぎ(時定数 {} s・利得 {})", v[0], v[1]);
        }
        if let Some(op) = &op {
            if self.med_from_operator {
                m.set_fixed_med(op.mu);
                println!("高さの基準 = 操作者の登録の中央値 {:.0} Hz", op.mu.exp());
            }
            if self.warp {
                let an = lightvc_core::enroll::Analyzer::load(&self.enroll_consts)?;
                let al = self.alpha.unwrap_or_else(|| an.fit_alpha(&m.target().table, &op.table, op.k));
                println!("伸縮率 α {al:.2}");
                m.set_operator_warp(op, an.warp_matrix(al))?;
                if self.global_map {
                    m.set_global_map(op)?;
                    m.gmap_scale = self.gmap_scale;
                    if self.gmap_floor {
                        m.gmap_floor = Some(an.dct25());
                    }
                    println!("全体写像: 包絡 = W_α(操作者の包絡) + 長時間平均の差");
                }
                if self.color {
                    m.set_color(op)?;
                    println!("動きの色づけ: 目標の共分散");
                }
                anyhow::ensure!((0.0..=1.0).contains(&self.resid), "--resid は 0〜1");
                m.resid = self.resid;
                if let Some(spec) = &self.resid_dims {
                    let mut w = vec![1.0f32; 24];
                    for part in spec.split(',') {
                        let (rng, v) = part.split_once(':').context("--resid-dims は 範囲:重み")?;
                        let (a, b) = rng.split_once('-').unwrap_or((rng, rng));
                        let (a, b): (usize, usize) = (a.parse()?, b.parse()?);
                        anyhow::ensure!(1 <= a && a <= b && b <= 24, "--resid-dims の範囲は 1..24");
                        let v: f32 = v.parse()?;
                        for r in a..=b {
                            w[r - 1] = v;
                        }
                    }
                    m.resid_dims = Some(w);
                }
                if let Some(g) = &self.gamma_adapt {
                    anyhow::ensure!(g.len() == 4, "--gamma-adapt は γ_min,γ_max,d_lo,d_hi の 4 つ");
                    m.gamma_adapt = Some((g[0], g[1], g[2], g[3]));
                }
            } else {
                anyhow::ensure!(!self.color, "--color には --warp が要る");
                m.set_operator(op)?;
            }
        }
        if let Some(ex) = &self.rr {
            let mem = self.rr_mem.as_ref().context("--rr には --rr-mem(目標の記憶)が要る")?;
            anyhow::ensure!((0.0..=3.0).contains(&self.rr_gain), "--rr-gain は 0〜3");
            m.set_rr(ex, mem, self.rr_gain, self.rr_stride, self.rr_sync)?;
            let r = self.rr_op_ref.as_ref().context("--rr には --rr-op-ref(操作者の参照の WAV)が要る")?;
            let cache = op_cache_path(r, "rrop", &format!("{:?}", (&self.f3, &self.lvl, &self.f0est, &self.c1, self.causal_ct)));
            let st = match std::fs::read_to_string(&cache).ok().and_then(|t| serde_json::from_str::<(Vec<f32>, f32, f32)>(&t).ok()) {
                Some((e, c0, lf)) if e.len() == 24 => {
                    let mut a = [0.0f32; 24];
                    a.copy_from_slice(&e);
                    (a, c0, lf)
                }
                _ => {
                    let (x, sr) = load_wav_mono(r)?;
                    anyhow::ensure!(sr == 48_000, "--rr-op-ref は 48 kHz が必要 (got {sr})");
                    let mut c = self.clone();
                    c.rr = None;
                    c.c0_ppg_ref = None;
                    c.save_enroll = None;
                    c.save_operator = None;
                    let st = c.load()?.rr_op_stats(&x).context("--rr-op-ref の有声が足りない")?;
                    let _ = std::fs::write(&cache, serde_json::to_string(&(st.0.to_vec(), st.1, st.2))?);
                    st
                }
            };
            println!("R の操作者の統計: c0n {:.3}・f0 中央値 {:.0} Hz・利得 {}", st.1, (st.2 as f64).exp(), self.rr_gain);
            m.set_rr_op(st);
        }
        if let Some(r) = &self.c0_ppg_ref {
            anyhow::ensure!(m.is_ppg(), "--c0-ppg-ref は内容 → 波形の生成器だけ");
            let cache = op_cache_path(r, "c0lv", &format!("{:?}", (&self.rvoc, &self.lvl, &self.f0est, &self.c1, &self.c2, self.causal_ct, self.mix, &self.f4, self.per_p75, &self.enroll, &self.operator_enroll)));
            let lv: Option<f32> = match std::fs::read_to_string(&cache).ok().and_then(|t| serde_json::from_str::<Option<f32>>(&t).ok()) {
                Some(v) => v,
                None => {
                    let (x, sr) = load_wav_mono(r)?;
                    anyhow::ensure!(sr == 48_000, "--c0-ppg-ref は 48 kHz が必要 (got {sr})");
                    let n: usize = 12 * 48_000;
                    let mut pre = vec![0.0f32; n.saturating_sub(x.len())];
                    pre.extend_from_slice(&x[x.len().saturating_sub(n)..]);
                    let mut c = self.clone();
                    c.c0_ppg_ref = None;
                    c.rr = None;
                    c.save_enroll = None;
                    c.save_operator = None;
                    let lv = c.load()?.c0_level(&pre);
                    let _ = std::fs::write(&cache, serde_json::to_string(&lv)?);
                    lv
                }
            };
            m.c0_ppg_shift = C0_FEM - lv.unwrap_or(C0_FEM);
            println!("音量の条件の水準ずらし {:+.3}(操作者の参照の発声中の c0n {})", m.c0_ppg_shift, lv.map_or("なし".to_string(), |v| format!("{v:.3}")));
        }
        if let Some(v) = self.c0_ppg_shift {
            anyhow::ensure!(m.is_ppg(), "--c0-ppg-shift は内容 → 波形の生成器だけ");
            m.c0_ppg_shift = v;
        }
        anyhow::ensure!(m.c0_ppg_shift.abs() <= 4.0, "音量の条件の水準ずらしは ±4 以内 (got {})", m.c0_ppg_shift);
        Ok(m)
    }

    fn track(&self) -> Option<(f32, f64)> {
        self.f0_track.then_some((10.0, 0.02))
    }

    fn hyst(&self) -> Option<(f32, f32)> {
        self.voicing_hyst.as_ref().map(|v| {
            assert!(v.len() == 2, "--voicing-hyst は 入る,抜ける の 2 つ");
            (v[0], v[1])
        })
    }

    fn enroll_wav(&self, r: &std::path::Path, save: Option<&std::path::Path>, who: &str) -> Result<lightvc_core::artic_vc::Enroll> {
        let (x, sr) = load_wav_mono(r)?;
        anyhow::ensure!(sr == 48_000, "{who}の参照は 48 kHz が必要 (got {sr})");
        anyhow::ensure!(x.len() >= 20 * 48_000, "{who}の参照は 20 秒以上が必要 (got {:.1} s)", x.len() as f64 / 48_000.0);
        let t0 = std::time::Instant::now();
        let en = lightvc_core::artic_vc::ArticVc::enroll_from_wav(&x, &self.f0est, &self.c1, &self.enroll_consts)?;
        if let Some(d) = save {
            en.save(d)?;
        }
        println!("{who}の登録: {:.1} s の参照 → 表 {}x{}・f0 中央値 {:.0} Hz({:.1} s)", x.len() as f64 / 48_000.0, en.d, en.k, en.mu.exp(), t0.elapsed().as_secs_f64());
        Ok(en)
    }

    fn load_inner(&self) -> Result<lightvc_core::artic_vc::ArticVc> {
        use lightvc_core::artic_vc::ArticVc;
        match &self.reference {
            Some(r) => {
                let en = self.enroll_wav(r, self.save_enroll.as_deref(), "目標")?;
                ArticVc::load_with_opts(&self.rvoc, &self.lvl, &self.f0est, &self.c1, en, 1, self.hyst(), self.track(), self.f0_conf)
            }
            None => ArticVc::load_with_opts(&self.rvoc, &self.lvl, &self.f0est, &self.c1, lightvc_core::artic_vc::Enroll::load(&self.enroll)?, 1, self.hyst(), self.track(), self.f0_conf),
        }
    }
}

#[derive(Parser)]
pub struct ArticCmd {
    #[arg(short, long)]
    pub input: std::path::PathBuf,
    #[arg(short, long, default_value = "artic_output.wav")]
    pub output: std::path::PathBuf,
    #[command(flatten)]
    pub model: ArticModel,
    /// 診断: 元の声の有声 log f0 の中央値を固定する(既定は最初の 2000 フレームの走る中央値)
    #[arg(long)]
    pub fixed_med: Option<f32>,
    /// 出荷検査: t 秒以降を時間反転した入力でも t より前の出力がビット一致するか(未来不変性・実音声)と 1 ブロックの処理時間の分布を JSON で出す
    #[arg(long)]
    pub probe_future: Option<f32>,
}

#[derive(Parser)]
pub struct RealtimeArticCmd {
    #[command(flatten)]
    pub model: ArticModel,
    /// 実行秒数 (0 = 停止まで)
    #[arg(long, default_value_t = 0)]
    pub seconds: u64,
    /// cpal のバッファ (フレーム数・0 = 既定)
    #[arg(long, default_value_t = 240)]
    pub buffer: u32,
}

#[derive(Parser)]
pub struct ArticEnrollCmd {
    /// 参照音声(48 kHz・20 秒以上)
    #[arg(short, long)]
    pub reference: std::path::PathBuf,
    /// 登録を書くディレクトリ
    #[arg(short, long)]
    pub out: std::path::PathBuf,
    #[command(flatten)]
    pub model: ArticModel,
}

pub fn run_artic_enroll(cmd: ArticEnrollCmd) -> Result<()> {
    let en = cmd.model.enroll_wav(&cmd.reference, Some(&cmd.out), "参照")?;
    if let Some(ex) = &cmd.model.render_artic {
        let (x, _) = load_wav_mono(&cmd.reference)?;
        let t0 = std::time::Instant::now();
        let v = lightvc_core::artic_vc::ArticVc::render_curve_from_wav(&x, &cmd.model.f0est, &cmd.model.c1, &cmd.model.enroll_consts, ex, &en.table)?;
        std::fs::write(cmd.out.join("render_curve.f32"), v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
        println!("構音の描画の目標ごとの曲線 → {}({:.1} s)", cmd.out.join("render_curve.f32").display(), t0.elapsed().as_secs_f64());
    }
    if let Some(rr) = &cmd.model.rr {
        let (x, _) = load_wav_mono(&cmd.reference)?;
        let t0 = std::time::Instant::now();
        let m = lightvc_core::artic_vc::ArticVc::enroll_rr_memory(&x, &cmd.model.f0est, &cmd.model.c1, &cmd.model.enroll_consts, rr, &cmd.out.join("rr_mem"))?;
        println!("R の目標の記憶: {m} フレーム → {}({:.1} s)", cmd.out.join("rr_mem").display(), t0.elapsed().as_secs_f64());
    }
    Ok(())
}

pub fn run_artic(cmd: ArticCmd) -> Result<()> {
    use lightvc_core::nvoc::HOP;
    let (x, sr) = load_wav_mono(&cmd.input)?;
    anyhow::ensure!(sr == 48_000, "入力は 48 kHz が必要 (got {sr})");
    let mut vc = cmd.model.load()?;
    if let Some(m) = cmd.fixed_med {
        vc.set_fixed_med(m);
    }
    let n = x.len() / HOP;
    if let Some(ts) = cmd.probe_future {
        let cut = ((ts * 48_000.0) as usize / HOP).min(n - 1) * HOP;
        let mut x2 = x[..n * HOP].to_vec();
        x2[cut..].reverse();
        let mut vc2 = cmd.model.load()?;
        if let Some(m) = cmd.fixed_med {
            vc2.set_fixed_med(m);
        }
        let (mut y1, mut y2) = (vec![0f32; n * HOP], vec![0f32; n * HOP]);
        let mut ms = Vec::with_capacity(n);
        let prof = std::env::var_os("LVC_PROF");
        if prof.is_some() {
            vc.prof = Some(Vec::with_capacity(n * 9));
        }
        for i in 0..n {
            let t = std::time::Instant::now();
            vc.process_block(&x[i * HOP..(i + 1) * HOP], &mut y1[i * HOP..(i + 1) * HOP]);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        for i in 0..n {
            vc2.process_block(&x2[i * HOP..(i + 1) * HOP], &mut y2[i * HOP..(i + 1) * HOP]);
        }
        if let (Some(p), Some(pf)) = (prof.as_ref(), vc.prof.as_ref()) {
            let mut v: Vec<u8> = Vec::with_capacity((pf.len() + n) * 4);
            for (i, &t) in ms.iter().enumerate() {
                v.extend_from_slice(&(t as f32).to_le_bytes());
                pf[i * 9..(i + 1) * 9].iter().for_each(|s| v.extend_from_slice(&s.to_le_bytes()));
            }
            std::fs::write(p, v)?;
        }
        let pre = y1[..cut].iter().zip(&y2[..cut]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let first = y1[..cut].iter().zip(&y2[..cut]).position(|(a, b)| a.to_bits() != b.to_bits());
        let post = (y1[cut..].iter().zip(&y2[cut..]).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / (n * HOP - cut) as f64).sqrt();
        let rms = (y1[cut..].iter().map(|a| (*a as f64).powi(2)).sum::<f64>() / (n * HOP - cut) as f64).sqrt();
        let mut q = ms.clone();
        q.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pc = |p: f64| q[((q.len() - 1) as f64 * p).round() as usize];
        let verdict = if pre > 0 { "FAIL" } else if post < 1e-6 * rms.max(1e-9) { "INCONCLUSIVE" } else { "PASS" };
        println!(
            "{{\"cut_s\": {:.3}, \"pre_mismatch\": {pre}, \"first_mismatch_ms\": {}, \"post_rms_diff\": {post:.3e}, \"post_rms\": {rms:.3e}, \"verdict\": \"{verdict}\", \"block_ms\": {{\"mean\": {:.4}, \"p50\": {:.4}, \"p95\": {:.4}, \"p99\": {:.4}, \"max\": {:.4}}}, \"rtf\": {:.4}, \"n_blocks\": {n}}}",
            cut as f64 / 48_000.0,
            first.map_or("null".to_string(), |f| format!("{:.2}", f as f64 / 48.0)),
            ms.iter().sum::<f64>() / n as f64, pc(0.5), pc(0.95), pc(0.99), q[q.len() - 1],
            ms.iter().sum::<f64>() / n as f64 / (HOP as f64 / 48.0)
        );
        return Ok(());
    }
    let mut y = vec![0f32; n * HOP];
    let dump = std::env::var_os("LIGHTVC_DUMP_COND");
    if dump.is_some() {
        vc.dump = Some(vec![]);
    }
    vc.cond_only = std::env::var_os("LIGHTVC_COND_ONLY").is_some();
    let front = std::env::var_os("LIGHTVC_DUMP_FRONT");
    if front.is_some() {
        vc.front = Some(vec![]);
    }
    let vcd = std::env::var_os("LIGHTVC_DUMP_VCOND");
    if vcd.is_some() {
        *vc.voc_vdump() = Some(vec![]);
    }
    let rcd = std::env::var_os("LIGHTVC_DUMP_RCOND");
    if rcd.is_some() {
        *vc.voc_rdump() = Some(vec![]);
    }
    let rrd = std::env::var_os("LIGHTVC_DUMP_RR");
    if let (Some(_), Some(r)) = (rrd.as_ref(), vc.rr.as_mut()) {
        r.dump = Some(vec![]);
    }
    let t0 = std::time::Instant::now();
    for i in 0..n {
        vc.process_block(&x[i * HOP..(i + 1) * HOP], &mut y[i * HOP..(i + 1) * HOP]);
    }
    let el = t0.elapsed().as_secs_f64();
    if let (Some(path), Some(d)) = (dump, vc.dump.as_ref()) {
        std::fs::write(path, d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    if let (Some(path), Some(d)) = (front, vc.front.as_ref()) {
        std::fs::write(path, d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    if let (Some(path), Some(d)) = (vcd, vc.voc_vdump().as_ref()) {
        std::fs::write(path, d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    if let (Some(path), Some(d)) = (rcd, vc.voc_rdump().as_ref()) {
        std::fs::write(path, d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    if let (Some(path), Some(d)) = (rrd, vc.rr.as_ref().and_then(|r| r.dump.as_ref())) {
        std::fs::write(path, d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let pk = y.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-9);
    if pk > 0.99 {
        y.iter_mut().for_each(|v| *v *= 0.99 / pk);
    }
    save_wav_mono(&cmd.output, &y, 48_000)?;
    println!("artic: {} s → {} | RTF {:.3}(壁時計)", n * HOP / 48_000, cmd.output.display(), el / (n as f64 * HOP as f64 / 48_000.0));
    Ok(())
}

pub fn run_realtime_artic(cmd: RealtimeArticCmd) -> Result<()> {
    use cpal::traits::DeviceTrait;
    use lightvc_core::nvoc::{DELAY, HOP, SR};
    let mut vc = cmd.model.load()?;
    let input = lightvc_audio::engine::default_input()?;
    let output = lightvc_audio::engine::default_output()?;
    let in_cfg = input.default_input_config()?;
    let out_cfg = output.default_output_config()?;
    let buf = if cmd.buffer == 0 { cpal::BufferSize::Default } else { cpal::BufferSize::Fixed(cmd.buffer) };
    let (engine, mut io, dev_sr) = match lightvc_audio::AudioEngine::start_with(
        &input, &output, SR as u32, SR as u32, in_cfg.channels(), out_cfg.channels(), buf) {
        Ok((e, io)) => (e, io, SR),
        Err(e48) => {
            let (isr, osr) = (in_cfg.sample_rate(), out_cfg.sample_rate());
            anyhow::ensure!(isr == osr, "48 kHz で開けず({e48})、入出力の既定レートも不一致 ({isr} / {osr})");
            let (e, io) = lightvc_audio::AudioEngine::start_with(&input, &output, isr, osr, in_cfg.channels(), out_cfg.channels(), buf)?;
            (e, io, isr as usize)
        }
    };
    let conv = dev_sr != SR;
    let mut up = if conv { Some(lightvc_audio::RateConv::new(dev_sr, SR, HOP)?) } else { None };
    let mut down = if conv { Some(lightvc_audio::RateConv::new(SR, dev_sr, HOP)?) } else { None };
    println!("構音 VC realtime | device {} Hz{} | algorithmic latency {} ms(出力部)+ ブロック {} ms + resampler + device buffers",
             dev_sr, if conv { " (48 kHz へ変換)" } else { "" }, 1000 * DELAY / SR, 1000 * HOP / SR);
    for _ in 0..HOP {
        let _ = io.playback.push(0.0);
    }
    let mut cap: Vec<f32> = Vec::new();
    let mut in48: Vec<f32> = Vec::new();
    let mut out48: Vec<f32> = Vec::new();
    let mut out = vec![0f32; HOP];
    let mut times: Vec<f64> = Vec::new();
    let t_start = std::time::Instant::now();
    let mut last_report = std::time::Instant::now();
    loop {
        if engine.is_disconnected() {
            anyhow::bail!("オーディオデバイスが切断された");
        }
        if cmd.seconds > 0 && t_start.elapsed().as_secs() >= cmd.seconds {
            break;
        }
        let n = io.capture.slots();
        if n == 0 {
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }
        for _ in 0..n {
            cap.push(io.capture.pop().unwrap_or(0.0));
        }
        match up.as_mut() {
            Some(u) => {
                while cap.len() >= u.input_frames_next() {
                    let k = u.input_frames_next();
                    in48.extend_from_slice(u.process(&cap[..k])?);
                    cap.drain(..k);
                }
            }
            None => in48.append(&mut cap),
        }
        while in48.len() >= HOP {
            let t0 = std::time::Instant::now();
            vc.process_block(&in48[..HOP], &mut out);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
            in48.drain(..HOP);
            out48.extend(out.iter().map(|v| v.clamp(-1.0, 1.0)));
        }
        match down.as_mut() {
            Some(d) => {
                while out48.len() >= d.input_frames_next() {
                    let k = d.input_frames_next();
                    for v in d.process(&out48[..k])? {
                        let _ = io.playback.push(*v);
                    }
                    out48.drain(..k);
                }
            }
            None => {
                for v in out48.drain(..) {
                    let _ = io.playback.push(v);
                }
            }
        }
        if last_report.elapsed().as_secs() >= 5 && !times.is_empty() {
            let mut t = times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| t[((t.len() as f64 - 1.0) * p) as usize];
            println!("  blocks {} | compute ms p50 {:.2} p95 {:.2} p99 {:.2} (予算 5.0) | overrun {} underrun {}",
                     t.len(), q(0.5), q(0.95), q(0.99), engine.overrun_count(), engine.underrun_count());
            times.clear();
            last_report = std::time::Instant::now();
        }
    }
    Ok(())
}

#[derive(Parser)]
pub struct GuiCmd {
    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: Option<PathBuf>,
    #[arg(long)]
    pub cuda: bool,
    #[arg(long)]
    pub metal: bool,
    /// Launch the GUI with mock data for screenshot capture. No model,
    /// no audio devices required. The user switches tabs inside the app.
    #[arg(long)]
    pub demo: bool,
}

#[derive(Parser)]
pub struct RoundtripCmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(short, long, default_value = "roundtrip_output.wav")]
    pub output: PathBuf,

    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: PathBuf,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub metal: bool,
}

#[derive(Parser)]
pub struct ConvertCmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(short, long)]
    pub reference: PathBuf,

    #[arg(short, long, default_value = "converted_output.wav")]
    pub output: PathBuf,

    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: PathBuf,

    #[arg(long, env = "LIGHTVC_CONVERTER_WEIGHTS")]
    pub converter_weights: PathBuf,

    #[arg(long)]
    pub converter_config: Option<PathBuf>,

    #[arg(
        long,
        default_value = "balanced",
        help = "strict | balanced | quality | full"
    )]
    pub mode: String,

    #[arg(
        long,
        default_value = "1.0",
        help = "Velocity scale (guidance). 1.0 = training-matched, >1 amplifies conversion"
    )]
    pub velocity_scale: f64,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub metal: bool,
}

#[derive(Parser)]
pub struct ConvertB1Cmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(short, long, help = "Pre-computed ECAPA timbre safetensors")]
    pub timbre: PathBuf,

    #[arg(short, long, default_value = "converted_b1.wav")]
    pub output: PathBuf,

    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: PathBuf,

    #[arg(long, default_value = "models/dac_quantizer.safetensors")]
    pub quantizer_weights: PathBuf,

    #[arg(long, default_value = "models/utte_adapter_b1.safetensors")]
    pub adapter_weights: PathBuf,

    #[arg(long, default_value = "balanced", help = "strict | balanced")]
    pub mode: String,

    #[arg(long, default_value = "5.0")]
    pub tau: f64,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub metal: bool,
}

#[derive(Parser)]
pub struct ResynthCmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(long, env = "LIGHTVC_FREEC_WEIGHTS", help = "freeC vocoder safetensors")]
    pub weights: PathBuf,

    #[arg(
        long,
        env = "LIGHTVC_MEL_BASIS",
        help = "librosa slaney mel filterbank safetensors (key `mel_basis`)"
    )]
    pub mel_basis: PathBuf,

    #[arg(short, long, default_value = "resynth_output.wav")]
    pub output: PathBuf,

    #[arg(long, default_value = "4", help = "Mel frames per streaming chunk (k*hop samples)")]
    pub k: usize,
}

pub fn select_device(cuda: bool, metal: bool) -> Result<Device> {
    if cuda {
        Ok(Device::new_cuda(0)?)
    } else if metal {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn run_roundtrip(cmd: RoundtripCmd) -> Result<()> {
    println!("LightVC Phase 0: DAC Round-Trip Test");
    println!("  Input:  {}", cmd.input.display());
    println!("  Output: {}", cmd.output.display());

    let device = select_device(cmd.cuda, cmd.metal)?;
    println!("  Device: {:?}", device);

    let (input_pcm, input_sr) = load_wav_mono(&cmd.input)?;
    println!(
        "  Loaded: {} samples at {} Hz ({:.1}s)",
        input_pcm.len(),
        input_sr,
        input_pcm.len() as f32 / input_sr as f32
    );

    let pcm_44k = if input_sr != 44_100 {
        println!("  Resampling {} → 44100 Hz...", input_sr);
        resample_to_44100(&input_pcm, input_sr)?
    } else {
        input_pcm
    };

    let padded = lightvc_core::codec::pad_to_hop(pcm_44k);
    println!("  Padded: {} samples", padded.len());

    println!("  Loading DAC weights...");
    let dac_config = DacConfig::default();
    let codec = lightvc_core::DacCodec::from_file(&cmd.dac_weights, &dac_config, device)?;

    println!("  Encoding...");
    let latent = codec.encode_pcm(&padded)?;
    println!("  Latent shape: {:?}", latent.shape());

    println!("  Decoding...");
    let output_pcm = codec.decode_to_pcm(&latent)?;
    println!("  Output: {} samples", output_pcm.len());

    let trim_len = padded.len().min(output_pcm.len());
    let output_trimmed = &output_pcm[..trim_len];

    save_wav_mono(&cmd.output, output_trimmed, 44_100)?;
    println!("  Saved: {}", cmd.output.display());

    let mse: f64 = output_trimmed
        .iter()
        .zip(padded.iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum::<f64>()
        / trim_len as f64;
    println!("  MSE: {:.6}", mse);
    println!("  RMSE: {:.6}", mse.sqrt());

    println!("Done.");
    Ok(())
}

pub fn run_convert(cmd: ConvertCmd) -> Result<()> {
    println!("LightVC Offline Conversion");

    let device = select_device(cmd.cuda, cmd.metal)?;

    let converter_config = if let Some(cfg_path) = &cmd.converter_config {
        let cfg_str = std::fs::read_to_string(cfg_path)?;
        serde_json::from_str(&cfg_str)?
    } else {
        ConverterConfig::default()
    };

    println!("Loading DAC...");
    let dac_config = DacConfig::default();

    let mode = match cmd.mode.as_str() {
        "strict" => LatencyMode::Strict,
        "quality" => LatencyMode::Quality,
        _ => LatencyMode::Balanced,
    };
    let use_full = cmd.mode == "full";

    println!("Loading converter...");
    let vb = lightvc_core::weights::load_varbuilder(&cmd.converter_weights, DType::F32, &device)?;
    let converter = AnyConverter::new(converter_config, vb)?;

    let mut pipeline = VcPipeline::new(&cmd.dac_weights, &dac_config, converter, mode, device)?;
    pipeline.velocity_scale = cmd.velocity_scale;

    println!("Loading reference: {}", cmd.reference.display());
    let (ref_pcm, ref_sr) = load_wav_mono(&cmd.reference)?;
    let ref_44k = if ref_sr != 44_100 {
        resample_to_44100(&ref_pcm, ref_sr)?
    } else {
        ref_pcm
    };
    pipeline.set_target(&ref_44k)?;

    println!("Loading source: {}", cmd.input.display());
    let (src_pcm, src_sr) = load_wav_mono(&cmd.input)?;
    let src_44k = if src_sr != 44_100 {
        resample_to_44100(&src_pcm, src_sr)?
    } else {
        src_pcm
    };
    let padded = lightvc_core::codec::pad_to_hop(src_44k);

    if use_full {
        println!("Mode: full (offline, no chunking) — SOTA quality");
        println!("Processing...");
        let output = pipeline.process_full(&padded)?;
        save_wav_mono(&cmd.output, &output, 44_100)?;
        println!("Saved: {}", cmd.output.display());
        return Ok(());
    }

    let chunk_size = pipeline.chunk_samples();
    println!(
        "Chunk size: {} samples ({:.1} ms)",
        chunk_size,
        pipeline.chunk_ms()
    );
    println!("Processing...");

    let mut output = Vec::with_capacity(padded.len());
    let mut i = 0;
    while i < padded.len() {
        let end = (i + chunk_size).min(padded.len());
        let chunk = &padded[i..end];

        let chunk_padded = if chunk.len() < chunk_size {
            let mut c = chunk.to_vec();
            c.resize(chunk_size, 0.0);
            c
        } else {
            chunk.to_vec()
        };

        let out_chunk = pipeline.process_chunk(&chunk_padded)?;
        output.extend_from_slice(&out_chunk[..end - i]);
        i = end;
    }

    save_wav_mono(&cmd.output, &output, 44_100)?;
    println!("Saved: {}", cmd.output.display());

    Ok(())
}

pub fn run_convert_b1(cmd: ConvertB1Cmd) -> Result<()> {
    println!("LightVC B1 Offline Conversion");
    println!("  Input:  {}", cmd.input.display());
    println!("  Timbre: {}", cmd.timbre.display());
    println!("  Output: {}", cmd.output.display());

    let device = select_device(cmd.cuda, cmd.metal)?;
    println!("  Device: {:?}", device);

    let (input_pcm, input_sr) = load_wav_mono(&cmd.input)?;
    println!(
        "  Loaded: {} samples at {} Hz ({:.1}s)",
        input_pcm.len(),
        input_sr,
        input_pcm.len() as f32 / input_sr as f32
    );

    let pcm_44k = resample_to_44100(&input_pcm, input_sr)?;

    let timbre = {
        let vb = lightvc_core::weights::load_varbuilder(&cmd.timbre, DType::F32, &device)?;
        vb.get((1, 192), "timbre")?
    };

    let chunk_mode = match cmd.mode.as_str() {
        "strict" => {
            eprintln!(
                "  WARNING: Strict mode (1-frame decode) produces C/D quality with B1.\n\
                 \x20          Balanced is recommended. See issue #7."
            );
            lightvc_core::streaming::ChunkMode::Strict
        }
        _ => lightvc_core::streaming::ChunkMode::Balanced,
    };

    let mut b1 = lightvc_core::b1_pipeline::B1Streaming::new(
        &cmd.dac_weights,
        &cmd.quantizer_weights,
        &cmd.adapter_weights,
        chunk_mode,
        device,
    )?;
    b1.set_timbre(timbre);
    b1.set_tau(cmd.tau);

    println!(
        "  Pipeline: B1 UTTE adapter ({}, tau={})",
        cmd.mode, cmd.tau
    );
    println!("  Processing...");
    let output = b1.process_full(&pcm_44k)?;
    println!(
        "  Output: {} samples ({:.1}s)",
        output.len(),
        output.len() as f32 / 44_100.0
    );

    save_wav_mono(&cmd.output, &output, 44_100)?;
    println!("  Saved: {}", cmd.output.display());

    Ok(())
}

#[derive(clap::Args)]
pub struct V2fCmd {
    /// Input WAV (44.1 kHz mono)
    pub input: std::path::PathBuf,
    /// Flat f32 weights (training/export_v2f.py)
    #[arg(short, long, default_value = "models/v2f.bin")]
    pub weights: std::path::PathBuf,
    /// Mel filterbank [80][513] flat f32
    #[arg(long, default_value = "models/mel_fb_1024_80.bin")]
    pub mel_fb: std::path::PathBuf,
    /// Mel->linear map [257][80] flat f32
    #[arg(long, default_value = "models/mel2lin_W.bin")]
    pub mel2lin: std::path::PathBuf,
    #[arg(short, long, default_value = "v2f_output.wav")]
    pub output: std::path::PathBuf,
}

#[derive(clap::Args)]
pub struct VcCmd {
    /// Input WAV (44.1 kHz mono, 男声)
    pub input: std::path::PathBuf,
    /// E 重み (training/export_eg.py --kind e)
    #[arg(long, default_value = "models/e1.bin")]
    pub e: std::path::PathBuf,
    /// G 重み = voice cartridge (training/export_eg.py --kind g)
    #[arg(long, default_value = "models/g1.bin")]
    pub g: std::path::PathBuf,
    /// V 重み
    #[arg(long, default_value = "models/v2f.bin")]
    pub v: std::path::PathBuf,
    #[arg(long, default_value = "models/mel_fb_1024_80.bin")]
    pub mel_fb: std::path::PathBuf,
    #[arg(long, default_value = "models/mel2lin_W.bin")]
    pub mel2lin: std::path::PathBuf,
    /// f0 シフト (半音・cartridge 定数)
    #[arg(long, default_value_t = 15.0)]
    pub shift: f32,
    /// E/G を f16 重みで実行 (帯域半減で ~2 倍速。要 F16C)。proxy 判定は f32 と
    /// 同値を確認済み (SECS 0.722 / CER 一致)。--f16=false で f32 に戻せる
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub f16: bool,
    #[arg(short, long, default_value = "vc_output.wav")]
    pub output: std::path::PathBuf,
}

pub fn run_vc(cmd: VcCmd) -> Result<()> {
    use lightvc_core::eg::Eg1d;
    use lightvc_core::v2f_infer::{V2fEngine, V2fStream};
    use lightvc_core::vc_stream::VcStream;
    let rdf = |p: &std::path::Path| -> Result<Vec<f32>> {
        Ok(std::fs::read(p)
            .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    println!("LightVC voice conversion (causal, streaming)");
    let e = std::sync::Arc::new(Eg1d::from_flat(&rdf(&cmd.e)?, 80, 768, 256, 8));
    let g = std::sync::Arc::new(Eg1d::from_flat(&rdf(&cmd.g)?, 770, 80, 256, 6));
    let eng = std::sync::Arc::new(
        V2fEngine::load(&cmd.v, &cmd.mel_fb, &cmd.mel2lin, 24, 8)
            .map_err(|e| anyhow::anyhow!("weights: {e}"))?);
    let fb = rdf(&cmd.mel_fb)?;
    let (x, sr) = load_wav_mono(&cmd.input)?;
    anyhow::ensure!(sr == 44_100, "44.1 kHz expected, got {sr}");
    // 学習系の慣習: front/E/G は ×32768 スケール、V 出力は [-1,1]
    let x32: Vec<f32> = x.iter().map(|v| v * 32768.0).collect();
    let v = V2fStream::with_threads(eng, 2);
    let mut vc = if cmd.f16 {
        use lightvc_core::eg::Eg1dH;
        use lightvc_core::vc_stream::EgAny;
        VcStream::new_any(
            EgAny::F16(std::sync::Arc::new(Eg1dH::from_net(&e))),
            EgAny::F16(std::sync::Arc::new(Eg1dH::from_net(&g))),
            v, fb, cmd.shift)
    } else {
        VcStream::new(e, g, v, fb, cmd.shift)
    };
    let n = (x32.len() / VcStream::BLOCK) * VcStream::BLOCK;
    let mut y = Vec::with_capacity(n);
    let t0 = std::time::Instant::now();
    for b in x32[..n].chunks(VcStream::BLOCK) {
        y.extend(vc.process_block(b));
    }
    let el = t0.elapsed().as_secs_f64();
    println!("  {} samples  shift {:+.1} st  RTF {:.3}", n, cmd.shift,
             el / (n as f64 / 44_100.0));
    save_wav_mono_f32(&cmd.output, &y, 44_100)?;
    println!("  Saved: {}", cmd.output.display());
    Ok(())
}

#[derive(Parser)]
pub struct RealtimeDdspCmd {
    /// DDSP-VC の書き出し (training/export_ddsp_vc.py の出力ディレクトリ)
    #[arg(long, default_value = "results/ddsp_vc/export")]
    pub model: std::path::PathBuf,
    /// 目標話者の参照音声 (48 kHz・3 秒程度)
    #[arg(long)]
    pub reference: std::path::PathBuf,
    /// 利用者の登録音声 (48 kHz・f0 レジスタ推定用。省略時は男声の既定値 120 Hz)
    #[arg(long)]
    pub enroll: Option<std::path::PathBuf>,
    /// 実行秒数 (0 = 停止まで)
    #[arg(long, default_value_t = 0)]
    pub seconds: u64,
    /// cpal のバッファ (フレーム数・0 = 既定)
    #[arg(long, default_value_t = 240)]
    pub buffer: u32,
}

pub fn run_realtime_ddsp(cmd: RealtimeDdspCmd) -> Result<()> {
    use cpal::traits::DeviceTrait;
    use lightvc_core::ddsp_vc::{map_register, register_from_pcm, CausalYin, DdspVc, DELAY, HOP, SR};
    // デバイスは 48 kHz で開く。開けなければ既定レートで開き、48 kHz との間を RateConv で変換する。
    let (rf, rsr) = load_wav_mono(&cmd.reference)?;
    anyhow::ensure!(rsr as usize == SR, "参照音声は 48 kHz が必要 (got {rsr})");
    let (mu_t, sd_t) = register_from_pcm(&rf);
    let (mu_s, sd_s) = match &cmd.enroll {
        Some(e) => {
            let (x, sr) = load_wav_mono(e)?;
            anyhow::ensure!(sr as usize == SR, "登録音声は 48 kHz が必要 (got {sr})");
            register_from_pcm(&x)
        }
        None => ((120f32).ln(), 0.15),
    };
    let mut vc = DdspVc::load(&cmd.model)?;
    vc.set_target_ref(&rf[..rf.len().min(3 * SR)]);
    let input = lightvc_audio::engine::default_input()?;
    let output = lightvc_audio::engine::default_output()?;
    let in_cfg = input.default_input_config()?;
    let out_cfg = output.default_output_config()?;
    let buf = if cmd.buffer == 0 { cpal::BufferSize::Default } else { cpal::BufferSize::Fixed(cmd.buffer) };
    let (engine, mut io, dev_sr) = match lightvc_audio::AudioEngine::start_with(
        &input, &output, SR as u32, SR as u32, in_cfg.channels(), out_cfg.channels(), buf) {
        Ok((e, io)) => (e, io, SR),
        Err(e48) => {
            let (isr, osr) = (in_cfg.sample_rate(), out_cfg.sample_rate());
            anyhow::ensure!(isr == osr, "48 kHz で開けず({e48})、入出力の既定レートも不一致 ({isr} / {osr})");
            let (e, io) = lightvc_audio::AudioEngine::start_with(&input, &output, isr, osr, in_cfg.channels(), out_cfg.channels(), buf)?;
            (e, io, isr as usize)
        }
    };
    let conv = dev_sr != SR;
    let mut up = if conv { Some(lightvc_audio::RateConv::new(dev_sr, SR, HOP)?) } else { None };
    let mut down = if conv { Some(lightvc_audio::RateConv::new(SR, dev_sr, HOP)?) } else { None };
    println!("DDSP-VC realtime | device {} Hz{} | f0 register {:.0} Hz → {:.0} Hz | algorithmic latency {} ms + resampler + device buffers",
             dev_sr, if conv { " (48 kHz へ変換)" } else { "" }, mu_s.exp(), mu_t.exp(), 1000 * DELAY / SR);
    for _ in 0..HOP {
        let _ = io.playback.push(0.0);
    }
    let mut yin = CausalYin::new(0.45);
    let mut cap: Vec<f32> = Vec::new();
    let mut in48: Vec<f32> = Vec::new();
    let mut out48: Vec<f32> = Vec::new();
    let mut out = vec![0f32; HOP];
    let mut times: Vec<f64> = Vec::new();
    let t_start = std::time::Instant::now();
    let mut last_report = std::time::Instant::now();
    loop {
        if engine.is_disconnected() {
            anyhow::bail!("オーディオデバイスが切断された");
        }
        if cmd.seconds > 0 && t_start.elapsed().as_secs() >= cmd.seconds {
            break;
        }
        let n = io.capture.slots();
        if n == 0 {
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }
        for _ in 0..n {
            cap.push(io.capture.pop().unwrap_or(0.0));
        }
        match up.as_mut() {
            Some(u) => {
                while cap.len() >= u.input_frames_next() {
                    let k = u.input_frames_next();
                    in48.extend_from_slice(u.process(&cap[..k])?);
                    cap.drain(..k);
                }
            }
            None => in48.append(&mut cap),
        }
        while in48.len() >= HOP {
            let t0 = std::time::Instant::now();
            let f0 = map_register(yin.push(&in48[..HOP]), mu_s, sd_s, mu_t, sd_t);
            vc.process_block(&in48[..HOP], f0, &mut out);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
            in48.drain(..HOP);
            out48.extend(out.iter().map(|v| v.clamp(-1.0, 1.0)));
        }
        match down.as_mut() {
            Some(d) => {
                while out48.len() >= d.input_frames_next() {
                    let k = d.input_frames_next();
                    for v in d.process(&out48[..k])? {
                        let _ = io.playback.push(*v);
                    }
                    out48.drain(..k);
                }
            }
            None => {
                for v in out48.drain(..) {
                    let _ = io.playback.push(v);
                }
            }
        }
        if last_report.elapsed().as_secs() >= 5 && !times.is_empty() {
            let mut t = times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| t[((t.len() as f64 - 1.0) * p) as usize];
            println!("  blocks {} | compute ms p50 {:.2} p95 {:.2} p99 {:.2} (予算 5.0) | overrun {} underrun {}",
                     t.len(), q(0.5), q(0.95), q(0.99), engine.overrun_count(), engine.underrun_count());
            times.clear();
            last_report = std::time::Instant::now();
        }
    }
    Ok(())
}

pub fn run_v2f(cmd: V2fCmd) -> Result<()> {
    use lightvc_core::v2f_infer::V2fEngine;
    println!("LightVC v2f resynthesis (causal, 30 ms class)");
    let eng = V2fEngine::load(&cmd.weights, &cmd.mel_fb, &cmd.mel2lin, 24, 8)
        .map_err(|e| anyhow::anyhow!("weights: {e}"))?;
    let (x, sr) = load_wav_mono(&cmd.input)?;
    anyhow::ensure!(sr == 44_100, "44.1 kHz expected, got {sr}");
    // prior の雑音は決定的に（xorshift。乱数品質は要らない、再現性が要る）
    let mut st = 0x9e3779b97f4a7c15u64;
    let noise: Vec<f32> = (0..x.len()).map(|_| {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 40) as f32 / 8388608.0) - 1.0
    }).collect();
    let t0 = std::time::Instant::now();
    let y = eng.process(&x, &noise);
    let el = t0.elapsed().as_secs_f64();
    println!("  {} samples  RTF {:.3}", y.len(), el / (x.len() as f64 / 44_100.0));
    save_wav_mono(&cmd.output, &y, 44_100)?;
    println!("  Saved: {}", cmd.output.display());
    Ok(())
}

pub fn run_resynth(cmd: ResynthCmd) -> Result<()> {
    println!("LightVC FreeVocoder Resynthesis");
    println!("  Input:     {}", cmd.input.display());
    println!("  Weights:   {}", cmd.weights.display());
    println!("  Mel basis: {}", cmd.mel_basis.display());
    println!("  Output:    {}", cmd.output.display());

    let mut rs = FreeResynth::new(&cmd.weights, &cmd.mel_basis, cmd.k, Device::Cpu)?;
    let chunk = rs.chunk_samples();
    println!(
        "  Streaming: K={} → {} samples/chunk ({:.1} ms), algorithmic latency {:.1} ms",
        cmd.k,
        chunk,
        chunk as f32 / 44.1,
        rs.algorithmic_latency_ms()
    );

    let (src_pcm, src_sr) = load_wav_mono(&cmd.input)?;
    let pcm_44k = if src_sr != 44_100 {
        println!("  Resampling {} → 44100 Hz...", src_sr);
        resample_to_44100(&src_pcm, src_sr)?
    } else {
        src_pcm
    };
    println!("  Loaded: {} samples @ 44.1 kHz", pcm_44k.len());

    // Feed the deployed realtime path in fixed `chunk_samples()` blocks. The
    // trailing remainder (< chunk, hence < one whole mel frame after the mel
    // state's carry) is fed as-is; any sub-frame tail stays buffered in the mel
    // state and is simply dropped when the stream ends (the honest streaming
    // semantics — no artificial zero-padding of the analysis window).
    let mut output = Vec::with_capacity(pcm_44k.len());
    for blk in pcm_44k.chunks(chunk) {
        output.extend_from_slice(&rs.process_chunk(blk)?);
    }

    save_wav_mono_f32(&cmd.output, &output, 44_100)?;
    println!("  Saved: {} ({} samples)", cmd.output.display(), output.len());
    Ok(())
}

pub fn run_gui(cmd: GuiCmd) -> Result<()> {
    println!("LightVC GUI starting...");

    let icon = crate::assets::load_icon();

    let dac_weights = cmd
        .dac_weights
        .unwrap_or_else(|| std::path::PathBuf::from("models/dac_44khz.safetensors"));
    let mut app = crate::app::LightVcApp::new(dac_weights);
    if cmd.demo {
        app.enable_demo();
    }
    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size([800.0, 600.0])
        .with_title("LightVC");
    if let Some(icon_data) = icon {
        viewport = viewport.with_icon(std::sync::Arc::new(icon_data));
    }
    let opts = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native("LightVC", opts, Box::new(move |_cc| Ok(Box::new(app))))?;

    Ok(())
}

// --- WAV I/O helpers ---

pub fn load_wav_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = match spec.bits_per_sample {
                16 => 32768.0f32,
                24 => 8388608.0,
                32 => 2147483648.0,
                _ => 32768.0,
            };
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

pub fn save_wav_mono(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    Ok(())
}

/// Float WAV writer that preserves the sample's native range (no `[-1, 1]`
/// clamp). Used for FreeVocoder resynthesis, whose output legitimately peaks
/// above unity (like the training reference); clamping would clip those peaks
/// and corrupt the resynthesis fidelity.
pub fn save_wav_mono_f32(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn resample_to_44100(input: &[f32], input_sr: u32) -> Result<Vec<f32>> {
    if input_sr == 44_100 {
        return Ok(input.to_vec());
    }
    let mut resampler = Resampler::new(input_sr as usize, 4096)?;
    let mut output = Vec::new();
    let chunk_size = resampler.input_frames_needed_up();

    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let chunk = &input[pos..end];
        if chunk.len() < chunk_size {
            let mut padded = chunk.to_vec();
            padded.resize(chunk_size, 0.0);
            let resampled = resampler.process_up(&padded)?;
            output.extend_from_slice(resampled);
        } else {
            let resampled = resampler.process_up(chunk)?;
            output.extend_from_slice(resampled);
        }
        pos = end;
    }

    Ok(output)
}
