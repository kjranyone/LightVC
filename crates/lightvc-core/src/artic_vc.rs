//! 構音 VC の製品経路(表の引き・G なし・current/artic_vc.md・converter.md の基準経路 P0C2 の処方)のストリーミング推論。
//!
//! 入力ブロック t(48kHz・HOP)→ C1(単位 u_t)・F1(f0)・F2(c0n + 因果 3 平均)→ 条件 [31] = [c0n, avg3(T[:, u]/10), log(f0o/200), 有声, per_mean·有声]
//! → 励起(f0o)→ 出力部 rvoc → 出力ブロック。f0o = exp(log f0 − med + mu)(med = 元の声の最初の 2000 フレームの有声 log f0 の走る中央値・以後固定)。
//! 登録(表 T・mu・per_mean)は training/export_artic_vc.py(Python の CheapTrick で作る)。先読み 0。

use std::path::Path;

use anyhow::{Context, Result};

use crate::c1::C1;
use crate::f0est::F0Est;
use crate::lvl::{EnvEst, Lvl};
use crate::nvoc::{HOP, SR};
use crate::rr::Rr;
use crate::rvoc::{Excite, RVoc, D_COND};

pub const W_SM: usize = 3;
pub const MED_FRAMES: usize = 2000;

/// 色づけの共分散の縮約 C + λ·(tr C / d)·I(伸縮で潰れる方向の増幅を抑える・λ 0 で最大 455 倍 → 0.1 で約 5 倍)
pub const COLOR_SHRINK: f64 = 0.1;

pub struct Enroll {
    pub table: Vec<f32>,
    pub d: usize,
    pub k: usize,
    pub mu: f32,
    pub per_mean: [f32; 4],
    /// 有声フレームの周期性の 75 パーセンタイル(古い登録では None)。平均は息の多い立ち上がり・語尾に引かれて低く、定数で入れると倍音間の雑音が増える(D49〜D51)
    pub per_p75: Option<[f32; 4]>,
    /// 単位ごとの観測フレーム数(縮約用・古い登録では空)
    pub counts: Vec<u32>,
    /// 単位の中の動き(包絡 − 表[単位])の有声フレームの共分散 [d][d](古い登録では空・D74 で話者ごとに違う)
    pub res_cov: Vec<f32>,
    /// 単位ごとの c0 の話者平均からの相対値 [k](c0n の単位・D121)。古い登録には無い
    pub t0: Option<Vec<f32>>,
}

impl Enroll {
    /// enroll.bin(表 [d][k] f32 LE)と enroll.json(d・k・mu・per_mean・counts)を書く。
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let bytes: Vec<u8> = self.table.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join("enroll.bin"), bytes)?;
        let j = serde_json::json!({"D": self.d, "K": self.k, "mu": self.mu, "per_mean": self.per_mean.to_vec(), "per_p75": self.per_p75.map(|v| v.to_vec()), "counts": self.counts, "res_cov": self.res_cov, "t0": self.t0});
        std::fs::write(dir.join("enroll.json"), serde_json::to_string_pretty(&j)?)?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("enroll.json"))?)?;
        let d = j["D"].as_u64().context("D")? as usize;
        let k = j["K"].as_u64().context("K")? as usize;
        let raw = std::fs::read(dir.join("enroll.bin"))?;
        let table: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        anyhow::ensure!(table.len() == d * k, "enroll.bin の大きさ {} != {}", table.len(), d * k);
        let pm = j["per_mean"].as_array().context("per_mean")?;
        let mut per_mean = [0.0f32; 4];
        for (i, v) in pm.iter().enumerate().take(4) {
            per_mean[i] = v.as_f64().unwrap_or(0.0) as f32;
        }
        let counts: Vec<u32> = j["counts"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect()).unwrap_or_default();
        let per_p75 = j["per_p75"].as_array().filter(|a| a.len() >= 4).map(|a| {
            let mut o = [0.0f32; 4];
            for (i, v) in a.iter().enumerate().take(4) {
                o[i] = v.as_f64().unwrap_or(0.0) as f32;
            }
            o
        });
        let res_cov: Vec<f32> = j["res_cov"].as_array().filter(|a| a.len() == d * d).map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default();
        let t0: Option<Vec<f32>> = j["t0"].as_array().filter(|a| a.len() == k).map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect());
        Ok(Self { table, d, k, mu: j["mu"].as_f64().context("mu")? as f32, per_mean, per_p75, counts, res_cov, t0 })
    }
}

/// 常駐のワーカースレッド(1 ブロックの中で並列に動かす = 遅延は増えない)。
struct Worker<R: Send + 'static> {
    tx: std::sync::mpsc::Sender<Vec<f32>>,
    rx: std::sync::mpsc::Receiver<(Vec<f32>, R)>,
}

impl<R: Send + 'static> Worker<R> {
    fn spawn<M: Send + 'static>(mut m: M, f: fn(&mut M, &[f32]) -> R) -> Self {
        let (tx, rx_in) = std::sync::mpsc::channel::<Vec<f32>>();
        let (tx_out, rx) = std::sync::mpsc::channel::<(Vec<f32>, R)>();
        std::thread::spawn(move || {
            crate::affinity::pin_current_to_perf();
            while let Ok(x) = crate::affinity::recv_spin(&rx_in) {
                let r = f(&mut m, &x);
                if tx_out.send((x, r)).is_err() {
                    break;
                }
            }
        });
        Self { tx, rx }
    }
}

/// 操作者の遅い音色の揺らぎ(D451・training/train_ppgvoc.py の opslow_causal): r = 操作者の包絡 − Σq T_op / 10 →
/// 過去 100 フレームの平均 s − s の指数移動平均(時定数 tau)→ 遅い揺らぎの基底へ射影 × 利得 を生成器へ渡す(RVoc::set_ext_slow・文脈の符号器の後に足す・乱数の揺らぎの代わり)。先読み 0・発話の統計なし。
pub struct OpSlow {
    t_op: Vec<f32>,
    b: Vec<f64>,
    kb: usize,
    a: f64,
    gain: f64,
    ring: Vec<f64>,
    pos: usize,
    m: Vec<f64>,
    n: usize,
}

impl OpSlow {
    pub fn new(t_op: Vec<f32>, b: Vec<f64>, kb: usize, tau_s: f64, gain: f64) -> Self {
        Self { t_op, b, kb, a: (-1.0 / (tau_s * 200.0)).exp(), gain, ring: vec![0.0; 100 * 24], pos: 0, m: vec![0.0; 24], n: 0 }
    }

    fn apply(&mut self, env_op: &[f32], q: &[f32], k: usize, z: &mut [f32]) {
        let mut r = [0.0f64; 24];
        for (i, v) in r.iter_mut().enumerate() {
            let base = self.t_op[i * k..(i + 1) * k].iter().zip(q.iter()).map(|(&t, &w)| t * w).sum::<f32>() / 10.0;
            *v = (env_op[i] - base) as f64;
        }
        if self.n == 0 {
            for j in 0..100 {
                self.ring[j * 24..(j + 1) * 24].copy_from_slice(&r);
            }
        } else {
            self.ring[self.pos * 24..(self.pos + 1) * 24].copy_from_slice(&r);
        }
        self.pos = (self.pos + 1) % 100;
        let mut s = [0.0f64; 24];
        for j in 0..100 {
            for i in 0..24 {
                s[i] += self.ring[j * 24 + i];
            }
        }
        for i in 0..24 {
            s[i] /= 100.0;
            self.m[i] = if self.n == 0 { s[i] } else { self.a * self.m[i] + (1.0 - self.a) * s[i] };
        }
        self.n += 1;
        let mut w = vec![0.0f64; self.kb];
        for (j, wj) in w.iter_mut().enumerate() {
            *wj = (0..24).map(|i| self.b[i * self.kb + j] * (s[i] - self.m[i])).sum();
        }
        for i in 0..24 {
            let zi: f64 = (0..self.kb).map(|j| self.b[i * self.kb + j] * w[j]).sum();
            z[i] = (self.gain * zi) as f32;
        }
    }
}

/// 参照を読む R の S 形(R − G)の製品の口(D579〜d587・training/export_rr.py)。生成器の条件の包絡の行に gain·ΔS/10 を足す(遅い揺らぎの口・opslow の代わり)。
/// 問い = [C1 の事後の上位 16・(F3 の包絡 − 操作者の平均)/10・F2 の c0n − 操作者の平均・log f0 − 操作者の中央値・有声]・表の引き = 目標の表 × C1 の事後。
pub struct RrPath {
    /// 同期(フレーム t の R を待つ・一致検査用)。None なら worker(R を別のスレッドで 1 フレーム遅れで回す = 生成器のフレーム t に R(t − 1) の差分・音声の遅延は増えない)。
    net: Option<Rr>,
    worker: Option<Worker<Vec<f32>>>,
    pending: bool,
    buf: Vec<f32>,
    pub gain: f32,
    op: Option<([f32; 24], f32, f32)>,
    qv: Vec<f32>,
    tab: Vec<f32>,
    ds: Vec<f32>,
    z: Vec<f32>,
    sel: Vec<(f32, usize)>,
    /// 診断: Some のとき毎フレーム [問い nq・表 24・ΔS 24・Δc0 1] を積む(LIGHTVC_DUMP_RR)
    pub dump: Option<Vec<f32>>,
}

/// 低い高さの基音の傾き(D610〜D614・training/train_ppgvoc.py の h1fix_delta と同じ): 有声の枠で、目標の声域 ft = exp(mu) から
/// いまの f0 までの mel の帯に σ dB/oct(f0 より下は平ら・f ≥ ft は 0)を c1..c24 / 10 の差として表の包絡の行に足す。
pub struct H1Fix {
    sigma: f64,
    ft: f64,
    d: Vec<f32>,
    fc: Vec<f64>,
    nm: usize,
    dl: Vec<f64>,
}

impl H1Fix {
    /// dct_rows = Analyzer::dct_rows()(DCT の行 1..24 [24][n_mel])。fc = librosa の slaney の mel_frequencies(n_mel + 2, 0, 24 kHz) の内側。
    pub fn new(sigma: f32, mu: f32, dct_rows: (Vec<f32>, usize)) -> Self {
        let (d, nm) = dct_rows;
        let ls = 6.4f64.ln() / 27.0;
        let mel = |f: f64| if f < 1000.0 { f / (200.0 / 3.0) } else { 15.0 + (f / 1000.0).ln() / ls };
        let hz = |m: f64| if m < 15.0 { m * 200.0 / 3.0 } else { 1000.0 * ((m - 15.0) * ls).exp() };
        let (m0, m1) = (mel(0.0), mel(24000.0));
        let fc = (1..=nm).map(|i| hz(m0 + (m1 - m0) * i as f64 / (nm + 1) as f64)).collect();
        Self { sigma: sigma as f64, ft: (mu as f64).exp(), d, fc, nm, dl: vec![0.0; nm] }
    }

    /// lf = ln(f0 / 200)(条件の列 25)・有声の枠だけ呼ぶ。out = 条件の行 1..25。
    pub fn add(&mut self, lf: f32, out: &mut [f32]) {
        let f0 = (200.0 * (lf as f64).exp()).min(self.ft);
        let s = self.sigma * 10f64.ln() / 20.0;
        for (v, &f) in self.dl.iter_mut().zip(self.fc.iter()) {
            *v = s * (self.ft / f.max(f0).min(self.ft)).log2();
        }
        for (r, o) in out.iter_mut().enumerate().take(24) {
            let acc: f64 = self.d[r * self.nm..(r + 1) * self.nm].iter().zip(self.dl.iter()).map(|(&w, &v)| w as f64 * v).sum();
            *o += (acc / 10.0) as f32;
        }
    }
}

/// 生成器の後ろの適応 H1 EQ(D619〜D623・training/d619_h1eq.py と同じ): ブロックごとに生成器の出力の過去 2 周期(ブロックの始まりまで)の
/// H1 − H2 を 1 フレーム前の f0 で最小二乗(Hann)に測り、g = 0.7 g + 0.3 clamp(目標 − H1H2, 0, 15)(無声・f0 ≥ 目標の声域では 0)の
/// RBJ ピーキング(Q 3・中心 f0)を状態を引き継いで掛ける。先読み 0。
pub struct H1Eq {
    target: f64,
    ft: f64,
    fmax: f64,
    g: f64,
    z: [f64; 2],
    hist: Vec<f64>,
    f1: f32,
    f2: f32,
}

impl H1Eq {
    const Q: f64 = 3.0;
    const GMAX: f64 = 15.0;
    const NH: usize = 2048;

    pub fn new(target: f32, mu: f32, fmax: Option<f32>) -> Self {
        Self { target: target as f64, ft: (mu as f64).exp(), fmax: fmax.map_or(f64::INFINITY, |v| v as f64), g: 0.0, z: [0.0; 2], hist: vec![0.0; Self::NH], f1: 0.0, f2: 0.0 }
    }

    fn h12(&self, f0: f64) -> Option<f64> {
        let p = (SR as f64 / f0).round() as usize;
        let m = 2 * p;
        if m > self.hist.len() {
            return None;
        }
        let y = &self.hist[self.hist.len() - m..];
        let mut a = [[0.0f64; 5]; 5];
        let mut b = [0.0f64; 5];
        for (n, &v) in y.iter().enumerate() {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (m - 1) as f64).cos();
            let ph = 2.0 * std::f64::consts::PI * f0 * n as f64 / SR as f64;
            let r = [w, w * ph.cos(), w * ph.sin(), w * (2.0 * ph).cos(), w * (2.0 * ph).sin()];
            for i in 0..5 {
                b[i] += r[i] * v * w;
                for j in 0..5 {
                    a[i][j] += r[i] * r[j];
                }
            }
        }
        for c in 0..5 {
            let piv = (c..5).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap()).unwrap();
            a.swap(c, piv);
            b.swap(c, piv);
            if a[c][c].abs() < 1e-300 {
                return None;
            }
            for r in c + 1..5 {
                let f = a[r][c] / a[c][c];
                for k in c..5 {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
        let mut x = [0.0f64; 5];
        for c in (0..5).rev() {
            x[c] = (b[c] - (c + 1..5).map(|k| a[c][k] * x[k]).sum::<f64>()) / a[c][c];
        }
        let (a1, a2) = (x[1].hypot(x[2]), x[3].hypot(x[4]));
        Some(20.0 * ((a1 + 1e-12) / (a2 + 1e-12)).log10())
    }

    /// out = 生成器の出力のブロック(その場で EQ を掛ける)・f0o = いまのフレームの出力の f0(0 = 無声)。
    pub fn process(&mut self, out: &mut [f32], f0o: f32) {
        let f = self.f1 as f64;
        let graw = if f > 0.0 && f < self.ft.min(self.fmax) { self.h12(f).map_or(0.0, |d| (self.target - d).clamp(0.0, Self::GMAX)) } else { 0.0 };
        self.g = 0.7 * self.g + 0.3 * graw;
        let fc = if self.f1 > 0.0 { self.f1 as f64 } else if self.f2 > 0.0 { self.f2 as f64 } else { 200.0 };
        let aa = 10f64.powf(self.g / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * fc / SR as f64;
        let al = w0.sin() / (2.0 * Self::Q);
        let c = w0.cos();
        let a0 = 1.0 + al / aa;
        let (b0, b1, b2) = ((1.0 + al * aa) / a0, -2.0 * c / a0, (1.0 - al * aa) / a0);
        let (a1, a2) = (-2.0 * c / a0, (1.0 - al / aa) / a0);
        let n = out.len();
        self.hist.drain(..n);
        for v in out.iter_mut() {
            let x = *v as f64;
            self.hist.push(x);
            let y = b0 * x + self.z[0];
            self.z[0] = b1 * x - a1 * y + self.z[1];
            self.z[1] = b2 * x - a2 * y;
            *v = y as f32;
        }
        self.f2 = self.f1;
        self.f1 = f0o;
    }
}

pub struct ArticVc {
    c1: Worker<(usize, Vec<f32>)>,
    /// 表の引き: None = 硬い引き(argmax)・Some(β) = 柔らかい引き Σ p^β(u) T[u] / Σ p^β(u)(D24・D25: 明瞭さが大きく改善)
    pub soft: Option<f32>,
    /// 表の引きの出力の因果平均の幅(硬い引き 3・柔らかい引きは 1 = 平滑なし: D27 で明瞭さ 0.43 → 0.35・同一性は同等)
    pub w_sm: usize,
    /// 『目標らしさ ↔ 聞き取りやすさ』のつまみ: 包絡 = (1 − mix)·表の引き + mix·元の声の包絡(F3・因果)。0 = 表だけ。D24〜D28
    pub mix: f32,
    f3: Option<EnvEst>,
    /// F4(因果な周期性推定器・training/export_env.py で cfg.out = 4)と足す差(目標の p75 − 操作者の p75)。Some のとき条件の周期性 = F4 + 差(登録の定数の代わり)
    f4: Option<(EnvEst, [f32; 4])>,
    /// 操作者の登録があるとき: 包絡 = 元の声の包絡(F3)+ (1 − mix)·Σ p(u)(T_目標[u] − T_操作者[u])(D30)。表は差 [d][k]
    delta: Option<Vec<f32>>,
    /// 伸縮つきの話者差: 包絡 = Σp T_目標 + W(元の声の包絡 − Σp T_操作者)(W = 周波数伸縮 [24][24]・2 つめ = W·T_操作者 [24][k])
    warp: Option<(Vec<f32>, Vec<f32>)>,
    /// 伸縮後の操作者の動きに目標の動きの共分散をかぶせる M = C_目標^½ (W C_操作者 Wᵀ)^−½ [d][d](D74・D75)
    color: Option<Vec<f32>>,
    /// 伸縮つき話者差で足す操作者の動きの量 γ(0 = 柔らかい引き・1 = 全部)。『目標らしさ ↔ 聞き取りやすさ』のつまみ
    pub resid: f32,
    /// 遷移で動きの量を変える(D147): Some((γ_min, γ_max, d_lo, d_hi))= γ_t = γ_min + (γ_max − γ_min)·clip((d̄_t − d_lo)/(d_hi − d_lo), 0, 1)・d_t = ½Σ|p_t − p_{t−1}|(C1 の事後の全変動)の因果 3 平均。None = 一定の resid
    pub gamma_adapt: Option<(f32, f32, f32, f32)>,
    /// 動きの量を DCT の次元ごとに掛ける重み [d](D153・None = 全次元 1)
    pub resid_dims: Option<Vec<f32>>,
    pprev: Vec<f32>,
    dring: [f32; 3],
    /// 条件の周期性に登録の 75 パーセンタイルを使う(false = 平均)
    pub per_p75: bool,
    /// 内容 → 波形の条件の周期性を F4 の動き(列 27..30 = 目標の per_p75 + F4(t) − 操作者の per_p75)に(diag_ppgvoc53)。--f4 が要る
    pub per_dyn: bool,
    /// 有声フレームの包絡(行 1..24)に足す較正(登録で作る定数・D114: 目標の参照の教師の有声平均 − 操作者の参照を同じ変換に通した有声平均)
    pub env_bias: Option<Vec<f32>>,
    /// c0 の単位ごとの話者差(D121): 目標の t0 − 操作者の t0 [k]。Some のとき c0 += 表の引きと同じ重みで引いた差
    pub c0tab: Option<Vec<f32>>,
    /// c0 のエネルギー整合(D119): DCT の行 1..24 [24][n_mel]・n_mel。Some のとき c0 += (√n_mel/2)(log Σ e^{2 形_元} − log Σ e^{2 形_出力})/10(元 = F3・出力 = 条件の行 1..24)
    pub c0e: Option<(Vec<f32>, usize)>,
    /// 包絡のポストフィルタ: c2..c24 を β 倍(フォルマントの山谷を鋭く・表の過平滑の対策・1 = なし)
    pub postfilter: f32,
    /// c0 の水準のずらし(D185/D186: 目標の参照の有声の c0n 平均 − 操作者の参照の有声の c0n 平均・登録の定数)
    pub c0_shift: f32,
    /// 内容 → 波形の生成器の音量の条件の水準ずらし(D558・登録の定数 = 4.40 − 操作者の参照の末尾 12 s の発声中の c0n 平均・c0_level)。無音(c0n ≤ 0.05)には足さない。0 = なし。
    pub c0_ppg_shift: f32,
    q: Vec<f32>,
    f1: Worker<(f32, f32)>,
    bufs: Vec<Vec<f32>>,
    f2: Lvl,
    ex: Excite,
    voc: RVoc,
    en: Enroll,
    ring: Vec<Vec<f32>>,
    pos: usize,
    frames: usize,
    voiced: Vec<f32>,
    fixed_med: Option<f32>,
    cond: Vec<f32>,
    /// 内容 → 波形の生成器(rvoc の d_cond > D_COND・training/train_ppgvoc.py)の条件 [c0n・Σp T / 10・log f0・有声・周期性 × 有声・事後 K]。
    cond_ppg: Vec<f32>,
    exc: Vec<f32>,
    pub last_f0o: f32,
    /// 診断用: Some のとき毎フレームの条件 [D_COND] を積む(LIGHTVC_DUMP_COND・既定 None)
    pub dump: Option<Vec<f32>>,
    pub cond_only: bool,
    pub front: Option<Vec<f32>>,
    /// 診断用: Some のとき毎ブロックの段ごとの時間 [9] ms(前段・C1・F1・CCT・C2・条件と R・励起・生成器・H1Eq)
    pub prof: Option<Vec<f32>>,
    /// 因果な CheapTrick の c0n・包絡で F2・F3 を置き換える(D253・`--causal-ct`)。None = 従来とビット一致。
    pub cct: Option<crate::enroll::CausalCt>,
    /// 全体写像(`--global-map`): 包絡 = W_α·(操作者の包絡) + Δ(Δ = 目標の表の数で重みづけた平均 − W_α·操作者の表の同じ平均)。表の単位ごとの引きを使わない。None = 従来。
    pub gmap: Option<Vec<f32>>,
    /// 全体写像の Δ の倍率(`--gmap-scale`・既定 1)。
    pub gmap_scale: f32,
    /// 診断: 高さを目標の声域へ移さない(`--no-f0-shift`・f0o = 元の声の f0)。
    pub no_f0_shift: bool,
    /// 全体写像の Δ を下限つきの log-mel の領域で足す(`--gmap-floor`・D262): DCT 行 0..24 [25][nm] と nm。帯ごとに下限からの高さで Δ の重み(下限 + 1 nat 以下 0・+ 2 nat 以上 1)・下限で切って DCT に戻す(c0 も整合)。
    pub gmap_floor: Option<(Vec<f32>, usize)>,
    /// どの変換でも、操作者の c0・包絡からの差を下限つきの log-mel の領域で足し直す(`--floor-apply`・D263)。DCT 行 0..24 と nm。
    pub floor_apply: Option<(Vec<f32>, usize)>,
    /// 操作者の遅い音色の揺らぎ(`--opslow tau,gain`・D451)。None = 従来(乱数の揺らぎ)。
    pub opslow: Option<OpSlow>,
    /// 連続の内容の特徴 C2(c1 と同じ骨格・出力 = 特徴 [cv_d]・diag_ppgvoc52)。生成器の条件の事後の後ろに足す。None = 従来。
    c2: Option<Worker<Vec<f32>>>,
    /// 診断: C2 の置き換え(LIGHTVC_DIAG_C2 = f32 LE のファイル・cv_d 個 = 定数・4 cv_d 個 = [μ_op, σ_op, μ_f, σ_f] で次元ごとに (x − μ_op)/σ_op·σ_f + μ_f)。
    c2_diag: Option<Vec<f32>>,
    /// 診断: 事後を最大の単位の one-hot にする(LIGHTVC_DIAG_Q=onehot)。
    q_onehot: bool,
    /// 診断: 出力の log f0 の走る平均(有声・2 s)からのずれを s 倍(LIGHTVC_DIAG_F0X=s)。(s, 走る平均)
    f0x: Option<(f32, Option<f32>)>,
    /// 診断: 流れの差し替え(LIGHTVC_DIAG_STREAM = f32 LE のファイル・フレームごとに [c0(生成器の条件の 0 行)・出力の f0 Hz(≤ 0 = 無声)]・NaN = そのまま・条件だけ流す登録の間は進めない)。(値, 次のフレーム)
    stream: Option<(Vec<f32>, usize)>,
    /// 参照を読む R の S 形(`--rr`)。None = 従来。
    pub rr: Option<RrPath>,
    /// 高さを固定の半音だけ移す(`--semitones`・男声 → 女声の運用 +0 / +5 / +7 / +12・オーナー判断 10-25)。None = 従来(目標の声域へ)。
    pub semitones: Option<f32>,
    /// 低い高さの基音の傾き(`--h1fix`・D614)。None = 従来。
    pub h1fix: Option<H1Fix>,
    /// 生成器の後ろの適応 H1 EQ(`--h1eq`・D623)。None = 従来。
    pub h1eq: Option<H1Eq>,
    /// R の操作者の統計を作るときの取り込み(フレームごと [F3 の包絡 ×10 (24)・F2 の c0n・F1 の f0])。
    rr_cap: Option<Vec<f32>>,
}

impl ArticVc {
    /// dirs: 各部品の書き出し(rvoc・lvl・f0est・c1)と登録。
    pub fn load(rvoc: &Path, lvl: &Path, f0est: &Path, c1: &Path, enroll: &Path, seed: u64) -> Result<Self> {
        Self::load_with(rvoc, lvl, f0est, c1, Enroll::load(enroll)?, seed)
    }

    /// 参照の WAV(48kHz・20s 以上)から Rust で登録する(enroll::Analyzer・F1・C1・コードブック)。consts = training/export_enroll.py の出力。
    pub fn enroll_from_wav(x: &[f32], f0est: &Path, c1: &Path, consts: &Path) -> Result<Enroll> {
        let an = crate::enroll::Analyzer::load(consts)?;
        let mut f1 = F0Est::load(f0est)?;
        let mut c = C1::load(c1)?;
        c.prime();
        let (ts, _) = crate::nvoc::Tensors::load(c1)?;
        let cb = ts.get("codebook")?.0;
        Ok(an.enroll(x, &mut f1, &mut c, &cb))
    }

    /// 参照の WAV から R の目標の記憶を作り <out>/mem_k{l}.f32・mem_v{l}.f32 に書く(rr = training/export_rr.py の出力・記憶の符号器つき)。
    pub fn enroll_rr_memory(x: &[f32], f0est: &Path, c1: &Path, consts: &Path, rr: &Path, out: &Path) -> Result<usize> {
        let an = crate::enroll::Analyzer::load(consts)?;
        let mut f1 = F0Est::load(f0est)?;
        let mut c = C1::load(c1)?;
        c.prime();
        let (rp, rv, m) = an.rr_inputs(x, &mut f1, &mut c, 5000);
        let net = Rr::load(rr)?;
        let (k, v) = net.encode_memory(&rp, &rv, m)?;
        std::fs::create_dir_all(out)?;
        for l in 0..k.len() {
            std::fs::write(out.join(format!("mem_k{l}.f32")), k[l].iter().flat_map(|z| z.to_le_bytes()).collect::<Vec<u8>>())?;
            std::fs::write(out.join(format!("mem_v{l}.f32")), v[l].iter().flat_map(|z| z.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        Ok(m)
    }

    pub fn load_with(rvoc: &Path, lvl: &Path, f0est: &Path, c1: &Path, en: Enroll, seed: u64) -> Result<Self> {
        Self::load_with_hyst(rvoc, lvl, f0est, c1, en, seed, None)
    }

    /// f1_hyst = F1 の有声の判定のヒステリシス(確率: 入る・抜ける。例 (0.7, 0.3) で切り替わりが教師と同じ毎秒 3 回・D57)。
    pub fn load_with_hyst(rvoc: &Path, lvl: &Path, f0est: &Path, c1: &Path, en: Enroll, seed: u64, f1_hyst: Option<(f32, f32)>) -> Result<Self> {
        Self::load_with_opts(rvoc, lvl, f0est, c1, en, seed, f1_hyst, None, None)
    }

    /// f1_track = F1 の音高の事後の因果の前向きフィルタ(σ ビン・ε)。例 (10, 0.02)(D62)。
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_opts(rvoc: &Path, lvl: &Path, f0est: &Path, c1: &Path, en: Enroll, seed: u64, f1_hyst: Option<(f32, f32)>, f1_track: Option<(f32, f64)>, f1_conf: Option<f32>) -> Result<Self> {
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(rvoc.join("manifest.json"))?)?;
        let c = &man["const"];
        let ex = Excite::new(c["H_MAX"].as_f64().context("H_MAX")? as f32, c["H_ROLL"].as_f64().context("H_ROLL")? as f32,
                             c["K_MAX"].as_u64().context("K_MAX")? as usize, seed);
        let mut c1m = C1::load(c1)?;
        c1m.prime();
        anyhow::ensure!(en.k == c1m.k && en.d == 24, "登録の表 {}x{} と C1 の K {} が合わない", en.d, en.k, c1m.k);
        let c1w = Worker::spawn(c1m, |m: &mut C1, x: &[f32]| {
            let u = m.process_block(x);
            (u, m.logits().to_vec())
        });
        let mut f1m = F0Est::load(f0est)?;
        f1m.hyst = f1_hyst.map(|(on, off)| ((on / (1.0 - on)).ln(), (off / (1.0 - off)).ln()));
        f1m.conf_min = f1_conf;
        if let Some((sig, eps)) = f1_track {
            f1m.set_track(sig, eps);
        }
        let f1w = Worker::spawn(f1m, |m: &mut F0Est, x: &[f32]| m.process_block(x));
        let mut voc = RVoc::load(rvoc)?;
        voc.enable_parallel();
        Ok(Self { c1: c1w, soft: None, w_sm: W_SM, mix: 0.0, f3: None, f4: None, delta: None, warp: None, color: None, resid: 1.0, gamma_adapt: None, resid_dims: None, pprev: vec![], dring: [0.0; 3], per_p75: false, per_dyn: false, env_bias: None, c0tab: None, c0e: None, postfilter: 1.0, c0_shift: 0.0, c0_ppg_shift: 0.0, q: vec![], f1: f1w, bufs: vec![vec![0.0; HOP], vec![0.0; HOP]], f2: Lvl::load(lvl)?, ex, voc, en, ring: vec![], pos: 0, frames: 0,
                  voiced: vec![], fixed_med: None, cond: vec![0.0; D_COND], cond_ppg: vec![], exc: vec![0.0; 2 * HOP], last_f0o: 0.0, dump: None, cond_only: false, front: None, prof: None, cct: None, gmap: None, gmap_scale: 1.0, no_f0_shift: false, gmap_floor: None, floor_apply: None, opslow: None, c2: None, c2_diag: None,
                  q_onehot: std::env::var("LIGHTVC_DIAG_Q").map(|v| v == "onehot").unwrap_or(false),
                  f0x: std::env::var("LIGHTVC_DIAG_F0X").ok().and_then(|v| v.parse::<f32>().ok()).map(|s| (s, None)),
                  stream: match std::env::var("LIGHTVC_DIAG_STREAM") {
                      Ok(p) => Some((std::fs::read(&p).with_context(|| format!("LIGHTVC_DIAG_STREAM {p}"))?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(), 0)),
                      Err(_) => None,
                  },
                  rr: None, rr_cap: None, semitones: None, h1fix: None, h1eq: None })
    }

    /// つまみ用の F3(training/export_env.py の出力)を読み込む。
    pub fn load_f3(&mut self, dir: &Path) -> Result<()> {
        self.f3 = Some(EnvEst::load(dir)?);
        Ok(())
    }

    /// 条件の周期性を F4 の推定から作る(off = 目標の per_p75 − 操作者の per_p75・操作者なしは 0 = 元の声の周期性のまま)。
    pub fn load_f4(&mut self, dir: &Path, off: [f32; 4]) -> Result<()> {
        let m = EnvEst::load(dir)?;
        anyhow::ensure!(m.n_out() == 4, "F4 の書き出しではない(出力 {})", m.n_out());
        self.f4 = Some((m, off));
        Ok(())
    }

    /// 操作者の登録(自分の声の参照から enroll_from_wav で作った表)。F3 が要る。
    pub fn set_operator(&mut self, op: &Enroll) -> Result<()> {
        anyhow::ensure!(op.d == self.en.d && op.k == self.en.k, "操作者の表 {}x{} が目標の表 {}x{} と合わない", op.d, op.k, self.en.d, self.en.k);
        anyhow::ensure!(self.f3.is_some(), "操作者の登録には F3 が要る(load_f3)");
        self.delta = Some(self.en.table.iter().zip(op.table.iter()).map(|(&a, &b)| a - b).collect());
        Ok(())
    }

    pub fn target(&self) -> &Enroll {
        &self.en
    }

    /// 内容 → 波形の生成器(d_cond > 31)か。
    pub fn set_h1eq(&mut self, target: f32, fmax: Option<f32>) {
        self.h1eq = Some(H1Eq::new(target, self.en.mu, fmax));
    }

    pub fn set_h1fix(&mut self, sigma: f32, dct_rows: (Vec<f32>, usize)) {
        self.h1fix = Some(H1Fix::new(sigma, self.en.mu, dct_rows));
    }

    pub fn is_ppg(&self) -> bool {
        self.voc.d_cond > D_COND
    }

    /// 参照 x を条件だけ流し、生成器の音量の条件 c0n の発声中(> 0.5)の平均を返す(D558 の登録の定数・読み込んだばかりの実体で呼ぶ = 状態が進む)。発声が 50 フレーム以下なら None。
    pub fn c0_level(&mut self, x: &[f32]) -> Option<f32> {
        assert!(self.is_ppg() && self.voc.c0low.is_none(), "c0_level は c0low の無い内容 → 波形の生成器だけ");
        let (co, sh) = (self.cond_only, self.c0_ppg_shift);
        let dump = self.dump.replace(vec![]);
        self.cond_only = true;
        self.c0_ppg_shift = 0.0;
        let mut out = vec![0.0f32; HOP];
        for b in x.chunks_exact(HOP) {
            self.process_block(b, &mut out);
        }
        let rows = std::mem::replace(&mut self.dump, dump).unwrap_or_default();
        self.cond_only = co;
        self.c0_ppg_shift = sh;
        let v: Vec<f64> = rows.chunks_exact(self.voc.d_cond).map(|r| r[0] as f64).filter(|&c| c > 0.5).collect();
        if v.len() > 50 { Some((v.iter().sum::<f64>() / v.len() as f64) as f32) } else { None }
    }

    /// 連続の内容の特徴 C2(training/export_c1.py の C2 の書き出し)を使う: 生成器の条件 = [31 行・事後 K・C2 の特徴 cv_d](内容 → 波形の生成器のみ)。
    pub fn set_c2(&mut self, dir: &Path) -> Result<()> {
        anyhow::ensure!(self.voc.d_cond > D_COND, "--c2 は内容 → 波形の生成器(d_cond > 31)だけ");
        let mut m = C1::load(dir)?;
        m.prime();
        anyhow::ensure!(self.voc.d_cond == D_COND + self.en.k + m.k, "生成器の d_cond {} != 31 + K {} + C2 {}", self.voc.d_cond, self.en.k, m.k);
        let n = m.k;
        self.c2 = Some(Worker::spawn(m, |m: &mut C1, x: &[f32]| {
            m.process_block(x);
            m.logits().to_vec()
        }));
        if let Ok(p) = std::env::var("LIGHTVC_DIAG_C2") {
            let raw = std::fs::read(&p)?;
            let v: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            anyhow::ensure!(v.len() == n || v.len() == 4 * n, "LIGHTVC_DIAG_C2 は {n} 個か {} 個", 4 * n);
            self.c2_diag = Some(v);
        }
        Ok(())
    }

    /// 操作者の遅い音色の揺らぎ(D451)を使う: 生成器の乱数の揺らぎを止め、操作者の包絡と表から因果に作る(内容 → 波形の生成器のみ)。
    pub fn set_opslow(&mut self, op: &Enroll, tau_s: f64, gain: f64) -> Result<()> {
        anyhow::ensure!(self.voc.d_cond > D_COND, "--opslow は内容 → 波形の生成器(d_cond > 31)だけ");
        anyhow::ensure!(op.d == 24 && op.k == self.en.k, "操作者の表 {}x{} が目標の表 {}x{} と合わない", op.d, op.k, self.en.d, self.en.k);
        let (b, kb) = self.voc.slow_basis().context("--opslow には生成器の遅い揺らぎの基底(cfg.slow)が要る")?;
        self.voc.slow_sigma(Some(0.0));
        self.opslow = Some(OpSlow::new(op.table.clone(), b, kb, tau_s, gain));
        Ok(())
    }

    /// 参照を読む R の S 形を使う(export = training/export_rr.py の出力・mem = 目標の記憶 mem_k{l}.f32 / mem_v{l}.f32)。内容 → 波形の生成器と F3 が要る。
    /// 診断: 生成器に入る実効の条件の書き出し(LIGHTVC_DUMP_VCOND)
    pub fn voc_vdump(&mut self) -> &mut Option<Vec<f32>> {
        &mut self.voc.vdump
    }

    /// 構音の描画(生成器なし・current/artic_render.md)を使う: 生成器の段を通らず、生の条件と遅い揺らぎ + R の ΔS から描く。export = training/export_artic_render.py の出力。
    pub fn set_render(&mut self, export: &std::path::Path, seed: u64) -> Result<()> {
        anyhow::ensure!(self.voc.d_cond > D_COND, "--render-artic は内容 → 波形の条件(事後・C2)が要る(--c2 と内容 → 波形の生成器)");
        self.voc.render = Some(crate::artic_render::ArticRender::load(export, seed)?);
        Ok(())
    }

    /// 構音の描画の目標ごとの曲線(dB [1025]・render_curve_from_wav)を使う。描画が無ければ何もしない。
    pub fn set_render_curve(&mut self, db: &[f32]) {
        if let Some(r) = self.voc.render.as_mut() {
            r.set_target_curve(db);
        }
    }

    /// 登録の時の目標ごとの曲線(training/d666_target_cal.py と同じ式)を参照の WAV と登録の表から作る(export = 構音の描画の書き出し)。
    pub fn render_curve_from_wav(x: &[f32], f0est: &Path, c1: &Path, consts: &Path, export: &Path, table: &[f32]) -> Result<Vec<f32>> {
        let an = crate::enroll::Analyzer::load(consts)?;
        let mut f1 = F0Est::load(f0est)?;
        let mut c = C1::load(c1)?;
        c.prime();
        let t = x.len() / HOP;
        let x = &x[..t * HOP];
        let (mut f0, mut lg) = (Vec::with_capacity(t), Vec::with_capacity(t));
        for i in 0..t {
            let b = &x[i * HOP..(i + 1) * HOP];
            f0.push(f1.process_block(b).1);
            c.process_block(b);
            lg.push(c.logits().to_vec());
        }
        let env = an.envelope(x, &f0);
        let r = crate::artic_render::ArticRender::load(export, 0)?;
        Ok(r.target_curve(&env, &lg, table))
    }

    /// 構音の描画の声道長のつまみ α(1.0 = 中立)。描画が無ければ何もしない。
    pub fn set_render_alpha(&mut self, alpha: f64) {
        if let Some(r) = self.voc.render.as_mut() {
            r.alpha = alpha;
        }
    }

    /// 構音の描画の大きさの音節の動きのつまみ γ(1.0 = 操作者のまま)。描画が無ければ何もしない。
    pub fn set_render_level(&mut self, gamma: f64) {
        if let Some(r) = self.voc.render.as_mut() {
            r.level_gamma = gamma;
        }
    }

    /// 診断: 文脈の符号器の前の生の条件と符号器の後の包絡の書き出し(LIGHTVC_DUMP_RCOND)
    pub fn voc_rdump(&mut self) -> &mut Option<Vec<f32>> {
        &mut self.voc.rdump
    }

    pub fn set_rr(&mut self, export: &Path, mem: &Path, gain: f32, stride: usize, sync: bool) -> Result<()> {
        anyhow::ensure!(self.voc.d_cond > D_COND, "--rr は内容 → 波形の生成器(d_cond > 31)だけ");
        anyhow::ensure!(self.f3.is_some(), "--rr には F3 が要る(操作者の登録で読み込まれる)");
        let mut net = Rr::load(export)?;
        net.load_memory(mem, stride)?;
        anyhow::ensure!(net.nq == self.en.k + 27, "R の問いの次元 {} != K {} + 27", net.nq, self.en.k);
        self.voc.slow_sigma(Some(0.0));
        let nq = net.nq;
        let (net, worker) = if sync {
            (Some(net), None)
        } else {
            let w = Worker::spawn(net, |m: &mut Rr, x: &[f32]| {
                let nq = m.nq;
                let mut ds = vec![0.0f32; 25];
                let dc = m.step(&x[..nq], &x[nq..nq + 24], &mut ds[..24]);
                ds[24] = dc;
                ds
            });
            (None, Some(w))
        };
        self.rr = Some(RrPath { net, worker, pending: false, buf: vec![0.0; nq + 24], gain, op: None, qv: vec![0.0; nq], tab: vec![0.0; 24], ds: vec![0.0; 24], z: vec![0.0; 24], sel: vec![], dump: None });
        Ok(())
    }

    /// R の操作者の統計(F3 の包絡 ×10 の有声平均 [24]・F2 の c0n の有声平均・有声の log f0 の中央値)。
    pub fn set_rr_op(&mut self, st: ([f32; 24], f32, f32)) {
        if let Some(r) = self.rr.as_mut() {
            r.op = Some(st);
        }
    }

    /// 操作者の参照 x を条件だけ流し、R の操作者の統計を返す(読み込んだばかりの実体で呼ぶ = 状態が進む)。F3 が要る。有声が 50 フレーム以下なら None。
    pub fn rr_op_stats(&mut self, x: &[f32]) -> Option<([f32; 24], f32, f32)> {
        assert!(self.f3.is_some(), "rr_op_stats には F3 が要る");
        let co = self.cond_only;
        let dump = self.dump.take();
        self.cond_only = true;
        self.rr_cap = Some(vec![]);
        let mut out = vec![0.0f32; HOP];
        for b in x.chunks_exact(HOP) {
            self.process_block(b, &mut out);
        }
        let cap = self.rr_cap.take().unwrap_or_default();
        self.cond_only = co;
        self.dump = dump;
        let rows: Vec<&[f32]> = cap.chunks_exact(26).filter(|r| r[25] > 0.0).collect();
        if rows.len() <= 50 {
            return None;
        }
        let n = rows.len() as f64;
        let mut e3 = [0.0f32; 24];
        for r in 0..24 {
            e3[r] = (rows.iter().map(|v| v[r] as f64).sum::<f64>() / n) as f32;
        }
        let c0 = (rows.iter().map(|v| v[24] as f64).sum::<f64>() / n) as f32;
        let mut lf: Vec<f64> = rows.iter().map(|v| (v[25] as f64).ln()).collect();
        lf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = lf.len();
        let med = if m % 2 == 1 { lf[m / 2] } else { 0.5 * (lf[m / 2 - 1] + lf[m / 2]) };
        Some((e3, c0, med as f32))
    }

    /// 伸縮つきの操作者の登録(w = enroll::Analyzer::warp_matrix)。F3 が要る。
    pub fn set_operator_warp(&mut self, op: &Enroll, w: Vec<f32>) -> Result<()> {
        let (d, k) = (self.en.d, self.en.k);
        anyhow::ensure!(op.d == d && op.k == k && w.len() == d * d, "操作者の表 {}x{}・伸縮 {} が合わない", op.d, op.k, w.len());
        anyhow::ensure!(self.f3.is_some(), "操作者の登録には F3 が要る(load_f3)");
        let mut wt = vec![0.0f32; d * k];
        for r in 0..d {
            for u in 0..k {
                wt[r * k + u] = (0..d).map(|c| w[r * d + c] * op.table[c * k + u]).sum();
            }
        }
        self.delta = None;
        self.warp = Some((w, wt));
        Ok(())
    }

    /// 全体写像の Δ を作る(set_operator_warp の後)。
    pub fn set_global_map(&mut self, op: &Enroll) -> Result<()> {
        let (d, k) = (self.en.d, self.en.k);
        let (_, wt) = self.warp.as_ref().context("--global-map には --warp が要る")?;
        let mean = |tab: &[f32], cnt: &[u32]| -> Vec<f32> {
            let z: f64 = cnt.iter().map(|&c| c as f64).sum::<f64>().max(1.0);
            (0..d).map(|r| ((0..k).map(|u| tab[r * k + u] as f64 * cnt.get(u).copied().unwrap_or(0) as f64).sum::<f64>() / z) as f32).collect()
        };
        let mt = mean(&self.en.table, &self.en.counts);
        let mo = mean(wt, &op.counts);
        self.gmap = Some((0..d).map(|r| (mt[r] - mo[r]) / 10.0).collect());
        Ok(())
    }

    /// 伸縮つきの操作者の動きを目標の動きの共分散へ色づけする(set_operator_warp の後・両方の登録に res_cov が要る)。
    pub fn set_color(&mut self, op: &Enroll) -> Result<()> {
        let d = self.en.d;
        let w = &self.warp.as_ref().context("--color には伸縮つきの操作者(set_operator_warp)が要る")?.0;
        anyhow::ensure!(self.en.res_cov.len() == d * d && op.res_cov.len() == d * d, "登録に動きの共分散が無い(古い登録・artic-enroll で作り直す)");
        let mut co = vec![0.0f64; d * d];
        for r in 0..d {
            for c in 0..d {
                let mut s = 0.0f64;
                for a in 0..d {
                    for b in 0..d {
                        s += w[r * d + a] as f64 * op.res_cov[a * d + b] as f64 * w[c * d + b] as f64;
                    }
                }
                co[r * d + c] = s;
            }
        }
        let mut ct: Vec<f64> = self.en.res_cov.iter().map(|&v| v as f64).collect();
        for c in [&mut ct, &mut co] {
            let tr = (0..d).map(|i| c[i * d + i]).sum::<f64>() / d as f64;
            for i in 0..d {
                c[i * d + i] += COLOR_SHRINK * tr;
            }
        }
        let a = crate::enroll::sym_pow(&ct, d, 0.5);
        let b = crate::enroll::sym_pow(&co, d, -0.5);
        let mut m = vec![0.0f32; d * d];
        for r in 0..d {
            for c in 0..d {
                m[r * d + c] = (0..d).map(|z| a[r * d + z] * b[z * d + c]).sum::<f64>() as f32;
            }
        }
        self.color = Some(m);
        Ok(())
    }

    /// 一致検査用: 元の声の中央値を固定する(製品は走る中央値)。
    pub fn set_fixed_med(&mut self, med: f32) {
        self.fixed_med = Some(med);
    }

    fn med(&mut self, f0: f32) -> f32 {
        if let Some(m) = self.fixed_med {
            return m;
        }
        if self.frames < MED_FRAMES && f0 > 0.0 {
            let v = f0.ln();
            let i = self.voiced.partition_point(|&a| a < v);
            self.voiced.insert(i, v);
        }
        let n = self.voiced.len();
        if n == 0 {
            return self.en.mu;
        }
        if n % 2 == 1 { self.voiced[n / 2] } else { 0.5 * (self.voiced[n / 2 - 1] + self.voiced[n / 2]) }
    }

    /// 入力ブロック [HOP] → 出力ブロック [HOP](出力部の DELAY 240 を含む)。
    pub fn process_block(&mut self, x: &[f32], out: &mut [f32]) {
        let mut tp = [std::time::Instant::now(); 10];
        let mut b1 = self.bufs.pop().unwrap_or_else(|| vec![0.0; HOP]);
        let mut b2 = self.bufs.pop().unwrap_or_else(|| vec![0.0; HOP]);
        b1.copy_from_slice(x);
        b2.copy_from_slice(x);
        self.c1.tx.send(b1).expect("C1 worker");
        self.f1.tx.send(b2).expect("F1 worker");
        if let Some(w) = self.c2.as_ref() {
            let mut b3 = self.bufs.pop().unwrap_or_else(|| vec![0.0; HOP]);
            b3.copy_from_slice(x);
            w.tx.send(b3).expect("C2 worker");
        }
        let (_raw, mut c0n) = self.f2.process_block(x);
        let c0f = c0n;
        let f3e: Option<Vec<f32>> = self.f3.as_mut().map(|f3| f3.process_block(x)[1..25].to_vec());
        let mut mixed: Option<Vec<f32>> = if self.mix > 0.0 || self.delta.is_some() || self.warp.is_some() { f3e.clone() } else { None };
        let p4: Option<[f32; 4]> = self.f4.as_mut().map(|(m, off)| {
            let o = m.process_block(x);
            [o[0] + off[0], o[1] + off[1], o[2] + off[2], o[3] + off[3]]
        });
        tp[1] = std::time::Instant::now();
        let (b1, (u, logit)) = crate::affinity::recv_spin(&self.c1.rx).expect("C1 worker");
        tp[2] = std::time::Instant::now();
        let gam = match self.gamma_adapt {
            None => self.resid,
            Some((gmin, gmax, dlo, dhi)) => {
                let mx = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = logit.iter().map(|&l| (l - mx).exp()).collect();
                let z: f32 = ex.iter().sum();
                let pc: Vec<f32> = ex.iter().map(|&e| e / z).collect();
                let d = if self.pprev.len() == pc.len() { 0.5 * pc.iter().zip(self.pprev.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() } else { 0.0 };
                self.pprev = pc;
                self.dring = [self.dring[1], self.dring[2], d];
                let db = (self.dring[0] + self.dring[1] + self.dring[2]) / 3.0;
                gmin + (gmax - gmin) * ((db - dlo) / (dhi - dlo).max(1e-6)).clamp(0.0, 1.0)
            }
        };
        let (b2, (_v, f0)) = crate::affinity::recv_spin(&self.f1.rx).expect("F1 worker");
        tp[3] = std::time::Instant::now();
        if let (Some(cap), Some(e)) = (self.rr_cap.as_mut(), f3e.as_ref()) {
            cap.extend(e.iter().map(|v| v * 10.0));
            cap.push(c0f);
            cap.push(f0);
        }
        if let Some(ct) = self.cct.as_mut() {
            let (c0c, ec) = ct.process_block(x, f0);
            c0n = c0c;
            mixed = Some(ec.to_vec());
        }
        tp[4] = std::time::Instant::now();
        self.bufs.push(b1);
        self.bufs.push(b2);
        let cvf: Option<Vec<f32>> = match self.c2.as_ref() {
            Some(w) => {
                let (b3, f) = crate::affinity::recv_spin(&w.rx).expect("C2 worker");
                self.bufs.push(b3);
                Some(f)
            }
            None => None,
        };
        tp[5] = std::time::Instant::now();
        let med = self.med(f0);
        self.frames += 1;
        let (d, k) = (self.en.d, self.en.k);
        let tb: &[f32] = self.delta.as_deref().unwrap_or(&self.en.table);
        let mut eo: Vec<f32> = vec![];
        let e: Vec<f32> = match self.soft {
            None => {
                if let Some((_, wt)) = &self.warp {
                    eo = (0..d).map(|r| wt[r * k + u] / 10.0).collect();
                }
                (0..d).map(|r| tb[r * k + u] / 10.0).collect()
            }
            Some(beta) => {
                let mx = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                self.q.resize(k, 0.0);
                let mut z = 0.0f32;
                for (j, &l) in logit.iter().enumerate() {
                    self.q[j] = ((l - mx) * beta).exp();
                    z += self.q[j];
                }
                if let Some((_, wt)) = &self.warp {
                    eo = (0..d).map(|r| wt[r * k..(r + 1) * k].iter().zip(self.q.iter()).map(|(&t, &w)| t * w).sum::<f32>() / z / 10.0).collect();
                }
                (0..d).map(|r| tb[r * k..(r + 1) * k].iter().zip(self.q.iter()).map(|(&t, &w)| t * w).sum::<f32>() / z / 10.0).collect()
            }
        };
        let dc0: f32 = match (&self.c0tab, self.soft) {
            (None, _) => 0.0,
            (Some(dt), None) => dt[u],
            (Some(dt), Some(_)) => dt.iter().zip(self.q.iter()).map(|(&a, &w)| a * w).sum::<f32>() / self.q.iter().sum::<f32>(),
        };
        let w = self.w_sm.max(1);
        if self.ring.len() != w {
            self.ring = vec![e.clone(); w];
            self.pos = 0;
        }
        self.ring[self.pos] = e;
        self.pos = (self.pos + 1) % w;
        let f0o = if f0 > 0.0 {
            match self.semitones {
                Some(st) => f0 * (st / 12.0).exp2(),
                None if self.no_f0_shift => f0,
                None => (f0.ln() - med + self.en.mu).exp(),
            }
        } else {
            0.0
        };
        let f0o = match self.f0x.as_mut() {
            Some((sc, m)) if f0o > 0.0 => {
                let l = f0o.ln();
                let a = (-1.0f32 / 400.0).exp();
                let mm = match *m {
                    Some(p) => a * p + (1.0 - a) * l,
                    None => l,
                };
                *m = Some(mm);
                (mm + *sc * (l - mm)).exp()
            }
            _ => f0o,
        };
        let co = self.cond_only;
        let srow: Option<[f32; 2]> = self.stream.as_mut().filter(|_| !co).map(|(v, i)| {
            let r = if 2 * *i + 1 < v.len() { [v[2 * *i], v[2 * *i + 1]] } else { [f32::NAN, f32::NAN] };
            *i += 1;
            r
        });
        let f0o = match srow {
            Some([_, f]) if !f.is_nan() => f.max(0.0),
            _ => f0o,
        };
        self.last_f0o = f0o;
        let vo = if f0o > 0.0 { 1.0 } else { 0.0 };
        self.cond[0] = c0n + dc0 + self.c0_shift;
        let dv: Vec<f32> = match (&mixed, &self.warp) {
            (Some(e), Some((wm, _))) => {
                let v: Vec<f32> = (0..d).map(|r| (0..d).map(|c| wm[r * d + c] * e[c]).sum::<f32>() - eo[r]).collect();
                match &self.color {
                    Some(cm) => (0..d).map(|r| (0..d).map(|c| cm[r * d + c] * v[c]).sum()).collect(),
                    None => v,
                }
            }
            _ => vec![],
        };
        let gm: Option<Vec<f32>> = match (&self.gmap, &mixed, &self.warp) {
            (Some(g), Some(e), Some((wm, _))) => {
                let mut v: Vec<f32> = (0..d).map(|r| (0..d).map(|c| wm[r * d + c] * e[c]).sum::<f32>()).collect();
                match &self.gmap_floor {
                    None => {
                        for r in 0..d {
                            v[r] += self.gmap_scale * g[r];
                        }
                    }
                    Some((dm, nm)) => {
                        let nm = *nm;
                        let c0_sil = (1e-5f64).ln() * (nm as f64).sqrt();
                        let fl = (1e-5f64).ln();
                        let c0raw = self.cond[0] as f64 * 10.0 + c0_sil;
                        let mut lm = vec![0.0f64; nm];
                        for b in 0..nm {
                            let mut s = dm[b] as f64 * c0raw;
                            let mut dl = 0.0f64;
                            for r in 0..d {
                                let bw = dm[(r + 1) * nm + b] as f64;
                                s += bw * 10.0 * v[r] as f64;
                                dl += bw * 10.0 * (self.gmap_scale * g[r]) as f64;
                            }
                            let wgt = ((s - fl - 1.0) / 1.0).clamp(0.0, 1.0);
                            lm[b] = (s + wgt * dl).max(fl);
                        }
                        let c0 = (0..nm).map(|b| dm[b] as f64 * lm[b]).sum::<f64>();
                        self.cond[0] = ((c0 - c0_sil) / 10.0) as f32;
                        for r in 0..d {
                            v[r] = ((0..nm).map(|b| dm[(r + 1) * nm + b] as f64 * lm[b]).sum::<f64>() / 10.0) as f32;
                        }
                    }
                }
                Some(v)
            }
            _ => None,
        };
        for r in 0..d {
            let tab = self.ring.iter().map(|v| v[r]).sum::<f32>() / w as f32;
            if let Some(v) = &gm {
                self.cond[1 + r] = v[r];
                continue;
            }
            self.cond[1 + r] = match (&mixed, &self.warp) {
                (Some(e), Some(_)) => (1.0 - self.mix) * (tab + gam * self.resid_dims.as_ref().map_or(1.0, |w| w[r]) * dv[r]) + self.mix * e[r],
                _ => match &mixed {
                Some(e) if self.delta.is_some() => e[r] + (1.0 - self.mix) * tab,
                Some(e) => (1.0 - self.mix) * tab + self.mix * e[r],
                None => tab,
                },
            };
        }
        if let Some(b) = &self.env_bias {
            for r in 0..d {
                self.cond[1 + r] += b[r] * vo;
            }
        }
        if self.postfilter != 1.0 {
            for r in 1..d {
                self.cond[1 + r] *= self.postfilter;
            }
        }
        if let (Some((dm, nm)), Some(e)) = (&self.c0e, &mixed) {
            let ls = |v: &dyn Fn(usize) -> f32| -> f64 {
                let mut acc = 0.0f64;
                for b in 0..*nm {
                    let mut l = 0.0f32;
                    for r in 0..d {
                        l += dm[r * nm + b] * v(r);
                    }
                    acc += (2.0 * l as f64).exp();
                }
                acc.ln()
            };
            let c = &self.cond;
            let s_src = ls(&|r| 10.0 * e[r]);
            let s_out = ls(&|r| 10.0 * c[1 + r]);
            self.cond[0] += ((*nm as f64).sqrt() / 2.0 * (s_src - s_out) / 10.0) as f32;
        }
        if let (Some((dm, nm)), Some(eo)) = (&self.floor_apply, &mixed) {
            let nm = *nm;
            let c0_sil = (1e-5f64).ln() * (nm as f64).sqrt();
            let fl = (1e-5f64).ln();
            let c0op = c0n as f64 * 10.0 + c0_sil;
            let dc = (self.cond[0] - c0n) as f64 * 10.0;
            let mut lm = vec![0.0f64; nm];
            for b in 0..nm {
                let mut s = dm[b] as f64 * c0op;
                let mut dl = dm[b] as f64 * dc;
                for r in 0..d {
                    let bw = dm[(r + 1) * nm + b] as f64;
                    s += bw * 10.0 * eo[r] as f64;
                    dl += bw * 10.0 * (self.cond[1 + r] - eo[r]) as f64;
                }
                let wgt = ((s - fl - 1.0) / 1.0).clamp(0.0, 1.0);
                lm[b] = (s + wgt * dl).max(fl);
            }
            let c0 = (0..nm).map(|b| dm[b] as f64 * lm[b]).sum::<f64>();
            self.cond[0] = ((c0 - c0_sil) / 10.0) as f32;
            for r in 0..d {
                self.cond[1 + r] = ((0..nm).map(|b| dm[(r + 1) * nm + b] as f64 * lm[b]).sum::<f64>() / 10.0) as f32;
            }
        }
        self.cond[25] = if f0o > 0.0 { (f0o / 200.0).ln() } else { 0.0 };
        self.cond[26] = vo;
        let pl = match (p4, self.per_p75, self.en.per_p75) {
            (Some(p), _, _) => p,
            (None, true, Some(p)) => p,
            _ => self.en.per_mean,
        };
        for i in 0..4 {
            self.cond[27 + i] = pl[i].clamp(0.0, 1.5) * vo;
        }
        let ppg = self.voc.d_cond > D_COND;
        if ppg {
            let k = logit.len();
            assert!(self.cct.is_some(), "内容 → 波形の生成器には --causal-ct が要る(学習の c0 と同じ分析)");
            let cv_d = cvf.as_ref().map_or(0, |f| f.len());
            assert_eq!(self.voc.d_cond, D_COND + k + cv_d, "生成器の d_cond と C1 の単位の数(+ C2 の次元)が合わない(C2 の生成器には --c2 が要る)");
            let mx = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            self.q.resize(k, 0.0);
            let mut z = 0.0f32;
            for (j, &l) in logit.iter().enumerate() {
                self.q[j] = (l - mx).exp();
                z += self.q[j];
            }
            self.q.iter_mut().for_each(|v| *v /= z);
            if self.q_onehot {
                let j = (0..k).max_by(|&a, &b| self.q[a].partial_cmp(&self.q[b]).unwrap()).unwrap_or(0);
                self.q.iter_mut().for_each(|v| *v = 0.0);
                self.q[j] = 1.0;
            }
            self.cond_ppg.resize(self.voc.d_cond, 0.0);
            self.cond_ppg[0] = match self.voc.c0low.as_ref() {
                Some((v, a)) => a * (self.cond[0] + (1..25).map(|r| v[r] * self.cond[r]).sum::<f32>()),
                None if c0n > 0.05 => c0n + self.c0_ppg_shift,
                None => c0n,
            };
            if let Some([c, _]) = srow {
                if !c.is_nan() {
                    self.cond_ppg[0] = c;
                }
            }
            for r in 0..d {
                self.cond_ppg[1 + r] = self.en.table[r * k..(r + 1) * k].iter().zip(self.q.iter()).map(|(&t, &w)| t * w).sum::<f32>() / 10.0;
            }
            self.cond_ppg[25] = self.cond[25];
            self.cond_ppg[26] = vo;
            if let (Some(h), true) = (self.h1fix.as_mut(), vo > 0.5) {
                let lf = self.cond_ppg[25];
                h.add(lf, &mut self.cond_ppg[1..25]);
            }
            if self.per_dyn {
                assert!(self.f4.is_some(), "--per-dyn には --f4 が要る");
                for i in 0..4 {
                    self.cond_ppg[27 + i] = self.cond[27 + i];
                }
            } else {
                let pt = self.en.per_p75.unwrap_or(self.en.per_mean);
                for i in 0..4 {
                    self.cond_ppg[27 + i] = pt[i].clamp(0.0, 1.5) * vo;
                }
            }
            self.cond_ppg[D_COND..D_COND + k].copy_from_slice(&self.q);
            if let Some(f) = cvf.as_ref() {
                let n = f.len();
                let dst = &mut self.cond_ppg[D_COND + k..];
                match self.c2_diag.as_deref() {
                    Some(c) if c.len() == n => dst.copy_from_slice(c),
                    Some(c) => {
                        for i in 0..n {
                            dst[i] = (f[i] - c[i]) / c[n + i].max(1e-6) * c[3 * n + i] + c[2 * n + i];
                        }
                    }
                    None => dst.copy_from_slice(f),
                }
            }
            if let (Some(rp), false) = (self.rr.as_mut(), self.rr_cap.is_some()) {
                let e3 = f3e.as_ref().expect("R には F3 が要る");
                let (oe, oc, ol) = rp.op.expect("R の操作者の統計が未設定(set_rr_op)");
                rp.sel.clear();
                rp.sel.extend(self.q.iter().cloned().zip(0..k));
                let top = 16.min(k);
                rp.sel.select_nth_unstable_by(top - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                rp.qv[..k].iter_mut().for_each(|v| *v = 0.0);
                for &(v, j) in rp.sel[..top].iter() {
                    rp.qv[j] = v;
                }
                for r in 0..24 {
                    rp.qv[k + r] = (10.0 * e3[r] - oe[r]) / 10.0;
                }
                rp.qv[k + 24] = c0f - oc;
                rp.qv[k + 25] = if f0 > 0.0 { f0.ln() - ol } else { 0.0 };
                rp.qv[k + 26] = if f0 > 0.0 { 1.0 } else { 0.0 };
                for r in 0..24 {
                    rp.tab[r] = self.en.table[r * k..(r + 1) * k].iter().zip(self.q.iter()).map(|(&t, &w)| t * w).sum::<f32>();
                }
                let dc = match (rp.net.as_mut(), rp.worker.as_ref()) {
                    (Some(net), _) => net.step(&rp.qv, &rp.tab, &mut rp.ds),
                    (None, Some(w)) => {
                        let mut dc = 0.0f32;
                        if rp.pending {
                            let (back, out) = crate::affinity::recv_spin(&w.rx).expect("R worker");
                            rp.ds.copy_from_slice(&out[..24]);
                            dc = out[24];
                            rp.buf = back;
                        } else {
                            rp.ds.iter_mut().for_each(|v| *v = 0.0);
                        }
                        let nq = rp.qv.len();
                        rp.buf.resize(nq + 24, 0.0);
                        rp.buf[..nq].copy_from_slice(&rp.qv);
                        rp.buf[nq..].copy_from_slice(&rp.tab);
                        w.tx.send(std::mem::take(&mut rp.buf)).expect("R worker");
                        rp.pending = true;
                        dc
                    }
                    _ => unreachable!(),
                };
                for r in 0..24 {
                    rp.z[r] = rp.gain * rp.ds[r] / 10.0;
                }
                self.voc.set_ext_slow(Some(&rp.z));
                if let Some(d) = rp.dump.as_mut() {
                    d.extend_from_slice(&rp.qv);
                    d.extend_from_slice(&rp.tab);
                    d.extend_from_slice(&rp.ds);
                    d.push(dc);
                }
            } else if let Some(os) = self.opslow.as_mut() {
                let mut z = [0.0f32; 24];
                os.apply(&self.cond[1..25], &self.q, k, &mut z);
                self.voc.set_ext_slow(Some(&z));
            }
        }
        if let Some(d) = self.dump.as_mut() {
            d.extend_from_slice(if ppg { &self.cond_ppg } else { &self.cond });
        }
        if let Some(fd) = self.front.as_mut() {
            fd.extend_from_slice(&logit);
            match &mixed {
                Some(e) => fd.extend_from_slice(e),
                None => fd.extend(std::iter::repeat(0.0f32).take(d)),
            }
        }
        if self.cond_only {
            out.fill(0.0);
            return;
        }
        tp[6] = std::time::Instant::now();
        self.ex.block(f0o, &mut self.exc);
        tp[7] = std::time::Instant::now();
        if ppg {
            self.voc.process_block(&self.cond_ppg, &self.exc, out);
        } else {
            self.voc.process_block(&self.cond, &self.exc, out);
        }
        tp[8] = std::time::Instant::now();
        if let Some(h) = self.h1eq.as_mut() {
            h.process(out, f0o);
        }
        tp[9] = std::time::Instant::now();
        if let Some(pf) = self.prof.as_mut() {
            pf.extend((1..10).map(|i| (tp[i] - tp[i - 1]).as_secs_f32() * 1e3));
        }
    }

    pub fn last_cond(&self) -> &[f32] {
        &self.cond
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        let raw = std::fs::read(p).unwrap();
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// ARTIC_VC_EXPORT=<export_artic_vc.py の出力> RVOC_EXPORT LVL_EXPORT F0EST_EXPORT C1_EXPORT も設定して
    /// cargo test --release -p lightvc-core artic_vc_parity -- --nocapture
    #[test]
    fn artic_vc_parity() {
        let env = |k: &str| std::env::var(k).ok();
        let (Some(a), Some(r), Some(l), Some(f), Some(c)) = (env("ARTIC_VC_EXPORT"), env("RVOC_EXPORT"), env("LVL_EXPORT"), env("F0EST_EXPORT"), env("C1_EXPORT")) else {
            eprintln!("書き出しの環境変数が未設定のためスキップ");
            return;
        };
        let a = Path::new(&a);
        let mut m = ArticVc::load(Path::new(&r), Path::new(&l), Path::new(&f), Path::new(&c), a, 1).unwrap();
        let fx = a.join("fixture");
        let x = read_f32(&fx.join("x.f32"));
        let med = read_f32(&fx.join("med.f32"))[0];
        let cref = read_f32(&fx.join("cond.f32"));
        let fref = read_f32(&fx.join("f0o.f32"));
        m.set_fixed_med(med);
        let t = x.len() / HOP;
        let mut out = vec![0.0f32; HOP];
        let mut mrow = [0.0f32; D_COND];
        let mut mf = 0.0f32;
        let mut lat = Vec::with_capacity(t);
        let t0 = std::time::Instant::now();
        for i in 0..t {
            let tb = std::time::Instant::now();
            m.process_block(&x[i * HOP..(i + 1) * HOP], &mut out);
            lat.push(tb.elapsed().as_secs_f64() * 1e3);
            for (j, &v) in m.last_cond().iter().enumerate() {
                mrow[j] = mrow[j].max((v - cref[i * D_COND + j]).abs());
            }
            if fref[i] > 0.0 && m.last_f0o > 0.0 {
                mf = mf.max((1200.0 * (m.last_f0o / fref[i]).log2()).abs());
            }
        }
        let el = t0.elapsed().as_secs_f64();
        lat.sort_by(|p, q| p.partial_cmp(q).unwrap());
        let q = |p: f64| lat[((lat.len() - 1) as f64 * p) as usize];
        let worst = mrow.iter().cloned().fold(0.0f32, f32::max);
        println!("artic_vc parity: 条件 max|差| {worst:.3e}(行ごと c0n {:.2e}・包絡 {:.2e}・log f0 {:.2e}・有声 {:.0}・周期性 {:.2e})・f0o max|cent| {mf:.3e}",
                 mrow[0], mrow[1..25].iter().cloned().fold(0.0f32, f32::max), mrow[25], mrow[26], mrow[27..].iter().cloned().fold(0.0f32, f32::max));
        println!("RTF(全経路・C1 と F1 はワーカースレッドで並列・壁時計){:.3}・ブロック p50 {:.2} p95 {:.2} p99 {:.2} ms(予算 5 ms)", el / (t as f64 * HOP as f64 / 48000.0), q(0.5), q(0.95), q(0.99));
        assert!(worst < 2e-3 && mf < 1.0, "artic_vc parity FAIL");
    }
}
