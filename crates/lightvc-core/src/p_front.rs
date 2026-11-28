//! P 線の前段 v3(training/snv_p0_front.py)のストリーミング版。48 kHz・H = 240・フレーム t の窓は (t+1)H で終わる(先読み 0)・発話の統計なし。
//! 高さと有声 = 因果の Butterworth 4 次 → 左寄せの CMNDF(基準窓 1200・遅れ 20〜960)→ 谷の候補 6(放物線の補間)→ 1 kHz 未満のパワーの比 → 前向きの動的計画。
//! 包絡 = 因果の位置 τ_t = min(tH, (t+1)H − 2 − ⌈1.5 T0 SR⌉) の CheapTrick(cheaptrick.rs の WORLD の移植)→ √ → mel 128 → log(床 1e-5)→ DCT 25。
//! 書き出しの定数 W_YIN(基準窓)・ENV_CAP_MS(有声の窓の半幅の上限・無ければ上限なし)・TAU_UV_HZ(無声の位置・無ければ 200)を読む(段 P100)。
//! 周期性 = 左寄せの Hann 2048 の 4 帯の倍音の山と谷のパワー比。定数と一致検査の固定の入出力 = training/export_pfront.py(paper_snv §6 段 P)。
//! 重み m [NB_P](振幅・p_gate の門が開いたフレーム)を渡すと training/train_snv_mask.features_with_mask と snv_p0_front.f0_track_myin の計算:
//! YIN の区間とパワーに m(2049 本へ線形の補間)を掛け、周期性は m² ⊙ P で、包絡の mel の帯の振幅に √((m² ⊙ P)·fb / P·fb) を掛ける。

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::cheaptrick::{cheaptrick_frame, Randn};
use crate::nvoc::{Tensors, HOP};
use crate::prof::prof;
use crate::fft::fft;

const SR: f64 = 48000.0;
const W_YIN_V3: usize = 1200;
const TMIN: usize = 20;
const TMAX: usize = 960;
const NFFT_X: usize = 4096;
const NFFT_S: usize = 4096;
const NB_S: usize = NFFT_S / 2 + 1;
const NFFT_P: usize = 2048;
const NB_P: usize = NFFT_P / 2 + 1;
const NFFT_CT: usize = 2048;
const NB_CT: usize = NFFT_CT / 2 + 1;
const K: usize = 6;
const C_MAX: f64 = 0.7;
const BETA: f64 = 0.03;
const LAMBDA: f64 = 1.0;
const GAMMA: f64 = 0.3;
const U_COST: f64 = 0.65;
const HOLD_BASE: f64 = 0.5;
const HOLD_STEP: f64 = 0.02;
const LB_HZ: f64 = 1000.0;
const LB_MIN: f64 = 0.7;
const E_FLOOR: f64 = 1e-7;
const RING: usize = 4096;
const W_YIN_V4: usize = 600;
const PRE_V4: usize = 64;
const DEC_V4: usize = 8;
const TS_HI_V4: usize = 120;
const TS_ULTRA_V4: usize = 64;
const TMAX_NET_V4: usize = 880;
const TMAX_VOC_V4: usize = 800;

#[derive(Clone, Copy, Debug)]
pub struct PFrame {
    pub f0: f32,
    pub f0_voc: f32,
    pub ap: f32,
    pub env: [f32; 25],
    pub per: [f32; 4],
}

struct DpV4 {
    prev: [f64; K + 2],
    prev_lf: [f64; K + 2],
    hold_n: usize,
}

impl DpV4 {
    fn new() -> Self {
        Self { prev: [f64::INFINITY; K + 2], prev_lf: [0.0; K + 2], hold_n: 0 }
    }

    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, f: &[f64; K], c: &[f64; K], ms: f64, lb: f64, efl: f64, ucost: f64, hold_base: f64, hold_min_c: f64, jump_cap: f64, sub_exit_c: f64) -> (f32, f32) {
        let live = ms > efl;
        let mut loc = [f64::INFINITY; K + 2];
        for j in 0..K {
            loc[j] = if c[j].is_finite() && live { c[j] } else { f64::INFINITY };
        }
        loc[K] = if live { ucost } else { 0.0 };
        let cmin = loc[..K].iter().copied().fold(f64::INFINITY, f64::min);
        if live && lb >= LB_MIN && cmin >= hold_min_c {
            loc[K + 1] = hold_base + HOLD_STEP * self.hold_n as f64;
        }
        let lf: [f64; K] = std::array::from_fn(|j| f[j].max(1.0).log2());
        let mut cur_lf = [0.0; K + 2];
        cur_lf[..K].copy_from_slice(&lf);
        let mut acc;
        if !self.prev.iter().any(|v| v.is_finite()) {
            acc = loc;
            acc[K + 1] = f64::INFINITY;
        } else {
            let vprev: [usize; K + 1] = std::array::from_fn(|j| if j < K { j } else { K + 1 });
            let mut from_v = [0.0; K];
            for i in 0..K {
                let mut best = f64::INFINITY;
                for &pj in vprev.iter() {
                    let dlf = lf[i] - self.prev_lf[pj];
                    let mut jc = (LAMBDA * dlf.abs()).min(jump_cap);
                    if sub_exit_c > 0.0 && loc[i] < sub_exit_c && ((dlf - 1.0).abs() < 0.12 || (dlf - 1.585).abs() < 0.12 || (dlf - 2.0).abs() < 0.12) {
                        jc = jc.min(0.12);
                    }
                    let v = self.prev[pj] + jc;
                    if v < best {
                        best = v;
                    }
                }
                from_v[i] = best.min(self.prev[K] + GAMMA);
            }
            let mut hp = vprev[0];
            for &pj in vprev.iter() {
                if self.prev[pj] < self.prev[hp] {
                    hp = pj;
                }
            }
            let from_u = (self.prev[hp] + GAMMA).min(self.prev[K]);
            acc = loc;
            for i in 0..K {
                acc[i] += from_v[i];
            }
            acc[K] += from_u;
            acc[K + 1] += self.prev[hp];
            cur_lf[K + 1] = self.prev_lf[hp];
        }
        let mn = acc.iter().copied().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min);
        for v in acc.iter_mut() {
            *v -= mn;
        }
        let mut j = 0;
        for i in 1..K + 2 {
            if acc[i] < acc[j] {
                j = i;
            }
        }
        self.hold_n = if j == K + 1 { self.hold_n + 1 } else { 0 };
        let f0 = if j != K && cur_lf[j] > 0.0 { 2f64.powf(cur_lf[j]) as f32 } else { 0.0 };
        let best = if j < K { c[j] as f32 } else { 1.0 };
        self.prev = acc;
        self.prev_lf = cur_lf;
        (f0, best)
    }
}

struct V4State {
    hann_w: Vec<f64>,
    w8: Vec<f64>,
    w4: Vec<f64>,
    dp_net: DpV4,
    dp_voc: DpV4,
    ext_prev: f64,
    ext_n: usize,
}

fn solve_linear<const N: usize>(mut m: [[f64; N]; N], mut b: [f64; N]) -> [f64; N] {
    for k in 0..N {
        let mut piv = k;
        for i in (k + 1)..N {
            if m[i][k].abs() > m[piv][k].abs() {
                piv = i;
            }
        }
        if piv != k {
            m.swap(k, piv);
            b.swap(k, piv);
        }
        let d = m[k][k];
        if d.abs() < 1e-30 {
            return [0.0; N];
        }
        for i in (k + 1)..N {
            let f = m[i][k] / d;
            b[i] -= f * b[k];
            let (top, rest) = m.split_at_mut(i);
            for (mj, &mkj) in rest[0][k..N].iter_mut().zip(&top[k][k..N]) {
                *mj -= f * mkj;
            }
        }
    }
    let mut x = [0.0; N];
    for i in (0..N).rev() {
        let mut s = b[i];
        for j in (i + 1)..N {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    x
}

pub struct PFront {
    fb: Vec<f32>,
    fbr: Vec<(usize, usize)>,
    sq: Vec<f64>,
    dct: Vec<f32>,
    nm: usize,
    hb: Vec<(f64, f64)>,
    b: [f64; 5],
    a: [f64; 5],
    z: [f64; 4],
    raw: Vec<f64>,
    lp: Vec<f64>,
    w_yin: usize,
    l_seg: usize,
    env_cap: f64,
    tau_uv_hz: f64,
    t: usize,
    prev: [f64; K + 2],
    prev_lf: [f64; K + 2],
    hold_n: usize,
    v4: Option<V4State>,
    rn: Randn,
    re: Vec<f64>,
    im: Vec<f64>,
    re2: Vec<f64>,
    im2: Vec<f64>,
    hann_s: Vec<f64>,
    hann_p: Vec<f64>,
}

fn hann_sym(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()).collect()
}

/// 重み m [n] を nb 本の格子へ線形の補間した k 本目(snv_p0_front.mask_to_grid)。
fn grid(m: &[f64], nb: usize, k: usize) -> f64 {
    let n = m.len() - 1;
    let pos = (k * n) as f64 / (nb - 1) as f64;
    let lo = (pos.floor() as usize).min(n);
    let hi = (lo + 1).min(n);
    let fr = pos - lo as f64;
    m[lo] * (1.0 - fr) + m[hi] * fr
}

impl PFront {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_mode(dir, true)
    }

    pub fn load_plain(dir: &Path) -> Result<Self> {
        Self::load_mode(dir, false)
    }

    fn load_mode(dir: &Path, allow_v4: bool) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let c = &man["const"];
        let num = |k: &str| c[k].as_f64().with_context(|| format!("const {k}"));
        for (k, v) in [("SR", SR), ("H", HOP as f64), ("TMIN", TMIN as f64), ("TMAX", TMAX as f64), ("NFFT_X", NFFT_X as f64), ("NFFT_S", NFFT_S as f64),
                       ("NFFT_P", NFFT_P as f64), ("K_CAND", K as f64), ("C_MAX", C_MAX), ("BETA", BETA), ("LAMBDA", LAMBDA), ("GAMMA", GAMMA), ("U_COST", U_COST),
                       ("HOLD_BASE", HOLD_BASE), ("HOLD_STEP", HOLD_STEP), ("LB_HZ", LB_HZ), ("LB_MIN", LB_MIN), ("E_FLOOR", E_FLOOR), ("ETA", 0.0)] {
            if (num(k)? - v).abs() > 1e-12 {
                bail!("前段の定数 {k} が Python({}) と Rust({v}) で違う", num(k)?);
            }
        }
        let w_yin = num("W_YIN")? as usize;
        if !(TMIN..=W_YIN_V3).contains(&w_yin) {
            bail!("前段の定数 W_YIN = {w_yin} は {TMIN}〜{W_YIN_V3} の外");
        }
        let opt = |k: &str, d: f64| c.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let (env_cap, tau_uv_hz) = (opt("ENV_CAP_MS", f64::INFINITY) * 1e-3, opt("TAU_UV_HZ", 200.0));
        let is_v4 = allow_v4 && (c.get("F0_MODE").and_then(|v| v.as_str()) == Some("v4_dual") || man.get("cfg").and_then(|g| g.get("f0_mode")).and_then(|v| v.as_str()) == Some("v4_dual"));
        if is_v4 && w_yin != W_YIN_V4 {
            bail!("v4_dual の W_YIN = {w_yin} は {W_YIN_V4} でない");
        }
        let l_seg = w_yin + TMAX;
        let l_lp = if is_v4 { l_seg + PRE_V4 } else { l_seg };
        let v4 = is_v4.then(|| V4State {
            hann_w: hann_sym(l_seg / DEC_V4),
            w8: vec![0.0; l_seg],
            w4: vec![0.0; l_seg],
            dp_net: DpV4::new(),
            dp_voc: DpV4::new(),
            ext_prev: 0.0,
            ext_n: 0,
        });
        let arr = |k: &str| -> Result<[f64; 5]> {
            let v: Vec<f64> = c[k].as_array().with_context(|| k.to_string())?.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
            v.try_into().map_err(|_| anyhow::anyhow!("{k} は 5 個"))
        };
        let hb = c["HB"].as_array().context("HB")?.iter().map(|b| (b[0].as_f64().unwrap_or(0.0), b[1].as_f64().unwrap_or(0.0))).collect();
        let fb = ts.get("mel.fb")?.0;
        let nm = fb.len() / NB_CT;
        let fbr = (0..nm)
            .map(|m| {
                let row = &fb[m * NB_CT..(m + 1) * NB_CT];
                let lo = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                (lo, row.iter().rposition(|&w| w != 0.0).map_or(lo, |h| h + 1))
            })
            .collect();
        Ok(Self {
            fbr, sq: vec![0.0; NB_CT], fb, dct: ts.get("dct")?.0, nm, hb, b: arr("butter_b")?, a: arr("butter_a")?, z: [0.0; 4],
            raw: vec![0.0; RING], lp: vec![0.0; l_lp], w_yin, l_seg, env_cap, tau_uv_hz, t: 0, prev: [f64::INFINITY; K + 2], prev_lf: [0.0; K + 2], hold_n: 0,
            v4, rn: Randn::new(), re: vec![0.0; NFFT_X], im: vec![0.0; NFFT_X], re2: vec![0.0; NFFT_X], im2: vec![0.0; NFFT_X],
            hann_s: hann_sym(NFFT_S), hann_p: hann_sym(NFFT_P),
        })
    }

    fn lowpass(&mut self, x: f64) -> f64 {
        let (b, a, z) = (&self.b, &self.a, &mut self.z);
        let y = b[0] * x + z[0];
        z[0] = b[1] * x - a[1] * y + z[1];
        z[1] = b[2] * x - a[2] * y + z[2];
        z[2] = b[3] * x - a[3] * y + z[3];
        z[3] = b[4] * x - a[4] * y;
        y
    }

    fn filtered(&mut self, m: &[f64]) -> Vec<f64> {
        let l_seg = self.l_seg;
        let off = self.lp.len() - l_seg;
        for i in 0..NFFT_X {
            self.re[i] = if i < l_seg { self.lp[off + i] } else { 0.0 };
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        for k in 0..NFFT_X {
            let g = grid(m, NFFT_X / 2 + 1, if k <= NFFT_X / 2 { k } else { NFFT_X - k });
            self.re[k] *= g;
            self.im[k] = -(self.im[k] * g);
        }
        fft(&mut self.re, &mut self.im);
        (0..l_seg).map(|i| self.re[i] / NFFT_X as f64).collect()
    }

    fn cmndf_slice(s: &[f64], w_yin: usize, clean_dc: bool, re: &mut [f64], im: &mut [f64], re2: &mut [f64], im2: &mut [f64]) -> (Vec<f64>, f64) {
        let l_seg = w_yin + TMAX;
        for i in 0..NFFT_X {
            re[i] = if i < l_seg { s[i] } else { 0.0 };
            im[i] = 0.0;
            re2[i] = if i < w_yin { s[TMAX + i] } else { 0.0 };
            im2[i] = 0.0;
        }
        fft(re, im);
        fft(re2, im2);
        for i in 0..NFFT_X {
            let (ar, ai, br, bi) = (re[i], im[i], re2[i], -im2[i]);
            re[i] = ar * br - ai * bi;
            im[i] = -(ar * bi + ai * br);
        }
        fft(re, im);
        let xk = |k: usize, r: &[f64]| r[k] / NFFT_X as f64;
        let mut cs = vec![0.0; l_seg + 1];
        for i in 0..l_seg {
            cs[i + 1] = cs[i] + s[i] * s[i];
        }
        let e_ref = cs[l_seg] - cs[TMAX];
        let mut d = vec![0.0; TMAX + 1];
        for tau in 0..=TMAX {
            let e_lag = cs[TMAX - tau + w_yin] - cs[TMAX - tau];
            let esum = e_ref + e_lag;
            let dr = (esum - 2.0 * xk(TMAX - tau, re)).max(0.0);
            d[tau] = if clean_dc && dr < 1e-12 * esum { 0.0 } else { dr };
        }
        let mut dn = vec![1.0; TMAX + 1];
        let mut cum = 0.0;
        for tau in 1..=TMAX {
            cum += d[tau];
            dn[tau] = if clean_dc && cum <= 1e-20 { 1.0 } else { d[tau] * tau as f64 / cum.max(1e-20) };
        }
        (dn, e_ref / w_yin as f64)
    }

    fn cmndf(&mut self, m: Option<&[f64]>, clean_dc: bool) -> (Vec<f64>, f64) {
        let sm = m.map(|m| self.filtered(m));
        let (l_seg, w_yin) = (self.l_seg, self.w_yin);
        let off = self.lp.len() - l_seg;
        let s = sm.as_deref().unwrap_or(&self.lp[off..]);
        Self::cmndf_slice(s, w_yin, clean_dc, &mut self.re, &mut self.im, &mut self.re2, &mut self.im2)
    }

    fn candidates(dn: &[f64]) -> ([f64; K], [f64; K]) {
        Self::candidates_with(dn, C_MAX, BETA)
    }

    fn candidates_with(dn: &[f64], c_max: f64, beta: f64) -> ([f64; K], [f64; K]) {
        let mut is_min = vec![false; TMAX];
        let mut shortest = TMAX;
        for tau in TMIN..TMAX {
            is_min[tau] = dn[tau] < dn[tau - 1] && dn[tau] <= dn[tau + 1] && dn[tau] < c_max;
            if is_min[tau] && tau < shortest {
                shortest = tau;
            }
        }
        let mut sc: Vec<(f64, usize)> = (TMIN..TMAX).map(|tau| (if is_min[tau] { dn[tau] + beta * (tau as f64 / shortest as f64).log2() } else { f64::INFINITY }, tau)).collect();
        sc.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal).then(x.1.cmp(&y.1)));
        let (mut f, mut cost) = ([0.0; K], [f64::INFINITY; K]);
        for (j, &(c, tau)) in sc.iter().take(K).enumerate() {
            cost[j] = c;
            if c.is_finite() {
                let (dm, d0, dp) = (dn[tau - 1], dn[tau], dn[(tau + 1).min(TMAX)]);
                let den = dm - 2.0 * d0 + dp;
                let off = if den.abs() > 1e-12 { (0.5 * (dm - dp) / den).clamp(-1.0, 1.0) } else { 0.0 };
                f[j] = SR / (tau as f64 + off);
            }
        }
        (f, cost)
    }

    /// 1 kHz 未満のパワーの比(snv_p0_front.lowband・左寄せの Hann 4096・m があればパワーに補間した m²)。
    fn lowband(&mut self, m: Option<&[f64]>) -> f64 {
        for i in 0..NFFT_S {
            self.re[i] = self.raw[RING - NFFT_S + i] * self.hann_s[i];
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        let k = (LB_HZ / (SR / NFFT_S as f64)) as usize;
        let p = |i: usize, re: &[f64], im: &[f64]| {
            let v = re[i] * re[i] + im[i] * im[i];
            m.map_or(v, |m| {
                let g = grid(m, NB_S, i);
                v * (g * g)
            })
        };
        let lo: f64 = (1..k).map(|i| p(i, &self.re, &self.im)).sum();
        let all: f64 = (1..NB_S).map(|i| p(i, &self.re, &self.im)).sum();
        lo / (all + 1e-20)
    }

    /// 前向きの動的計画の 1 歩(snv_p0_front.online_dp)。
    fn dp_step(&mut self, f: &[f64; K], c: &[f64; K], ms: f64, lb: f64) -> (f32, f32) {
        let live = ms > E_FLOOR;
        let mut loc = [f64::INFINITY; K + 2];
        for j in 0..K {
            loc[j] = if c[j].is_finite() && live { c[j] } else { f64::INFINITY };
        }
        loc[K] = if live { U_COST } else { 0.0 };
        if live && lb >= LB_MIN {
            loc[K + 1] = HOLD_BASE + HOLD_STEP * self.hold_n as f64;
        }
        let lf: [f64; K] = std::array::from_fn(|j| f[j].max(1.0).log2());
        let mut cur_lf = [0.0; K + 2];
        cur_lf[..K].copy_from_slice(&lf);
        let mut acc;
        if !self.prev.iter().any(|v| v.is_finite()) {
            acc = loc;
            acc[K + 1] = f64::INFINITY;
        } else {
            let vprev: [usize; K + 1] = std::array::from_fn(|j| if j < K { j } else { K + 1 });
            let mut from_v = [0.0; K];
            for i in 0..K {
                let mut best = f64::INFINITY;
                for &pj in vprev.iter() {
                    let v = self.prev[pj] + LAMBDA * (lf[i] - self.prev_lf[pj]).abs();
                    if v < best {
                        best = v;
                    }
                }
                from_v[i] = best.min(self.prev[K] + GAMMA);
            }
            let mut hp = vprev[0];
            for &pj in vprev.iter() {
                if self.prev[pj] < self.prev[hp] {
                    hp = pj;
                }
            }
            let from_u = (self.prev[hp] + GAMMA).min(self.prev[K]);
            acc = loc;
            for i in 0..K {
                acc[i] += from_v[i];
            }
            acc[K] += from_u;
            acc[K + 1] += self.prev[hp];
            cur_lf[K + 1] = self.prev_lf[hp];
        }
        let mn = acc.iter().copied().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min);
        for v in acc.iter_mut() {
            *v -= mn;
        }
        let mut j = 0;
        for i in 1..K + 2 {
            if acc[i] < acc[j] {
                j = i;
            }
        }
        self.hold_n = if j == K + 1 { self.hold_n + 1 } else { 0 };
        let f0 = if j != K && cur_lf[j] > 0.0 { 2f64.powf(cur_lf[j]) as f32 } else { 0.0 };
        let best = if j < K { c[j] as f32 } else { 1.0 };
        self.prev = acc;
        self.prev_lf = cur_lf;
        (f0, best)
    }

    fn envelope(&mut self, f0: f32, g: Option<&[f64]>) -> [f32; 25] {
        let end = ((self.t + 1) * HOP) as f64;
        let t0 = 1.0 / if f0 > 0.0 { f0 as f64 } else { self.tau_uv_hz };
        let capped = f0 > 0.0 && 1.5 * t0 > self.env_cap;
        let tau = ((self.t * HOP) as f64).min(end - 2.0 - if capped { (self.env_cap * SR).ceil() } else { (1.5 * t0 * SR).ceil() });
        let pos = (tau - (end - RING as f64)) / SR;
        let f0_ct = if f0 <= 0.0 { 0.0 } else if capped { 1.5 / self.env_cap } else { f0 as f64 };
        let sp = cheaptrick_frame(&self.raw, f0_ct, pos, &mut self.rn, &mut self.re[..NFFT_CT], &mut self.im[..NFFT_CT]);
        let nm = self.nm;
        for (q, &p) in self.sq.iter_mut().zip(sp.iter()) {
            *q = p.sqrt();
        }
        let mut mel = vec![0.0; nm];
        for (m, mv) in mel.iter_mut().enumerate() {
            let (lo, hi) = self.fbr[m];
            let row = &self.fb[m * NB_CT + lo..m * NB_CT + hi];
            let s: f64 = row.iter().zip(self.sq[lo..hi].iter()).map(|(&w, &q)| w as f64 * q).sum();
            *mv = g.map_or(s, |g| s * g[m]).max(1e-5).ln();
        }
        let mut c = [0.0f32; 25];
        for (k, ck) in c.iter_mut().enumerate() {
            let row = &self.dct[k * nm..(k + 1) * nm];
            *ck = row.iter().zip(mel.iter()).map(|(&d, &v)| d as f64 * v).sum::<f64>() as f32;
        }
        c
    }

    /// 左寄せの Hann 2048 のパワー [NB_P](snv_p0_front.power_p)。
    fn power_p(&mut self) -> Vec<f64> {
        for i in 0..NFFT_P {
            self.re[i] = self.raw[RING - NFFT_P + i] * self.hann_p[i];
            self.im[i] = 0.0;
        }
        fft(&mut self.re[..NFFT_P], &mut self.im[..NFFT_P]);
        (0..NB_P).map(|i| self.re[i] * self.re[i] + self.im[i] * self.im[i]).collect()
    }

    /// 包絡の mel の帯の利得 √((m² ⊙ P)·fb / P·fb)(帯のパワーが 0 なら 1)。
    fn band_gain(&self, pw: &[f64], pm: &[f64]) -> Vec<f64> {
        (0..self.nm)
            .map(|m| {
                let (lo, hi) = self.fbr[m];
                let row = &self.fb[m * NB_CT + lo..m * NB_CT + hi];
                let num: f64 = row.iter().zip(&pm[lo..hi]).map(|(&w, &q)| w as f64 * q).sum();
                let den: f64 = row.iter().zip(&pw[lo..hi]).map(|(&w, &q)| w as f64 * q).sum();
                if den > 0.0 { (num / den).sqrt() } else { 1.0 }
            })
            .collect()
    }

    fn periodicity(&self, f0: f32, p: &[f64]) -> [f32; 4] {
        let df = SR / NFFT_P as f64;
        let fv = if f0 > 0.0 { f0 as f64 } else { 500.0 };
        let kmax = (self.hb.last().map(|b| b.1).unwrap_or(10000.0) / 60.0) as usize + 1;
        let interp = |fbin: f64| -> f64 {
            let fb = fbin.clamp(0.0, (NB_P - 1) as f64 - 1e-6);
            let i0 = fb.floor() as usize;
            let w = fb - i0 as f64;
            p[i0] * (1.0 - w) + p[i0 + 1] * w
        };
        let mut r = [0.0f32; 4];
        for (bi, &(lo, hi)) in self.hb.iter().enumerate().take(4) {
            let (mut num, mut den, mut cnt) = (0.0, 0.0, 0usize);
            for k in 1..=kmax {
                let fk = fv * k as f64;
                if !(fk >= lo && fk < hi && fk < SR / 2.0 - 200.0) {
                    continue;
                }
                num += [-0.2, -0.1, 0.0, 0.1, 0.2].iter().map(|o| interp((fk + o * fv) / df)).fold(f64::NEG_INFINITY, f64::max);
                den += [0.4, 0.5, 0.6].iter().map(|o| interp((fk + o * fv) / df)).fold(f64::INFINITY, f64::min);
                cnt += 1;
            }
            let hnr = 10.0 * ((num + 1e-20) / (den + 1e-20)).log10();
            r[bi] = if cnt > 0 && f0 > 0.0 && num > 1e-12 { (hnr / 30.0).clamp(0.0, 1.5) as f32 } else { 0.0 };
        }
        r
    }

    fn v4_step(&mut self, raw_dn: &[f64], raw_ms: f64, lb_u: f64) -> (f32, f32, f32) {
        let l_seg = self.l_seg;
        let ms_fb = self.raw[RING - W_YIN_V4..RING].iter().map(|&v| v * v).sum::<f64>() / W_YIN_V4 as f64;
        {
            let v4 = self.v4.as_mut().unwrap();
            let n_d = v4.hann_w.len();
            let mut d = vec![0.0; n_d];
            for (i, w) in v4.hann_w.iter().enumerate() {
                d[i] = self.lp[PRE_V4 + i * DEC_V4] * w;
            }
            let mut r = [0.0; 9];
            for k in 0..=8 {
                r[k] = (0..(n_d - k)).map(|m| d[m] * d[m + k]).sum();
            }
            r[0] = r[0] * 1.001 + 1e-12;
            let mut t8 = [[0.0; 8]; 8];
            let mut b8 = [0.0; 8];
            for i in 0..8 {
                b8[i] = -r[i + 1];
                for j in 0..8 {
                    t8[i][j] = r[i.abs_diff(j)];
                }
            }
            let a8 = solve_linear(t8, b8);
            let mut t4 = [[0.0; 4]; 4];
            let mut b4 = [0.0; 4];
            for i in 0..4 {
                b4[i] = -r[i + 1];
                for j in 0..4 {
                    t4[i][j] = r[i.abs_diff(j)];
                }
            }
            let a4 = solve_linear(t4, b4);
            for m in 0..l_seg {
                let mut s8 = self.lp[PRE_V4 + m];
                let mut s4 = s8;
                for (k, &ak) in a8.iter().enumerate() {
                    s8 += ak * self.lp[PRE_V4 + m - DEC_V4 * (k + 1)];
                }
                for (k, &ak) in a4.iter().enumerate() {
                    s4 += ak * self.lp[PRE_V4 + m - DEC_V4 * (k + 1)];
                }
                v4.w8[m] = s8;
                v4.w4[m] = s4;
            }
        }
        let (dn8, dn4) = {
            let v4 = self.v4.as_ref().unwrap();
            let (dn8, _) = Self::cmndf_slice(&v4.w8, W_YIN_V4, true, &mut self.re, &mut self.im, &mut self.re2, &mut self.im2);
            let (dn4, _) = Self::cmndf_slice(&v4.w4, W_YIN_V4, true, &mut self.re, &mut self.im, &mut self.re2, &mut self.im2);
            (dn8, dn4)
        };
        let mut mu = [0.0; TS_HI_V4];
        for tau in 0..TS_HI_V4 {
            let m2 = [-1isize, 0, 1].iter().map(|&o| raw_dn[(2 * tau as isize + o).clamp(0, TMAX as isize) as usize]).fold(f64::INFINITY, f64::min);
            let m3 = [-2isize, -1, 0, 1, 2].iter().map(|&o| raw_dn[(3 * tau as isize + o).clamp(0, TMAX as isize) as usize]).fold(f64::INFINITY, f64::min);
            mu[tau] = raw_dn[tau].max(m2).max(m3);
        }
        let raw_lo_min = raw_dn[TS_HI_V4..400].iter().copied().fold(f64::INFINITY, f64::min);
        let any_n = mu.iter().any(|&v| v < 0.15);
        let mut dn_n = vec![1.0; TMAX + 1];
        if any_n || !(lb_u < 0.32 && raw_lo_min > 0.35) {
            for tau in 0..TS_HI_V4 {
                let hb = if tau < TS_ULTRA_V4 { 1.0 } else { raw_dn[tau].max(dn8[tau]) };
                dn_n[tau] = if mu[tau] < 0.15 { mu[tau] } else if any_n { 1.0 } else { hb };
            }
            dn_n[TS_HI_V4..TMAX_NET_V4].copy_from_slice(&dn8[TS_HI_V4..TMAX_NET_V4]);
        }
        let (fn_c, cn_c) = Self::candidates_with(&dn_n, 0.70, 0.05);
        let ms_net = raw_ms.max(0.20 * ms_fb);

        let any_v = mu.iter().any(|&v| v < 0.18);
        let mut dn_v = vec![1.0; TMAX + 1];
        if any_v || !(lb_u < 0.30 && raw_lo_min > 0.35) {
            for tau in 0..TS_HI_V4 {
                let hb = if tau < TS_ULTRA_V4 { 1.0 } else { raw_dn[tau].max(dn4[tau]) };
                dn_v[tau] = if mu[tau] < 0.18 { mu[tau] } else if any_v { 1.0 } else { hb };
            }
            for tau in TS_HI_V4..TMAX_VOC_V4 {
                dn_v[tau] = 0.88 * raw_dn[tau] + 0.12 * dn4[tau];
            }
        }
        let (fv_c, cv_c) = Self::candidates_with(&dn_v, 0.70, 0.05);
        let ms_voc = raw_ms.max(0.25 * ms_fb);

        let v4 = self.v4.as_mut().unwrap();
        let (mut f_net, mut b_net) = v4.dp_net.step(&fn_c, &cn_c, ms_net, lb_u, 1e-7, 0.50, 0.50, 0.40, 0.45, 0.22);
        if f_net > 0.0 {
            v4.ext_prev = f_net as f64;
            v4.ext_n = 0;
        } else if ms_net <= 1e-7 {
            v4.ext_prev = 0.0;
            v4.ext_n = 0;
        } else if v4.ext_prev > 0.0 && v4.ext_n < 8 {
            let cc = SR / v4.ext_prev;
            let lo = TMIN.max((cc * 2f64.powf(-0.18)) as usize);
            let hi_i = TMAX.min((cc * 2f64.powf(0.18)).ceil() as usize);
            let mut j = lo;
            for tau in (lo + 1)..=hi_i {
                if raw_dn[tau] < raw_dn[j] {
                    j = tau;
                }
            }
            if raw_dn[j] < 0.35 && dn8[j] < 0.72 && lo < j && j < hi_i {
                f_net = (SR / j as f64) as f32;
                b_net = raw_dn[j] as f32;
                v4.ext_prev = f_net as f64;
                v4.ext_n += 1;
            } else {
                v4.ext_prev = 0.0;
                v4.ext_n = 0;
            }
        } else if v4.ext_prev <= 0.0 {
            let mut min_ok = f64::INFINITY;
            for tau in (TS_HI_V4 + 1)..TMAX_NET_V4 {
                if raw_dn[tau] < raw_dn[tau - 1] && raw_dn[tau] <= raw_dn[tau + 1] && raw_dn[tau] < 0.22 && dn8[tau] < 0.60 {
                    min_ok = min_ok.min(raw_dn[tau]);
                }
            }
            if min_ok.is_finite() {
                let mut j = TS_HI_V4 + 1;
                for tau in (TS_HI_V4 + 1)..TMAX_NET_V4 {
                    if raw_dn[tau] < raw_dn[tau - 1] && raw_dn[tau] <= raw_dn[tau + 1] && raw_dn[tau] < 0.22 && dn8[tau] < 0.60 && raw_dn[tau] <= min_ok + 0.10 {
                        j = tau;
                    }
                }
                f_net = (SR / j as f64) as f32;
                b_net = raw_dn[j] as f32;
                v4.ext_prev = f_net as f64;
            } else {
                v4.ext_prev = 0.0;
                v4.ext_n = 0;
            }
        } else {
            v4.ext_prev = 0.0;
            v4.ext_n = 0;
        }
        let (f_voc, _) = v4.dp_voc.step(&fv_c, &cv_c, ms_voc, lb_u, 1e-9, 0.65, 0.50, 0.45, 0.38, 0.25);
        (f_net, f_voc, b_net)
    }

    pub fn process_block(&mut self, x: &[f32]) -> PFrame {
        self.process_block_masked(x, None)
    }

    pub fn process_block_masked(&mut self, x: &[f32], m: Option<&[f64]>) -> PFrame {
        assert_eq!(x.len(), HOP, "ブロックは HOP サンプル");
        assert!(m.is_none_or(|m| m.len() == NB_P), "重みは NB_P 本");
        prof!(F_BUF, {
            self.raw.drain(..HOP);
            self.raw.extend(x.iter().map(|&v| v as f64));
            self.lp.drain(..HOP);
            for &v in x {
                let y = self.lowpass(v as f64);
                self.lp.push(y);
            }
        });
        let mf = m.filter(|m| m.iter().any(|&v| v < 1.0));
        let clean_main = self.v4.is_some() && m.is_none();
        let (dn, ms) = prof!(F_CMNDF, self.cmndf(mf, clean_main));
        let (f, c) = prof!(F_DP, Self::candidates(&dn));
        let lb = prof!(F_LOWBAND, self.lowband(mf));
        let (fm, bm) = prof!(F_DP, self.dp_step(&f, &c, ms, lb));
        let (f0, f0_voc, ap) = if self.v4.is_some() {
            let (raw_dn, raw_ms, lb_u) = if !clean_main {
                let (rd, rm) = prof!(F_CMNDF, self.cmndf(None, true));
                let lu = if mf.is_some() { prof!(F_LOWBAND, self.lowband(None)) } else { lb };
                (rd, rm, lu)
            } else {
                (dn, ms, lb)
            };
            let (fn_u, fv_u, bn_u) = prof!(F_DP, self.v4_step(&raw_dn, raw_ms, lb_u));
            if m.is_some() { (fm, fm, bm) } else { (fn_u, fv_u, bn_u) }
        } else {
            (fm, fm, bm)
        };
        let f_ep = if f0_voc <= 0.0 && f0 > 0.0 { f0 } else { f0_voc };
        let pw = prof!(F_POWER, self.power_p());
        let (env, per) = match m {
            None => (prof!(F_ENV, self.envelope(f_ep, None)), prof!(F_PER, self.periodicity(f_ep, &pw))),
            Some(m) => {
                let (pm, g) = prof!(F_PER, {
                    let pm: Vec<f64> = pw.iter().zip(m).map(|(&p, &w)| p * (w * w)).collect();
                    let g = self.band_gain(&pw, &pm);
                    (pm, g)
                });
                (prof!(F_ENV, self.envelope(f_ep, Some(&g))), prof!(F_PER, self.periodicity(f_ep, &pm)))
            }
        };
        self.t += 1;
        PFrame { f0, f0_voc, ap, env, per }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    #[test]
    #[cfg_attr(not(feature = "model-tests"), ignore = "モデルの試験(推論コードか書き出しを替えたときだけ): cargo test --release -p lightvc-core --features model-tests")]
    fn parity_with_python_front_p124() {
        parity("results/pfront_export_p124");
    }

    fn parity(rel: &str) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: {} が無い(training/export_pfront.py)", dir.display());
            return;
        }
        for name in ["uirou_0", "ef000_t03", "uirou_enroll"] {
            let fx = dir.join("fixture").join(name);
            if !fx.exists() {
                continue;
            }
            let x = read_f32(&fx.join("x.f32"));
            let (f0p, envp, perp) = (read_f32(&fx.join("f0.f32")), read_f32(&fx.join("env.f32")), read_f32(&fx.join("per.f32")));
            let fvp = fx.join("f0_voc.f32").exists().then(|| read_f32(&fx.join("f0_voc.f32")));
            let mut fr = PFront::load(&dir).unwrap();
            let t_n = x.len() / HOP;
            let tm = std::time::Instant::now();
            let (mut f0_same, mut fv_same, mut env_max, mut per_max, mut env_n) = (0usize, 0usize, 0f32, 0f32, 0usize);
            let mut f0_rel = 0f64;
            for t in 0..t_n {
                let o = fr.process_block(&x[t * HOP..(t + 1) * HOP]);
                let fp = f0p[t];
                let ok_f = (o.f0 > 0.0) == (fp > 0.0) && (o.f0 - fp).abs() <= 1e-3 * fp.max(1.0);
                if ok_f {
                    f0_same += 1;
                }
                let ok_v = match &fvp {
                    Some(vp) => (o.f0_voc > 0.0) == (vp[t] > 0.0) && (o.f0_voc - vp[t]).abs() <= 1e-3 * vp[t].max(1.0),
                    None => ok_f,
                };
                if ok_v {
                    fv_same += 1;
                }
                if o.f0 > 0.0 && fp > 0.0 {
                    f0_rel = f0_rel.max(((o.f0 - fp).abs() / fp) as f64);
                }
                if ok_f && ok_v {
                    for k in 0..25 {
                        env_max = env_max.max((o.env[k] - envp[t * 25 + k]).abs());
                    }
                    for k in 0..4 {
                        per_max = per_max.max((o.per[k] - perp[t * 4 + k]).abs());
                    }
                    env_n += 1;
                }
            }
            let rtf = tm.elapsed().as_secs_f64() / (t_n as f64 * HOP as f64 / SR);
            let agree = f0_same as f64 / t_n as f64;
            let agree_v = fv_same as f64 / t_n as f64;
            eprintln!("{name}: frames {t_n} f0 一致 {agree:.4} f0_voc 一致 {agree_v:.4}(最大の相対差 {f0_rel:.2e})包絡の最大差 {env_max:.2e} 周期性の最大差 {per_max:.2e}(比べたフレーム {env_n})・RTF {rtf:.3}(比較込み)");
            assert!(agree >= 0.995 && agree_v >= 0.995, "{name}: f0 一致 {agree} / {agree_v} < 0.995");
            assert!(env_max < 1e-2, "{name}: 包絡の差 {env_max}");
            assert!(per_max < 1e-2, "{name}: 周期性の差 {per_max}");
        }
    }
}
