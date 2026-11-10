//! P 線の前段 v3(training/snv_p0_front.py)のストリーミング版。48 kHz・H = 240・フレーム t の窓は (t+1)H で終わる(先読み 0)・発話の統計なし。
//! 高さと有声 = 因果の Butterworth 4 次 → 左寄せの CMNDF(基準窓 1200・遅れ 20〜960)→ 谷の候補 6(放物線の補間)→ 1 kHz 未満のパワーの比 → 前向きの動的計画。
//! 包絡 = 因果の位置 τ_t = min(tH, (t+1)H − 2 − ⌈1.5 T0 SR⌉) の CheapTrick(enroll.rs の WORLD の移植)→ √ → mel 128 → log(床 1e-5)→ DCT 25。
//! 周期性 = 左寄せの Hann 2048 の 4 帯の倍音の山と谷のパワー比。定数と一致検査の固定の入出力 = training/export_pfront.py(paper_snv §6 段 P)。

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::enroll::{cheaptrick_frame, Randn};
use crate::nvoc::{Tensors, HOP};
use crate::ship_front::fft;

const SR: f64 = 48000.0;
const W_YIN: usize = 1200;
const TMIN: usize = 20;
const TMAX: usize = 960;
const L_SEG: usize = W_YIN + TMAX;
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

/// 1 フレームの出力(snv_p0_front.analyze と同じ量)。
#[derive(Clone, Copy, Debug)]
pub struct PFrame {
    pub f0: f32,
    pub ap: f32,
    pub env: [f32; 25],
    pub per: [f32; 4],
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
    t: usize,
    prev: [f64; K + 2],
    prev_lf: [f64; K + 2],
    hold_n: usize,
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

impl PFront {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let c = &man["const"];
        let num = |k: &str| c[k].as_f64().with_context(|| format!("const {k}"));
        for (k, v) in [("SR", SR), ("H", HOP as f64), ("W_YIN", W_YIN as f64), ("TMIN", TMIN as f64), ("TMAX", TMAX as f64), ("NFFT_X", NFFT_X as f64), ("NFFT_S", NFFT_S as f64),
                       ("NFFT_P", NFFT_P as f64), ("K_CAND", K as f64), ("C_MAX", C_MAX), ("BETA", BETA), ("LAMBDA", LAMBDA), ("GAMMA", GAMMA), ("U_COST", U_COST),
                       ("HOLD_BASE", HOLD_BASE), ("HOLD_STEP", HOLD_STEP), ("LB_HZ", LB_HZ), ("LB_MIN", LB_MIN), ("E_FLOOR", E_FLOOR), ("ETA", 0.0)] {
            if (num(k)? - v).abs() > 1e-12 {
                bail!("前段の定数 {k} が Python({}) と Rust({v}) で違う", num(k)?);
            }
        }
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
            raw: vec![0.0; RING], lp: vec![0.0; L_SEG], t: 0, prev: [f64::INFINITY; K + 2], prev_lf: [0.0; K + 2], hold_n: 0,
            rn: Randn::new(), re: vec![0.0; NFFT_X], im: vec![0.0; NFFT_X], re2: vec![0.0; NFFT_X], im2: vec![0.0; NFFT_X],
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

    /// 左寄せの CMNDF(snv_p0_front.cmndf): dn [TMAX+1]・ms = 基準窓の平均パワー。
    fn cmndf(&mut self) -> (Vec<f64>, f64) {
        let s = &self.lp;
        for i in 0..NFFT_X {
            self.re[i] = if i < L_SEG { s[i] } else { 0.0 };
            self.im[i] = 0.0;
            self.re2[i] = if i < W_YIN { s[TMAX + i] } else { 0.0 };
            self.im2[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        fft(&mut self.re2, &mut self.im2);
        for i in 0..NFFT_X {
            let (ar, ai, br, bi) = (self.re[i], self.im[i], self.re2[i], -self.im2[i]);
            self.re[i] = ar * br - ai * bi;
            self.im[i] = -(ar * bi + ai * br);
        }
        fft(&mut self.re, &mut self.im);
        let xk = |k: usize, re: &[f64]| re[k] / NFFT_X as f64;
        let mut cs = vec![0.0; L_SEG + 1];
        for i in 0..L_SEG {
            cs[i + 1] = cs[i] + s[i] * s[i];
        }
        let e_ref = cs[L_SEG] - cs[TMAX];
        let mut d = vec![0.0; TMAX + 1];
        for tau in 0..=TMAX {
            let e_lag = cs[TMAX - tau + W_YIN] - cs[TMAX - tau];
            let c = xk(TMAX - tau, &self.re);
            d[tau] = (e_ref + e_lag - 2.0 * c).max(0.0);
        }
        let mut dn = vec![1.0; TMAX + 1];
        let mut cum = 0.0;
        for tau in 1..=TMAX {
            cum += d[tau];
            dn[tau] = d[tau] * tau as f64 / cum.max(1e-20);
        }
        (dn, e_ref / W_YIN as f64)
    }

    /// 谷の候補(snv_p0_front.candidates): f [K]・費用 [K](候補なしは f = 0・費用 ∞)。
    fn candidates(dn: &[f64]) -> ([f64; K], [f64; K]) {
        let mut is_min = vec![false; TMAX];
        let mut shortest = TMAX;
        for tau in TMIN..TMAX {
            is_min[tau] = dn[tau] < dn[tau - 1] && dn[tau] <= dn[tau + 1] && dn[tau] < C_MAX;
            if is_min[tau] && tau < shortest {
                shortest = tau;
            }
        }
        let mut sc: Vec<(f64, usize)> = (TMIN..TMAX).map(|tau| (if is_min[tau] { dn[tau] + BETA * (tau as f64 / shortest as f64).log2() } else { f64::INFINITY }, tau)).collect();
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

    /// 1 kHz 未満のパワーの比(snv_p0_front.lowband・左寄せの Hann 4096)。
    fn lowband(&mut self) -> f64 {
        for i in 0..NFFT_S {
            self.re[i] = self.raw[RING - NFFT_S + i] * self.hann_s[i];
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        let k = (LB_HZ / (SR / NFFT_S as f64)) as usize;
        let p = |i: usize, re: &[f64], im: &[f64]| re[i] * re[i] + im[i] * im[i];
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

    fn envelope(&mut self, f0: f32) -> [f32; 25] {
        let end = ((self.t + 1) * HOP) as f64;
        let t0 = 1.0 / if f0 > 0.0 { f0 as f64 } else { 200.0 };
        let tau = ((self.t * HOP) as f64).min(end - 2.0 - (1.5 * t0 * SR).ceil());
        let pos = (tau - (end - RING as f64)) / SR;
        let sp = cheaptrick_frame(&self.raw, if f0 > 0.0 { f0 as f64 } else { 0.0 }, pos, &mut self.rn, &mut self.re[..NFFT_CT], &mut self.im[..NFFT_CT]);
        let nm = self.nm;
        for (q, &p) in self.sq.iter_mut().zip(sp.iter()) {
            *q = p.sqrt();
        }
        let mut mel = vec![0.0; nm];
        for (m, mv) in mel.iter_mut().enumerate() {
            let (lo, hi) = self.fbr[m];
            let row = &self.fb[m * NB_CT + lo..m * NB_CT + hi];
            let s: f64 = row.iter().zip(self.sq[lo..hi].iter()).map(|(&w, &q)| w as f64 * q).sum();
            *mv = s.max(1e-5).ln();
        }
        let mut c = [0.0f32; 25];
        for (k, ck) in c.iter_mut().enumerate() {
            let row = &self.dct[k * nm..(k + 1) * nm];
            *ck = row.iter().zip(mel.iter()).map(|(&d, &v)| d as f64 * v).sum::<f64>() as f32;
        }
        c
    }

    fn periodicity(&mut self, f0: f32) -> [f32; 4] {
        for i in 0..NFFT_P {
            self.re[i] = self.raw[RING - NFFT_P + i] * self.hann_p[i];
            self.im[i] = 0.0;
        }
        fft(&mut self.re[..NFFT_P], &mut self.im[..NFFT_P]);
        let p: Vec<f64> = (0..NB_P).map(|i| self.re[i] * self.re[i] + self.im[i] * self.im[i]).collect();
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

    /// 入力の 1 ブロック(HOP = 240 サンプル・48 kHz)→ このフレームの前段の出力(未来の入力を見ない)。
    pub fn process_block(&mut self, x: &[f32]) -> PFrame {
        assert_eq!(x.len(), HOP, "ブロックは HOP サンプル");
        self.raw.drain(..HOP);
        self.raw.extend(x.iter().map(|&v| v as f64));
        self.lp.drain(..HOP);
        for &v in x {
            let y = self.lowpass(v as f64);
            self.lp.push(y);
        }
        let (dn, ms) = self.cmndf();
        let (f, c) = Self::candidates(&dn);
        let lb = self.lowband();
        let (f0, ap) = self.dp_step(&f, &c, ms, lb);
        let env = self.envelope(f0);
        let per = self.periodicity(f0);
        self.t += 1;
        PFrame { f0, ap, env, per }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    #[test]
    fn parity_with_python_front_v3() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/pfront_export");
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: {} が無い(training/export_pfront.py)", dir.display());
            return;
        }
        for name in ["uirou_0", "ef000_t03"] {
            let fx = dir.join("fixture").join(name);
            let x = read_f32(&fx.join("x.f32"));
            let (f0p, envp, perp) = (read_f32(&fx.join("f0.f32")), read_f32(&fx.join("env.f32")), read_f32(&fx.join("per.f32")));
            let mut fr = PFront::load(&dir).unwrap();
            let t_n = x.len() / HOP;
            let tm = std::time::Instant::now();
            let (mut f0_same, mut env_max, mut per_max, mut env_n) = (0usize, 0f32, 0f32, 0usize);
            let mut f0_rel = 0f64;
            for t in 0..t_n {
                let o = fr.process_block(&x[t * HOP..(t + 1) * HOP]);
                let fp = f0p[t];
                if (o.f0 > 0.0) == (fp > 0.0) && (o.f0 - fp).abs() <= 1e-3 * fp.max(1.0) {
                    f0_same += 1;
                }
                if o.f0 > 0.0 && fp > 0.0 {
                    f0_rel = f0_rel.max(((o.f0 - fp).abs() / fp) as f64);
                }
                if (o.f0 > 0.0) == (fp > 0.0) && (o.f0 - fp).abs() <= 1e-3 * fp.max(1.0) {
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
            eprintln!("{name}: frames {t_n} f0 一致 {agree:.4}(同じ判定のフレームの最大の相対差 {f0_rel:.2e})包絡の最大差 {env_max:.2e} 周期性の最大差 {per_max:.2e}(比べたフレーム {env_n})・RTF {rtf:.3}(比較込み)");
            assert!(agree >= 0.995, "{name}: f0 の一致 {agree} < 0.995");
            assert!(env_max < 1e-2, "{name}: 包絡の差 {env_max}");
            assert!(per_max < 1e-2, "{name}: 周期性の差 {per_max}");
        }
    }
}
