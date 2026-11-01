//! F1: 因果な f0・有声推定器(training/f0est.py と同じ・重みは training/export_f0est.py)。
//!
//! 特徴(フレーム t は入力の (t+1)·HOP サンプルまで): 直近 CTX48 サンプルを時間反転した区間で、窓 W ∈ WS の YIN の累積平均正規化差分(CMNDF・遅れ 0..TMAX)を
//! 対数の遅れ格子 N_LAG 点へ線形補間 + log エネルギー/10、と因果 log-mel((ln − ln 1e-5)/10)。ネット = 因果 conv(k 3)→ GELU の拡張畳み込みの残差 → 1 + N_BIN。
//! f0 = 最大の bin の ±4 bin の確率の重みつき log Hz の平均・有声 = ロジット > 0。先読み 0。

use std::path::Path;

use anyhow::{Context, Result};

use crate::nvoc::{CConv, MelFront, Tensors, HOP, N_MEL};
use crate::ship_front::fft;

const NFFT_C: usize = 4096;

fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}

fn gelu(dst: &mut [f32], src: &[f32]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        let x = v as f64;
        *d = (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32;
    }
}

pub struct F0Est {
    /// 有声の判定のヒステリシス(ロジットの閾値: 入る・抜ける)。None = ロジット > 0(確率 0.5)
    pub hyst: Option<(f32, f32)>,
    /// 音高の確信度(最大ビンの近傍 ±4 の事後の和)がこれ未満のフレームの f0 を 0(無声)にする。None = 従来。0.3 でオクターブ誤り 2.1 → 0.8%(実の男声)・新たに無声にしたフレームの 7〜8 割は pYIN も無声(D108)
    pub conf_min: Option<f32>,
    voiced: bool,
    /// 音高の事後の因果の前向きフィルタ(σ ビン・一様の跳び ε)。None = フレームごとの argmax(従来)。有声区間の中の 6 半音超の跳び 2.2 → 0.7 回/秒(D62)
    track: Option<(Vec<f32>, f64)>,
    alpha: Vec<f64>,
    pred: Vec<f64>,
    colinv: Vec<f64>,
    ws: Vec<usize>,
    tmax: usize,
    ctx: usize,
    lag_i0: Vec<usize>,
    lag_w: Vec<f64>,
    e_floor: f64,
    log_hz: Vec<f64>,
    n_bin: usize,
    ring: Vec<f32>,
    front: MelFront,
    mel: Vec<f32>,
    feat: Vec<f32>,
    inp: CConv,
    blocks: Vec<(CConv, CConv)>,
    out: CConv,
    x: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
    o: Vec<f32>,
    sre: Vec<f64>,
    sim: Vec<f64>,
    wre: Vec<f64>,
    wim: Vec<f64>,
    s: Vec<f64>,
    cs: Vec<f64>,
    d: Vec<f64>,
    dn: Vec<f64>,
}

impl F0Est {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let c = &man["const"];
        let us = |k: &str| -> Result<usize> { Ok(c[k].as_u64().with_context(|| format!("const.{k}"))? as usize) };
        let fl = |k: &str| -> Result<f64> { c[k].as_f64().with_context(|| format!("const.{k}")) };
        let ws: Vec<usize> = c["WS"].as_array().context("WS")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
        let (tmax, ctx, n_lag, n_bin) = (us("TMAX")?, us("CTX48")?, us("N_LAG")?, us("N_BIN")?);
        let (lag_min, f_lo, cents, e_floor) = (fl("LAG_MIN")?, fl("F_LO")?, fl("CENTS")?, fl("E_FLOOR")?);
        let (a, b) = (lag_min.ln(), ((tmax - 1) as f64).ln());
        let mut lag_i0 = Vec::with_capacity(n_lag);
        let mut lag_w = Vec::with_capacity(n_lag);
        for i in 0..n_lag {
            let l = (a + (b - a) * i as f64 / (n_lag - 1) as f64).exp();
            let i0 = (l.floor() as usize).min(tmax - 1);
            lag_i0.push(i0);
            lag_w.push(l - i0 as f64);
        }
        let log_hz: Vec<f64> = (0..n_bin).map(|k| (f_lo * 2f64.powf(k as f64 * cents / 1200.0)).ln() as f32 as f64).collect();
        let dils: Vec<usize> = man["cfg"]["dils"].as_array().context("cfg.dils")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
        let conv = |name: &str, d: usize| -> Result<CConv> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (bb, _) = ts.get(&format!("{name}.b"))?;
            Ok(CConv::from_raw(w, bb, sh[0], sh[1], sh[2], d))
        };
        let inp = conv("inp", 1)?;
        let mut blocks = Vec::new();
        for (i, &d) in dils.iter().enumerate() {
            blocks.push((conv(&format!("blocks.{i}.c1"), d)?, conv(&format!("blocks.{i}.c2"), 1)?));
        }
        let out = conv("out", 1)?;
        let front = MelFront::new(ts.get("mel.fb")?.0, ts.get("mel.win")?.0);
        let d_in = ws.len() * (n_lag + 1) + N_MEL;
        Ok(Self {
            hyst: None,
            conf_min: None,
            voiced: false,
            track: None,
            alpha: vec![],
            pred: vec![],
            colinv: vec![],
            ws, tmax, ctx, lag_i0, lag_w, e_floor, log_hz, n_bin, ring: vec![0.0; ctx], front, mel: vec![0.0; N_MEL], feat: vec![0.0; d_in], inp, blocks, out,
            x: vec![], t1: vec![], t2: vec![], t3: vec![], o: vec![0.0; 1 + n_bin], sre: vec![0.0; NFFT_C], sim: vec![0.0; NFFT_C], wre: vec![0.0; NFFT_C],
            wim: vec![0.0; NFFT_C], s: vec![0.0; ctx], cs: vec![0.0; ctx + 1], d: vec![0.0; tmax + 1], dn: vec![0.0; tmax + 1],
        })
    }

    fn cmndf_feats(&mut self) {
        let l = self.ctx;
        for j in 0..l {
            self.s[j] = self.ring[l - 1 - j] as f64;
        }
        self.cs[0] = 0.0;
        for j in 0..l {
            self.cs[j + 1] = self.cs[j] + self.s[j] * self.s[j];
        }
        for i in 0..NFFT_C {
            self.sre[i] = if i < l { self.s[i] } else { 0.0 };
            self.sim[i] = 0.0;
        }
        fft(&mut self.sre, &mut self.sim);
        let n_lag = self.lag_i0.len();
        assert_eq!(self.ws.len(), 2, "窓 2 本を 1 本の複素 FFT に詰める前提");
        let (w0, w1) = (self.ws[0], self.ws[1]);
        for i in 0..NFFT_C {
            self.wre[i] = if i < w0 { self.s[i] } else { 0.0 };
            self.wim[i] = if i < w1 { self.s[i] } else { 0.0 };
        }
        fft(&mut self.wre, &mut self.wim);
        let n = NFFT_C;
        let mut pre = vec![0.0f64; n];
        let mut pim = vec![0.0f64; n];
        for i in 0..n {
            let jn = (n - i) % n;
            let (zr, zi, cr, ci) = (self.wre[i], self.wim[i], self.wre[jn], -self.wim[jn]);
            let (a0r, a0i) = (0.5 * (zr + cr), 0.5 * (zi + ci));
            let (a1r, a1i) = (0.5 * (zi - ci), -0.5 * (zr - cr));
            let (br, bi) = (self.sre[i], self.sim[i]);
            let (p0r, p0i) = (a0r * br + a0i * bi, a0r * bi - a0i * br);
            let (p1r, p1i) = (a1r * br + a1i * bi, a1r * bi - a1i * br);
            pre[i] = p0r - p1i;
            pim[i] = p0i + p1r;
        }
        for i in 0..n {
            pim[i] = -pim[i];
        }
        fft(&mut pre, &mut pim);
        let inv = 1.0 / n as f64;
        let ws = self.ws.clone();
        for (wi, &w) in ws.iter().enumerate() {
            let e0 = self.cs[w];
            for tau in 0..=self.tmax {
                let r = if wi == 0 { pre[tau] * inv } else { -pim[tau] * inv };
                let et = self.cs[w + tau] - self.cs[tau];
                self.d[tau] = (e0 + et - 2.0 * r).max(0.0);
            }
            self.d[0] = 0.0;
            let mut cum = 0.0;
            self.dn[0] = 1.0;
            for tau in 1..=self.tmax {
                cum += self.d[tau];
                let cm = cum / tau as f64;
                self.dn[tau] = self.d[tau] / cm.max(1e-12);
            }
            let silent = e0 / (w as f64) < self.e_floor;
            let base = wi * (n_lag + 1);
            for i in 0..n_lag {
                let (i0, ww) = (self.lag_i0[i], self.lag_w[i]);
                let v = if silent { 1.0 } else { self.dn[i0] * (1.0 - ww) + self.dn[i0 + 1] * ww };
                self.feat[base + i] = v as f32;
            }
            self.feat[base + n_lag] = ((e0 / w as f64).max(self.e_floor).ln() / 10.0) as f32;
        }
    }

    /// 入力ブロック [HOP] → (有声のロジット, f0 Hz(無声 0))
    pub fn process_block(&mut self, xin: &[f32]) -> (f32, f32) {
        assert_eq!(xin.len(), HOP);
        self.ring.copy_within(HOP.., 0);
        let l = self.ring.len();
        self.ring[l - HOP..].copy_from_slice(xin);
        self.cmndf_feats();
        self.front.push(xin, &mut self.mel);
        let l1e5 = (1e-5f32).ln();
        let off = self.ws.len() * (self.lag_i0.len() + 1);
        for (k, &v) in self.mel.iter().enumerate() {
            self.feat[off + k] = (v - l1e5) / 10.0;
        }
        let ch = self.inp.co_len();
        self.x.resize(ch, 0.0);
        self.inp.run(&self.feat, 1, &mut self.x);
        for v in [&mut self.t1, &mut self.t2, &mut self.t3] {
            v.resize(ch, 0.0);
        }
        for (c1, c2) in self.blocks.iter_mut() {
            gelu(&mut self.t1, &self.x);
            c1.run(&self.t1, 1, &mut self.t2);
            gelu(&mut self.t3, &self.t2);
            c2.run(&self.t3, 1, &mut self.t1);
            for (a, &b) in self.x.iter_mut().zip(self.t1.iter()) {
                *a += b;
            }
        }
        gelu(&mut self.t1, &self.x);
        self.out.run(&self.t1, 1, &mut self.o);
        let v = self.o[0];
        let p = &self.o[1..];
        let nb = self.n_bin;
        let mx = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = p.iter().map(|&q| ((q - mx) as f64).exp()).sum();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        if let Some((ker, eps)) = &self.track {
            let half = ker.len() / 2;
            if self.alpha.len() != nb {
                self.alpha = vec![1.0 / nb as f64; nb];
                self.pred = vec![0.0; nb];
                self.colinv = (0..nb)
                    .map(|i| {
                        let lo = i.saturating_sub(half);
                        let hi = (i + half).min(nb - 1);
                        1.0 / (lo..=hi).map(|jj| ker[jj + half - i] as f64).sum::<f64>()
                    })
                    .collect();
            }
            for j in 0..nb {
                let mut acc = 0.0f64;
                for (o, &kv) in ker.iter().enumerate() {
                    let i = j as i64 + o as i64 - half as i64;
                    if i < 0 || i as usize >= nb {
                        continue;
                    }
                    let i = i as usize;
                    acc += kv as f64 * self.colinv[i] * self.alpha[i];
                }
                self.pred[j] = (1.0 - eps) * acc + eps / nb as f64;
            }
            let mut s = 0.0f64;
            for j in 0..nb {
                self.alpha[j] = self.pred[j] * ((p[j] - mx) as f64).exp() / z;
                s += self.alpha[j];
            }
            for a in self.alpha.iter_mut() {
                *a /= s.max(1e-300);
            }
            let k = (0..nb).fold(0usize, |b, j| if self.alpha[j] > self.alpha[b] { j } else { b }) as i64;
            for dk in -4i64..=4 {
                let i = (k + dk).clamp(0, nb as i64 - 1) as usize;
                num += self.alpha[i] * self.log_hz[i];
                den += self.alpha[i];
            }
        } else {
            let k = p.iter().position(|&q| q == mx).unwrap_or(0) as i64;
            for dk in -4i64..=4 {
                let i = (k + dk).clamp(0, nb as i64 - 1) as usize;
                let w = ((p[i] - mx) as f64).exp() / z;
                num += w * self.log_hz[i];
                den += w;
            }
        }
        let on = match self.hyst {
            None => v > 0.0,
            Some((hi, lo)) => if self.voiced { v >= lo } else { v > hi },
        };
        self.voiced = on;
        let conf_ok = self.conf_min.map_or(true, |m| den >= m as f64);
        let f0 = if on && conf_ok { (num / den.max(1e-9)).exp() as f32 } else { 0.0 };
        (v, f0)
    }

    /// 前向きフィルタを有効にする(σ = 遷移のガウスの幅[ビン]・ε = 一様の跳びの確率)。
    pub fn set_track(&mut self, sigma: f32, eps: f64) {
        let half = (4.0 * sigma).ceil() as usize;
        let ker: Vec<f32> = (0..=2 * half).map(|o| (-0.5 * ((o as f32 - half as f32) / sigma).powi(2)).exp()).collect();
        self.track = Some((ker, eps));
        self.alpha.clear();
    }

    pub fn last_feat(&self) -> &[f32] {
        &self.feat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        let raw = std::fs::read(p).unwrap();
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// F0EST_EXPORT=<export_f0est.py の出力> cargo test --release -p lightvc-core f0est_parity -- --nocapture
    #[test]
    fn f0est_parity() {
        let Ok(dir) = std::env::var("F0EST_EXPORT") else {
            eprintln!("F0EST_EXPORT 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let mut m = F0Est::load(dir).unwrap();
        let fx = dir.join("fixture");
        let x = read_f32(&fx.join("x.f32"));
        let feat = read_f32(&fx.join("feat.f32"));
        let vref = read_f32(&fx.join("v.f32"));
        let fref = read_f32(&fx.join("f0.f32"));
        let t = x.len() / HOP;
        let d_in = feat.len() / t;
        let (mut mf, mut mv) = (0.0f32, 0.0f32);
        let (mut vagree, mut cents) = (0usize, Vec::new());
        let t0 = std::time::Instant::now();
        for i in 0..t {
            let (v, f0) = m.process_block(&x[i * HOP..(i + 1) * HOP]);
            let ft = m.last_feat();
            for k in 0..d_in {
                mf = mf.max((ft[k] - feat[i * d_in + k]).abs());
            }
            mv = mv.max((v - vref[i]).abs());
            if (f0 > 0.0) == (fref[i] > 0.0) {
                vagree += 1;
            }
            if f0 > 0.0 && fref[i] > 0.0 {
                cents.push((1200.0 * (f0 / fref[i]).log2()).abs());
            }
        }
        let el = t0.elapsed().as_secs_f64();
        let mc = cents.iter().cloned().fold(0.0f32, f32::max);
        println!("f0est parity: 特徴 max|差| {mf:.3e}・有声ロジット max|差| {mv:.3e}・有声の一致 {}/{t}・f0 max|cent| {mc:.3e}・RTF {:.4}", vagree,
                 el / (t as f64 * HOP as f64 / 48000.0));
        assert!(mf < 1e-3 && vagree == t && mc < 1.0, "f0est parity FAIL");
    }

    /// F0EST_EXPORT=… F0_TRACK_REF=<Python の前向きフィルタの f0 [T] f32> cargo test --release -p lightvc-core f0_track_parity -- --nocapture
    #[test]
    fn f0_track_parity() {
        let (Ok(dir), Ok(rf)) = (std::env::var("F0EST_EXPORT"), std::env::var("F0_TRACK_REF")) else {
            eprintln!("F0EST_EXPORT / F0_TRACK_REF 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let mut m = F0Est::load(dir).unwrap();
        m.set_track(10.0, 0.02);
        let x = read_f32(&dir.join("fixture").join("x.f32"));
        let r = read_f32(Path::new(&rf));
        let t = x.len() / HOP;
        let (mut vm, mut md) = (0usize, 0.0f64);
        for i in 0..t.min(r.len()) {
            let (_, f0) = m.process_block(&x[i * HOP..(i + 1) * HOP]);
            if (f0 > 0.0) == (r[i] > 0.0) {
                vm += 1;
            }
            if f0 > 0.0 && r[i] > 0.0 {
                md = md.max((1200.0 * (f0 as f64 / r[i] as f64).log2()).abs());
            }
        }
        println!("f0 track parity: voicing {vm}/{} · max |cent| {md:.4}", t.min(r.len()));
        assert!(vm == t.min(r.len()) && md < 1.0, "track parity FAIL");
    }
}
