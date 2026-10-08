//! nvoc: 因果 NSF 型ボコーダ（training/nvoc.py と parity・重みは training/export_nvoc.py）。
//!
//! 48kHz・hop 240。`process_block(x, f0, out)` は入力ブロック t（x[tH..(t+1)H]）とフレーム t の f0 を受け取り、
//! 出力ブロック t（入力時刻 tH − DELAY からの 240 サンプルの再構成）を返す。アルゴリズム遅延 10ms。
//! `process_mel(mel, f0, out)` は log-mel を外から与える（VC の音響モデル出力用）。
//! manifest の const.NOSRC が true（調波源なしで学習）なら調波源を 0 にし f0 を使わない（`needs_f0()`）。
//! 畳み込みは全て左詰めの状態つき。ConvTranspose(2r, r) は後半 r を次の入力へ持ち越して重ね合わせる。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::ship_front::fft;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const SR: usize = 48000;
pub const HOP: usize = 240;
pub const DELAY: usize = 240;
pub const N_MEL: usize = 128;
pub const NFFT: usize = 2048;
pub const WIN: usize = 1024;
const NBIN: usize = NFFT / 2 + 1;

struct Tensors {
    t: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Tensors {
    fn load(dir: &Path) -> Result<(Self, serde_json::Value)> {
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
        let raw = std::fs::read(dir.join("weights.bin"))?;
        let all: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let mut t = HashMap::new();
        for (k, v) in man["tensors"].as_object().context("manifest.tensors")? {
            let off = v[0].as_u64().context("offset")? as usize;
            let shape: Vec<usize> = v[1].as_array().context("shape")?.iter().map(|s| s.as_u64().unwrap_or(0) as usize).collect();
            let n: usize = shape.iter().product();
            t.insert(k.clone(), (all[off..off + n].to_vec(), shape));
        }
        Ok((Self { t }, man))
    }

    fn get(&self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        self.t.get(k).cloned().with_context(|| format!("missing tensor {k}"))
    }
}

fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn lrelu(dst: &mut [f32], src: &[f32], s: f32) {
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = if v >= 0.0 { v } else { v * s };
    }
}

/// out[o][t] = b[o] + Σ_c Σ_j w[o][c][j]·buf[c][t + j·d]（buf 行幅 ls・out 行幅 n）。
#[allow(clippy::too_many_arguments)]
fn conv_scalar(buf: &[f32], ls: usize, w: &[f32], b: &[f32], ci: usize, co: usize, k: usize, d: usize, n: usize, out: &mut [f32]) {
    for o in 0..co {
        let row = &mut out[o * n..(o + 1) * n];
        row.iter_mut().for_each(|v| *v = b[o]);
        for c in 0..ci {
            for j in 0..k {
                let wv = w[(o * ci + c) * k + j];
                let src = &buf[c * ls + j * d..c * ls + j * d + n];
                for t in 0..n {
                    row[t] += wv * src[t];
                }
            }
        }
    }
}

/// 時間方向ベクトル版（n % 16 == 0・co % 4 == 0）。出力 4 チャネル × 16 時刻をレジスタに保持。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_tv16(buf: &[f32], ls: usize, w: &[f32], b: &[f32], ci: usize, co: usize, k: usize, d: usize, n: usize, out: &mut [f32]) {
    unsafe {
        let mut o0 = 0;
        while o0 + 4 <= co {
            let mut t0 = 0;
            while t0 < n {
                let mut acc = [_mm256_setzero_ps(); 8];
                for q in 0..4 {
                    let bb = _mm256_set1_ps(b[o0 + q]);
                    acc[2 * q] = bb;
                    acc[2 * q + 1] = bb;
                }
                for c in 0..ci {
                    let base = buf.as_ptr().add(c * ls + t0);
                    let w0 = w.as_ptr().add((o0 * ci + c) * k);
                    let w1 = w.as_ptr().add(((o0 + 1) * ci + c) * k);
                    let w2 = w.as_ptr().add(((o0 + 2) * ci + c) * k);
                    let w3 = w.as_ptr().add(((o0 + 3) * ci + c) * k);
                    for j in 0..k {
                        let p = base.add(j * d);
                        let s0 = _mm256_loadu_ps(p);
                        let s1 = _mm256_loadu_ps(p.add(8));
                        let v0 = _mm256_set1_ps(*w0.add(j));
                        acc[0] = _mm256_fmadd_ps(v0, s0, acc[0]);
                        acc[1] = _mm256_fmadd_ps(v0, s1, acc[1]);
                        let v1 = _mm256_set1_ps(*w1.add(j));
                        acc[2] = _mm256_fmadd_ps(v1, s0, acc[2]);
                        acc[3] = _mm256_fmadd_ps(v1, s1, acc[3]);
                        let v2 = _mm256_set1_ps(*w2.add(j));
                        acc[4] = _mm256_fmadd_ps(v2, s0, acc[4]);
                        acc[5] = _mm256_fmadd_ps(v2, s1, acc[5]);
                        let v3 = _mm256_set1_ps(*w3.add(j));
                        acc[6] = _mm256_fmadd_ps(v3, s0, acc[6]);
                        acc[7] = _mm256_fmadd_ps(v3, s1, acc[7]);
                    }
                }
                for q in 0..4 {
                    let op = out.as_mut_ptr().add((o0 + q) * n + t0);
                    _mm256_storeu_ps(op, acc[2 * q]);
                    _mm256_storeu_ps(op.add(8), acc[2 * q + 1]);
                }
                t0 += 16;
            }
            o0 += 4;
        }
    }
}

/// チャネル方向ベクトル版（co % 16 == 0）。wt は [c][j][co]。出力 16 チャネル × TT 時刻をレジスタに保持。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_cv<const TT: usize>(buf: &[f32], ls: usize, wt: &[f32], b: &[f32], ci: usize, co: usize, k: usize, d: usize,
                                   n: usize, t0: usize, out: &mut [f32]) {
    unsafe {
        let mut o0 = 0;
        while o0 < co {
            let b0 = _mm256_loadu_ps(b.as_ptr().add(o0));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(o0 + 8));
            let mut acc = [[b0, b1]; TT];
            for c in 0..ci {
                let xp = buf.as_ptr().add(c * ls + t0);
                for j in 0..k {
                    let wp = wt.as_ptr().add((c * k + j) * co + o0);
                    let w0 = _mm256_loadu_ps(wp);
                    let w1 = _mm256_loadu_ps(wp.add(8));
                    let xj = xp.add(j * d);
                    for (u, a) in acc.iter_mut().enumerate() {
                        let xv = _mm256_set1_ps(*xj.add(u));
                        a[0] = _mm256_fmadd_ps(w0, xv, a[0]);
                        a[1] = _mm256_fmadd_ps(w1, xv, a[1]);
                    }
                }
            }
            let mut tmp = [0f32; 16];
            for (u, a) in acc.iter().enumerate() {
                _mm256_storeu_ps(tmp.as_mut_ptr(), a[0]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), a[1]);
                for (i, &v) in tmp.iter().enumerate() {
                    *out.get_unchecked_mut((o0 + i) * n + t0 + u) = v;
                }
            }
            o0 += 16;
        }
    }
}

/// 状態つき因果 Conv1d（stride 1・dilation d）。
pub(crate) struct CConv {
    ci: usize,
    co: usize,
    k: usize,
    d: usize,
    w: Vec<f32>,
    wt: Vec<f32>,
    b: Vec<f32>,
    hist: Vec<f32>,
    buf: Vec<f32>,
}

impl CConv {
    fn new(ts: &Tensors, name: &str, d: usize) -> Result<Self> {
        let (w, sh) = ts.get(&format!("{name}.w"))?;
        let (b, _) = ts.get(&format!("{name}.b"))?;
        let (co, ci, k) = (sh[0], sh[1], sh[2]);
        Ok(Self::from_raw(w, b, co, ci, k, d))
    }

    /// w [co][ci][k]・b [co] から作る（A2 回路と共用）。
    pub(crate) fn from_raw(w: Vec<f32>, b: Vec<f32>, co: usize, ci: usize, k: usize, d: usize) -> Self {
        let mut wt = vec![0.0; w.len()];
        for o in 0..co {
            for c in 0..ci {
                for j in 0..k {
                    wt[(c * k + j) * co + o] = w[(o * ci + c) * k + j];
                }
            }
        }
        Self { ci, co, k, d, w, wt, b, hist: vec![0.0; ci * (k - 1) * d], buf: Vec::new() }
    }

    pub(crate) fn reset(&mut self) {
        self.hist.iter_mut().for_each(|v| *v = 0.0);
    }

    /// inp [ci][n] → out [co][n]
    pub(crate) fn run(&mut self, inp: &[f32], n: usize, out: &mut [f32]) {
        let l = (self.k - 1) * self.d;
        let ls = l + n;
        self.buf.resize(self.ci * ls, 0.0);
        for c in 0..self.ci {
            self.buf[c * ls..c * ls + l].copy_from_slice(&self.hist[c * l..(c + 1) * l]);
            self.buf[c * ls + l..(c + 1) * ls].copy_from_slice(&inp[c * n..(c + 1) * n]);
            if l > 0 {
                self.hist[c * l..(c + 1) * l].copy_from_slice(&self.buf[c * ls + n..(c + 1) * ls]);
            }
        }
        let (ci, co, k, d) = (self.ci, self.co, self.k, self.d);
        #[cfg(target_arch = "x86_64")]
        if have_avx2() {
            if n % 16 == 0 && co % 4 == 0 {
                unsafe { conv_tv16(&self.buf, ls, &self.w, &self.b, ci, co, k, d, n, out) };
                return;
            }
            if co % 16 == 0 {
                let mut t0 = 0;
                while t0 + 4 <= n {
                    unsafe { conv_cv::<4>(&self.buf, ls, &self.wt, &self.b, ci, co, k, d, n, t0, out) };
                    t0 += 4;
                }
                while t0 < n {
                    unsafe { conv_cv::<1>(&self.buf, ls, &self.wt, &self.b, ci, co, k, d, n, t0, out) };
                    t0 += 1;
                }
                return;
            }
        }
        conv_scalar(&self.buf, ls, &self.w, &self.b, ci, co, k, d, n, out);
    }
}

/// 状態つき因果 ConvTranspose1d(k = 2r, stride r)。後半 r を次の入力ステップへ持ち越す。
struct Up {
    ci: usize,
    co: usize,
    r: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    carry: Vec<f32>,
    tmp: Vec<f32>,
}

impl Up {
    fn new(ts: &Tensors, name: &str) -> Result<Self> {
        let (w, sh) = ts.get(&format!("{name}.w"))?;
        let (b, _) = ts.get(&format!("{name}.b"))?;
        let (ci, co, k) = (sh[0], sh[1], sh[2]);
        let r = k / 2;
        Ok(Self { ci, co, r, w, b, carry: vec![0.0; co * r], tmp: vec![0.0; co * k] })
    }

    fn reset(&mut self) {
        self.carry.iter_mut().for_each(|v| *v = 0.0);
    }

    /// inp [ci][n_in] → out [co][n_in·r]
    fn run(&mut self, inp: &[f32], n_in: usize, out: &mut [f32]) {
        let (r, co, ci) = (self.r, self.co, self.ci);
        let k = 2 * r;
        let n = n_in * r;
        for p in 0..n_in {
            self.tmp.iter_mut().for_each(|v| *v = 0.0);
            for c in 0..ci {
                let xv = inp[c * n_in + p];
                crate::simd::axpy(&mut self.tmp, &self.w[c * co * k..(c + 1) * co * k], xv);
            }
            for o in 0..co {
                for j in 0..r {
                    out[o * n + p * r + j] = self.carry[o * r + j] + self.tmp[o * k + j] + self.b[o];
                    self.carry[o * r + j] = self.tmp[o * k + r + j];
                }
            }
        }
    }
}

/// 励起(調波源・雑音 2ch)の strided 因果 conv。出力 j の窓は励起 [jS − lpad, jS − lpad + k)。
struct SrcConv {
    co: usize,
    k: usize,
    s: usize,
    lpad: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    hist: Vec<f32>,
    buf: Vec<f32>,
}

impl SrcConv {
    fn new(ts: &Tensors, name: &str, s: usize) -> Result<Self> {
        let (w, sh) = ts.get(&format!("{name}.w"))?;
        let (b, _) = ts.get(&format!("{name}.b"))?;
        let (co, k) = (sh[0], sh[2]);
        let lpad = if s > 1 { s } else { k - 1 };
        Ok(Self { co, k, s, lpad, w, b, hist: vec![0.0; 2 * lpad], buf: vec![0.0; 2 * (lpad + HOP)] })
    }

    fn reset(&mut self) {
        self.hist.iter_mut().for_each(|v| *v = 0.0);
    }

    /// exc [2][HOP] を読み out [co][HOP/s] に加算する。
    fn add(&mut self, exc: &[f32], out: &mut [f32]) {
        let (lp, ls) = (self.lpad, self.lpad + HOP);
        for c in 0..2 {
            self.buf[c * ls..c * ls + lp].copy_from_slice(&self.hist[c * lp..(c + 1) * lp]);
            self.buf[c * ls + lp..(c + 1) * ls].copy_from_slice(&exc[c * HOP..(c + 1) * HOP]);
            self.hist[c * lp..(c + 1) * lp].copy_from_slice(&self.buf[c * ls + HOP..(c + 1) * ls]);
        }
        let n = HOP / self.s;
        for o in 0..self.co {
            for jj in 0..n {
                let mut acc = self.b[o];
                for c in 0..2 {
                    let wr = &self.w[(o * 2 + c) * self.k..(o * 2 + c + 1) * self.k];
                    let xr = &self.buf[c * ls + jj * self.s..c * ls + jj * self.s + self.k];
                    for q in 0..self.k {
                        acc += wr[q] * xr[q];
                    }
                }
                out[o * n + jj] += acc;
            }
        }
    }
}

struct ResBlock {
    c1: Vec<CConv>,
    c2: Vec<CConv>,
}

struct Stage {
    up: Up,
    src: SrcConv,
    res: Vec<ResBlock>,
    co: usize,
    x: Vec<f32>,
    xr: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
    acc: Vec<f32>,
    inp: Vec<f32>,
}

impl Stage {
    fn run(&mut self, xin: &[f32], n_in: usize, exc: &[f32]) -> usize {
        let n = n_in * self.up.r;
        let sz = self.co * n;
        self.inp.resize(xin.len(), 0.0);
        lrelu(&mut self.inp, xin, 0.1);
        self.x.resize(sz, 0.0);
        self.up.run(&self.inp, n_in, &mut self.x);
        self.src.add(exc, &mut self.x);
        for v in [&mut self.xr, &mut self.t1, &mut self.t2, &mut self.t3] {
            v.resize(sz, 0.0);
        }
        self.acc.clear();
        self.acc.resize(sz, 0.0);
        for rb in self.res.iter_mut() {
            self.xr.copy_from_slice(&self.x);
            for (a, b) in rb.c1.iter_mut().zip(rb.c2.iter_mut()) {
                lrelu(&mut self.t1, &self.xr, 0.1);
                a.run(&self.t1, n, &mut self.t2);
                self.t2.iter_mut().for_each(|v| {
                    if *v < 0.0 {
                        *v *= 0.1
                    }
                });
                b.run(&self.t2, n, &mut self.t3);
                for (x, &y) in self.xr.iter_mut().zip(self.t3.iter()) {
                    *x += y;
                }
            }
            for (x, &y) in self.acc.iter_mut().zip(self.xr.iter()) {
                *x += y;
            }
        }
        let inv = 1.0 / self.res.len() as f32;
        self.acc.iter_mut().for_each(|v| *v *= inv);
        n
    }
}

struct MelFront {
    fb: Vec<f32>,
    win: Vec<f32>,
    ring: Vec<f32>,
    re: Vec<f64>,
    im: Vec<f64>,
    mag: Vec<f32>,
}

impl MelFront {
    fn push(&mut self, x: &[f32], mel: &mut [f32]) {
        self.ring.copy_within(HOP.., 0);
        self.ring[WIN - HOP..].copy_from_slice(x);
        for i in 0..NFFT {
            self.re[i] = if i < WIN { (self.ring[i] * self.win[i]) as f64 } else { 0.0 };
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        for k in 0..NBIN {
            self.mag[k] = (self.re[k] * self.re[k] + self.im[k] * self.im[k]).sqrt() as f32;
        }
        for (m, o) in mel.iter_mut().enumerate() {
            let row = &self.fb[m * NBIN..(m + 1) * NBIN];
            let mut s = 0.0f32;
            for (a, b) in row.iter().zip(self.mag.iter()) {
                s += a * b;
            }
            *o = s.max(1e-5).ln();
        }
    }
}

/// 帯域制限パルス列（単位 RMS）。ブロック t は f0[t−1] → f0[t] の直線補間。位相は f64。
struct HarmSrc {
    ph: f64,
    prev: f32,
    started: bool,
    h_max: f32,
    h_roll: f32,
    k_max: usize,
}

impl HarmSrc {
    fn block(&mut self, f0: f32, out: &mut [f32]) {
        let prev = if self.started { self.prev } else { f0 };
        self.started = true;
        self.prev = f0;
        let fa = if prev > 0.0 { prev } else { f0 };
        let fb = if f0 > 0.0 { f0 } else { prev };
        let va = if prev > 0.0 { 1.0f32 } else { 0.0 };
        let vb = if f0 > 0.0 { 1.0f32 } else { 0.0 };
        let dfa = (fb - fa) as f64;
        for (j, o) in out.iter_mut().enumerate().take(HOP) {
            let w64 = (j as f64 + 1.0) / HOP as f64;
            let f = fa as f64 + dfa * w64;
            self.ph += f / SR as f64;
            let w32 = (j as f32 + 1.0) / HOP as f32;
            let v = va + (vb - va) * w32;
            if v == 0.0 {
                *o = 0.0;
                continue;
            }
            let f32v = f as f32;
            let kmax = ((self.h_max / f32v).ceil() as usize).min(self.k_max);
            let phi = 2.0 * std::f64::consts::PI * (self.ph - self.ph.floor());
            let c2 = 2.0 * phi.cos();
            let (mut s1, mut s2) = (phi.sin(), 0.0f64);
            let (mut acc, mut nrm) = (0.0f64, 0.0f64);
            for k in 1..=kmax {
                let a = ((self.h_max - k as f32 * f32v) / self.h_roll).clamp(0.0, 1.0) as f64;
                acc += a * s1;
                nrm += a * a;
                let s0 = c2 * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            *o = v * (acc / (nrm / 2.0).sqrt().max(1e-3)) as f32;
        }
    }
}

pub struct NVoc {
    front: MelFront,
    src: HarmSrc,
    pre: CConv,
    stages: Vec<Stage>,
    post: CConv,
    mel: Vec<f32>,
    exc: Vec<f32>,
    x0: Vec<f32>,
    tail: Vec<f32>,
    noise_fix: Option<Vec<f32>>,
    noise_pos: usize,
    rng: u64,
    ch_last: usize,
    nosrc: bool,
}

impl NVoc {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let cfg = &man["cfg"];
        let cst = &man["const"];
        let ch = cfg["ch"].as_u64().context("cfg.ch")? as usize;
        let ups: Vec<usize> = cfg["ups"].as_array().context("ups")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
        let dils: Vec<usize> = cfg["dils"].as_array().context("dils")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
        let kernels: Vec<Vec<usize>> = cfg["kernels"].as_array().context("kernels")?.iter()
            .map(|a| a.as_array().map(|b| b.iter().map(|v| v.as_u64().unwrap_or(3) as usize).collect()).unwrap_or_default()).collect();
        let mut stages = Vec::new();
        let mut tot = 1;
        for (i, &r) in ups.iter().enumerate() {
            tot *= r;
            let s = HOP / tot;
            let mut res = Vec::new();
            for j in 0..kernels[i].len() {
                let mut c1 = Vec::new();
                let mut c2 = Vec::new();
                for (q, &d) in dils.iter().enumerate() {
                    c1.push(CConv::new(&ts, &format!("res{i}.{j}.c1.{q}"), d)?);
                    c2.push(CConv::new(&ts, &format!("res{i}.{j}.c2.{q}"), 1)?);
                }
                res.push(ResBlock { c1, c2 });
            }
            stages.push(Stage {
                up: Up::new(&ts, &format!("up{i}"))?,
                src: SrcConv::new(&ts, &format!("src{i}"), s)?,
                res,
                co: ch >> (i + 1),
                x: Vec::new(),
                xr: Vec::new(),
                t1: Vec::new(),
                t2: Vec::new(),
                t3: Vec::new(),
                acc: Vec::new(),
                inp: Vec::new(),
            });
        }
        let front = MelFront {
            fb: ts.get("mel.fb")?.0,
            win: ts.get("mel.win")?.0,
            ring: vec![0.0; WIN],
            re: vec![0.0; NFFT],
            im: vec![0.0; NFFT],
            mag: vec![0.0; NBIN],
        };
        let src = HarmSrc {
            ph: 0.0,
            prev: 0.0,
            started: false,
            h_max: cst["H_MAX"].as_f64().context("H_MAX")? as f32,
            h_roll: cst["H_ROLL"].as_f64().context("H_ROLL")? as f32,
            k_max: cst["K_MAX"].as_u64().context("K_MAX")? as usize,
        };
        Ok(Self {
            front,
            src,
            pre: CConv::new(&ts, "pre", 1)?,
            stages,
            post: CConv::new(&ts, "post", 1)?,
            mel: vec![0.0; N_MEL],
            exc: vec![0.0; 2 * HOP],
            x0: vec![0.0; ch],
            tail: vec![0.0; ch >> ups.len()],
            noise_fix: None,
            noise_pos: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
            ch_last: ch >> ups.len(),
            nosrc: cst["NOSRC"].as_bool().unwrap_or(false),
        })
    }

    pub fn reset(&mut self) {
        self.front.ring.iter_mut().for_each(|v| *v = 0.0);
        self.src.ph = 0.0;
        self.src.started = false;
        self.pre.reset();
        self.post.reset();
        for st in self.stages.iter_mut() {
            st.up.reset();
            st.src.reset();
            for rb in st.res.iter_mut() {
                rb.c1.iter_mut().for_each(|c| c.reset());
                rb.c2.iter_mut().for_each(|c| c.reset());
            }
        }
        self.noise_pos = 0;
    }

    /// 調波源なしの重み(manifest の NOSRC)なら false。false のとき f0 は無視される。
    pub fn needs_f0(&self) -> bool {
        !self.nosrc
    }

    pub fn set_noise_fixture(&mut self, v: Vec<f32>) {
        self.noise_fix = Some(v);
        self.noise_pos = 0;
    }

    fn gauss(&mut self) -> f32 {
        let mut u = || {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            ((self.rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (a, b) = (u(), u());
        ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
    }

    /// 入力ブロック t（HOP サンプル）とフレーム t の f0 → 出力ブロック t（HOP サンプル）。
    pub fn process_block(&mut self, x: &[f32], f0: f32, out: &mut [f32]) {
        let mut mel = std::mem::take(&mut self.mel);
        self.front.push(x, &mut mel);
        self.process_mel(&mel, f0, out);
        self.mel = mel;
    }

    /// log-mel（N_MEL）とフレーム t の f0 → 出力ブロック t。
    pub fn process_mel(&mut self, mel: &[f32], f0: f32, out: &mut [f32]) {
        let mut exc = std::mem::take(&mut self.exc);
        if self.nosrc {
            exc[..HOP].iter_mut().for_each(|v| *v = 0.0);
        } else {
            self.src.block(f0, &mut exc[..HOP]);
        }
        if let Some(nf) = &self.noise_fix {
            exc[HOP..].copy_from_slice(&nf[self.noise_pos..self.noise_pos + HOP]);
            self.noise_pos += HOP;
        } else {
            for i in 0..HOP {
                exc[HOP + i] = self.gauss();
            }
        }
        self.pre.run(mel, 1, &mut self.x0);
        let mut cur = std::mem::take(&mut self.x0);
        let mut n_in = 1;
        for st in self.stages.iter_mut() {
            n_in = st.run(&cur, n_in, &exc);
            cur.clear();
            cur.extend_from_slice(&st.acc);
        }
        self.tail.resize(self.ch_last * HOP, 0.0);
        lrelu(&mut self.tail, &cur, 0.01);
        self.post.run(&self.tail, HOP, out);
        self.x0 = cur;
        self.x0.resize(self.pre.co, 0.0);
        self.exc = exc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        let raw = std::fs::read(p).unwrap();
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    #[test]
    fn nvoc_parity_and_rtf() {
        let Ok(dir) = std::env::var("NVOC_EXPORT") else {
            eprintln!("NVOC_EXPORT 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let mut m = NVoc::load(dir).unwrap();
        let fx = dir.join("fixture");
        let x = read_f32(&fx.join("x.f32"));
        let f0 = read_f32(&fx.join("f0.f32"));
        let noise = read_f32(&fx.join("noise.f32"));
        let y = read_f32(&fx.join("y.f32"));
        let t = x.len() / HOP;
        m.set_noise_fixture(noise);
        let mut out = vec![0.0f32; t * HOP];
        let mut times = Vec::with_capacity(t);
        let t_all = std::time::Instant::now();
        for i in 0..t {
            let t0 = std::time::Instant::now();
            m.process_block(&x[i * HOP..(i + 1) * HOP], f0[i], &mut out[i * HOP..(i + 1) * HOP]);
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let el = t_all.elapsed().as_secs_f64();
        let md = out.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
        println!("parity: max|Rust − Python| {md:.3e}  (出力 RMS {rms:.3e})");
        println!("RTF (1 thread, streaming 240-sample blocks) {:.3}  per-block ms p50 {:.3} p95 {:.3} p99 {:.3} (予算 5 ms)",
                 el / (t as f64 * HOP as f64 / SR as f64), q(0.5), q(0.95), q(0.99));
        assert!(md < 1e-3 * rms.max(1e-3) + 1e-4, "parity FAIL {md}");
    }
}
