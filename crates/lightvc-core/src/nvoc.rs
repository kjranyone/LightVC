//! 重みの読み込み(Tensors)・状態つき因果 Conv1d(CConv)・帯域制限パルス列(HarmSrc)と、製品の経路が共有する定数。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const SR: usize = 48000;
pub const HOP: usize = 240;
pub const DELAY: usize = 240;
pub const N_MEL: usize = 128;

pub(crate) struct Tensors {
    t: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Tensors {
    pub(crate) fn load(dir: &Path) -> Result<(Self, serde_json::Value)> {
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

    pub(crate) fn get(&self, k: &str) -> Result<(Vec<f32>, Vec<usize>)> {
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
    /// w [co][ci][k]・b [co] から作る。
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

    pub(crate) fn co_len(&self) -> usize {
        self.co
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

/// 帯域制限パルス列（単位 RMS）。ブロック t は f0[t−1] → f0[t] の直線補間。位相は f64。
pub(crate) struct HarmSrc {
    ph: f64,
    prev: f32,
    started: bool,
    h_max: f32,
    h_roll: f32,
    k_max: usize,
}

impl HarmSrc {
    pub(crate) fn new(h_max: f32, h_roll: f32, k_max: usize) -> Self {
        Self { ph: 0.0, prev: 0.0, started: false, h_max, h_roll, k_max }
    }

    pub(crate) fn block(&mut self, f0: f32, out: &mut [f32]) {
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
