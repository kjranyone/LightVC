//! DDSP-VC ストリーミング推論（training/ddsp_vc.py と parity・重みは training/export_ddsp_vc.py）。
//!
//! 48kHz・hop 240。`process_block(x, f0)` は入力ブロック t（x[tH..(t+1)H]）と、そのブロック末で既知の f0 を受け取り、
//! 出力 y[(t+1)H..(t+2)H) を返す。y[n] は入力時刻 n − DELAY(480) の変換音声（アルゴリズム遅延 10ms）。
//! 最初のブロックでは Python の frames_to_samples が先頭 H サンプルを p[0] で保持するのに合わせ、
//! 出力しない「ブロック 0」を後段と位相に通して状態を揃える。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::ship_front::fft;

pub const SR: usize = 48000;
pub const HOP: usize = 240;
pub const DELAY: usize = 480;
pub const N_MEL: usize = 80;
pub const MEL_NFFT: usize = 1024;
pub const K_HARM: usize = 160;
pub const F_MAX: f32 = 22000.0;
pub const NOISE_NFFT: usize = 512;
const NBIN_M: usize = MEL_NFFT / 2 + 1;
const NBIN_N: usize = NOISE_NFFT / 2 + 1;
const PC: usize = 32;
const POST_CH: usize = 24;

pub struct Weights {
    t: HashMap<String, (Vec<f32>, Vec<usize>)>,
    norm: Option<f64>,
}

impl Weights {
    pub fn load(dir: &Path) -> Result<Self> {
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
        let c = &man["const"];
        let norm = if c["NORM"].as_bool().unwrap_or(false) { Some(c["NORM_PRIOR"].as_f64().unwrap_or(100.0)) } else { None };
        Ok(Self { t, norm })
    }

    fn get(&self, k: &str) -> Result<Vec<f32>> {
        Ok(self.t.get(k).with_context(|| format!("missing tensor {k}"))?.0.clone())
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), s1);
            i += 16;
        }
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            i += 8;
        }
        let mut buf = [0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), _mm256_add_ps(s0, s1));
        let mut r: f32 = buf.iter().sum();
        while i < n {
            r += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        r
    }
}

/// out[o] = b[o] + w[o, :] · x（w は [out][in] 行優先）。
fn gemv(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let cin = x.len();
    for (o, y) in out.iter_mut().enumerate() {
        *y = b[o] + dot(&w[o * cin..(o + 1) * cin], x);
    }
}

fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
        .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

#[inline]
fn gelu(x: f32) -> f32 {
    (0.5 * x as f64 * erfc(-(x as f64) / std::f64::consts::SQRT_2)) as f32
}

#[inline]
fn leaky(x: f32) -> f32 {
    if x >= 0.0 { x } else { 0.1 * x }
}

fn layer_norm(u: &mut [f32], w: &[f32], b: &[f32]) {
    let n = u.len() as f64;
    let mean = u.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = u.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for (i, v) in u.iter_mut().enumerate() {
        *v = ((*v as f64 - mean) * inv) as f32 * w[i] + b[i];
    }
}

/// k=3 の因果畳み込みの重み [o][i][k] を [o][k·cin + i] へ（入力 [x(t−2d), x(t−d), x(t)] を連結して 1 回の内積）。
fn rearrange_k3(w: &[f32], cout: usize, cin: usize) -> Vec<f32> {
    let mut r = vec![0f32; cout * 3 * cin];
    for o in 0..cout {
        for i in 0..cin {
            for k in 0..3 {
                r[o * 3 * cin + k * cin + i] = w[(o * cin + i) * 3 + k];
            }
        }
    }
    r
}

struct CRes {
    ch: usize,
    d: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    pw_w: Vec<f32>,
    pw_b: Vec<f32>,
    film_w: Option<(Vec<f32>, Vec<f32>)>,
    g: Vec<f32>,
    bb: Vec<f32>,
    hist: Vec<f32>,
    pos: usize,
    cat: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}

impl CRes {
    fn new(wt: &Weights, pre: &str, ch: usize, d: usize, film: bool) -> Result<Self> {
        Ok(Self {
            ch,
            d,
            w: rearrange_k3(&wt.get(&format!("{pre}.conv.weight"))?, ch, ch),
            b: wt.get(&format!("{pre}.conv.bias"))?,
            ln_w: wt.get(&format!("{pre}.norm.weight"))?,
            ln_b: wt.get(&format!("{pre}.norm.bias"))?,
            pw_w: wt.get(&format!("{pre}.pw.weight"))?,
            pw_b: wt.get(&format!("{pre}.pw.bias"))?,
            film_w: if film { Some((wt.get(&format!("{pre}.film.weight"))?, wt.get(&format!("{pre}.film.bias"))?)) } else { None },
            g: vec![0.0; ch],
            bb: vec![0.0; ch],
            hist: vec![0.0; 2 * d * ch],
            pos: 0,
            cat: vec![0.0; 3 * ch],
            u: vec![0.0; ch],
            v: vec![0.0; ch],
        })
    }

    fn set_speaker(&mut self, s: &[f32]) {
        if let Some((w, b)) = &self.film_w {
            let mut gb = vec![0f32; 2 * self.ch];
            gemv(w, b, s, &mut gb);
            self.g.copy_from_slice(&gb[..self.ch]);
            self.bb.copy_from_slice(&gb[self.ch..]);
        }
    }

    fn reset(&mut self) {
        self.hist.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
    }

    fn step(&mut self, h: &mut [f32]) {
        let (ch, d2) = (self.ch, 2 * self.d);
        let old = self.pos;
        let mid = (self.pos + self.d) % d2;
        self.cat[..ch].copy_from_slice(&self.hist[old * ch..(old + 1) * ch]);
        self.cat[ch..2 * ch].copy_from_slice(&self.hist[mid * ch..(mid + 1) * ch]);
        self.cat[2 * ch..].copy_from_slice(h);
        gemv(&self.w, &self.b, &self.cat, &mut self.u);
        layer_norm(&mut self.u, &self.ln_w, &self.ln_b);
        if self.film_w.is_some() {
            for i in 0..ch {
                self.u[i] = self.u[i] * (1.0 + self.g[i]) + self.bb[i];
            }
        }
        self.u.iter_mut().for_each(|v| *v = gelu(*v));
        gemv(&self.pw_w, &self.pw_b, &self.u, &mut self.v);
        self.hist[old * ch..(old + 1) * ch].copy_from_slice(h);
        self.pos = (self.pos + 1) % d2;
        for i in 0..ch {
            h[i] += self.v[i];
        }
    }
}

struct Front {
    fb: Vec<f32>,
    win: Vec<f32>,
    ring: Vec<f32>,
    re: Vec<f64>,
    im: Vec<f64>,
    pow: Vec<f32>,
}

impl Front {
    fn new(wt: &Weights) -> Result<Self> {
        Ok(Self {
            fb: wt.get("front.fb")?,
            win: wt.get("front.win")?,
            ring: vec![0.0; MEL_NFFT],
            re: vec![0.0; MEL_NFFT],
            im: vec![0.0; MEL_NFFT],
            pow: vec![0.0; NBIN_M],
        })
    }

    fn reset(&mut self) {
        self.ring.iter_mut().for_each(|v| *v = 0.0);
    }

    fn push(&mut self, x: &[f32], mel: &mut [f32]) {
        self.ring.copy_within(HOP.., 0);
        self.ring[MEL_NFFT - HOP..].copy_from_slice(x);
        for i in 0..MEL_NFFT {
            self.re[i] = (self.ring[i] * self.win[i]) as f64;
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        for k in 0..NBIN_M {
            self.pow[k] = (self.re[k] * self.re[k] + self.im[k] * self.im[k]) as f32;
        }
        for (m, o) in mel.iter_mut().enumerate() {
            *o = dot(&self.fb[m * NBIN_M..(m + 1) * NBIN_M], &self.pow).max(1e-8).ln();
        }
    }
}

struct Post {
    inp_w: Vec<f32>,
    inp_b: Vec<f32>,
    dils: Vec<usize>,
    conv_w: Vec<Vec<f32>>,
    conv_b: Vec<Vec<f32>>,
    mix_w: Vec<Vec<f32>>,
    mix_b: Vec<Vec<f32>>,
    hist: Vec<Vec<f32>>,
    pos: Vec<usize>,
    out_w: Vec<f32>,
    out_b: f32,
    x: Vec<f32>,
    h: Vec<f32>,
    cat: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}

impl Post {
    fn new(wt: &Weights) -> Result<Self> {
        let dils = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let mut conv_w = vec![];
        let mut conv_b = vec![];
        let mut mix_w = vec![];
        let mut mix_b = vec![];
        for i in 0..dils.len() {
            conv_w.push(rearrange_k3(&wt.get(&format!("post.convs.{i}.weight"))?, POST_CH, POST_CH));
            conv_b.push(wt.get(&format!("post.convs.{i}.bias"))?);
            mix_w.push(wt.get(&format!("post.mix.{i}.weight"))?);
            mix_b.push(wt.get(&format!("post.mix.{i}.bias"))?);
        }
        Ok(Self {
            inp_w: wt.get("post.inp.weight")?,
            inp_b: wt.get("post.inp.bias")?,
            hist: dils.iter().map(|d| vec![0.0; 2 * d * POST_CH]).collect(),
            pos: vec![0; dils.len()],
            dils,
            conv_w,
            conv_b,
            mix_w,
            mix_b,
            out_w: wt.get("post.out.weight")?,
            out_b: wt.get("post.out.bias")?[0],
            x: vec![0.0; 3 + PC],
            h: vec![0.0; POST_CH],
            cat: vec![0.0; 3 * POST_CH],
            u: vec![0.0; POST_CH],
            v: vec![0.0; POST_CH],
        })
    }

    fn reset(&mut self) {
        self.hist.iter_mut().for_each(|h| h.iter_mut().for_each(|v| *v = 0.0));
        self.pos.iter_mut().for_each(|p| *p = 0);
    }

    fn step(&mut self, s0: f32, harm: f32, noise: f32, pcs: &[f32]) -> f32 {
        self.x[0] = s0;
        self.x[1] = harm;
        self.x[2] = noise;
        self.x[3..].copy_from_slice(pcs);
        gemv(&self.inp_w, &self.inp_b, &self.x, &mut self.h);
        let c = POST_CH;
        for l in 0..self.dils.len() {
            let d = self.dils[l];
            let d2 = 2 * d;
            let old = self.pos[l];
            let mid = (old + d) % d2;
            let hist = &mut self.hist[l];
            self.cat[..c].copy_from_slice(&hist[old * c..(old + 1) * c]);
            self.cat[c..2 * c].copy_from_slice(&hist[mid * c..(mid + 1) * c]);
            self.cat[2 * c..].copy_from_slice(&self.h);
            gemv(&self.conv_w[l], &self.conv_b[l], &self.cat, &mut self.u);
            self.u.iter_mut().for_each(|v| *v = leaky(*v));
            gemv(&self.mix_w[l], &self.mix_b[l], &self.u, &mut self.v);
            hist[old * c..(old + 1) * c].copy_from_slice(&self.h);
            self.pos[l] = (old + 1) % d2;
            for i in 0..c {
                self.h[i] += self.v[i];
            }
        }
        let mut o = self.out_b;
        for i in 0..c {
            o += self.out_w[i] * leaky(self.h[i]);
        }
        s0 + o
    }
}

/// 対数メル周波数格子での直線内挿の添字と重み（Python の harmonic_synth / noise_synth と同じ式）。
fn mel_interp(lc: &[f32], f: f32) -> (usize, f32) {
    let n = lc.len();
    let idx = ((f.max(1.0).ln() - lc[0]) / (lc[n - 1] - lc[0]) * (n - 1) as f32).clamp(0.0, (n - 1) as f32 - 1e-6);
    let lo = idx.floor() as usize;
    (lo, idx - lo as f32)
}

pub struct DdspVc {
    front: Front,
    lc: Vec<f32>,
    c_inp_w: Vec<f32>,
    c_inp_b: Vec<f32>,
    c_blocks: Vec<CRes>,
    c_out_w: Vec<f32>,
    c_out_b: Vec<f32>,
    g_inp_w: Vec<f32>,
    g_inp_b: Vec<f32>,
    g_blocks: Vec<CRes>,
    harm_w: Vec<f32>,
    harm_b: Vec<f32>,
    noise_w: Vec<f32>,
    noise_b: Vec<f32>,
    pc_w: Vec<f32>,
    pc_b: Vec<f32>,
    spk_convs: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
    spk_proj_w: Vec<f32>,
    spk_proj_b: Vec<f32>,
    post: Post,
    win_n: Vec<f32>,
    /// content の因果インスタンス正規化（Python causal_norm と同式・事前値の重み）。None なら正規化しない（v1）。
    norm: Option<f64>,
    ns1: Vec<f64>,
    ns2: Vec<f64>,
    ncnt: f64,
    // 走行状態
    first: bool,
    phase: f64,
    f0_prev: f32,
    vuv_prev: f32,
    amp_prev: Vec<f32>,
    pc_prev: Vec<f32>,
    ola: Vec<f32>,
    rng: u64,
    noise_fixture: Option<(Vec<f32>, usize)>,
    // 作業領域
    mel: Vec<f32>,
    hc: Vec<f32>,
    c: Vec<f32>,
    gin: Vec<f32>,
    hg: Vec<f32>,
    he: Vec<f32>,
    ne: Vec<f32>,
    pc: Vec<f32>,
    amp: Vec<f32>,
    pcs: Vec<f32>,
}

impl DdspVc {
    pub fn load(dir: &Path) -> Result<Self> {
        let wt = Weights::load(dir)?;
        let centers = wt.get("centers")?;
        let mut c_blocks = vec![];
        for (i, d) in [1, 2, 4, 8, 1, 2, 4, 8].iter().enumerate() {
            c_blocks.push(CRes::new(&wt, &format!("content.blocks.{i}"), 256, *d, false)?);
        }
        let mut g_blocks = vec![];
        for (i, d) in [1, 2, 4, 8, 16, 1, 2, 4, 8, 16].iter().enumerate() {
            g_blocks.push(CRes::new(&wt, &format!("gen.blocks.{i}"), 384, *d, true)?);
        }
        let mut spk_convs = vec![];
        for (i, cin) in [(0usize, N_MEL), (2, 256), (4, 256)] {
            spk_convs.push((wt.get(&format!("spk.convs.{i}.weight"))?, wt.get(&format!("spk.convs.{i}.bias"))?, cin, 256));
        }
        let win_n: Vec<f32> = (0..NOISE_NFFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / NOISE_NFFT as f64).cos()) as f32)
            .collect();
        Ok(Self {
            front: Front::new(&wt)?,
            lc: centers.iter().map(|v| v.ln()).collect(),
            c_inp_w: wt.get("content.inp.weight")?,
            c_inp_b: wt.get("content.inp.bias")?,
            c_blocks,
            c_out_w: wt.get("content.out.weight")?,
            c_out_b: wt.get("content.out.bias")?,
            g_inp_w: wt.get("gen.inp.weight")?,
            g_inp_b: wt.get("gen.inp.bias")?,
            g_blocks,
            harm_w: wt.get("gen.harm.weight")?,
            harm_b: wt.get("gen.harm.bias")?,
            noise_w: wt.get("gen.noise.weight")?,
            noise_b: wt.get("gen.noise.bias")?,
            pc_w: wt.get("gen.pcond.weight")?,
            pc_b: wt.get("gen.pcond.bias")?,
            spk_convs,
            spk_proj_w: wt.get("spk.proj.weight")?,
            spk_proj_b: wt.get("spk.proj.bias")?,
            post: Post::new(&wt)?,
            win_n,
            norm: wt.norm,
            ns1: vec![0.0; 192],
            ns2: vec![0.0; 192],
            ncnt: 0.0,
            first: true,
            phase: 0.0,
            f0_prev: 0.0,
            vuv_prev: 0.0,
            amp_prev: vec![0.0; K_HARM],
            pc_prev: vec![0.0; PC],
            ola: vec![0.0; NOISE_NFFT + HOP],
            rng: 0x9E37_79B9_7F4A_7C15,
            noise_fixture: None,
            mel: vec![0.0; N_MEL],
            hc: vec![0.0; 256],
            c: vec![0.0; 192],
            gin: vec![0.0; 195],
            hg: vec![0.0; 384],
            he: vec![0.0; N_MEL],
            ne: vec![0.0; N_MEL],
            pc: vec![0.0; PC],
            amp: vec![0.0; K_HARM],
            pcs: vec![0.0; PC],
        })
    }

    pub fn reset(&mut self) {
        self.front.reset();
        self.c_blocks.iter_mut().for_each(|b| b.reset());
        self.g_blocks.iter_mut().for_each(|b| b.reset());
        self.post.reset();
        self.first = true;
        self.phase = 0.0;
        self.ola.iter_mut().for_each(|v| *v = 0.0);
        self.ns1.iter_mut().for_each(|v| *v = 0.0);
        self.ns2.iter_mut().for_each(|v| *v = 0.0);
        self.ncnt = 0.0;
    }

    /// 目標の参照音声（48k）→ 話者埋め込み（登録時に 1 回・非因果でよい）。FiLM を前計算する。
    pub fn set_target_ref(&mut self, pcm: &[f32]) -> Vec<f32> {
        let mut fr = Front { fb: self.front.fb.clone(), win: self.front.win.clone(), ring: vec![0.0; MEL_NFFT],
                             re: vec![0.0; MEL_NFFT], im: vec![0.0; MEL_NFFT], pow: vec![0.0; NBIN_M] };
        let t = pcm.len() / HOP;
        let mut seq: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut m = vec![0f32; N_MEL];
        for i in 0..t {
            fr.push(&pcm[i * HOP..(i + 1) * HOP], &mut m);
            seq.push(m.iter().map(|v| (v + 5.0) / 4.0).collect());
        }
        for (w, b, cin, cout) in &self.spk_convs {
            let mut out = vec![vec![0f32; *cout]; t];
            let mut col = vec![0f32; cin * 5];
            let mut wr = vec![0f32; cout * cin * 5];
            for o in 0..*cout {
                for i in 0..*cin {
                    for k in 0..5 {
                        wr[o * cin * 5 + k * cin + i] = w[(o * cin + i) * 5 + k];
                    }
                }
            }
            for ti in 0..t {
                for k in 0..5 {
                    let src = ti as isize + k as isize - 2;
                    let dst = &mut col[k * cin..(k + 1) * cin];
                    if src >= 0 && (src as usize) < t {
                        dst.copy_from_slice(&seq[src as usize]);
                    } else {
                        dst.iter_mut().for_each(|v| *v = 0.0);
                    }
                }
                gemv(&wr, b, &col, &mut out[ti]);
                out[ti].iter_mut().for_each(|v| *v = gelu(*v));
            }
            seq = out;
        }
        let ch = 256;
        let mut pooled = vec![0f32; 2 * ch];
        for c in 0..ch {
            let mu = seq.iter().map(|r| r[c] as f64).sum::<f64>() / t as f64;
            let var = seq.iter().map(|r| (r[c] as f64 - mu).powi(2)).sum::<f64>() / t as f64;
            pooled[c] = mu as f32;
            pooled[ch + c] = var.max(1e-6).sqrt() as f32;
        }
        let mut s = vec![0f32; 256];
        gemv(&self.spk_proj_w, &self.spk_proj_b, &pooled, &mut s);
        self.set_speaker(&s);
        s
    }

    pub fn set_speaker(&mut self, s: &[f32]) {
        self.g_blocks.iter_mut().for_each(|b| b.set_speaker(s));
    }

    /// parity 用: 雑音枝の白色雑音を外から与える（[T][512] を連結）。
    pub fn set_noise_fixture(&mut self, v: Vec<f32>) {
        self.noise_fixture = Some((v, 0));
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

    fn noise_frame(&mut self) {
        let mut re = vec![0f64; NOISE_NFFT];
        let mut im = vec![0f64; NOISE_NFFT];
        for i in 0..NOISE_NFFT {
            let z = match &mut self.noise_fixture {
                Some((v, p)) => {
                    let z = v[*p];
                    *p += 1;
                    z
                }
                None => self.gauss(),
            };
            re[i] = (z * self.win_n[i]) as f64;
        }
        fft(&mut re, &mut im);
        for k in 0..NBIN_N {
            let (lo, fr) = mel_interp(&self.lc, k as f32 * SR as f32 / NOISE_NFFT as f32);
            let hi = (lo + 1).min(N_MEL - 1);
            let g = (self.ne[lo] * (1.0 - fr) + self.ne[hi] * fr).exp() as f64;
            re[k] *= g;
            im[k] *= g;
            if k > 0 && k < NOISE_NFFT / 2 {
                re[NOISE_NFFT - k] = re[k];
                im[NOISE_NFFT - k] = -im[k];
            }
        }
        im[0] = 0.0;
        im[NOISE_NFFT / 2] = 0.0;
        for v in im.iter_mut() {
            *v = -*v;
        }
        fft(&mut re, &mut im);
        for i in 0..NOISE_NFFT {
            self.ola[i] += (re[i] / NOISE_NFFT as f64) as f32 * self.win_n[i];
        }
    }

    fn amplitudes(lc: &[f32], he: &[f32], f0f: f32, out: &mut [f32]) {
        for k in 0..K_HARM {
            let fk = f0f * (k + 1) as f32;
            if f0f > 0.0 && fk < F_MAX {
                let (lo, fr) = mel_interp(lc, fk);
                let hi = (lo + 1).min(N_MEL - 1);
                out[k] = (he[lo] * (1.0 - fr) + he[hi] * fr).exp();
            } else {
                out[k] = 0.0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render(&mut self, f0_a: f32, f0_b: f32, v_a: f32, v_b: f32, amp_a: &[f32], amp_b: &[f32], pc_a: &[f32], pc_b: &[f32],
              noise: &[f32], out: &mut [f32]) {
        let kmax = (0..K_HARM).rev().find(|&k| amp_a[k] != 0.0 || amp_b[k] != 0.0).map(|k| k + 1).unwrap_or(0);
        for j in 0..HOP {
            let w = (j + 1) as f32 / HOP as f32;
            let f0l = f0_a + (f0_b - f0_a) * w;
            let vs = v_a + (v_b - v_a) * w;
            let f0s = if vs > 0.5 { f0l } else { 0.0 };
            self.phase += f0s as f64 / SR as f64;
            self.phase -= self.phase.floor();
            let phf = (self.phase as f32) * (2.0 * std::f32::consts::PI);
            let (s1, c1) = (phf as f64).sin_cos();
            let (mut sp, mut sc) = (0.0f64, s1);
            let mut harm = 0f64;
            for k in 0..kmax {
                let a = amp_a[k] + (amp_b[k] - amp_a[k]) * w;
                harm += a as f64 * sc;
                let nx = 2.0 * c1 * sc - sp;
                sp = sc;
                sc = nx;
            }
            let harm = harm as f32 * vs;
            for i in 0..PC {
                self.pcs[i] = pc_a[i] + (pc_b[i] - pc_a[i]) * w;
            }
            out[j] = self.post.step(harm + noise[j], harm, noise[j], &self.pcs);
        }
    }

    /// 入力ブロック（HOP サンプル）と、そのブロック末で既知の f0（Hz・0=無声）→ 出力 HOP サンプル。
    pub fn process_block(&mut self, x: &[f32], f0: f32, out: &mut [f32]) {
        let mut mel = std::mem::take(&mut self.mel);
        self.front.push(x, &mut mel);
        let lev = (mel.iter().sum::<f32>() / N_MEL as f32 + 5.0) / 4.0;
        let xin: Vec<f32> = mel.iter().map(|v| (v + 5.0) / 4.0).collect();
        gemv(&self.c_inp_w, &self.c_inp_b, &xin, &mut self.hc);
        for b in self.c_blocks.iter_mut() {
            b.step(&mut self.hc);
        }
        gemv(&self.c_out_w, &self.c_out_b, &self.hc, &mut self.c);
        if let Some(prior) = self.norm {
            self.ncnt += 1.0;
            let n = self.ncnt + prior;
            for i in 0..192 {
                let c = self.c[i] as f64;
                self.ns1[i] += c;
                self.ns2[i] += c * c;
                let m1 = self.ns1[i] / n;
                let m2 = (self.ns2[i] + prior) / n;
                self.c[i] = ((c - m1) / (m2 - m1 * m1).max(1e-4).sqrt()) as f32;
            }
        }
        let vuv = if f0 > 0.0 { 1.0 } else { 0.0 };
        let lf0 = if f0 > 0.0 { (f0.max(1.0) / 200.0).ln() } else { 0.0 };
        self.gin[..192].copy_from_slice(&self.c);
        self.gin[192] = lf0;
        self.gin[193] = vuv;
        self.gin[194] = lev;
        gemv(&self.g_inp_w, &self.g_inp_b, &self.gin, &mut self.hg);
        for b in self.g_blocks.iter_mut() {
            b.step(&mut self.hg);
        }
        gemv(&self.harm_w, &self.harm_b, &self.hg, &mut self.he);
        gemv(&self.noise_w, &self.noise_b, &self.hg, &mut self.ne);
        gemv(&self.pc_w, &self.pc_b, &self.hg, &mut self.pc);
        self.mel = mel;
        let f0_gated = if vuv > 0.5 { f0 } else { 0.0 };
        if self.first {
            let mut amp0 = vec![0f32; K_HARM];
            Self::amplitudes(&self.lc, &self.he, f0_gated, &mut amp0);
            let pc0 = self.pc.clone();
            let zeros = vec![0f32; HOP];
            let mut discard = vec![0f32; HOP];
            self.render(f0, f0, vuv, vuv, &amp0.clone(), &amp0, &pc0.clone(), &pc0, &zeros, &mut discard);
            self.amp_prev = amp0;
            self.pc_prev = pc0;
            self.f0_prev = f0;
            self.vuv_prev = vuv;
            self.first = false;
        }
        let f0f = if self.vuv_prev > 0.5 { self.f0_prev } else { 0.0 };
        let mut amp = std::mem::take(&mut self.amp);
        Self::amplitudes(&self.lc, &self.he, f0f, &mut amp);
        self.noise_frame();
        let noise: Vec<f32> = self.ola[..HOP].iter().map(|v| v / 1.5).collect();
        self.ola.copy_within(HOP.., 0);
        let n = self.ola.len();
        self.ola[n - HOP..].iter_mut().for_each(|v| *v = 0.0);
        let (amp_a, pc_a, pc_b) = (self.amp_prev.clone(), self.pc_prev.clone(), self.pc.clone());
        let (f0_a, v_a) = (self.f0_prev, self.vuv_prev);
        self.render(f0_a, f0, v_a, vuv, &amp_a, &amp, &pc_a, &pc_b, &noise, out);
        self.amp_prev.copy_from_slice(&amp);
        self.amp = amp;
        self.pc_prev.copy_from_slice(&pc_b);
        self.f0_prev = f0;
        self.vuv_prev = vuv;
    }
}

/// 左寄せ YIN（training/artic_dsp.causal_yin と同じ式・窓 1536・hop 240・f64）。push 1 回 = 1 フレーム。
pub struct CausalYin {
    ring: Vec<f64>,
    thr: f64,
    voi_max: f64,
    floor_db: f64,
    re: Vec<f64>,
    im: Vec<f64>,
}

pub const YIN_W: usize = 1536;
const YIN_NFFT: usize = 4096;
const YIN_TMIN: usize = 48;
const YIN_TMAX: usize = 800;

impl CausalYin {
    pub fn new(voi_max: f64) -> Self {
        Self { ring: vec![0.0; YIN_W], thr: 0.15, voi_max, floor_db: -60.0, re: vec![0.0; YIN_NFFT], im: vec![0.0; YIN_NFFT] }
    }

    pub fn push(&mut self, x: &[f32]) -> f32 {
        self.ring.copy_within(HOP.., 0);
        for (i, v) in x.iter().enumerate() {
            self.ring[YIN_W - HOP + i] = *v as f64;
        }
        self.re.iter_mut().for_each(|v| *v = 0.0);
        self.im.iter_mut().for_each(|v| *v = 0.0);
        self.re[..YIN_W].copy_from_slice(&self.ring);
        fft(&mut self.re, &mut self.im);
        for k in 0..YIN_NFFT {
            self.re[k] = self.re[k] * self.re[k] + self.im[k] * self.im[k];
            self.im[k] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        let r: Vec<f64> = (0..YIN_TMAX + 2).map(|t| self.re[t] / YIN_NFFT as f64).collect();
        let mut cs = vec![0f64; YIN_W + 1];
        for i in 0..YIN_W {
            cs[i + 1] = cs[i] + self.ring[i] * self.ring[i];
        }
        let mut d = vec![0f64; YIN_TMAX + 2];
        for tau in 1..YIN_TMAX + 2 {
            d[tau] = (cs[YIN_W - tau] + (cs[YIN_W] - cs[tau]) - 2.0 * r[tau]).max(0.0);
        }
        let mut dn = vec![1f64; YIN_TMAX + 2];
        let mut acc = 0f64;
        for tau in 1..YIN_TMAX + 2 {
            acc += d[tau];
            dn[tau] = d[tau] / (acc / tau as f64).max(1e-12);
        }
        let seg = &dn[YIN_TMIN..YIN_TMAX + 1];
        let n = seg.len();
        let first_below = |th: f64| seg.iter().position(|&v| v < th);
        let k = match first_below(self.thr).or_else(|| first_below(self.voi_max)) {
            Some(k0) => {
                let mut k = k0;
                while k + 1 < n && seg[k + 1] < seg[k] {
                    k += 1;
                }
                k
            }
            None => seg.iter().enumerate().fold((0, f64::INFINITY), |b, (i, &v)| if v < b.1 { (i, v) } else { b }).0,
        };
        let dmin = if k < n - 2 { seg[k] } else { 1.0 };
        let mut tk = (k + YIN_TMIN) as f64;
        if k >= 1 && k < n - 1 {
            let (a, b, c) = (seg[k - 1], seg[k], seg[k + 1]);
            let den = a - 2.0 * b + c;
            if den.abs() > 1e-12 {
                tk += 0.5 * (a - c) / den;
            }
        }
        let rms_db = 10.0 * (cs[YIN_W] / YIN_W as f64).max(1e-20).log10();
        if dmin < self.voi_max && rms_db > self.floor_db { (SR as f64 / tk) as f32 } else { 0.0 }
    }
}

/// f0 レジスタの写像 log f0' = μ_T + (σ_T/σ_S)(log f0 − μ_S)（登録時の定数・発話統計ではない）。
pub fn map_register(f0: f32, mu_s: f32, sd_s: f32, mu_t: f32, sd_t: f32) -> f32 {
    if f0 <= 0.0 {
        return 0.0;
    }
    (mu_t + (sd_t / sd_s.max(1e-3)) * (f0.ln() - mu_s)).exp()
}

/// 登録音声（48k）→ f0 レジスタの定数（log f0 の中央値・標準偏差）。有声が少なければ男声の既定値。
pub fn register_from_pcm(x: &[f32]) -> (f32, f32) {
    let mut yin = CausalYin::new(0.25);
    let lf: Vec<f64> = x.chunks_exact(HOP).map(|c| yin.push(c)).filter(|f| *f > 0.0).map(|f| (f as f64).ln()).collect();
    if lf.len() < 20 {
        return ((120f64).ln() as f32, 0.15);
    }
    let mut s = lf.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = lf.iter().sum::<f64>() / lf.len() as f64;
    let sd = (lf.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / lf.len() as f64).sqrt();
    (s[s.len() / 2] as f32, sd as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    #[test]
    fn ddsp_profile() {
        let Ok(dir) = std::env::var("DDSP_EXPORT") else {
            return;
        };
        let mut m = DdspVc::load(Path::new(&dir)).unwrap();
        m.set_speaker(&vec![0.1f32; 256]);
        let x: Vec<f32> = (0..HOP).map(|i| (i as f32 * 0.05).sin() * 0.1).collect();
        let n = 2000;
        let mut out = vec![0f32; HOP];
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            m.process_block(&x, 180.0, &mut out);
        }
        let full = t0.elapsed().as_secs_f64() / n as f64 * 1000.0;
        let pcs = vec![0.1f32; PC];
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            for j in 0..HOP {
                out[j] = m.post.step(0.1, 0.1, 0.0, &pcs);
            }
        }
        let post = t1.elapsed().as_secs_f64() / n as f64 * 1000.0;
        let t2 = std::time::Instant::now();
        for _ in 0..n {
            for b in m.g_blocks.iter_mut() {
                b.step(&mut m.hg);
            }
        }
        let gen_ms = t2.elapsed().as_secs_f64() / n as f64 * 1000.0;
        let t3 = std::time::Instant::now();
        for _ in 0..n {
            for b in m.c_blocks.iter_mut() {
                b.step(&mut m.hc);
            }
        }
        let cont = t3.elapsed().as_secs_f64() / n as f64 * 1000.0;
        eprintln!("per block ms: full {full:.3} | post {post:.3} | gen blocks {gen_ms:.3} | content blocks {cont:.3} | rest {:.3}",
                  full - post - gen_ms - cont);
    }

    #[test]
    fn ddsp_parity_and_rtf() {
        let Ok(dir) = std::env::var("DDSP_EXPORT") else {
            eprintln!("DDSP_EXPORT 未設定: skip");
            return;
        };
        let dir = Path::new(&dir);
        let fx = dir.join("fixture");
        let (x, f0, rf, noise, y, spk) = (read_f32(&fx.join("x.f32")), read_f32(&fx.join("f0.f32")), read_f32(&fx.join("ref.f32")),
                                          read_f32(&fx.join("noise.f32")), read_f32(&fx.join("y.f32")), read_f32(&fx.join("spk.f32")));
        let mut m = DdspVc::load(dir).unwrap();
        let s = m.set_target_ref(&rf);
        let ds = s.iter().zip(&spk).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("speaker emb max|Δ| {ds:.2e}");
        m.set_noise_fixture(noise);
        let t = x.len() / HOP;
        let mut out = vec![0f32; t * HOP];
        for i in 0..t {
            m.process_block(&x[i * HOP..(i + 1) * HOP], f0[i], &mut out[i * HOP..(i + 1) * HOP]);
        }
        let n = (t - 1) * HOP;
        let err = out[..n].iter().zip(&y[HOP..HOP + n]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
        eprintln!("parity: max|Rust − Python| {err:.3e}  (出力 RMS {rms:.3e})");
        let mut yin = CausalYin::new(0.45);
        let mut agree = 0;
        for i in 0..t {
            let fr = yin.push(&x[i * HOP..(i + 1) * HOP]);
            if (fr - f0[i]).abs() <= 1e-2 * f0[i].max(1.0) {
                agree += 1;
            }
        }
        eprintln!("causal YIN agreement {agree}/{t}");
        let mut m2 = DdspVc::load(dir).unwrap();
        m2.set_speaker(&s);
        let secs = 10usize;
        let blocks = secs * SR / HOP;
        let mut ob = vec![0f32; HOP];
        let t0 = std::time::Instant::now();
        for i in 0..blocks {
            let j = i % t;
            m2.process_block(&x[j * HOP..(j + 1) * HOP], f0[j], &mut ob);
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("RTF (1 thread, streaming 240-sample blocks) {:.3}  per-block p50≈{:.3} ms (予算 5 ms)", dt / secs as f64,
                  1000.0 * dt / blocks as f64);
        assert!(ds < 1e-3, "speaker embedding parity");
        assert!(err < 1e-3 * rms.max(1e-3) * 100.0, "output parity");
    }
}
