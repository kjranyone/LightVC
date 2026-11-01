// CheapTrick の部分は WORLD の移植。以下は WORLD のライセンス(modified BSD)。
// /* ----------------------------------------------------------------- */
// /*           WORLD: High-quality speech analysis,                    */
// /*           manipulation and synthesis system                       */
// /*           developed by M. Morise                                  */
// /*           http://www.kisc.meiji.ac.jp/~mmorise/world/english/     */
// /* ----------------------------------------------------------------- */
// /*                                                                   */
// /*  Copyright (c) 2010  M. Morise                                    */
// /*                                                                   */
// /* All rights reserved.                                              */
// /*                                                                   */
// /* Redistribution and use in source and binary forms, with or        */
// /* without modification, are permitted provided that the following   */
// /* conditions are met:                                               */
// /*                                                                   */
// /* - Redistributions of source code must retain the above copyright  */
// /*   notice, this list of conditions and the following disclaimer.   */
// /* - Redistributions in binary form must reproduce the above         */
// /*   copyright notice, this list of conditions and the following     */
// /*   disclaimer in the documentation and/or other materials provided */
// /*   with the distribution.                                          */
// /* - Neither the name of the M. Morise nor the names of its          */
// /*   contributors may be used to endorse or promote products derived */
// /*   from this software without specific prior written permission.   */
// /*                                                                   */
// /* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            */
// /* CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       */
// /* INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          */
// /* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          */
// /* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS */
// /* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          */
// /* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   */
// /* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     */
// /* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON */
// /* ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   */
// /* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    */
// /* OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
// /* POSSIBILITY OF SUCH DAMAGE.                                       */
// /* ----------------------------------------------------------------- */

//! 登録: 目標話者の参照音声(48kHz・20s 以上)→ 構音目標表(単位ごとの CheapTrick の DCT 包絡 c1..c24 の平均)・mu(有声 log f0 の中央値)・per_mean(4 帯の周期性の有声平均)。
//! training/pae.py(envelope・periodicity)・conv_c0.table と同じ手順。f0 は F1(D22: 教師の f0 と同一性が同じ)・単位は C1。オフライン(中心揃えの窓で可)。
//!
//! CheapTrick は WORLD(M. Morise, modified BSD・NOTICE を参照)の cheaptrick.cpp / common.cpp / matlabfunctions.cpp を Rust へ移したもの。

use std::path::Path;

use anyhow::{Context, Result};

use crate::artic_vc::Enroll;
use crate::c1::C1;
use crate::f0est::F0Est;
use crate::nvoc::{Tensors, HOP};
use crate::ship_front::fft;

const SR: usize = 48000;
const NFFT: usize = 2048;
const A: usize = NFFT / 2;
const NB: usize = NFFT / 2 + 1;
const K_PI: f64 = 3.1415926535897932384;
const K_SAFE: f64 = 0.000000000001;
const K_EPS: f64 = 0.00000000000000022204460492503131;
const K_DEFAULT_F0: f64 = 500.0;
const Q1: f64 = -0.15;

struct Randn {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl Randn {
    fn new() -> Self {
        Self { x: 123456789, y: 362436069, z: 521288629, w: 88675123 }
    }

    fn step(&mut self) -> u32 {
        let t = self.x ^ (self.x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w = (self.w ^ (self.w >> 19)) ^ (t ^ (t >> 8));
        self.w
    }

    fn next(&mut self) -> f64 {
        let mut tmp = self.step() >> 4;
        for _ in 0..11 {
            tmp = tmp.wrapping_add(self.step() >> 4);
        }
        tmp as f64 / 268435456.0 - 6.0
    }
}

fn mround(x: f64) -> i64 {
    if x > 0.0 { (x + 0.5) as i64 } else { (x - 0.5) as i64 }
}

/// WORLD interp1Q: 等間隔(起点 x・間隔 shift)の y を xi で線形補間。
fn interp1q(x: f64, shift: f64, y: &[f64], xi: &[f64], yi: &mut [f64]) {
    let n = y.len();
    let mut dy = vec![0.0; n];
    for i in 0..n - 1 {
        dy[i] = y[i + 1] - y[i];
    }
    dy[n - 1] = 0.0;
    for (o, &v) in yi.iter_mut().zip(xi.iter()) {
        let q = (v - x) / shift;
        let b = q as i64 as usize;
        let fr = q - b as f64;
        *o = y[b] + dy[b] * fr;
    }
}

fn dc_correction(p: &mut [f64], f0: f64) {
    let upper = 2 + (f0 * NFFT as f64 / SR as f64) as usize;
    let axis: Vec<f64> = (0..upper).map(|i| i as f64 * SR as f64 / NFFT as f64).collect();
    let rep_n = upper - 1;
    let mut rep = vec![0.0; rep_n];
    interp1q(f0 - axis[0], -(SR as f64) / NFFT as f64, &p[..upper + 1], &axis[..rep_n], &mut rep);
    for i in 0..rep_n {
        p[i] += rep[i];
    }
}

fn linear_smoothing(p: &[f64], width: f64) -> Vec<f64> {
    let boundary = (width * NFFT as f64 / SR as f64) as usize + 1;
    let half = NFFT / 2;
    let len = half + boundary * 2 + 1;
    let mut ms = vec![0.0; len];
    for i in 0..boundary {
        ms[i] = p[boundary - i];
    }
    for i in boundary..half + boundary {
        ms[i] = p[i - boundary];
    }
    for i in half + boundary..=half + boundary * 2 {
        ms[i] = p[half - (i - (half + boundary))];
    }
    let df = SR as f64 / NFFT as f64;
    let mut seg = vec![0.0; len];
    seg[0] = ms[0] * df;
    for i in 1..len {
        seg[i] = ms[i] * df + seg[i - 1];
    }
    let mut fa: Vec<f64> = (0..=half).map(|i| i as f64 / NFFT as f64 * SR as f64 - width / 2.0).collect();
    let origin = -(boundary as f64 - 0.5) * df;
    let mut lo = vec![0.0; half + 1];
    let mut hi = vec![0.0; half + 1];
    interp1q(origin, df, &seg, &fa, &mut lo);
    fa.iter_mut().for_each(|v| *v += width);
    interp1q(origin, df, &seg, &fa, &mut hi);
    lo.iter().zip(hi.iter()).map(|(l, h)| (h - l) / width).collect()
}

/// CheapTrick の 1 フレーム(位置 pos 秒・f0 Hz)→ パワー包絡 [NB]。
fn cheaptrick_frame(x: &[f64], f0_in: f64, pos: f64, rn: &mut Randn, re: &mut [f64], im: &mut [f64]) -> Vec<f64> {
    let f0_floor = 3.0 * SR as f64 / (NFFT as f64 - 3.0);
    let f0 = if f0_in <= f0_floor { K_DEFAULT_F0 } else { f0_in };
    let hwl = mround(1.5 * SR as f64 / f0) as i64;
    let origin = mround(pos * SR as f64 + 0.001);
    let n = (2 * hwl + 1) as usize;
    let mut win = vec![0.0; n];
    let mut avg = 0.0;
    for i in 0..n {
        let b = i as i64 - hwl;
        let p = b as f64 / 1.5 / SR as f64;
        win[i] = 0.5 * (K_PI * p * f0).cos() + 0.5;
        avg += win[i] * win[i];
    }
    let avg = avg.sqrt();
    win.iter_mut().for_each(|v| *v /= avg);
    let xl = x.len() as i64;
    let (mut s1, mut s2) = (0.0, 0.0);
    for i in 0..NFFT {
        re[i] = 0.0;
        im[i] = 0.0;
    }
    for i in 0..n {
        let idx = (origin + i as i64 - hwl).clamp(0, xl - 1) as usize;
        re[i] = x[idx] * win[i] + rn.next() * K_SAFE;
        s1 += re[i];
        s2 += win[i];
    }
    let wc = s1 / s2;
    for i in 0..n {
        re[i] -= win[i] * wc;
    }
    fft(re, im);
    let mut p: Vec<f64> = (0..NB).map(|i| re[i] * re[i] + im[i] * im[i]).collect();
    p.resize(NB + 1, 0.0);
    dc_correction(&mut p, f0);
    p.truncate(NB);
    let mut p = linear_smoothing(&p, f0 * 2.0 / 3.0);
    for v in p.iter_mut() {
        *v += rn.next().abs() * K_EPS;
    }
    for i in 0..NFFT {
        let k = if i <= NFFT / 2 { i } else { NFFT - i };
        re[i] = p[k].ln();
        im[i] = 0.0;
    }
    fft(re, im);
    let mut cep = vec![0.0; NFFT];
    for i in 0..=NFFT / 2 {
        let (sl, cl) = if i == 0 {
            (1.0, 1.0)
        } else {
            let q = i as f64 / SR as f64;
            ((K_PI * f0 * q).sin() / (K_PI * f0 * q), (1.0 - 2.0 * Q1) + 2.0 * Q1 * (2.0 * K_PI * q * f0).cos())
        };
        cep[i] = re[i] * sl * cl / NFFT as f64;
    }
    for i in 1..NFFT / 2 {
        cep[NFFT - i] = cep[i];
    }
    for i in 0..NFFT {
        re[i] = cep[i];
        im[i] = 0.0;
    }
    fft(re, im);
    (0..NB).map(|i| re[i].exp()).collect()
}

pub struct Analyzer {
    fb: Vec<f32>,
    dct: Vec<f32>,
    hb: Vec<(f64, f64)>,
    sm_w: f32,
}

impl Analyzer {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let hb = man["const"]["HB"].as_array().context("HB")?.iter().map(|b| (b[0].as_f64().unwrap_or(0.0), b[1].as_f64().unwrap_or(0.0))).collect();
        Ok(Self { fb: ts.get("mel.fb")?.0, dct: ts.get("dct")?.0, hb, sm_w: man["const"]["SM_W"].as_f64().unwrap_or(0.25) as f32 })
    }

    /// DCT の行 0..24 [25][n_mel] と n_mel(--gmap-floor)。
    pub fn dct25(&self) -> (Vec<f32>, usize) {
        let nm = self.fb.len() / NB;
        (self.dct[..25 * nm].to_vec(), nm)
    }

    /// DCT の行 1..24 [24][n_mel](c1..c24 → log 振幅の mel 帯)と n_mel。
    pub fn dct_rows(&self) -> (Vec<f32>, usize) {
        let nm = self.fb.len() / NB;
        (self.dct[nm..25 * nm].to_vec(), nm)
    }

    /// メル軸の周波数伸縮 f → f/α を c1..c24 に作用させる行列 [24][24](= B·I_α·Bᵀ・B = DCT の行 1..24)。training/d34_parallel_map.py の wmat と同じ。
    pub fn warp_matrix(&self, alpha: f32) -> Vec<f32> {
        let nm = self.fb.len() / NB;
        let fc: Vec<f64> = (0..nm)
            .map(|i| {
                let row = &self.fb[i * NB..(i + 1) * NB];
                let j = row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (j, &v)| if v > b.1 { (j, v) } else { b }).0;
                j as f64 * (SR as f64 / 2.0) / (NB - 1) as f64
            })
            .collect();
        let mut im = vec![0.0f64; nm * nm];
        for i in 0..nm {
            let src = (fc[i] / alpha as f64).clamp(fc[0], fc[nm - 1]);
            let j = fc.partition_point(|&v| v < src).saturating_sub(1).min(nm - 2);
            let w = ((src - fc[j]) / (fc[j + 1] - fc[j]).max(1e-9)).clamp(0.0, 1.0);
            im[i * nm + j] += 1.0 - w;
            im[i * nm + j + 1] += w;
        }
        let b = |k: usize, i: usize| self.dct[(k + 1) * nm + i] as f64;
        let mut bi = vec![0.0f64; 24 * nm];
        for k in 0..24 {
            for i in 0..nm {
                bi[k * nm + i] = (0..nm).map(|q| b(k, q) * im[q * nm + i]).sum();
            }
        }
        let mut w = vec![0.0f32; 24 * 24];
        for k in 0..24 {
            for l in 0..24 {
                w[k * 24 + l] = (0..nm).map(|i| bi[k * nm + i] * b(l, i)).sum::<f64>() as f32;
            }
        }
        w
    }

    /// 目標の表と操作者の表(各 [24][k])から伸縮率 α を当てる: メル 128 帯に戻した単位ごとの形(帯の平均を引く)の正規化相関が最大の α(1.00〜1.40・0.02 刻み)。
    pub fn fit_alpha(&self, tf: &[f32], top: &[f32], k: usize) -> f32 {
        let nm = self.fb.len() / NB;
        let spec = |t: &[f32]| -> Vec<f64> {
            let mut s = vec![0.0f64; nm * k];
            for u in 0..k {
                let mut m = 0.0;
                for i in 0..nm {
                    let v: f64 = (0..24).map(|r| self.dct[(r + 1) * nm + i] as f64 * t[r * k + u] as f64).sum();
                    s[u * nm + i] = v;
                    m += v;
                }
                for i in 0..nm {
                    s[u * nm + i] -= m / nm as f64;
                }
            }
            s
        };
        let sf = spec(tf);
        let nf: f64 = sf.iter().map(|v| v * v).sum();
        let mut best = (f64::NEG_INFINITY, 1.0f32);
        for g in 0..=20 {
            let al = 1.0 + 0.02 * g as f32;
            let w = self.warp_matrix(al);
            let mut wt = vec![0.0f32; 24 * k];
            for r in 0..24 {
                for u in 0..k {
                    wt[r * k + u] = (0..24).map(|c| w[r * 24 + c] * top[c * k + u]).sum();
                }
            }
            let sm = spec(&wt);
            let nmv: f64 = sm.iter().map(|v| v * v).sum();
            let ncc = sf.iter().zip(sm.iter()).map(|(a, b)| a * b).sum::<f64>() / (nf * nmv).sqrt().max(1e-30);
            if ncc > best.0 {
                best = (ncc, al);
            }
        }
        best.1
    }

    /// x(48kHz・長さ T·HOP)・f0 [T] → c [T][25](pae.envelope = CheapTrick → √ → mel 128 → log(床 1e-5)→ DCT 25)。平滑の前。
    pub fn envelope(&self, x: &[f32], f0: &[f32]) -> Vec<[f32; 25]> {
        let xa: Vec<f64> = std::iter::repeat(0.0).take(A).chain(x.iter().map(|&v| v as f64)).chain(std::iter::repeat(0.0).take(A)).collect();
        let mut rn = Randn::new();
        let (mut re, mut im) = (vec![0.0; NFFT], vec![0.0; NFFT]);
        let nm = self.fb.len() / NB;
        let mut out = Vec::with_capacity(f0.len());
        let mut mel = vec![0.0f64; nm];
        for (t, &f) in f0.iter().enumerate() {
            let pos = (A + t * HOP) as f64 / SR as f64;
            let sp = cheaptrick_frame(&xa, f as f64, pos, &mut rn, &mut re, &mut im);
            for m in 0..nm {
                let row = &self.fb[m * NB..(m + 1) * NB];
                let s: f64 = row.iter().zip(sp.iter()).map(|(&w, &p)| w as f64 * p.sqrt()).sum();
                mel[m] = s.max(1e-5).ln();
            }
            let mut c = [0.0f32; 25];
            for (k, ck) in c.iter_mut().enumerate() {
                let row = &self.dct[k * nm..(k + 1) * nm];
                *ck = row.iter().zip(mel.iter()).map(|(&d, &v)| d as f64 * v).sum::<f64>() as f32;
            }
            out.push(c);
        }
        out
    }

    /// pae.periodicity: x・f0 → per [T][4](倍音の山と谷のパワー比 dB/30・[0, 1.5]・無声 0)。
    pub fn periodicity(&self, x: &[f32], f0: &[f32]) -> Vec<[f32; 4]> {
        let xa: Vec<f64> = std::iter::repeat(0.0).take(A).chain(x.iter().map(|&v| v as f64)).chain(std::iter::repeat(0.0).take(A)).collect();
        let win: Vec<f64> = (0..NFFT).map(|i| 0.5 - 0.5 * (2.0 * K_PI * i as f64 / (NFFT - 1) as f64).cos()).collect();
        let (mut re, mut im) = (vec![0.0; NFFT], vec![0.0; NFFT]);
        let df = SR as f64 / NFFT as f64;
        let kmax = (self.hb.last().map(|b| b.1).unwrap_or(10000.0) / 60.0) as usize + 1;
        let interp = |p: &[f64], fbin: f64| -> f64 {
            let f = fbin.clamp(0.0, (NB - 1) as f64 - 1e-6);
            let i0 = f.floor() as usize;
            let w = f - i0 as f64;
            p[i0] * (1.0 - w) + p[i0 + 1] * w
        };
        let mut out = Vec::with_capacity(f0.len());
        for (t, &f) in f0.iter().enumerate() {
            for i in 0..NFFT {
                re[i] = xa[t * HOP + i] * win[i];
                im[i] = 0.0;
            }
            fft(&mut re, &mut im);
            let p: Vec<f64> = (0..NB).map(|i| re[i] * re[i] + im[i] * im[i]).collect();
            let fv = if f > 0.0 { f as f64 } else { 500.0 };
            let mut r = [0.0f32; 4];
            for (bi, &(lo, hi)) in self.hb.iter().enumerate().take(4) {
                let (mut num, mut den, mut cnt) = (0.0, 0.0, 0usize);
                for k in 1..=kmax {
                    let fk = fv * k as f64;
                    if !(fk >= lo && fk < hi && fk < SR as f64 / 2.0 - 200.0) {
                        continue;
                    }
                    let pk = [-0.2, -0.1, 0.0, 0.1, 0.2].iter().map(|o| interp(&p, (fk + o * fv) / df)).fold(f64::NEG_INFINITY, f64::max);
                    let vl = [0.4, 0.5, 0.6].iter().map(|o| interp(&p, (fk + o * fv) / df)).fold(f64::INFINITY, f64::min);
                    num += pk;
                    den += vl;
                    cnt += 1;
                }
                let hnr = 10.0 * ((num + 1e-20) / (den + 1e-20)).log10();
                let ok = cnt > 0 && f > 0.0 && num > 1e-12;
                r[bi] = if ok { (hnr / 30.0).clamp(0.0, 1.5) as f32 } else { 0.0 };
            }
            out.push(r);
        }
        out
    }

    /// R の記憶の入力(training/d579_rr_dev.py の rmem・F1 の f0 の CheapTrick・中央揃えの 3 点平滑・先頭 max_m フレーム): rp [K][m](C1 の事後の上位 16)・rv [25][m](c1..c24 / 10・相対 c0)。
    /// f1・c1 は新しい(状態のない)インスタンス(c1 は prime 済み)。F1 の f0 の記憶と教師の f0 の記憶の R の差分の相関 0.999(dev2 20 組)。
    pub fn rr_inputs(&self, x: &[f32], f1: &mut F0Est, c1: &mut C1, max_m: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let t = x.len() / HOP;
        let x = &x[..t * HOP];
        let k = c1.k;
        let m = t.min(max_m);
        let mut f0 = Vec::with_capacity(t);
        let mut rp = vec![0.0f32; k * m];
        let mut sel: Vec<(f32, usize)> = Vec::with_capacity(k);
        for i in 0..t {
            let b = &x[i * HOP..(i + 1) * HOP];
            f0.push(f1.process_block(b).1);
            c1.process_block(b);
            if i < m {
                let lg = c1.logits();
                let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = lg.iter().map(|&l| (l - mx).exp()).sum();
                sel.clear();
                sel.extend(lg.iter().map(|&l| (l - mx).exp() / z).zip(0..k));
                sel.select_nth_unstable_by(15, |a, b| b.0.partial_cmp(&a.0).unwrap());
                for &(v, j) in sel[..16].iter() {
                    rp[j * m + i] = v;
                }
            }
        }
        let env = self.envelope(x, &f0);
        let w = self.sm_w;
        let c0_sil = (1e-5f32).ln() * 128f32.sqrt();
        let mut rv = vec![0.0f32; 25 * m];
        let mut rc0 = vec![0.0f32; m];
        for i in 0..m {
            let (a, c) = (&env[i.saturating_sub(1)], &env[(i + 1).min(t - 1)]);
            let sm = |q: usize| w * a[q] + (1.0 - 2.0 * w) * env[i][q] + w * c[q];
            for r in 0..24 {
                rv[r * m + i] = sm(1 + r) / 10.0;
            }
            rc0[i] = (sm(0) - c0_sil) / 10.0;
        }
        let mean = rc0.iter().map(|&v| v as f64).sum::<f64>() / m.max(1) as f64;
        for i in 0..m {
            rv[24 * m + i] = rc0[i] - mean as f32;
        }
        (rp, rv, m)
    }

    /// 参照 x(48kHz)→ 登録。f1・c1 は新しい(状態のない)インスタンスを渡す(c1 は prime 済み)。cb = C1 のコードブック [K][D]。
    pub fn enroll(&self, x: &[f32], f1: &mut F0Est, c1: &mut C1, cb: &[f32]) -> Enroll {
        let t = x.len() / HOP;
        let x = &x[..t * HOP];
        let mut f0 = Vec::with_capacity(t);
        let mut u = Vec::with_capacity(t);
        for i in 0..t {
            let b = &x[i * HOP..(i + 1) * HOP];
            f0.push(f1.process_block(b).1);
            u.push(c1.process_block(b));
        }
        let env = self.envelope(x, &f0);
        let w = self.sm_w;
        let sm: Vec<[f32; 25]> = (0..t)
            .map(|i| {
                let (a, c) = (&env[i.saturating_sub(1)], &env[(i + 1).min(t - 1)]);
                let mut o = [0.0f32; 25];
                for k in 0..25 {
                    o[k] = w * a[k] + (1.0 - 2.0 * w) * env[i][k] + w * c[k];
                }
                o
            })
            .collect();
        let k = c1.k;
        let d = 24;
        let mut sum = vec![0.0f64; d * k];
        let mut cnt = vec![0usize; k];
        for i in 0..t {
            cnt[u[i]] += 1;
            for r in 0..d {
                sum[r * k + u[i]] += sm[i][1 + r] as f64;
            }
        }
        let c0m = (0..t).map(|i| sm[i][0] as f64).sum::<f64>() / t.max(1) as f64;
        let mut s0 = vec![0.0f64; k];
        for i in 0..t {
            s0[u[i]] += sm[i][0] as f64 - c0m;
        }
        let mut t0 = vec![0.0f32; k];
        let dim = cb.len() / k;
        let mut table = vec![0.0f32; d * k];
        for j in 0..k {
            let src = if cnt[j] > 0 {
                j
            } else {
                let mut best = (f32::NEG_INFINITY, j);
                for (q, &c) in cnt.iter().enumerate() {
                    if c == 0 {
                        continue;
                    }
                    let s: f32 = (0..dim).map(|z| cb[j * dim + z] * cb[q * dim + z]).sum();
                    if s > best.0 {
                        best = (s, q);
                    }
                }
                best.1
            };
            for r in 0..d {
                table[r * k + j] = if cnt[src] > 0 { (sum[r * k + src] / cnt[src] as f64) as f32 } else { 0.0 };
            }
            t0[j] = if cnt[src] > 0 { (s0[src] / cnt[src] as f64 / 10.0) as f32 } else { 0.0 };
        }
        let mut lf: Vec<f32> = f0.iter().filter(|&&f| f > 0.0).map(|f| f.ln()).collect();
        lf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mu = if lf.is_empty() { 5.6 } else if lf.len() % 2 == 1 { lf[lf.len() / 2] } else { 0.5 * (lf[lf.len() / 2 - 1] + lf[lf.len() / 2]) };
        let per = self.periodicity(x, &f0);
        let mut pm = [0.0f32; 4];
        let nv = f0.iter().filter(|&&f| f > 0.0).count().max(1);
        for (i, &f) in f0.iter().enumerate() {
            if f > 0.0 {
                for b in 0..4 {
                    pm[b] += per[i][b] / nv as f32;
                }
            }
        }
        let mut p75 = [0.0f32; 4];
        for b in 0..4 {
            let mut v: Vec<f32> = f0.iter().enumerate().filter(|(_, f)| **f > 0.0).map(|(i, _)| per[i][b]).collect();
            if v.is_empty() {
                continue;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = 0.75 * (v.len() - 1) as f32;
            let lo = q.floor() as usize;
            let hi = (lo + 1).min(v.len() - 1);
            p75[b] = v[lo] + (q - lo as f32) * (v[hi] - v[lo]);
        }
        let mut res_cov = vec![0.0f32; d * d];
        let vi: Vec<usize> = (0..t).filter(|&i| f0[i] > 0.0).collect();
        if vi.len() > 1 {
            let rs: Vec<Vec<f64>> = vi.iter().map(|&i| (0..d).map(|r| sm[i][1 + r] as f64 - table[r * k + u[i]] as f64).collect()).collect();
            let mean: Vec<f64> = (0..d).map(|r| rs.iter().map(|v| v[r]).sum::<f64>() / rs.len() as f64).collect();
            for a in 0..d {
                for b in 0..d {
                    res_cov[a * d + b] = (rs.iter().map(|v| (v[a] - mean[a]) * (v[b] - mean[b])).sum::<f64>() / rs.len() as f64) as f32;
                }
            }
        }
        Enroll { table, d, k, mu, per_mean: pm, per_p75: Some(p75), counts: cnt.iter().map(|&c| c as u32).collect(), res_cov, t0: Some(t0) }
    }
}

/// 対称行列 a [n][n] の冪 a^p(Jacobi の固有分解・固有値は最大の 1e-6 倍で下から抑える)。
/// 因果な CheapTrick の c0n・包絡(D253・`--causal-ct`): 入力ブロック t を受け、フレーム t の窓は (t+1)H で終わる(先読み 0)。
/// 分析の位置 = min(tH, (t+1)H − 1.5/f0)(f0 = F1・無声は 200 Hz)・CheapTrick → √ → mel → log(床 1e-5)→ DCT 25 → 因果の 3 点 [w, 1−2w, w] を (t−2, t−1, t)。
/// 出力 = (c0n = (c0 − C0_SIL)/10, c1..c24 / 10)。Python の対応 = training/d253_causal_timed.py の PQP。
pub struct CausalCt {
    fb: Vec<f32>,
    dct: Vec<f32>,
    nm: usize,
    w: f32,
    ring: Vec<f64>,
    t: usize,
    hist: Vec<[f32; 25]>,
    rn: Randn,
    re: Vec<f64>,
    im: Vec<f64>,
    mel: Vec<f64>,
}

const CT_RING: usize = 4096;

impl CausalCt {
    pub fn new(an: &Analyzer) -> Self {
        let nm = an.fb.len() / NB;
        Self { fb: an.fb.clone(), dct: an.dct.clone(), nm, w: an.sm_w, ring: vec![0.0; CT_RING], t: 0, hist: vec![], rn: Randn::new(), re: vec![0.0; NFFT], im: vec![0.0; NFFT], mel: vec![0.0; nm] }
    }

    pub fn process_block(&mut self, x: &[f32], f0: f32) -> (f32, [f32; 24]) {
        self.ring.drain(..HOP);
        self.ring.extend(x.iter().map(|&v| v as f64));
        let end = (self.t + 1) * HOP;
        let t0 = 1.0 / if f0 > 0.0 { f0 as f64 } else { 200.0 };
        let pos_abs = ((self.t * HOP) as f64).min(end as f64 - 1.5 * t0 * SR as f64).max(0.0);
        let pos = (pos_abs - (end as f64 - CT_RING as f64)) / SR as f64;
        let sp = cheaptrick_frame(&self.ring, if f0 > 0.0 { f0 as f64 } else { 0.0 }, pos, &mut self.rn, &mut self.re, &mut self.im);
        let nm = self.nm;
        for m in 0..nm {
            let row = &self.fb[m * NB..(m + 1) * NB];
            let s: f64 = row.iter().zip(sp.iter()).map(|(&w, &p)| w as f64 * p.sqrt()).sum();
            self.mel[m] = s.max(1e-5).ln();
        }
        let mut c = [0.0f32; 25];
        for (k, ck) in c.iter_mut().enumerate() {
            let row = &self.dct[k * nm..(k + 1) * nm];
            *ck = row.iter().zip(self.mel.iter()).map(|(&d, &v)| d as f64 * v).sum::<f64>() as f32;
        }
        if self.hist.is_empty() {
            self.hist = vec![c; 3];
        } else {
            self.hist.remove(0);
            self.hist.push(c);
        }
        self.t += 1;
        let w = self.w;
        let sm: Vec<f32> = (0..25).map(|k| w * self.hist[0][k] + (1.0 - 2.0 * w) * self.hist[1][k] + w * self.hist[2][k]).collect();
        let c0_sil = (1e-5f64).ln() * (nm as f64).sqrt();
        let mut e = [0.0f32; 24];
        for r in 0..24 {
            e[r] = sm[1 + r] / 10.0;
        }
        (((sm[0] as f64 - c0_sil) / 10.0) as f32, e)
    }
}

pub fn sym_pow(a: &[f64], n: usize, p: f64) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| m[i * n + j] * m[i * n + j]).sum();
        if off < 1e-20 {
            break;
        }
        for pi in 0..n {
            for q in pi + 1..n {
                let apq = m[pi * n + q];
                if apq.abs() < 1e-30 {
                    continue;
                }
                let th = (m[q * n + q] - m[pi * n + pi]) / (2.0 * apq);
                let t = if th == 0.0 { 1.0 } else { th.signum() / (th.abs() + (th * th + 1.0).sqrt()) };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for kk in 0..n {
                    let (akp, akq) = (m[kk * n + pi], m[kk * n + q]);
                    m[kk * n + pi] = c * akp - s * akq;
                    m[kk * n + q] = s * akp + c * akq;
                }
                for kk in 0..n {
                    let (apk, aqk) = (m[pi * n + kk], m[q * n + kk]);
                    m[pi * n + kk] = c * apk - s * aqk;
                    m[q * n + kk] = s * apk + c * aqk;
                }
                for kk in 0..n {
                    let (vkp, vkq) = (v[kk * n + pi], v[kk * n + q]);
                    v[kk * n + pi] = c * vkp - s * vkq;
                    v[kk * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let ev: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    let fl = ev.iter().cloned().fold(0.0f64, f64::max) * 1e-6;
    let pw: Vec<f64> = ev.iter().map(|&e| e.max(fl).max(1e-30).powf(p)).collect();
    let mut out = vec![0.0f64; n * n];
    for r in 0..n {
        for c in 0..n {
            out[r * n + c] = (0..n).map(|z| v[r * n + z] * pw[z] * v[c * n + z]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn sym_pow_inverse_sqrt() {
        let n = 6;
        let mut b = vec![0.0f64; n * n];
        let mut st = 12345u64;
        for v in b.iter_mut() {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((st >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        }
        let a: Vec<f64> = (0..n * n).map(|i| (0..n).map(|z| b[(i / n) * n + z] * b[(i % n) * n + z]).sum::<f64>() + if i / n == i % n { 0.1 } else { 0.0 }).collect();
        let h = super::sym_pow(&a, n, 0.5);
        let hi = super::sym_pow(&a, n, -0.5);
        for r in 0..n {
            for c in 0..n {
                let hh: f64 = (0..n).map(|z| h[r * n + z] * h[z * n + c]).sum();
                let id: f64 = (0..n).map(|z| h[r * n + z] * hi[z * n + c]).sum();
                assert!((hh - a[r * n + c]).abs() < 1e-9, "h·h != a");
                assert!((id - if r == c { 1.0 } else { 0.0 }).abs() < 1e-9, "h·h⁻¹ != I");
            }
        }
    }

    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        let raw = std::fs::read(p).unwrap();
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// ENROLL_EXPORT=<export_enroll.py の出力> cargo test --release -p lightvc-core enroll_parity -- --nocapture
    #[test]
    fn enroll_parity() {
        let Ok(dir) = std::env::var("ENROLL_EXPORT") else {
            eprintln!("ENROLL_EXPORT 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let an = Analyzer::load(dir).unwrap();
        let fx = dir.join("fixture");
        let x = read_f32(&fx.join("x.f32"));
        let f0 = read_f32(&fx.join("f0.f32"));
        let env = read_f32(&fx.join("env.f32"));
        let per = read_f32(&fx.join("per.f32"));
        let t0 = std::time::Instant::now();
        let e = an.envelope(&x, &f0);
        let el = t0.elapsed().as_secs_f64();
        let mut me = 0.0f32;
        for (i, c) in e.iter().enumerate() {
            for k in 0..25 {
                me = me.max((c[k] - env[i * 25 + k]).abs());
            }
        }
        let p = an.periodicity(&x, &f0);
        let mut mp = 0.0f32;
        for (i, r) in p.iter().enumerate() {
            for b in 0..4 {
                mp = mp.max((r[b] - per[i * 4 + b]).abs());
            }
        }
        println!("enroll parity: CheapTrick の DCT 包絡 max|差| {me:.3e}・周期性 max|差| {mp:.3e}・包絡の時間 {:.2} s / {:.1} s の音声", el, x.len() as f64 / 48000.0);
        assert!(me < 1e-3 && mp < 1e-3, "enroll parity FAIL");
    }

    /// ENROLL_EXPORT・F0EST_EXPORT・C1_EXPORT を設定して cargo test --release -p lightvc-core enroll_full_parity -- --nocapture
    #[test]
    fn enroll_full_parity() {
        let env = |k: &str| std::env::var(k).ok();
        let (Some(d), Some(f), Some(c)) = (env("ENROLL_EXPORT"), env("F0EST_EXPORT"), env("C1_EXPORT")) else {
            eprintln!("書き出しの環境変数が未設定のためスキップ");
            return;
        };
        let an = Analyzer::load(Path::new(&d)).unwrap();
        let mut f1 = F0Est::load(Path::new(&f)).unwrap();
        let mut c1 = C1::load(Path::new(&c)).unwrap();
        c1.prime();
        let (ts, _) = Tensors::load(Path::new(&c)).unwrap();
        let cb = ts.get("codebook").unwrap().0;
        let fx = Path::new(&d).join("fixture_full");
        let x = read_f32(&fx.join("x.f32"));
        let tab = read_f32(&fx.join("table.f32"));
        let mp = read_f32(&fx.join("mu_per.f32"));
        let t0 = std::time::Instant::now();
        let e = an.enroll(&x, &mut f1, &mut c1, &cb);
        let el = t0.elapsed().as_secs_f64();
        let mt = e.table.iter().zip(tab.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mpd = e.per_mean.iter().zip(mp[1..].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("enroll full: 表 max|差| {mt:.3e}・mu 差 {:.3e}・per_mean max|差| {mpd:.3e}・登録の時間 {el:.2} s({:.1} s の参照)", (e.mu - mp[0]).abs(),
                 x.len() as f64 / 48000.0);
        assert!(mt < 2e-3 && (e.mu - mp[0]).abs() < 1e-3 && mpd < 1e-3, "enroll full parity FAIL");
    }
}
