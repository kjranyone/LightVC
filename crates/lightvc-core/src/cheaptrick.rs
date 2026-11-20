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

//! CheapTrick の 1 フレームの包絡(前段 v3 = p_front が使う)。
//! WORLD(M. Morise, modified BSD・NOTICE を参照)の cheaptrick.cpp / common.cpp / matlabfunctions.cpp を Rust へ移したもの。

use crate::fft::fft;

const SR: usize = 48000;
const NFFT: usize = 2048;
const NB: usize = NFFT / 2 + 1;
const K_PI: f64 = std::f64::consts::PI;
const K_SAFE: f64 = 0.000000000001;
const K_EPS: f64 = 0.00000000000000022204460492503131;
const K_DEFAULT_F0: f64 = 500.0;
const Q1: f64 = -0.15;

pub(crate) struct Randn {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl Randn {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn next(&mut self) -> f64 {
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
pub(crate) fn cheaptrick_frame(x: &[f64], f0_in: f64, pos: f64, rn: &mut Randn, re: &mut [f64], im: &mut [f64]) -> Vec<f64> {
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
