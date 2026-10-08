//! A2-VC 回路（training/a2vc.py と同じ構造）。48kHz・アップサンプリングなし・因果拡張畳み込み。
//!
//! `process_block(carriers, ctrl, out)`: 搬送波 [3][HOP]（c_in・c_pulse・c_noise）とフレーム t の制御 [d_ctrl + d_spk] から
//! 出力ブロック t を作る。各層の γ, β はフレーム t−1 → t を直線補間（t=0 は t の値）。

use crate::nvoc::{CConv, HOP};

pub const DILS: [usize; 16] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 1, 3, 8, 21];
const D_HID: usize = 256;

pub struct A2Circuit {
    ch: usize,
    d_in: usize,
    inp: CConv,
    convs: Vec<CConv>,
    pws: Vec<CConv>,
    out: CConv,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    prev: Vec<f32>,
    cur: Vec<f32>,
    hid: Vec<f32>,
    has_prev: bool,
    h: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    ramp: Vec<f32>,
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm_erf(x / std::f32::consts::SQRT_2))
}

fn libm_erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

impl A2Circuit {
    /// 重みを並べて作る。w_inp [ch][3]・convs[i] [ch][ch][3]・pws[i] [ch][ch]・w_out [1][ch]・w1 [256][d_in]・w2 [2·ch·L][256]。
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(ch: usize, d_in: usize, inp: (Vec<f32>, Vec<f32>), convs: Vec<(Vec<f32>, Vec<f32>)>, pws: Vec<(Vec<f32>, Vec<f32>)>,
                      out: (Vec<f32>, Vec<f32>), w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>) -> Self {
        let nl = DILS.len();
        let inp = CConv::from_raw(inp.0, inp.1, ch, 3, 1, 1);
        let convs = convs.into_iter().zip(DILS).map(|((w, b), d)| CConv::from_raw(w, b, ch, ch, 3, d)).collect();
        let pws = pws.into_iter().map(|(w, b)| CConv::from_raw(w, b, ch, ch, 1, 1)).collect();
        let out = CConv::from_raw(out.0, out.1, 1, ch, 1, 1);
        let nc = 2 * ch * nl;
        Self {
            ch, d_in, inp, convs, pws, out, w1, b1, w2, b2,
            prev: vec![0.0; nc], cur: vec![0.0; nc], hid: vec![0.0; D_HID], has_prev: false,
            h: vec![0.0; ch * HOP], u: vec![0.0; ch * HOP], v: vec![0.0; ch * HOP],
            ramp: (0..HOP).map(|i| (i + 1) as f32 / HOP as f32).collect(),
        }
    }

    /// 乱数の重み（RTF 計測用）。
    pub fn random(ch: usize, d_in: usize, seed: u64) -> Self {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
            }).collect()
        };
        let nl = DILS.len();
        let inp = (rnd(ch * 3, 0.5), rnd(ch, 0.1));
        let convs = (0..nl).map(|_| (rnd(ch * ch * 3, 1.0 / (3.0 * ch as f32).sqrt()), rnd(ch, 0.1))).collect();
        let pws = (0..nl).map(|_| (rnd(ch * ch, 0.3 / (ch as f32).sqrt()), rnd(ch, 0.01))).collect();
        let out = (rnd(ch, 1.0 / (ch as f32).sqrt()), rnd(1, 0.0));
        let w1 = rnd(D_HID * d_in, 1.0 / (d_in as f32).sqrt());
        let b1 = rnd(D_HID, 0.1);
        let w2 = rnd(2 * ch * nl * D_HID, 0.05 / (D_HID as f32).sqrt());
        let b2 = rnd(2 * ch * nl, 0.01);
        Self::from_parts(ch, d_in, inp, convs, pws, out, w1, b1, w2, b2)
    }

    pub fn reset(&mut self) {
        self.inp.reset();
        self.convs.iter_mut().for_each(|c| c.reset());
        self.pws.iter_mut().for_each(|c| c.reset());
        self.out.reset();
        self.has_prev = false;
    }

    pub fn macs_per_second(&self) -> f64 {
        let ch = self.ch as f64;
        let per = 3.0 * ch + DILS.len() as f64 * (3.0 * ch * ch + ch * ch + 2.0 * ch) + ch;
        per * 48000.0
    }

    fn control(&mut self, ctrl: &[f32]) {
        let d = self.d_in;
        for (o, hv) in self.hid.iter_mut().enumerate() {
            let w = &self.w1[o * d..(o + 1) * d];
            *hv = gelu(self.b1[o] + w.iter().zip(ctrl).map(|(a, b)| a * b).sum::<f32>());
        }
        for (o, cv) in self.cur.iter_mut().enumerate() {
            let w = &self.w2[o * D_HID..(o + 1) * D_HID];
            *cv = self.b2[o] + w.iter().zip(&self.hid).map(|(a, b)| a * b).sum::<f32>();
        }
        if !self.has_prev {
            self.prev.copy_from_slice(&self.cur);
            self.has_prev = true;
        }
    }

    /// carriers [3][HOP]・ctrl [d_in] → out [HOP]
    pub fn process_block(&mut self, carriers: &[f32], ctrl: &[f32], out: &mut [f32]) {
        let ch = self.ch;
        self.control(ctrl);
        self.inp.run(carriers, HOP, &mut self.h);
        for i in 0..DILS.len() {
            self.convs[i].run(&self.h, HOP, &mut self.u);
            for c in 0..ch {
                let gi = 2 * i * ch + c;
                let bi = (2 * i + 1) * ch + c;
                let (pg, dg) = (self.prev[gi], self.cur[gi] - self.prev[gi]);
                let (pb, db) = (self.prev[bi], self.cur[bi] - self.prev[bi]);
                let row = &mut self.u[c * HOP..(c + 1) * HOP];
                for (t, x) in row.iter_mut().enumerate() {
                    let w = self.ramp[t];
                    let y = *x * (1.0 + pg + dg * w) + pb + db * w;
                    *x = if y >= 0.0 { y } else { 0.1 * y };
                }
            }
            self.pws[i].run(&self.u, HOP, &mut self.v);
            for (a, b) in self.h.iter_mut().zip(&self.v) {
                *a += *b;
            }
        }
        self.out.run(&self.h, HOP, out);
        self.prev.copy_from_slice(&self.cur);
    }
}
