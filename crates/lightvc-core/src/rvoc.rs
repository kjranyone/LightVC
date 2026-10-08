//! rvoc(出力部・training/rvoc.py と同じ構造)のストリーミング推論。現状は RTF 計測用(乱数の重み)。
//!
//! 条件 [D_COND](フレーム)と励起 [2][HOP](パルス列・雑音)→ 出力 [HOP]。段 = LeakyReLU → 直線補間 ×r(前のサンプルからの補間・因果)
//! → 因果 conv(k = 2r+1)→ + 励起(段のレートへ間引いた 2ch の conv)→ ResBlock(k 3/7/11・dil 1/3/5)の平均。

use crate::nvoc::{CConv, HOP};

pub const UPS: [usize; 4] = [5, 4, 4, 3];
pub const D_COND: usize = 31;

struct Rb {
    c1: Vec<CConv>,
    c2: Vec<CConv>,
}

struct Stage {
    r: usize,
    ci: usize,
    co: usize,
    prev: Vec<f32>,
    up: CConv,
    src: CConv,
    res: Vec<Rb>,
    xi: Vec<f32>,
    x: Vec<f32>,
    xr: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
    acc: Vec<f32>,
    e: Vec<f32>,
    es: Vec<f32>,
}

fn lrelu_inplace(v: &mut [f32], s: f32) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x *= s;
        }
    }
}

impl Stage {
    fn run(&mut self, xin: &[f32], n_in: usize, exc: &[f32], stride: usize) -> usize {
        let (r, ci, co) = (self.r, self.ci, self.co);
        let n = n_in * r;
        self.xi.resize(ci * n, 0.0);
        for c in 0..ci {
            let mut p = self.prev[c];
            for t in 0..n_in {
                let mut v = xin[c * n_in + t];
                if v < 0.0 {
                    v *= 0.1;
                }
                for j in 0..r {
                    let w = (j + 1) as f32 / r as f32;
                    self.xi[c * n + t * r + j] = p + (v - p) * w;
                }
                p = v;
            }
            self.prev[c] = p;
        }
        self.x.resize(co * n, 0.0);
        self.up.run(&self.xi, n, &mut self.x);
        self.e.resize(2 * n, 0.0);
        for k in 0..2 {
            for t in 0..n {
                self.e[k * n + t] = exc[k * HOP + t * stride];
            }
        }
        self.es.resize(co * n, 0.0);
        self.src.run(&self.e, n, &mut self.es);
        for (a, b) in self.x.iter_mut().zip(self.es.iter()) {
            *a += *b;
        }
        let sz = co * n;
        for v in [&mut self.xr, &mut self.t1, &mut self.t2, &mut self.t3] {
            v.resize(sz, 0.0);
        }
        self.acc.clear();
        self.acc.resize(sz, 0.0);
        for rb in self.res.iter_mut() {
            self.xr.copy_from_slice(&self.x);
            for (a, b) in rb.c1.iter_mut().zip(rb.c2.iter_mut()) {
                self.t1.copy_from_slice(&self.xr);
                lrelu_inplace(&mut self.t1, 0.1);
                a.run(&self.t1, n, &mut self.t2);
                lrelu_inplace(&mut self.t2, 0.1);
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

pub struct RVoc {
    pre: CConv,
    stages: Vec<Stage>,
    post: CConv,
    h0: Vec<f32>,
    out_in: Vec<f32>,
    pub ch: usize,
}

impl RVoc {
    pub fn random(ch: usize, seed: u64) -> Self {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
                })
                .collect()
        };
        let mut conv = |ci: usize, co: usize, k: usize, d: usize| {
            let sc = 1.0 / ((ci * k) as f32).sqrt();
            CConv::from_raw(rnd(co * ci * k, sc), rnd(co, 0.01), co, ci, k, d)
        };
        let pre = conv(D_COND, ch, 3, 1);
        let mut stages = Vec::new();
        for (i, &r) in UPS.iter().enumerate() {
            let (ci, co) = (ch >> i, ch >> (i + 1));
            let up = conv(ci, co, 2 * r + 1, 1);
            let src = conv(2, co, 3, 1);
            let res = [3usize, 7, 11]
                .iter()
                .map(|&k| Rb { c1: [1usize, 3, 5].iter().map(|&d| conv(co, co, k, d)).collect(), c2: (0..3).map(|_| conv(co, co, k, 1)).collect() })
                .collect();
            stages.push(Stage { r, ci, co, prev: vec![0.0; ci], up, src, res, xi: vec![], x: vec![], xr: vec![], t1: vec![], t2: vec![], t3: vec![], acc: vec![], e: vec![], es: vec![] });
        }
        let post = conv(ch >> UPS.len(), 1, 7, 1);
        Self { pre, stages, post, h0: vec![0.0; ch], out_in: vec![], ch }
    }

    /// cond [D_COND]・exc [2][HOP] → out [HOP]
    pub fn process_block(&mut self, cond: &[f32], exc: &[f32], out: &mut [f32]) {
        self.pre.run(cond, 1, &mut self.h0);
        let mut cur = self.h0.clone();
        let mut n = 1usize;
        let mut tot = 1usize;
        for st in self.stages.iter_mut() {
            tot *= st.r;
            let stride = HOP / tot;
            n = st.run(&cur, n, exc, stride);
            cur = st.acc.clone();
        }
        self.out_in.resize(cur.len(), 0.0);
        for (a, &b) in self.out_in.iter_mut().zip(cur.iter()) {
            *a = if b < 0.0 { b * 0.01 } else { b };
        }
        self.post.run(&self.out_in, n, out);
    }
}
