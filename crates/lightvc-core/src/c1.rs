//! C1: 因果な内容符号器(training/c1_content.py と同じ・重みは training/export_c1.py)。入力ブロック → 単位(K 個のうち最大)とロジット。
//!
//! 開始状態の規約(train_c1.prime): 受容野ぶん以上の無音を先に流す。`new` の後に `prime()` を 1 回呼ぶ(PRE を HOP の倍数へ切り上げた無音)。先読み 0。

use std::path::Path;

use anyhow::{Context, Result};

use crate::nvoc::{CConv, MelFront, Tensors, HOP, N_MEL};

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

pub struct C1 {
    front: MelFront,
    mel: Vec<f32>,
    inp: CConv,
    blocks: Vec<(CConv, CConv)>,
    out: CConv,
    pre: usize,
    x: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
    logit: Vec<f32>,
    pub k: usize,
}

impl C1 {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let dils: Vec<usize> = man["cfg"]["dils"].as_array().context("cfg.dils")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
        let k = man["cfg"]["k"].as_u64().context("cfg.k")? as usize;
        let pre = man["const"]["PRE"].as_u64().context("const.PRE")? as usize;
        let conv = |name: &str, d: usize| -> Result<CConv> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (b, _) = ts.get(&format!("{name}.b"))?;
            Ok(CConv::from_raw(w, b, sh[0], sh[1], sh[2], d))
        };
        let inp = conv("inp", 1)?;
        let mut blocks = Vec::new();
        for (i, &d) in dils.iter().enumerate() {
            blocks.push((conv(&format!("blocks.{i}.c1"), d)?, conv(&format!("blocks.{i}.c2"), 1)?));
        }
        let out = conv("out", 1)?;
        let front = MelFront::new(ts.get("mel.fb")?.0, ts.get("mel.win")?.0);
        Ok(Self { front, mel: vec![0.0; N_MEL], inp, blocks, out, pre, x: vec![], t1: vec![], t2: vec![], t3: vec![], logit: vec![0.0; k], k })
    }

    /// 開始状態: 無音を ceil(PRE / HOP) ブロック流す。
    pub fn prime(&mut self) {
        let z = vec![0.0f32; HOP];
        for _ in 0..self.pre.div_ceil(HOP) {
            self.process_block(&z);
        }
    }

    /// 入力ブロック [HOP] → 単位(argmax)。ロジットは `logits()`。
    pub fn process_block(&mut self, xin: &[f32]) -> usize {
        self.front.push(xin, &mut self.mel);
        let l1e5 = (1e-5f32).ln();
        for v in self.mel.iter_mut() {
            *v = (*v - l1e5) / 10.0;
        }
        let ch = self.inp.co_len();
        self.x.resize(ch, 0.0);
        self.inp.run(&self.mel, 1, &mut self.x);
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
        self.out.run(&self.t1, 1, &mut self.logit);
        let mut best = 0usize;
        for (i, &v) in self.logit.iter().enumerate() {
            if v > self.logit[best] {
                best = i;
            }
        }
        best
    }

    pub fn logits(&self) -> &[f32] {
        &self.logit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        let raw = std::fs::read(p).unwrap();
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// C1_EXPORT=<export_c1.py の出力> cargo test --release -p lightvc-core c1_parity -- --nocapture
    #[test]
    fn c1_parity() {
        let Ok(dir) = std::env::var("C1_EXPORT") else {
            eprintln!("C1_EXPORT 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let mut m = C1::load(dir).unwrap();
        m.prime();
        let fx = dir.join("fixture");
        let x = read_f32(&fx.join("x.f32"));
        let lg = read_f32(&fx.join("logit.f32"));
        let un = read_f32(&fx.join("unit.f32"));
        let t = x.len() / HOP;
        let k = m.k;
        let (mut ml, mut agree) = (0.0f32, 0usize);
        let t0 = std::time::Instant::now();
        for i in 0..t {
            let u = m.process_block(&x[i * HOP..(i + 1) * HOP]);
            for j in 0..k {
                ml = ml.max((m.logits()[j] - lg[i * k + j]).abs());
            }
            if u == un[i] as usize {
                agree += 1;
            }
        }
        let el = t0.elapsed().as_secs_f64();
        println!("c1 parity: ロジット max|差| {ml:.3e}・単位の一致 {agree}/{t}・RTF {:.4}", el / (t as f64 * HOP as f64 / 48000.0));
        assert!(ml < 1e-2 && agree * 1000 >= t * 995, "c1 parity FAIL");
    }
}
