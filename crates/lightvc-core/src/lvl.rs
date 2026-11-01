//! F2: 因果な c0 推定器(training/lvl.py と同じ構造・重みは training/export_lvl.py)。製品の処方 = 出力の因果 3 フレーム平均(train_c3.Chain.c0_avg)。
//!
//! 入力ブロック t(48kHz・HOP サンプル)→ フレーム t の c0n((c0 − C0_SIL)/10)。対数メルは nvoc の因果メル((ln − ln 1e-5)/10 に正規化)。先読み 0。

use std::path::Path;

use anyhow::{Context, Result};

use crate::nvoc::{CConv, MelFront, Tensors, HOP, N_MEL};

pub struct Lvl {
    front: MelFront,
    inp: CConv,
    blks: Vec<(CConv, CConv)>,
    out: CConv,
    mel: Vec<f32>,
    x: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
    o: Vec<f32>,
    avg: Vec<f32>,
    c0_avg: usize,
}

fn lrelu(dst: &mut [f32], src: &[f32]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = if v >= 0.0 { v } else { v * 0.1 };
    }
}

impl Lvl {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let cfg = &man["cfg"];
        let dils: Vec<usize> = cfg["dils"].as_array().context("cfg.dils")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
        let c0_avg = man["const"]["C0_AVG"].as_u64().unwrap_or(1) as usize;
        let conv = |name: &str, d: usize| -> Result<CConv> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (b, _) = ts.get(&format!("{name}.b"))?;
            Ok(CConv::from_raw(w, b, sh[0], sh[1], sh[2], d))
        };
        let inp = conv("inp", 1)?;
        let mut blks = Vec::new();
        for (i, &d) in dils.iter().enumerate() {
            blks.push((conv(&format!("blks.{i}.c1"), d)?, conv(&format!("blks.{i}.c2"), 1)?));
        }
        let out = conv("out", 1)?;
        let front = MelFront::new(ts.get("mel.fb")?.0, ts.get("mel.win")?.0);
        Ok(Self { front, inp, blks, out, mel: vec![0.0; N_MEL], x: vec![], t1: vec![], t2: vec![], t3: vec![], o: vec![0.0], avg: vec![], c0_avg })
    }

    /// 入力ブロック [HOP] → (F2 の生の c0n, 製品の処方の c0n = 因果 c0_avg 平均)
    pub fn process_block(&mut self, x: &[f32]) -> (f32, f32) {
        assert_eq!(x.len(), HOP);
        self.front.push(x, &mut self.mel);
        let l1e5 = (1e-5f32).ln();
        let mut lev = 0.0f32;
        for v in self.mel.iter_mut() {
            *v = (*v - l1e5) / 10.0;
            lev += *v;
        }
        lev = 11.3137085 * lev / N_MEL as f32;
        let ch = self.inp.co_len();
        self.x.resize(ch, 0.0);
        self.inp.run(&self.mel, 1, &mut self.x);
        for v in [&mut self.t1, &mut self.t2, &mut self.t3] {
            v.resize(ch, 0.0);
        }
        for (c1, c2) in self.blks.iter_mut() {
            lrelu(&mut self.t1, &self.x);
            c1.run(&self.t1, 1, &mut self.t2);
            lrelu(&mut self.t3, &self.t2);
            c2.run(&self.t3, 1, &mut self.t1);
            for (a, &b) in self.x.iter_mut().zip(self.t1.iter()) {
                *a += b;
            }
        }
        lrelu(&mut self.t1, &self.x);
        self.out.run(&self.t1, 1, &mut self.o);
        let c0n = lev - 2.7 + self.o[0];
        if self.avg.is_empty() {
            self.avg = vec![c0n; self.c0_avg.max(1)];
        }
        self.avg.rotate_left(1);
        let k = self.avg.len();
        self.avg[k - 1] = c0n;
        let a = self.avg.iter().sum::<f32>() / k as f32;
        (c0n, a)
    }
}

/// F3: 因果な包絡推定器(training/lvl.EnvEst・重みは training/export_env.py)。cfg.out = 4 のときは F4(lvl.PerEst・因果な周期性推定器・出力 = 周期性 4 帯・lev を足さない)。入力ブロック → [c0n, c1/10..c24/10](出力部の条件の行 0..24 と同じ単位)。
/// 『目標らしさ ↔ 聞き取りやすさ』のつまみ(表の引きに元の声の包絡を混ぜる)の元の声の包絡を先読み 0 で与える(D28: CheapTrick と CER の差 ≤ 0.014)。
pub struct EnvEst {
    front: MelFront,
    inp: CConv,
    blks: Vec<(CConv, CConv)>,
    out: CConv,
    mel: Vec<f32>,
    x: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
    o: Vec<f32>,
}

impl EnvEst {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let dils: Vec<usize> = man["cfg"]["dils"].as_array().context("cfg.dils")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
        let conv = |name: &str, d: usize| -> Result<CConv> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (b, _) = ts.get(&format!("{name}.b"))?;
            Ok(CConv::from_raw(w, b, sh[0], sh[1], sh[2], d))
        };
        let inp = conv("inp", 1)?;
        let mut blks = Vec::new();
        for (i, &d) in dils.iter().enumerate() {
            blks.push((conv(&format!("blks.{i}.c1"), d)?, conv(&format!("blks.{i}.c2"), 1)?));
        }
        let out = conv("out", 1)?;
        let n_out = man["cfg"]["out"].as_u64().unwrap_or(25) as usize;
        anyhow::ensure!(out.co_len() == n_out && (n_out == 25 || n_out == 4), "EnvEst の出力は 25(F3)か 4(F4 の周期性)");
        let front = MelFront::new(ts.get("mel.fb")?.0, ts.get("mel.win")?.0);
        Ok(Self { front, inp, blks, out, mel: vec![0.0; N_MEL], x: vec![], t1: vec![], t2: vec![], t3: vec![], o: vec![0.0; n_out] })
    }

    /// 入力ブロック [HOP] → [25](c0n・c1/10..c24/10)
    pub fn process_block(&mut self, xin: &[f32]) -> &[f32] {
        assert_eq!(xin.len(), HOP);
        self.front.push(xin, &mut self.mel);
        let l1e5 = (1e-5f32).ln();
        let mut lev = 0.0f32;
        for v in self.mel.iter_mut() {
            *v = (*v - l1e5) / 10.0;
            lev += *v;
        }
        lev = 11.3137085 * lev / N_MEL as f32;
        let ch = self.inp.co_len();
        self.x.resize(ch, 0.0);
        self.inp.run(&self.mel, 1, &mut self.x);
        for v in [&mut self.t1, &mut self.t2, &mut self.t3] {
            v.resize(ch, 0.0);
        }
        for (c1, c2) in self.blks.iter_mut() {
            lrelu(&mut self.t1, &self.x);
            c1.run(&self.t1, 1, &mut self.t2);
            lrelu(&mut self.t3, &self.t2);
            c2.run(&self.t3, 1, &mut self.t1);
            for (a, &b) in self.x.iter_mut().zip(self.t1.iter()) {
                *a += b;
            }
        }
        lrelu(&mut self.t1, &self.x);
        self.out.run(&self.t1, 1, &mut self.o);
        if self.o.len() == 25 {
            self.o[0] += lev - 2.7;
        }
        &self.o
    }

    pub fn n_out(&self) -> usize {
        self.o.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        let raw = std::fs::read(p).unwrap();
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// LVL_EXPORT=<export_lvl.py の出力> cargo test --release -p lightvc-core lvl_parity -- --nocapture
    #[test]
    fn lvl_parity() {
        let Ok(dir) = std::env::var("LVL_EXPORT") else {
            eprintln!("LVL_EXPORT 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let mut m = Lvl::load(dir).unwrap();
        let fx = dir.join("fixture");
        let x = read_f32(&fx.join("x.f32"));
        let c0 = read_f32(&fx.join("c0n.f32"));
        let ca = read_f32(&fx.join("c0n_avg3.f32"));
        let t = x.len() / HOP;
        let (mut m1, mut m2) = (0.0f32, 0.0f32);
        let t0 = std::time::Instant::now();
        for i in 0..t {
            let (a, b) = m.process_block(&x[i * HOP..(i + 1) * HOP]);
            m1 = m1.max((a - c0[i]).abs());
            m2 = m2.max((b - ca[i]).abs());
        }
        let el = t0.elapsed().as_secs_f64();
        println!("lvl parity: c0n max|差| {m1:.3e}・3 平均 max|差| {m2:.3e}・RTF {:.4}", el / (t as f64 * HOP as f64 / 48000.0));
        assert!(m1 < 2e-3 && m2 < 2e-3, "lvl parity FAIL {m1} {m2}");
    }

    /// ENV_EXPORT=<export_env.py の出力> cargo test --release -p lightvc-core env_parity -- --nocapture
    #[test]
    fn env_parity() {
        let Ok(dir) = std::env::var("ENV_EXPORT") else {
            eprintln!("ENV_EXPORT 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let mut m = EnvEst::load(dir).unwrap();
        let fx = dir.join("fixture");
        let x = read_f32(&fx.join("x.f32"));
        let per = fx.join("per.f32").exists();
        let e = read_f32(&fx.join(if per { "per.f32" } else { "env.f32" }));
        let n = if per { 4 } else { 25 };
        let t = x.len() / HOP;
        let mut md = 0.0f32;
        let t0 = std::time::Instant::now();
        for i in 0..t {
            let o = m.process_block(&x[i * HOP..(i + 1) * HOP]);
            assert_eq!(o.len(), n);
            for k in 0..n {
                md = md.max((o[k] - e[i * n + k]).abs());
            }
        }
        let el = t0.elapsed().as_secs_f64();
        println!("env parity: max|差| {md:.3e}・RTF {:.4}", el / (t as f64 * HOP as f64 / 48000.0));
        assert!(md < 2e-3, "env parity FAIL {md}");
    }
}
