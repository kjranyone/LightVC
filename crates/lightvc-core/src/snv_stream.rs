//! paper_snv の製品経路(段 P): 48 kHz・240 サンプルのブロック → 前段 v3(p_front)→ E / C(p_model)→ rvoc3hi(rvoc)→ 240 サンプル。
//! 特徴の変換は training/train_snv_p1.to_y と snv_p1_eval.convert と同じ: Y = [(c0 − C0_SIL)/10, 包絡 1..25 / 10, 周期性 ×有声]・lf = ln(f0 / 200)・C の高さと励起は k 半音上。
//! 登録(操作者・目標の参照)は波形を同じ前段に通した特徴から(発話ごとに新しい前段)。端から端の一致検査の固定の入出力 = training/export_snv_stream.py。
//! アルゴリズム遅延 = ブロック HOP + rvoc の DELAY(前段と E / C の先読み 0)。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::nvoc::{DELAY, HOP, N_MEL};
use crate::p_front::{PFrame, PFront};
use crate::p_model::{PModel, D_A, D_Y};
use crate::rvoc::{Excite, RVoc, D_COND};

const LN2_12: f32 = std::f32::consts::LN_2 / 12.0;

fn c0_sil() -> f32 {
    (1e-5f32).ln() * (N_MEL as f32).sqrt()
}

/// 前段の 1 フレーム → (Y [D_Y]・有声・lf)。
fn to_y(fr: &PFrame, y: &mut [f32]) -> (f32, f32) {
    let v = if fr.f0 > 0.0 { 1.0 } else { 0.0 };
    y[0] = (fr.env[0] - c0_sil()) / 10.0;
    for k in 1..25 {
        y[k] = fr.env[k] / 10.0;
    }
    for k in 0..4 {
        y[25 + k] = fr.per[k].clamp(0.0, 1.5) * v;
    }
    let lf = if v > 0.0 { (fr.f0.max(1.0) / 200.0).ln() } else { 0.0 };
    (v, lf)
}

pub struct SnvStream {
    front_dir: PathBuf,
    front: PFront,
    model: PModel,
    exc: Excite,
    voc: RVoc,
    k: f32,
    y: Vec<f32>,
    a: Vec<f32>,
    yo: Vec<f32>,
    cond: Vec<f32>,
    e: Vec<f32>,
    noise: Option<(Vec<f32>, usize)>,
}

impl SnvStream {
    pub const BLOCK: usize = HOP;
    pub const LATENCY_SAMPLES: usize = HOP + DELAY;

    pub fn load(front: &Path, model: &Path, voc: &Path) -> Result<Self> {
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(voc.join("manifest.json")).context("rvoc の manifest")?)?;
        let c = &man["const"];
        let num = |k: &str| c[k].as_f64().with_context(|| format!("const {k}"));
        let exc = Excite::new(num("H_MAX")? as f32, num("H_ROLL")? as f32, num("K_MAX")? as usize, 1);
        Ok(Self {
            front_dir: front.to_path_buf(), front: PFront::load(front)?, model: PModel::load(model)?, exc, voc: RVoc::load(voc)?, k: 0.0,
            y: vec![0.0; D_Y], a: vec![0.0; D_A], yo: vec![0.0; D_Y], cond: vec![0.0; D_COND], e: vec![0.0; 2 * HOP], noise: None,
        })
    }

    /// E / C を n スレッドに分け(n = 1・2・4)、描画器の最も重い残差ブロックをワーカーへ移す。
    pub fn enable_parallel(&mut self, n: usize) {
        self.model.enable_parallel(n);
        self.voc.enable_parallel();
    }

    /// 固定の半音(運用 = +0 / +5 / +7 / +12)。
    pub fn set_semitones(&mut self, k: f32) {
        self.k = k;
    }

    /// 発話の波形(48 kHz)を新しい前段に通した特徴: (Y [D_Y][t] チャンネル優先・有声 [t]・t)。
    fn analyze(&self, pcm: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let mut fr = PFront::load(&self.front_dir)?;
        let t = pcm.len() / HOP;
        let (mut y, mut v) = (vec![0.0f32; D_Y * t], vec![0.0f32; t]);
        let mut yy = vec![0.0f32; D_Y];
        for tt in 0..t {
            let f = fr.process_block(&pcm[tt * HOP..(tt + 1) * HOP]);
            v[tt] = to_y(&f, &mut yy).0;
            for c in 0..D_Y {
                y[c * t + tt] = yy[c];
            }
        }
        Ok((y, v, t))
    }

    fn concat(parts: &[(Vec<f32>, Vec<f32>, usize)]) -> (Vec<f32>, Vec<f32>, usize) {
        let t: usize = parts.iter().map(|p| p.2).sum();
        let (mut y, mut v) = (vec![0.0f32; D_Y * t], Vec::with_capacity(t));
        let mut o = 0;
        for (py, pv, pt) in parts {
            for c in 0..D_Y {
                y[c * t + o..c * t + o + pt].copy_from_slice(&py[c * pt..(c + 1) * pt]);
            }
            v.extend_from_slice(pv);
            o += pt;
        }
        (y, v, t)
    }

    /// 操作者の登録(発話の波形の組・48 kHz)。
    pub fn enroll_operator(&mut self, utts: &[&[f32]]) -> Result<()> {
        let parts = utts.iter().map(|u| self.analyze(u)).collect::<Result<Vec<_>>>()?;
        let (y, v, t) = Self::concat(&parts);
        self.model.enroll_operator(&y, &v, t);
        Ok(())
    }

    /// 目標の参照(発話の波形の組・48 kHz)。
    pub fn enroll_target(&mut self, utts: &[&[f32]]) -> Result<()> {
        let parts = utts.iter().map(|u| self.analyze(u)).collect::<Result<Vec<_>>>()?;
        let (y, v, t) = Self::concat(&parts);
        self.model.enroll_target(&y, &v, t);
        Ok(())
    }

    /// 励起の雑音を外から与える(一致検査だけ・[n] を順に使う)。
    pub fn set_noise_fixture(&mut self, v: Vec<f32>) {
        self.noise = Some((v, 0));
    }

    /// 入力 [HOP] → 出力 [HOP](未来の入力を見ない)。出力 RVoc の時間 = 入力 − DELAY。
    pub fn process_block(&mut self, x: &[f32], out: &mut [f32]) {
        let fr = self.front.process_block(x);
        let (v, lf) = to_y(&fr, &mut self.y);
        let lfk = lf + self.k * LN2_12 * v;
        self.model.step(&self.y, v, lfk, &mut self.a, &mut self.yo);
        self.cond[..25].copy_from_slice(&self.yo[..25]);
        self.cond[25] = lfk;
        self.cond[26] = v;
        for k in 0..4 {
            self.cond[27 + k] = self.yo[25 + k].clamp(0.0, 1.5) * v;
        }
        let f0k = if fr.f0 > 0.0 { fr.f0 * (self.k / 12.0).exp2() } else { 0.0 };
        self.exc.block(f0k, &mut self.e);
        if let Some((nz, i)) = self.noise.as_mut() {
            self.e[HOP..].copy_from_slice(&nz[*i..*i + HOP]);
            *i += HOP;
        }
        self.voc.process_block(&self.cond, &self.e, out);
    }

    /// 直前のブロックの C の出力 Y [D_Y]。
    pub fn last_y(&self) -> &[f32] {
        &self.yo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rd(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn open() -> Option<(PathBuf, serde_json::Value)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fx = root.join("results/snv_stream_export/fixture");
        if !fx.join("shapes.json").exists() {
            eprintln!("skip: {} が無い(training/export_snv_stream.py)", fx.display());
            return None;
        }
        Some((root, serde_json::from_str(&std::fs::read_to_string(fx.join("shapes.json")).unwrap()).unwrap()))
    }

    fn build(root: &Path, sh: &serde_json::Value) -> SnvStream {
        let p = |k: &str| root.join(sh[k].as_str().unwrap());
        let fx = root.join("results/snv_stream_export/fixture");
        let mut s = SnvStream::load(&p("front"), &p("model"), &p("voc")).unwrap();
        s.set_semitones(sh["k"].as_f64().unwrap() as f32);
        s.enroll_operator(&[&rd(&fx.join("op_x.f32"))]).unwrap();
        let refs: Vec<Vec<f32>> = (0..sh["refs"].as_u64().unwrap()).map(|j| rd(&fx.join(format!("ref{j}_x.f32")))).collect();
        s.enroll_target(&refs.iter().map(|r| r.as_slice()).collect::<Vec<_>>()).unwrap();
        s
    }

    fn run(s: &mut SnvStream, x: &[f32], ys: Option<&mut Vec<f32>>) -> (Vec<f32>, Vec<f64>) {
        let t = x.len() / HOP;
        let mut out = vec![0.0f32; t * HOP];
        let mut ms = Vec::with_capacity(t);
        let mut yv = Vec::new();
        for i in 0..t {
            let t0 = std::time::Instant::now();
            s.process_block(&x[i * HOP..(i + 1) * HOP], &mut out[i * HOP..(i + 1) * HOP]);
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            yv.extend_from_slice(s.last_y());
        }
        if let Some(y) = ys {
            *y = yv;
        }
        (out, ms)
    }

    /// 端から端の一致(Python の snv_p0_front → E / C → rvoc3hi と同じ波形)。
    #[test]
    fn parity_end_to_end() {
        let Some((root, sh)) = open() else { return };
        let fx = root.join("results/snv_stream_export/fixture");
        let mut s = build(&root, &sh);
        s.set_noise_fixture(rd(&fx.join("noise.f32")));
        let x = rd(&fx.join("src_x.f32"));
        let mut ys = Vec::new();
        let (out, _) = run(&mut s, &x, Some(&mut ys));
        let (w, yo) = (rd(&fx.join("w.f32")), rd(&fx.join("yo.f32")));
        let ry = ys.iter().zip(&yo).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / yo.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let md = out.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rms = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
        eprintln!("端から端: C の出力 Y の相対の最大差 {ry:.2e}・波形 max|Rust − Python| {md:.3e}(RMS {rms:.3e})・{} フレーム", x.len() / HOP);
        assert!(ry < 2e-3, "Y の差 {ry}");
        assert!(md < 1e-3 * rms + 1e-5, "波形の差 {md}");
    }

    /// 出荷の関門: 未来不変性(実音声・入力の n 以降を別の実音声に差し替えて n より前の出力が 1 bit も動かない・n 以降が動く)・静的な遅延の台帳・RTF。
    #[test]
    fn ship_gate() {
        let Some((root, sh)) = open() else { return };
        let fx = root.join("results/snv_stream_export/fixture");
        let (x, alt) = (rd(&fx.join("src_x.f32")), rd(&fx.join("alt_x.f32")));
        let t = (x.len() / HOP).min(alt.len() / HOP);
        let mut worst = 0usize;
        for &cut in &[t / 4, t / 2, 3 * t / 4] {
            let n = cut * HOP;
            let mut xe = x[..t * HOP].to_vec();
            xe[n..].copy_from_slice(&alt[n..t * HOP]);
            let (o1, _) = run(&mut build(&root, &sh), &x[..t * HOP], None);
            let (o2, _) = run(&mut build(&root, &sh), &xe, None);
            let first = o1.iter().zip(&o2).position(|(a, b)| a.to_bits() != b.to_bits());
            let Some(first) = first else { panic!("INCONCLUSIVE: 差し替えで出力が動かない(n = {n})") };
            let la = n.saturating_sub(first);
            worst = worst.max(la);
            eprintln!("未来不変性: 差し替え {n} サンプル目 → 最初に動く出力 {first}(先読み {la} サンプル)");
        }
        assert_eq!(worst, 0, "先読み {worst} サンプル");
        let lat_ms = SnvStream::LATENCY_SAMPLES as f64 / 48.0;
        let ledger = lat_ms + 10.0;
        eprintln!("静的な遅延の台帳: ブロック {HOP} + rvoc の DELAY {DELAY} = {lat_ms:.1} ms + 内容の符号器の予備 10 ms = {ledger:.1} ms(< 30 ms)");
        assert!(ledger < 30.0);
        for par in [0usize, 2, 4] {
            let mut s = build(&root, &sh);
            if par > 0 {
                s.enable_parallel(par);
            }
            let (_, mut ms) = run(&mut s, &x, None);
            let tot: f64 = ms.iter().sum();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| ms[((ms.len() - 1) as f64 * p) as usize];
            eprintln!("RTF(並列 {par}・0 = 逐次): {:.3}・ブロックあたり ms p50 {:.2} p95 {:.2} p99 {:.2} 最大 {:.2}(予算 5 ms)", tot / (ms.len() as f64 * 5.0), q(0.5), q(0.95), q(0.99), ms[ms.len() - 1]);
        }
    }
}
