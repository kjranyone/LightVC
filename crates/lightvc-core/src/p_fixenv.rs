//! 段 P91 の固定の窓の包絡(training/snv_p0_front.fixed_env)のストリーミング版。48 kHz・H = 240・フレーム t の窓は (t+1)H で終わる(先読み 0・流しの前は 0)。
//! 左寄せの対称のハン窓(40 ms = 1920 標本)→ |FFT 2048| → 前段と同じ mel 128 → log(床 1e-5)→ DCT 25。発話の統計なし。
//! 研究の候補 K3G(k3g_model)の、固定の窓の包絡を読む組の符号の入力だけに使う。窓の定数は模型の manifest の cfg.fixed_env・mel.fb と dct は同じ書き出しから。

use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::nvoc::{Tensors, HOP};
use crate::fft::fft;

pub const D_ENV: usize = 25;

#[derive(Clone)]
pub struct FixEnv {
    win: Vec<f64>,
    raw: Vec<f64>,
    fb: Vec<f32>,
    fbr: Vec<(usize, usize)>,
    dct: Vec<f32>,
    nm: usize,
    nfft: usize,
    floor: f64,
    re: Vec<f64>,
    im: Vec<f64>,
    mel: Vec<f64>,
}

impl FixEnv {
    /// 模型の書き出し(cfg.fixed_env・mel.fb・dct)から。
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir).context("固定の窓の包絡の定数")?;
        Self::new(&ts, &man["cfg"]["fixed_env"])
    }

    pub(crate) fn new(ts: &Tensors, cfg: &serde_json::Value) -> Result<Self> {
        let num = |k: &str| cfg[k].as_f64().with_context(|| format!("cfg.fixed_env {k}"));
        let (w, nfft) = (num("win")? as usize, num("nfft")? as usize);
        ensure!(num("hop")? as usize == HOP && cfg["window"].as_str() == Some("hann_sym") && nfft.is_power_of_two() && (HOP..=nfft).contains(&w), "固定の窓の包絡の cfg");
        let nb = nfft / 2 + 1;
        let (fb, sf) = ts.get("mel.fb")?;
        let (dct, sd) = ts.get("dct")?;
        ensure!(sf.len() == 2 && sf[1] == nb && sd == [D_ENV, sf[0]], "mel.fb {sf:?}・dct {sd:?}");
        let nm = sf[0];
        let fbr = (0..nm)
            .map(|m| {
                let row = &fb[m * nb..(m + 1) * nb];
                let lo = row.iter().position(|&v| v != 0.0).unwrap_or(0);
                (lo, row.iter().rposition(|&v| v != 0.0).map_or(lo, |h| h + 1))
            })
            .collect();
        let win = (0..w).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (w - 1) as f64).cos()).collect();
        Ok(Self { win, raw: vec![0.0; w], fb, fbr, dct, nm, nfft, floor: num("floor")?, re: vec![0.0; nfft], im: vec![0.0; nfft], mel: vec![0.0; nm] })
    }

    /// 同じ定数の、流しの初めの状態の写し(登録の発話ごとに使う)。
    pub fn fresh(&self) -> Self {
        let mut f = self.clone();
        f.raw.iter_mut().for_each(|v| *v = 0.0);
        f
    }

    /// 入力の 1 ブロック(HOP 標本)→ このフレームの包絡 [D_ENV](未来の入力を見ない)。
    pub fn step(&mut self, x: &[f32]) -> [f32; D_ENV] {
        assert_eq!(x.len(), HOP, "ブロックは HOP サンプル");
        let w = self.raw.len();
        self.raw.copy_within(HOP.., 0);
        for (r, &v) in self.raw[w - HOP..].iter_mut().zip(x) {
            *r = v as f64;
        }
        for i in 0..self.nfft {
            self.re[i] = if i < w { self.raw[i] * self.win[i] } else { 0.0 };
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        let nb = self.nfft / 2 + 1;
        for k in 0..nb {
            self.re[k] = (self.re[k] * self.re[k] + self.im[k] * self.im[k]).sqrt();
        }
        for (m, mv) in self.mel.iter_mut().enumerate() {
            let (lo, hi) = self.fbr[m];
            let s: f64 = self.fb[m * nb + lo..m * nb + hi].iter().zip(&self.re[lo..hi]).map(|(&f, &a)| f as f64 * a).sum();
            *mv = s.max(self.floor).ln();
        }
        let mut c = [0.0f32; D_ENV];
        for (k, ck) in c.iter_mut().enumerate() {
            *ck = self.dct[k * self.nm..(k + 1) * self.nm].iter().zip(&self.mel).map(|(&d, &v)| d as f64 * v).sum::<f64>() as f32;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rd(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// snv_p0_front.fixed_env との一致(源 = 音楽を足した操作者の発話)と、フレームの枠(フレーム t は (t+1)H 以降の標本で動かない・(t+1)H − 10 の標本で動く。窓の端の重みは 0 なので端の少し内側で見る)。
    #[test]
    fn parity_with_python_fixed_env() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/snv_k3g_export");
        if !dir.join("fixture/fenv.f32").exists() {
            eprintln!("skip: {} が無い(training/export_snv_k3g.py)", dir.display());
            return;
        }
        let (x, e) = (rd(&dir.join("fixture/x.f32")), rd(&dir.join("fixture/fenv.f32")));
        let proto = FixEnv::load(&dir).unwrap();
        let t_n = x.len() / HOP;
        assert_eq!(e.len(), t_n * D_ENV);
        let mut f = proto.fresh();
        let tm = std::time::Instant::now();
        let mut out = vec![0.0f32; t_n * D_ENV];
        for (t, blk) in x.chunks_exact(HOP).enumerate() {
            out[t * D_ENV..(t + 1) * D_ENV].copy_from_slice(&f.step(blk));
        }
        let ms = tm.elapsed().as_secs_f64() * 1e3 / t_n as f64;
        let md = out.iter().zip(&e).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mr = e.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let cut = t_n / 2;
        let run = |x: &[f32]| {
            let mut f = proto.fresh();
            x.chunks_exact(HOP).flat_map(|b| f.step(b)).collect::<Vec<f32>>()
        };
        let (mut x1, mut x2) = (x.clone(), x.clone());
        x1[cut * HOP..].iter_mut().for_each(|v| *v = -*v);
        x2[cut * HOP - 10] += 0.25;
        let (o1, o2) = (run(&x1), run(&x2));
        let first = |o: &[f32]| o.iter().zip(&out).position(|(a, b)| a.to_bits() != b.to_bits()).map(|i| i / D_ENV);
        eprintln!("固定の窓の包絡: {t_n} フレーム・max|Rust − Python| {md:.3e}(値の最大 {mr:.1})・フレームあたり {ms:.4} ms・標本 {} 以降の差し替えで最初に動くフレーム {:?}・標本 {} の差し替えで {:?}", cut * HOP, first(&o1), cut * HOP - 10, first(&o2));
        assert!(md < 1e-3, "包絡の差 {md}");
        assert_eq!(first(&o1), Some(cut), "未来の標本で前のフレームが動いた");
        assert_eq!(first(&o2), Some(cut - 1), "フレームの枠が (t+1)H で終わっていない");
    }
}
