//! paper_snv の研究の候補 K3G のストリーミング推論(SnvStream は manifest の cfg.kind = "k3g" で選ぶ・製品の既定は変えない別の型)。
//! K3G = snv_p30_c と同じ型の組(p30_model・符号 → 演奏 → 回路 C)を N 個・回路の出力 Y(29)を平均する(training/snv_p1_eval.Ens)。組ごとに操作者の登録の統計と目標の構造を持つ。
//! envc の印の組の符号だけ固定の窓の包絡(p_fixenv)の特徴 yc を読む(源・操作者の登録・目標の構造の鍵)。C の構造の z と値・錨は元の包絡 y。
//! 門が開いたフレームは、平均の代わりに源の包絡の一律の伸縮(錨・training/snv_p0c_anchor.warp_env・α は登録の定数 = 操作者と目標の有声の平均の包絡から fit_alpha)を出す(System.fallback の anch)。
//! 錨は全ての組で同じなので平均の後に 1 回だけ差し替える(Ens との差は f32 の丸めの 4.8e-7 = export_snv_k3g.py が確かめる)。重みと一致検査 = training/export_snv_k3g.py。

use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::affinity::Team;
use crate::nvoc::Tensors;
use crate::p30_model::P30Model;
use crate::p_fixenv::{FixEnv, D_ENV};
use crate::p_model::{D_A, D_Y};
use crate::prof::prof;

/// 錨: 源の包絡(DCT 25)を mel の帯の対数の振幅に戻し、周波数の軸を α 倍にして DCT に戻す線形の写し。
struct Anchor {
    dct: Vec<f64>,
    nm: usize,
    mel_hz: Vec<f64>,
    alphas: Vec<f64>,
    fit_hz: f64,
    c0_sil: f32,
    mu_o: Option<Vec<f64>>,
    mu_t: Option<Vec<f64>>,
    alpha: f64,
    m: Vec<f64>,
}

impl Anchor {
    fn new(ts: &Tensors, cfg: &serde_json::Value) -> Result<Self> {
        let arr = |k: &str| -> Result<Vec<f64>> { cfg[k].as_array().with_context(|| format!("cfg.anchor {k}"))?.iter().map(|v| v.as_f64().context("数")).collect() };
        let (dct, sd) = ts.get("dct")?;
        let (mel_hz, alphas) = (arr("mel_hz")?, arr("alphas")?);
        ensure!(sd.len() == 2 && sd[0] == D_ENV && sd[1] == mel_hz.len() && mel_hz.len() >= 2 && !alphas.is_empty(), "錨の定数(dct {sd:?}・mel_hz {}・alphas {})", mel_hz.len(), alphas.len());
        let c0_sil = cfg["c0_sil"].as_f64().context("cfg.anchor c0_sil")? as f32;
        Ok(Self { dct: dct.iter().map(|&v| v as f64).collect(), nm: sd[1], mel_hz, alphas, fit_hz: cfg["fit_hz"].as_f64().context("cfg.anchor fit_hz")?, c0_sil, mu_o: None, mu_t: None, alpha: 1.0, m: Vec::new() })
    }

    /// 包絡 c [D_ENV] → mel の帯の対数の振幅 [nm](dct の転置)。
    fn bands(&self, c: &[f64]) -> Vec<f64> {
        (0..self.nm).map(|i| (0..D_ENV).map(|k| self.dct[k * self.nm + i] * c[k]).sum()).collect()
    }

    /// snv_p0c_anchor.warp_env: 帯 i の値 = 周波数 mel_hz[i] / α の位置の線形の補間(端は端の値)。
    fn warp(&self, c: &[f64], alpha: f64) -> Vec<f64> {
        let (l, fm, n) = (self.bands(c), &self.mel_hz, self.nm);
        let lw: Vec<f64> = (0..n)
            .map(|i| {
                let x = fm[i] / alpha;
                let idx = if x <= fm[0] {
                    0.0
                } else if x >= fm[n - 1] {
                    (n - 1) as f64
                } else {
                    let j = fm.partition_point(|&f| f <= x) - 1;
                    (x - fm[j]) / (fm[j + 1] - fm[j]) + j as f64
                };
                let i0 = (idx.floor() as usize).min(n - 2);
                let w = idx - i0 as f64;
                l[i0] * (1.0 - w) + l[i0 + 1] * w
            })
            .collect();
        (0..D_ENV).map(|k| self.dct[k * n..(k + 1) * n].iter().zip(&lw).map(|(d, v)| d * v).sum()).collect()
    }

    /// snv_p0c_anchor.fit_alpha: fit_hz 以下の帯で、平均を引いた対数の振幅の二乗の差が最小の α(格子の最初の最小)。
    fn fit(&self, mu_o: &[f64], mu_t: &[f64]) -> f64 {
        let nb = self.mel_hz.iter().filter(|&&f| f <= self.fit_hz).count().max(1);
        let cen = |l: Vec<f64>| -> Vec<f64> {
            let m = l[..nb].iter().sum::<f64>() / nb as f64;
            l[..nb].iter().map(|v| v - m).collect()
        };
        let lt = cen(self.bands(mu_t));
        let (mut best, mut ba) = (f64::INFINITY, 1.0);
        for &a in &self.alphas {
            let wv: Vec<f64> = self.warp(mu_o, a).iter().map(|&v| v as f32 as f64).collect();
            let d = cen(self.bands(&wv)).iter().zip(&lt).map(|(p, q)| (p - q) * (p - q)).sum::<f64>() / nb as f64;
            if d < best {
                (best, ba) = (d, a);
            }
        }
        ba
    }

    /// y [D_Y][t]・有声 [t] の有声の平均(y の尺度・System.structure の hmu)。
    fn vmean(y: &[f32], v: &[f32], t: usize) -> Vec<f64> {
        let n = v.iter().map(|&w| w as f64).sum::<f64>().max(1.0);
        (0..D_ENV).map(|k| y[k * t..(k + 1) * t].iter().zip(v).map(|(&a, &w)| a as f64 * w as f64).sum::<f64>() / n).collect()
    }

    fn refit(&mut self) {
        let (Some(o), Some(t)) = (&self.mu_o, &self.mu_t) else { return };
        self.alpha = self.fit(o, t);
        let mut m = vec![0.0f64; D_ENV * D_ENV];
        let mut e = vec![0.0f64; D_ENV];
        for j in 0..D_ENV {
            e[j] = 1.0;
            for (k, v) in self.warp(&e, self.alpha).into_iter().enumerate() {
                m[k * D_ENV + j] = v;
            }
            e[j] = 0.0;
        }
        self.m = m;
    }

    /// 操作者の登録の有声の平均の包絡(包絡の尺度・snv_p1_eval.mu_of = 有声のフレームの和 / max(1, 数))。
    fn set_op(&mut self, y: &[f32], v: &[f32], t: usize) {
        let c0 = self.c0_sil as f64;
        let n = v.iter().filter(|&&w| w > 0.0).count().max(1) as f64;
        let mu = (0..D_ENV).map(|k| y[k * t..(k + 1) * t].iter().zip(v).filter(|(_, w)| **w > 0.0).map(|(&a, _)| a as f64 * 10.0 + if k == 0 { c0 } else { 0.0 }).sum::<f64>() / n).collect();
        self.mu_o = Some(mu);
        self.refit();
    }

    /// 目標の参照の有声の平均の包絡(y の尺度の平均を包絡の尺度へ = System.anchor の mu_t)。
    fn set_tgt(&mut self, y: &[f32], v: &[f32], t: usize) {
        let c0 = self.c0_sil as f64;
        self.mu_t = Some(Self::vmean(y, v, t).iter().enumerate().map(|(k, &m)| m * 10.0 + if k == 0 { c0 } else { 0.0 }).collect());
        self.refit();
    }

    /// 源の y [D_Y] → 錨 [D_Y](包絡 = 伸縮・周期性 = 源のまま)。
    fn apply(&self, y: &[f32], out: &mut [f32]) {
        assert!(!self.m.is_empty(), "enroll_operator と enroll_target が先");
        let mut e = [0.0f64; D_ENV];
        e[0] = (y[0] * 10.0 + self.c0_sil) as f64;
        for k in 1..D_ENV {
            e[k] = (y[k] * 10.0) as f64;
        }
        for (k, o) in out[..D_ENV].iter_mut().enumerate() {
            let w = self.m[k * D_ENV..(k + 1) * D_ENV].iter().zip(&e).map(|(a, b)| a * b).sum::<f64>() as f32;
            *o = if k == 0 { (w - self.c0_sil) / 10.0 } else { w / 10.0 };
        }
        out[D_ENV..D_Y].copy_from_slice(&y[D_ENV..D_Y]);
    }
}

struct Member {
    m: P30Model,
    envc: bool,
    a: Vec<f32>,
    yo: Vec<f32>,
}

/// 組の並列の手順が触る生のポインタ(rank ごとに互いに素な組・Team::run の前後で同期)。
struct Ptr(*mut Member);

unsafe impl Sync for Ptr {}

impl Ptr {
    fn at(&self, i: usize) -> *mut Member {
        self.0.wrapping_add(i)
    }
}

struct C0Bg {
    lam: f32,
    thr: f64,
    thr_lo: f64,
    thr_hi: f64,
    soft: bool,
    fix20: FixEnv,
    m0: f32,
    s0: f32,
    mt: f32,
    st: f32,
    since_voiced: usize,
}

impl C0Bg {
    fn new(ts: &Tensors, cfg: &serde_json::Value) -> Result<Self> {
        let lam = cfg["lam"].as_f64().context("cfg.c0bg.lam")? as f32;
        let thr = cfg["thr"].as_f64().context("cfg.c0bg.thr")?;
        let thr_lo = cfg.get("thr_lo").and_then(|v| v.as_f64()).unwrap_or(thr);
        let thr_hi = cfg.get("thr_hi").and_then(|v| v.as_f64()).unwrap_or(thr);
        let soft = cfg.get("soft").and_then(|v| v.as_bool()).unwrap_or(false);
        let fix20 = FixEnv::new(ts, &cfg["fixed_env"])?;
        Ok(Self { lam, thr, thr_lo, thr_hi, soft, fix20, m0: 0.0, s0: 1.0, mt: 0.0, st: 1.0, since_voiced: 999 })
    }

    fn set_op(&mut self, c0_20: &[f32], v: &[f32]) {
        let vals: Vec<f64> = c0_20.iter().zip(v).filter(|(_, w)| **w > 0.5).map(|(&x, _)| x as f64).collect();
        if vals.is_empty() {
            (self.m0, self.s0) = (0.0, 1.0);
        } else {
            let n = vals.len() as f64;
            let m = vals.iter().sum::<f64>() / n;
            let s = (vals.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n).sqrt().max(1e-3);
            (self.m0, self.s0) = (m as f32, s as f32);
        }
        self.since_voiced = 999;
    }

    fn set_tgt(&mut self, y: &[f32], v: &[f32], t: usize) {
        let den = v.iter().map(|&w| w as f64).sum::<f64>().max(1.0);
        let y0 = &y[..t];
        let m = y0.iter().zip(v).map(|(&x, &w)| x as f64 * w as f64).sum::<f64>() / den;
        let s = (y0.iter().zip(v).map(|(&x, &w)| ((x as f64 - m).powi(2)) * w as f64).sum::<f64>() / den).sqrt().max(1e-3);
        (self.mt, self.st) = (m as f32, s as f32);
        self.since_voiced = 999;
    }
}

/// 変換器(K3G): 組ごとの操作者の登録の統計と目標の構造で、フレームごとに Y(29)の平均(門が開いたフレームは錨)を出す。
pub struct K3gModel {
    ms: Vec<Member>,
    anchor: Anchor,
    fix: Option<FixEnv>,
    c0bg: Option<C0Bg>,
    team: Team,
}

impl K3gModel {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir).context("k3g の重み")?;
        let cfg = &man["cfg"];
        let fb = cfg["fb"].as_str().unwrap_or("");
        ensure!(cfg["kind"].as_str() == Some("k3g") && (fb == "anch" || fb == "c0bg+anch"), "{}: cfg.kind が k3g(fb = anch / c0bg+anch)でない", dir.display());
        let num = |c: &serde_json::Value, k: &str| c[k].as_f64().with_context(|| format!("cfg {k}"));
        let (ln_eps, tau) = (num(cfg, "ln_eps")?, num(cfg, "tau")?);
        let mut ms = Vec::new();
        for mc in cfg["members"].as_array().context("cfg.members")? {
            ensure!(mc["lags"] == serde_json::json!([0]), "遅れ [0] の型だけ");
            let pre = format!("{}.", mc["prefix"].as_str().context("members.prefix")?);
            let mut m = P30Model::from_tensors(&ts, &pre, ln_eps, num(mc, "cb_s2")? as f32, tau, num(mc, "shrink")?)?;
            if cfg["half"].as_bool().unwrap_or(false) {
                m.set_half();
            }
            if cfg["kv16"].as_bool().unwrap_or(false) {
                m.set_kv16();
            }
            ms.push(Member { m, envc: mc["envc"].as_bool().context("members.envc")?, a: vec![0.0; D_A], yo: vec![0.0; D_Y] });
        }
        ensure!(!ms.is_empty(), "組が無い");
        let fix = if ms.iter().any(|m| m.envc) { Some(FixEnv::new(&ts, &cfg["fixed_env"])?) } else { None };
        let c0bg = if fb == "c0bg+anch" { Some(C0Bg::new(&ts, &cfg["c0bg"])?) } else { None };
        Ok(Self { ms, anchor: Anchor::new(&ts, &cfg["anchor"])?, fix, c0bg, team: Team::new(1) })
    }

    /// 組の数。
    pub fn members(&self) -> usize {
        self.ms.len()
    }

    /// 固定の窓の包絡の流し(その組があるときだけ・流しの初めの状態の写し)。
    pub fn fixenv(&self) -> Option<FixEnv> {
        self.fix.as_ref().map(FixEnv::fresh)
    }

    pub fn fix20(&self) -> Option<FixEnv> {
        self.c0bg.as_ref().map(|c| c.fix20.fresh())
    }

    /// 錨の伸縮 α(登録の定数・両方の登録の後)。
    pub fn alpha(&self) -> f64 {
        self.anchor.alpha
    }

    /// 組を min(n, 組の数) 個のスレッドに分ける(rank r = 組 r, r + n, …・組の中は 1 スレッド・出力は 1 スレッドと 1 bit 同じ)。
    pub fn enable_parallel(&mut self, n: usize) {
        self.team = Team::new(n.clamp(1, self.ms.len()));
    }

    fn code<'a>(envc: bool, y: &'a [f32], yc: Option<&'a [f32]>) -> &'a [f32] {
        if envc { yc.expect("固定の窓の包絡の組には yc が要る") } else { y }
    }

    /// 操作者の登録: y [D_Y][t]・固定の窓の包絡つきの yc [D_Y][t]・有声 [t](チャンネル優先)→ 組ごとの統計と錨の平均。
    pub fn enroll_operator(&mut self, y: &[f32], yc: Option<&[f32]>, c0_20: Option<&[f32]>, v: &[f32], t: usize) {
        for mb in self.ms.iter_mut() {
            mb.m.enroll_operator(Self::code(mb.envc, y, yc), v, t);
        }
        self.anchor.set_op(y, v, t);
        if let Some(cb) = self.c0bg.as_mut() {
            cb.set_op(c0_20.unwrap_or(&y[..t]), v);
        }
    }

    /// 目標の参照 → 組ごとの (z, 参照の演奏 a_ref [D_A][t])。鍵 = 符号の特徴(envc の組は yc)・z と値 = y。
    pub fn enroll_target(&mut self, y: &[f32], yc: Option<&[f32]>, v: &[f32], t: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let out = self.ms.iter_mut().map(|mb| mb.m.enroll_target_keys(y, Self::code(mb.envc, y, yc), v, t)).collect();
        self.anchor.set_tgt(y, v, t);
        if let Some(cb) = self.c0bg.as_mut() {
            cb.set_tgt(y, v, t);
        }
        out
    }

    /// 1 フレーム: 操作者の Y・yc・有声・C に渡す高さ lf・門が開いているか → 出力 Y [D_Y](a_out = 最初の組の演奏)。
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, y: &[f32], yc: Option<&[f32]>, c0_20: Option<f32>, v: f32, lf: f32, open: bool, gate_p: f64, a_out: &mut [f32], y_out: &mut [f32]) {
        let (n, sz) = (self.ms.len(), self.team.size());
        let p = Ptr(self.ms.as_mut_ptr());
        self.team.run(&|r| {
            let mut i = r;
            while i < n {
                let mb = unsafe { &mut *p.at(i) };
                #[cfg(test)]
                crate::prof::slot(i);
                prof!(member M_ALL, mb.m.step(Self::code(mb.envc, y, yc), v, lf, &mut mb.a, &mut mb.yo));
                i += sz;
            }
        });
        prof!(MIX, self.mix(y, c0_20, v, open, gate_p, a_out, y_out));
    }

    #[allow(clippy::too_many_arguments)]
    fn mix(&mut self, y: &[f32], c0_20: Option<f32>, v: f32, open: bool, gate_p: f64, a_out: &mut [f32], y_out: &mut [f32]) {
        let act_c0 = if let Some(cb) = self.c0bg.as_mut() {
            cb.since_voiced = if v > 0.5 { 0 } else { cb.since_voiced.saturating_add(1) };
            let src_c0 = c0_20.unwrap_or(y[0]);
            let z_src = (src_c0 - cb.m0) / cb.s0;
            if cb.soft {
                let w_gp = (((cb.thr_hi - gate_p) / (cb.thr_hi - cb.thr_lo).max(1e-6)).clamp(0.0, 1.0)) as f32;
                let w_act = if cb.since_voiced <= 6 { 1.0 } else { ((z_src + 1.4) / 0.4).clamp(0.0, 1.0) };
                let w = cb.lam * w_gp * w_act;
                (w > 0.0).then_some((w, cb.mt + cb.st * z_src))
            } else {
                let act = cb.since_voiced <= 6 || z_src > -1.2;
                (gate_p < cb.thr && act).then_some((cb.lam, cb.mt + cb.st * z_src))
            }
        } else {
            None
        };
        a_out.copy_from_slice(&self.ms[0].a);
        if open {
            self.anchor.apply(y, y_out);
            return;
        }
        let inv = self.ms.len() as f32;
        for (k, o) in y_out.iter_mut().enumerate() {
            *o = self.ms.iter().map(|mb| mb.yo[k]).fold(0.0f32, |s, x| s + x) / inv;
        }
        if let Some((lam, d)) = act_c0 {
            y_out[0] = (1.0 - lam) * y_out[0] + lam * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CH: usize = 256;

    fn rd(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn tpose(x: &[f32], t: usize, c: usize) -> Vec<f32> {
        let mut y = vec![0.0; x.len()];
        for tt in 0..t {
            for cc in 0..c {
                y[cc * t + tt] = x[tt * c + cc];
            }
        }
        y
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "長さ");
        let m = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let s = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        m / s
    }

    /// 錨の線形の写し: α = 1 は恒等・伸縮した包絡から α が戻る(snv_p0c_anchor.selftest と同じ線)。
    #[test]
    fn anchor_selftest() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/snv_k3g_export");
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: {} が無い(training/export_snv_k3g.py)", dir.display());
            return;
        }
        let (ts, man) = Tensors::load(&dir).unwrap();
        let an = Anchor::new(&ts, &man["cfg"]["anchor"]).unwrap();
        let mut st = 0x2545f4914f6cdd1du64;
        let c: Vec<f64> = (0..D_ENV)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                (st >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect();
        let d1 = an.warp(&c, 1.0).iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let back = an.fit(&c, &an.warp(&c, 1.2));
        eprintln!("錨: α = 1 の最大差 {d1:.2e}・同じ包絡の α {}・1.2 倍の包絡から戻る α {back}", an.fit(&c, &c));
        assert!(d1 < 1e-4 && an.fit(&c, &c) == 1.0 && (back - 1.2).abs() <= 0.02);
    }

    /// 模型の段の一致(組ごとの操作者の統計・目標の構造・符号の文脈・演奏・回路の出力、錨の平均と α、平均と錨の差し替えの後の出力)。
    #[test]
    fn parity_with_python_model() {
        parity_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/snv_k3g_export"));
    }

    /// 同じ一致を、回路と CPC の重みを f16 で持つ K3G(段 P94・SNV_WF16=1 の書き出し)で。
    #[test]
    fn parity_with_python_model_f16() {
        parity_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/snv_k3g16_export"));
    }

    /// 同じ一致を、鍵と値も f16 で持つ K3G(段 P94 step c)で。
    #[test]
    fn parity_with_python_model_f16kv() {
        parity_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/snv_k3g16kv_export"));
    }

    #[test]
    fn parity_with_python_model_p126() {
        parity_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/snv_p126_k3g16kv_export"));
    }

    fn parity_dir(dir: &Path) {
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: {} が無い(training/export_snv_k3g.py)", dir.display());
            return;
        }
        let fx = dir.join("fixture");
        let sh: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx.join("shapes.json")).unwrap()).unwrap();
        let (t, tro, trt) = (sh["T"].as_u64().unwrap() as usize, sh["Tr_op"].as_u64().unwrap() as usize, sh["Tr_t"].as_u64().unwrap() as usize);
        let f = |n: &str| rd(&fx.join(format!("{n}.f32")));
        let opt_f = |n: &str| fx.join(format!("{n}.f32")).exists().then(|| rd(&fx.join(format!("{n}.f32"))));
        let (sy, syc, sv, slf, sg) = (f("src_y"), f("src_yc"), f("src_v"), f("src_lf_k"), f("src_gate"));
        let (sc20, sgp, oc20) = (opt_f("src_c0_20"), opt_f("src_gate_p"), opt_f("op_c0_20"));
        let (pyo, pyw) = (f("yo"), f("yw"));
        let n_open = sg.iter().filter(|&&g| g > 0.5).count();
        let mut seq: Option<Vec<f32>> = None;
        for par in [1usize, 2, 4] {
            let mut m = K3gModel::load(dir).unwrap();
            let nm = m.members();
            assert_eq!(nm as u64, sh["members"].as_u64().unwrap());
            assert_eq!(m.ms.iter().map(|mb| serde_json::Value::Bool(mb.envc)).collect::<Vec<_>>(), *sh["envc"].as_array().unwrap());
            m.enable_parallel(par);
            m.enroll_operator(&tpose(&f("op_y"), tro, D_Y), Some(&tpose(&f("op_yc"), tro, D_Y)), oc20.as_deref(), &f("op_v"), tro);
            let refs = m.enroll_target(&tpose(&f("ref_y"), trt, D_Y), Some(&tpose(&f("ref_yc"), trt, D_Y)), &f("ref_v"), trt);
            let tof = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            let (r_mo, r_mt) = (rel(&tof(m.anchor.mu_o.as_ref().unwrap()), &f("mu_o")), rel(&tof(m.anchor.mu_t.as_ref().unwrap()), &f("mu_t")));
            eprintln!("スレッド {par}: 錨 操作者の平均 {r_mo:.2e} 目標の平均 {r_mt:.2e}・α Rust {} Python {}", m.alpha(), sh["alpha"]);
            assert!(r_mo < 1e-4 && r_mt < 1e-4 && m.alpha() == sh["alpha"].as_f64().unwrap(), "錨の登録の定数");
            let mut worst = Vec::new();
            for (i, (zc, a_ref)) in refs.iter().enumerate() {
                let g = |n: &str| f(&format!("m{i}_{n}"));
                let (op, mem) = (m.ms[i].m.op.as_ref().unwrap(), m.ms[i].m.mem_c.as_ref().unwrap());
                let rs = [("mu", rel(&op.mu, &g("op_mu"))), ("wm", rel(&op.wm, &g("op_wm"))), ("delta", rel(&op.delta, &g("op_delta"))), ("aref", rel(a_ref, &tpose(&g("a_ref"), trt, D_A))),
                          ("zc", rel(zc, &g("sC_z"))), ("kc", rel(&mem.k, &g("sC_k"))), ("vc", rel(&mem.v, &g("sC_v")))];
                eprintln!("スレッド {par}・組 {i}(envc {}): 操作者の統計 mu {:.2e} 白くする行列 {:.2e} delta {:.2e} / 参照の演奏 {:.2e} / 構造 C z {:.2e} 鍵 {:.2e} 値 {:.2e}", m.ms[i].envc, rs[0].1, rs[1].1, rs[2].1, rs[3].1, rs[4].1, rs[5].1, rs[6].1);
                worst.extend(rs.iter().map(|&(n, r)| (format!("組 {i} {n}"), r)));
            }
            let mut c_all = vec![vec![0.0f32; t * CH]; nm];
            let mut a_all = vec![vec![0.0f32; t * D_A]; nm];
            let mut y_mem = vec![vec![0.0f32; t * D_Y]; nm];
            let mut y_all = vec![0.0f32; t * D_Y];
            let (mut a, mut yo) = (vec![0.0; D_A], vec![0.0; D_Y]);
            let mut ms = Vec::with_capacity(t);
            for tt in 0..t {
                let r = tt * D_Y..(tt + 1) * D_Y;
                let t0 = std::time::Instant::now();
                let c20_t = sc20.as_ref().map(|v| v[tt]);
                let gp_t = sgp.as_ref().map_or(if sg[tt] > 0.5 { 1.0 } else { 0.0 }, |v| v[tt] as f64);
                m.step(&sy[r.clone()], Some(&syc[r.clone()]), c20_t, sv[tt], slf[tt], sg[tt] > 0.5, gp_t, &mut a, &mut yo);
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
                for (i, mb) in m.ms.iter().enumerate() {
                    c_all[i][tt * CH..(tt + 1) * CH].copy_from_slice(&mb.m.ctx);
                    a_all[i][tt * D_A..(tt + 1) * D_A].copy_from_slice(&mb.a);
                    y_mem[i][r.clone()].copy_from_slice(&mb.yo);
                }
                y_all[r].copy_from_slice(&yo);
            }
            for i in 0..nm {
                let g = |n: &str| f(&format!("m{i}_{n}"));
                let rs = [("c", rel(&c_all[i], &g("c"))), ("a", rel(&a_all[i], &g("a"))), ("y", rel(&y_mem[i], &g("yo")))];
                eprintln!("スレッド {par}・組 {i}: 流し 文脈 c {:.2e} 演奏 a {:.2e} 回路の出力 Y {:.2e}(相対の最大差)", rs[0].1, rs[1].1, rs[2].1);
                worst.extend(rs.iter().map(|&(n, r)| (format!("組 {i} {n}"), r)));
            }
            let pick = |x: &[f32], open: bool| -> Vec<f32> { x.chunks_exact(D_Y).zip(&sg).filter(|(_, g)| (**g > 0.5) == open).flat_map(|(c, _)| c.iter().copied()).collect() };
            let (r_y, r_cl, r_an) = (rel(&y_all, &pyo), rel(&pick(&y_all, false), &pick(&pyo, false)), rel(&pick(&y_all, true), &pick(&pyw, true)));
            let same = seq.as_ref().map(|s| s.iter().zip(&y_all).all(|(p, q)| p.to_bits() == q.to_bits()));
            let tot: f64 = ms.iter().sum();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| ms[((ms.len() - 1) as f64 * p) as usize];
            eprintln!("スレッド {par}: 平均と錨の後の出力 Y {r_y:.2e}(門が閉じたフレーム = 平均 {r_cl:.2e}・開いた {n_open} フレーム = 錨 {r_an:.2e})・1 スレッドと 1 bit 一致 {same:?}・RTF {:.3}・フレームあたり ms p50 {:.2} p95 {:.2} p99 {:.2}(参照 {trt}・登録 {tro} フレーム)", tot / (t as f64 * 5.0), q(0.5), q(0.95), q(0.99));
            assert!(n_open > 0 && n_open < t, "門が開いたフレームと閉じたフレームの両方が要る");
            for (name, r) in worst.into_iter().chain([("平均と錨の後".to_string(), r_y), ("平均".to_string(), r_cl), ("錨".to_string(), r_an)]) {
                assert!(r < 2e-3, "{name}: 相対の差 {r}");
            }
            if let Some(s) = same {
                assert!(s, "スレッド {par} の出力が 1 スレッドと違う");
            }
            seq.get_or_insert(y_all);
        }
    }
}
