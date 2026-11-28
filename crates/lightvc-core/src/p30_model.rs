//! paper_snv の研究の基準の型 snv_p30_c のストリーミング推論(製品の既定 P11 = p_model と選べる別の型・SnvStream は manifest の cfg.kind = "p30" で選ぶ)。
//! 演奏 = 凍結した因果の CPC の文脈 c(training/train_snv_p2_cpc.CPC.encode)を train_snv_p3.PlayW で読む: 登録の平均 mu を引く → PCA 32 →
//! 縮みつきの共分散の逆平方根で白くする → 級の正規化(training/snv_cmn.apply)→ proj 64。mu・白くする行列・級の正規化の delta は登録の定数。
//! 回路 C = train_snv_p1.Circ(p_model の Net / StructEnc・C の鍵 = 目標の参照を目標自身の統計で読んだ演奏)。重みと一致検査 = training/export_snv_p30.py。

use std::ops::Range;
use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::affinity::Team;
use crate::nvoc::Tensors;
use crate::p_model::{mv, Lin, Memory, Net, StructEnc, Wm, D_A, D_Y};
use crate::prof::prof;

const CH: usize = 256;
const D_XP: usize = 32;
const PK: usize = 32;
const DILS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// 環 [R][ci] の時刻 t の行(R = 2 の冪)。
pub(crate) fn at(len: usize, ci: usize, t: usize) -> Range<usize> {
    let s = t & (len / ci - 1);
    s * ci..(s + 1) * ci
}

/// 因果の Conv1d(左に (k−1)·d の 0 埋め・先読み 0・p_gate も使う): 重み [co][k·ci](列 = j·ci + c・タップ j = 時刻 t − (k−1−j)·d・ci は 0 埋めで ci_pad へ)。
pub(crate) struct CConv {
    w: Wm,
    b: Vec<f32>,
    ci: usize,
    k: usize,
    d: usize,
}

impl CConv {
    pub(crate) fn new(ts: &Tensors, name: &str, d: usize, ci_pad: usize) -> Result<Self> {
        let (cw, sh) = ts.get(&format!("{name}.weight"))?;
        let (b, _) = ts.get(&format!("{name}.bias"))?;
        let (co, ci, k) = (sh[0], sh[1], sh[2]);
        ensure!(k <= 3, "{name} {sh:?}");
        let cp = ci.max(ci_pad);
        let mut w = vec![0.0f32; co * k * cp];
        for o in 0..co {
            for c in 0..ci {
                for j in 0..k {
                    w[(o * k + j) * cp + c] = cw[(o * ci + c) * k + j];
                }
            }
        }
        Ok(Self { w: Wm::new(w, false), b, ci: cp, k, d })
    }

    /// 重みを f16 で持つ(段 P94)。
    pub(crate) fn half(&mut self) {
        let f = std::mem::replace(&mut self.w, Wm::new(Vec::new(), false));
        self.w = Wm::new(f.into_f32(), true);
    }

    pub(crate) fn ring(&self) -> Vec<f32> {
        vec![0.0; ((self.k - 1) * self.d + 1).next_power_of_two() * self.ci]
    }

    pub(crate) fn run(&self, ring: &[f32], t: usize, out: &mut [f32]) {
        let mut taps: [&[f32]; 3] = [&[]; 3];
        for (j, tp) in taps.iter_mut().enumerate().take(self.k) {
            *tp = &ring[at(ring.len(), self.ci, t.wrapping_sub((self.k - 1 - j) * self.d))];
        }
        self.w.mv(&taps[..self.k], Some(&self.b), out);
    }

    pub(crate) fn one(&self, x: &[f32], out: &mut [f32]) {
        self.w.mv(&[x], Some(&self.b), out);
    }
}

fn relu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = x.max(0.0));
}

fn layer_norm(z: &mut [f32], eps: f64) {
    let n = z.len() as f64;
    let m = z.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = z.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n;
    let r = 1.0 / (var + eps).sqrt();
    z.iter_mut().for_each(|v| *v = ((*v as f64 - m) * r) as f32);
}

/// 凍結した CPC の符号器の流し(e1 → e2 → e3 → 層の正規化 → ar_in → 6 段のゲートつき膨張畳み込み)。入力 = [Y 1..28 / sig・有声・Δc0 / sig 29](train_snv_p2_cpc.x_torch)。
struct Cpc {
    sig: Vec<f32>,
    e1: CConv,
    e2: CConv,
    e3: CConv,
    ar_in: CConv,
    conv: Vec<CConv>,
    out: Vec<CConv>,
    eps: f64,
    rx: Vec<f32>,
    rh: Vec<f32>,
    rb: Vec<Vec<f32>>,
    t: usize,
    y0: Option<f32>,
    h: Vec<f32>,
    z: Vec<f32>,
    ab: Vec<f32>,
    g: Vec<f32>,
}

impl Cpc {
    fn new(ts: &Tensors, p: &str, eps: f64) -> Result<Self> {
        let (sig, sh) = ts.get(&format!("{p}.sig"))?;
        ensure!(sh == [D_Y + 1], "sig {sh:?}");
        let cv = |n: &str, d: usize, cp: usize| CConv::new(ts, &format!("{p}.cpc.{n}"), d, cp);
        let e1 = cv("e1", 1, D_XP)?;
        ensure!(e1.ci == D_XP && e1.b.len() == CH, "e1 の形");
        let conv = DILS.iter().enumerate().map(|(i, &d)| cv(&format!("ar.{i}.conv"), d, 0)).collect::<Result<Vec<_>>>()?;
        let out = (0..DILS.len()).map(|i| cv(&format!("ar.{i}.out"), 1, 0)).collect::<Result<Vec<_>>>()?;
        let (e2, e3, ar_in) = (cv("e2", 2, 0)?, cv("e3", 1, 0)?, cv("ar_in", 1, 0)?);
        let (rx, rh, rb) = (e1.ring(), e2.ring(), conv.iter().map(CConv::ring).collect());
        Ok(Self { sig, e1, e2, e3, ar_in, conv, out, eps, rx, rh, rb, t: 0, y0: None, h: vec![0.0; CH], z: vec![0.0; CH], ab: vec![0.0; 2 * CH], g: vec![0.0; CH] })
    }

    /// フレームごとに読む重みを f16 にする(段 P94・training/snv_p1_eval.half_weights と同じ層)。
    fn half(&mut self) {
        for c in [&mut self.e1, &mut self.e2, &mut self.e3, &mut self.ar_in].into_iter().chain(self.conv.iter_mut()).chain(self.out.iter_mut()) {
            c.half();
        }
    }

    fn reset(&mut self) {
        for r in [&mut self.rx, &mut self.rh].into_iter().chain(self.rb.iter_mut()) {
            r.iter_mut().for_each(|v| *v = 0.0);
        }
        self.t = 0;
        self.y0 = None;
    }

    /// 1 フレーム: Y [D_Y]・有声 → 文脈 c [CH](流しの最初のフレームの Δc0 = 0)。
    fn step(&mut self, y: &[f32], v: f32, c: &mut [f32]) {
        let t = self.t;
        let r = at(self.rx.len(), D_XP, t);
        let x = &mut self.rx[r];
        for (i, xi) in x.iter_mut().enumerate().take(D_Y - 1) {
            *xi = y[i + 1] / self.sig[i + 1];
        }
        x[D_Y - 1] = v;
        x[D_Y] = self.y0.map_or(0.0, |p| y[0] - p) / self.sig[D_Y];
        self.y0 = Some(y[0]);
        self.e1.run(&self.rx, t, &mut self.h);
        relu(&mut self.h);
        let r = at(self.rh.len(), CH, t);
        self.rh[r].copy_from_slice(&self.h);
        self.e2.run(&self.rh, t, &mut self.h);
        relu(&mut self.h);
        self.e3.one(&self.h, &mut self.z);
        layer_norm(&mut self.z, self.eps);
        let r = at(self.rb[0].len(), CH, t);
        self.ar_in.one(&self.z, &mut self.rb[0][r]);
        for i in 0..DILS.len() {
            self.conv[i].run(&self.rb[i], t, &mut self.ab);
            let (a, b) = self.ab.split_at(CH);
            for ((g, a), b) in self.g.iter_mut().zip(a).zip(b) {
                *g = a.tanh() * (1.0 / (1.0 + (-b).exp()));
            }
            self.out[i].one(&self.g, &mut self.z);
            let r = at(self.rb[i].len(), CH, t);
            for (z, h) in self.z.iter_mut().zip(&self.rb[i][r]) {
                *z += h;
            }
            if i + 1 < DILS.len() {
                let r = at(self.rb[i + 1].len(), CH, t);
                self.rb[i + 1][r].copy_from_slice(&self.z);
            } else {
                c.copy_from_slice(&self.z);
            }
        }
        self.t += 1;
    }
}

/// 登録の統計(登録の定数): mu [CH]・白くする行列 wm [PK][PK]・級の正規化の delta [級][PK]。
pub(crate) struct Stat {
    pub(crate) mu: Vec<f32>,
    pub(crate) wm: Vec<f32>,
    pub(crate) delta: Vec<f32>,
}

/// d = c − mu・s = Pᵀ d。
fn center(pt: &[f32], c: &[f32], mu: &[f32], d: &mut [f32], s: &mut [f32]) {
    for ((d, &c), &m) in d.iter_mut().zip(c).zip(mu) {
        *d = c - m;
    }
    mv(pt, &[d], None, s);
}

/// 級の事後確率 γ_k(x) ∝ exp(−‖x − c_k‖² / (2 s2))(snv_cmn.posterior)。
fn post(cb: &[f32], s2: f32, x: &[f32], p: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for (pk, ck) in p.iter_mut().zip(cb.chunks_exact(PK)) {
        let d: f32 = x.iter().zip(ck).map(|(a, b)| (a - b) * (a - b)).sum();
        *pk = -d / (2.0 * s2);
        mx = mx.max(*pk);
    }
    let mut z = 0.0f32;
    for pk in p.iter_mut() {
        *pk = (*pk - mx).exp();
        z += *pk;
    }
    let iz = 1.0 / z;
    p.iter_mut().for_each(|pk| *pk *= iz);
}

/// 対称な正定値の行列 a [n][n] の逆平方根 Q diag(1 / √max(e, 1e-8)) Qᵀ(巡回 Jacobi 法の固有値分解・train_snv_p3.PlayW.mean_of の式)。
fn inv_sqrt_sym(a: &[f64], n: usize) -> Vec<f32> {
    let mut a = a.to_vec();
    let mut q = vec![0.0f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    for _ in 0..64 {
        let all: f64 = a.iter().map(|v| v * v).sum();
        let diag: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum();
        if all - diag <= 1e-28 * all {
            break;
        }
        for p in 0..n {
            for r in p + 1..n {
                let apr = a[p * n + r];
                if apr == 0.0 {
                    continue;
                }
                let th = (a[r * n + r] - a[p * n + p]) / (2.0 * apr);
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (a[k * n + p], a[k * n + r]);
                    a[k * n + p] = c * x - s * y;
                    a[k * n + r] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (a[p * n + k], a[r * n + k]);
                    a[p * n + k] = c * x - s * y;
                    a[r * n + k] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (q[k * n + p], q[k * n + r]);
                    q[k * n + p] = c * x - s * y;
                    q[k * n + r] = s * x + c * y;
                }
            }
        }
    }
    let f: Vec<f64> = (0..n).map(|k| 1.0 / a[k * n + k].max(1e-8).sqrt()).collect();
    let mut w = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            w[i * n + j] = (0..n).map(|k| q[i * n + k] * f[k] * q[j * n + k]).sum::<f64>() as f32;
        }
    }
    w
}

/// 演奏の読み(train_snv_p3.PlayW・遅れ 0 だけ): x = Wm·Pᵀ(c − mu)・x' = x − γ(x)·delta・a = proj(x')。
struct Play {
    pt: Vec<f32>,
    cb: Vec<f32>,
    s2: f32,
    tau: f64,
    shrink: f64,
    proj: Lin,
    d: Vec<f32>,
    s: Vec<f32>,
    x: Vec<f32>,
    p: Vec<f32>,
}

impl Play {
    fn new(ts: &Tensors, p: &str, s2: f32, tau: f64, shrink: f64) -> Result<Self> {
        let (pm, sh) = ts.get(&format!("{p}.P"))?;
        ensure!(sh == [CH, PK], "P {sh:?}");
        let mut pt = vec![0.0f32; PK * CH];
        for c in 0..CH {
            for k in 0..PK {
                pt[k * CH + c] = pm[c * PK + k];
            }
        }
        let (cb, sc) = ts.get(&format!("{p}.cb"))?;
        ensure!(sc.len() == 2 && sc[1] == PK, "cb {sc:?}");
        let (_, sp) = ts.get(&format!("{p}.proj.weight"))?;
        ensure!(sp == [D_A, PK], "proj {sp:?}");
        Ok(Self { pt, s2, tau, shrink, proj: Lin::new(ts, &format!("{p}.proj"))?, d: vec![0.0; CH], s: vec![0.0; PK], x: vec![0.0; PK], p: vec![0.0; sc[0]], cb })
    }

    /// 登録(重み = 有声・PlayW.mean_of と snv_cmn.offsets): c [t][CH]・有声 [t]。
    fn stat(&mut self, c: &[f32], v: &[f32], t: usize) -> Stat {
        let n = v.iter().map(|&w| w as f64).sum::<f64>().max(1.0);
        let mut mu = vec![0.0f64; CH];
        for (ct, &w) in c.chunks_exact(CH).zip(v) {
            for (m, &x) in mu.iter_mut().zip(ct) {
                *m += x as f64 * w as f64;
            }
        }
        let mu: Vec<f32> = mu.iter().map(|m| (m / n) as f32).collect();
        let mut s = vec![0.0f32; t * PK];
        let mut cov = vec![0.0f64; PK * PK];
        for ((ct, &w), st) in c.chunks_exact(CH).zip(v).zip(s.chunks_exact_mut(PK)) {
            center(&self.pt, ct, &mu, &mut self.d, st);
            for (i, &si) in st.iter().enumerate() {
                let a = si as f64 * w as f64;
                for (cv, &sj) in cov[i * PK..(i + 1) * PK].iter_mut().zip(st.iter()) {
                    *cv += a * sj as f64;
                }
            }
        }
        let tr = (0..PK).map(|i| cov[i * PK + i]).sum::<f64>() / n / PK as f64;
        for (i, cv) in cov.iter_mut().enumerate() {
            *cv = (1.0 - self.shrink) * (*cv / n) + if i % (PK + 1) == 0 { self.shrink * tr } else { 0.0 };
        }
        let wm = inv_sqrt_sym(&cov, PK);
        let nc = self.p.len();
        let (mut num, mut den) = (vec![0.0f64; nc * PK], vec![0.0f64; nc]);
        for (st, &w) in s.chunks_exact(PK).zip(v) {
            mv(&wm, &[st], None, &mut self.x);
            post(&self.cb, self.s2, &self.x, &mut self.p);
            for ((pk, dk), nk) in self.p.iter().zip(den.iter_mut()).zip(num.chunks_exact_mut(PK)) {
                let g = *pk as f64 * w as f64;
                *dk += g;
                for (nm, &x) in nk.iter_mut().zip(&self.x) {
                    *nm += g * x as f64;
                }
            }
        }
        let delta = (0..nc * PK).map(|i| ((num[i] + self.tau * self.cb[i] as f64) / (den[i / PK] + self.tau) - self.cb[i] as f64) as f32).collect();
        Stat { mu, wm, delta }
    }

    /// 1 フレーム: 文脈 c [CH] → 演奏 a [D_A]。
    fn run(&mut self, c: &[f32], st: &Stat, a: &mut [f32]) {
        center(&self.pt, c, &st.mu, &mut self.d, &mut self.s);
        mv(&st.wm, &[&self.s], None, &mut self.x);
        post(&self.cb, self.s2, &self.x, &mut self.p);
        for (pk, dk) in self.p.iter().zip(st.delta.chunks_exact(PK)) {
            for (x, &d) in self.x.iter_mut().zip(dk) {
                *x -= pk * d;
            }
        }
        self.proj.run(&self.x, a);
    }
}

/// 変換器(snv_p30_c): 操作者の登録の統計と目標の参照の構造で、フレームごとに Y(29)を出す。
pub struct P30Model {
    cpc: Cpc,
    play: Play,
    c: Net,
    sc: StructEnc,
    team: Team,
    pub(crate) op: Option<Stat>,
    pub(crate) mem_c: Option<Memory>,
    pub(crate) ctx: Vec<f32>,
    xc: Vec<f32>,
    kv16: bool,
}

impl P30Model {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir).context("p30 の重み")?;
        let cfg = &man["cfg"];
        ensure!(cfg["kind"].as_str() == Some("p30"), "{}: cfg.kind が p30 でない", dir.display());
        ensure!(cfg["lags"] == serde_json::json!([0]) && cfg["pca_k"].as_u64() == Some(PK as u64), "遅れ [0]・PCA {PK} の型だけ");
        let num = |k: &str| cfg[k].as_f64().with_context(|| format!("cfg {k}"));
        Self::from_tensors(&ts, "", num("ln_eps")?, num("cb_s2")? as f32, num("tau")?, num("shrink")?)
    }

    /// 重みの名前の前に pre を付けた組(k3g_model の組 = "m0." など・p30 は "")。
    pub(crate) fn from_tensors(ts: &Tensors, pre: &str, ln_eps: f64, cb_s2: f32, tau: f64, shrink: f64) -> Result<Self> {
        let (pl, c) = (format!("{pre}Pl"), format!("{pre}C"));
        Ok(Self {
            cpc: Cpc::new(ts, &pl, ln_eps)?, play: Play::new(ts, &pl, cb_s2, tau, shrink)?,
            c: Net::new(ts, &c)?, sc: StructEnc::new(ts, &c, false)?, team: Team::new(1), op: None, mem_c: None, ctx: vec![0.0; CH], xc: vec![0.0; D_A + 2], kv16: false,
        })
    }

    /// C の各段の出力チャンネルを n 個のスレッドに分ける(n = 1・2・4・状態は保つ・CPC と演奏の読みは呼び出しのスレッド)。
    pub fn enable_parallel(&mut self, n: usize) {
        self.c.split(n);
        self.team = Team::new(n);
    }

    /// 新しい流しの CPC で y [D_Y][t]・有声 [t] を通した文脈 [t][CH](終わったら流しの状態を初めに戻す)。
    fn ctx_all(&mut self, y: &[f32], v: &[f32], t: usize) -> Vec<f32> {
        self.cpc.reset();
        let mut c = vec![0.0f32; t * CH];
        let mut yi = vec![0.0f32; D_Y];
        for (tt, ct) in c.chunks_exact_mut(CH).enumerate() {
            for (ch, y1) in yi.iter_mut().enumerate() {
                *y1 = y[ch * t + tt];
            }
            self.cpc.step(&yi, v[tt], ct);
        }
        self.cpc.reset();
        c
    }

    /// 操作者の登録: y [D_Y][t]・有声 [t](チャンネル優先)→ mu・白くする行列・級の正規化の delta。
    pub fn enroll_operator(&mut self, y: &[f32], v: &[f32], t: usize) {
        let c = self.ctx_all(y, v, t);
        self.op = Some(self.play.stat(&c, &v[..t], t));
    }

    /// 目標の参照: y [D_Y][t]・有声 [t] → 参照の演奏 a_ref [D_A][t](目標自身の統計で読む)→ C の構造。
    pub fn enroll_target(&mut self, y: &[f32], v: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        self.enroll_target_keys(y, y, v, t)
    }

    /// 目標の参照で、鍵の演奏だけ別の特徴 yk [D_Y][t] から読む(k3g_model の固定の窓の包絡の組)。C の構造の z と値は y から。
    pub(crate) fn enroll_target_keys(&mut self, y: &[f32], yk: &[f32], v: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let c = self.ctx_all(yk, v, t);
        let st = self.play.stat(&c, &v[..t], t);
        let mut a_ref = vec![0.0f32; D_A * t];
        let mut a = vec![0.0f32; D_A];
        for (tt, ct) in c.chunks_exact(CH).enumerate() {
            self.play.run(ct, &st, &mut a);
            for (ch, &av) in a.iter().enumerate() {
                a_ref[ch * t + tt] = av;
            }
        }
        let (zc, mut mem_c) = self.sc.run(y, Some(&a_ref), t);
        if self.kv16 {
            mem_c.half();
        }
        self.c.set_z(&zc);
        self.c.reset();
        self.mem_c = Some(mem_c);
        (zc, a_ref)
    }

    /// 目標の構造の注意の鍵と値を f16 で持つ(段 P94 step c・目標の登録の前に呼ぶ)。
    pub fn set_kv16(&mut self) {
        self.kv16 = true;
    }

    /// 符号器と回路のフレームごとに読む重みを f16 で持つ(段 P94・登録の前に呼ぶ)。
    pub fn set_half(&mut self) {
        self.cpc.half();
        self.c.set_half();
    }

    /// 1 フレーム: 操作者の Y(29)・有声・C に渡す高さ lf(= log(f0 / 200) + k 半音)→ (演奏 a [D_A], 出力 Y [D_Y])。
    pub fn step(&mut self, y: &[f32], v: f32, lf: f32, a_out: &mut [f32], y_out: &mut [f32]) {
        let (op, mc) = (self.op.as_ref().expect("enroll_operator が先"), self.mem_c.as_ref().expect("enroll_target が先"));
        prof!(member M_CPC, self.cpc.step(y, v, &mut self.ctx));
        prof!(member M_PLAY, self.play.run(&self.ctx, op, a_out));
        self.xc[..D_A].copy_from_slice(a_out);
        self.xc[D_A] = lf;
        self.xc[D_A + 1] = v;
        self.c.step(&self.team, &self.xc, mc, y_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 逆平方根の定義: W·A·W = I・W は対称(固有値の幅 1e4 の乱数の正定値の行列)。
    #[test]
    fn inv_sqrt_identity() {
        let n = PK;
        let mut st = 0x2545f4914f6cdd1du64;
        let mut r = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let b: Vec<f64> = (0..n * n).map(|_| r()).collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum::<f64>() + if i == j { 1e-4 * (1 + i) as f64 } else { 0.0 };
            }
        }
        let w: Vec<f64> = inv_sqrt_sym(&a, n).iter().map(|&v| v as f64).collect();
        let mut e = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let v: f64 = (0..n).flat_map(|k| (0..n).map(move |l| (k, l))).map(|(k, l)| w[i * n + k] * a[k * n + l] * w[l * n + j]).sum();
                e = e.max((v - if i == j { 1.0 } else { 0.0 }).abs()).max((w[i * n + j] - w[j * n + i]).abs());
            }
        }
        assert!(e < 1e-3, "W·A·W − I の最大 {e}");
    }
}
