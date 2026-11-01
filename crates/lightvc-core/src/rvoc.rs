//! rvoc(出力部・training/rvoc.py と同じ構造)のストリーミング推論。重みは training/export_rvoc.py(`RVoc::load`)、乱数の重みは RTF 計測用。
//!
//! 条件 [D_COND](フレーム)と励起 [2][HOP](48kHz のパルス列・雑音)→ 出力 [HOP]。段 = LeakyReLU → 直線補間 ×r(前のフレームからの補間・因果)
//! → 因果 conv(k = 2r+1)→ + 励起(48kHz の励起に stride s・k = 2s・左 s の因果 conv。最後の段は k = 3 の因果 conv)→ ResBlock の平均。

use std::path::Path;

use anyhow::{Context, Result};

use crate::nvoc::{CConv, HarmSrc, Tensors, HOP};

pub const UPS: [usize; 4] = [5, 4, 4, 3];
pub const D_COND: usize = 31;

/// 状態つきの因果 strided Conv1d: out[o][j] = b[o] + Σ_c Σ_q w[o][c][q]·xp[c][j·s + q]・xp = 前のブロックの末尾 lpad サンプル ++ 入力。
struct SConv {
    ci: usize,
    co: usize,
    k: usize,
    s: usize,
    lpad: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    hist: Vec<f32>,
    buf: Vec<f32>,
}

impl SConv {
    fn from_raw(w: Vec<f32>, b: Vec<f32>, co: usize, ci: usize, k: usize, s: usize, lpad: usize) -> Self {
        Self { ci, co, k, s, lpad, w, b, hist: vec![0.0; ci * lpad], buf: Vec::new() }
    }

    /// inp [ci][l] → out [co][l / s]
    fn run(&mut self, inp: &[f32], l: usize, out: &mut [f32]) {
        let (ci, co, k, s, lp) = (self.ci, self.co, self.k, self.s, self.lpad);
        let ls = lp + l;
        self.buf.resize(ci * ls, 0.0);
        for c in 0..ci {
            self.buf[c * ls..c * ls + lp].copy_from_slice(&self.hist[c * lp..(c + 1) * lp]);
            self.buf[c * ls + lp..(c + 1) * ls].copy_from_slice(&inp[c * l..(c + 1) * l]);
            if lp > 0 {
                self.hist[c * lp..(c + 1) * lp].copy_from_slice(&self.buf[(c + 1) * ls - lp..(c + 1) * ls]);
            }
        }
        let n = l / s;
        for o in 0..co {
            let row = &mut out[o * n..(o + 1) * n];
            for (j, r) in row.iter_mut().enumerate() {
                let mut acc = self.b[o];
                for c in 0..ci {
                    let wr = &self.w[(o * ci + c) * k..(o * ci + c + 1) * k];
                    let xr = &self.buf[c * ls + j * s..c * ls + j * s + k];
                    for q in 0..k {
                        acc += wr[q] * xr[q];
                    }
                }
                *r = acc;
            }
        }
    }
}

struct Rb {
    c1: Vec<CConv>,
    c2: Vec<CConv>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
}

impl Rb {
    fn new(c1: Vec<CConv>, c2: Vec<CConv>) -> Self {
        Self { c1, c2, t1: vec![], t2: vec![], t3: vec![] }
    }

    /// xr(入力・出力を兼ねる)[co][n] に残差の列を通す。
    fn run(&mut self, xr: &mut [f32], n: usize) {
        let sz = xr.len();
        for v in [&mut self.t1, &mut self.t2, &mut self.t3] {
            v.resize(sz, 0.0);
        }
        for (a, b) in self.c1.iter_mut().zip(self.c2.iter_mut()) {
            self.t1.copy_from_slice(xr);
            lrelu_inplace(&mut self.t1, 0.1);
            a.run(&self.t1, n, &mut self.t2);
            lrelu_inplace(&mut self.t2, 0.1);
            b.run(&self.t2, n, &mut self.t3);
            for (x, &y) in xr.iter_mut().zip(self.t3.iter()) {
                *x += y;
            }
        }
    }
}

/// 各段の残差ブロックのうち 1 本(最も重い k)を常駐のワーカースレッドが持つ(同じブロックの中で並列 = 遅延は増えない)。
struct RbWorker {
    tx: std::sync::mpsc::Sender<(usize, Vec<f32>, usize)>,
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
}

impl RbWorker {
    fn spawn(mut rbs: Vec<Rb>) -> Self {
        let (tx, rx_in) = std::sync::mpsc::channel::<(usize, Vec<f32>, usize)>();
        let (tx_out, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        std::thread::spawn(move || {
            crate::affinity::pin_current_to_perf();
            while let Ok((i, mut x, n)) = crate::affinity::recv_spin(&rx_in) {
                rbs[i].run(&mut x, n);
                if tx_out.send(x).is_err() {
                    break;
                }
            }
        });
        Self { tx, rx }
    }
}

struct Stage {
    r: usize,
    ci: usize,
    co: usize,
    prev: Vec<f32>,
    up: CConv,
    src: SConv,
    res: Vec<Rb>,
    n_res: usize,
    xi: Vec<f32>,
    x: Vec<f32>,
    xr: Vec<f32>,
    acc: Vec<f32>,
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
    fn run(&mut self, xin: &[f32], n_in: usize, exc: &[f32], idx: usize, par: Option<&RbWorker>) -> usize {
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
        self.es.resize(co * n, 0.0);
        self.src.run(exc, HOP, &mut self.es);
        for (a, b) in self.x.iter_mut().zip(self.es.iter()) {
            *a += *b;
        }
        let sz = co * n;
        self.xr.resize(sz, 0.0);
        self.acc.clear();
        self.acc.resize(sz, 0.0);
        if let Some(w) = par {
            w.tx.send((idx, self.x.clone(), n)).expect("rvoc worker");
        }
        for rb in self.res.iter_mut() {
            self.xr.copy_from_slice(&self.x);
            rb.run(&mut self.xr, n);
            for (x, &y) in self.acc.iter_mut().zip(self.xr.iter()) {
                *x += y;
            }
        }
        if let Some(w) = par {
            let y = crate::affinity::recv_spin(&w.rx).expect("rvoc worker");
            for (x, &v) in self.acc.iter_mut().zip(y.iter()) {
                *x += v;
            }
        }
        let inv = 1.0 / self.n_res as f32;
        self.acc.iter_mut().for_each(|v| *v *= inv);
        n
    }
}

/// 励起 [2][HOP] = [帯域制限パルス列 × 0.1, 雑音 × 0.01](training/train_rvoc.excitation と同じ。パルス列は nvoc.harmonic_source と一致・雑音は別系列の正規乱数)。
pub struct Excite {
    src: HarmSrc,
    rng: u64,
    h: Vec<f32>,
}

impl Excite {
    pub fn new(h_max: f32, h_roll: f32, k_max: usize, seed: u64) -> Self {
        Self { src: HarmSrc::new(h_max, h_roll, k_max), rng: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1, h: vec![0.0; HOP] }
    }

    fn gauss(&mut self) -> f32 {
        let mut u = || {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            ((self.rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (a, b) = (u(), u());
        ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
    }

    /// f0(Hz・0 = 無声)→ out [2][HOP]
    pub fn block(&mut self, f0: f32, out: &mut [f32]) {
        let mut h = std::mem::take(&mut self.h);
        self.src.block(f0, &mut h);
        for j in 0..HOP {
            out[j] = h[j] * 0.1;
            out[HOP + j] = self.gauss() * 0.01;
        }
        self.h = h;
    }
}

/// 文脈の符号器(training/train_ppgvoc.py の CtxEnc・因果・先読み 0): [事後 K・c0n・log f0・有声・包絡 24] → 包絡のずれ 24。
struct Ctx {
    k: usize,
    inp: CConv,
    blocks: Vec<CConv>,
    out: CConv,
    x: Vec<f32>,
    h: Vec<f32>,
    t: Vec<f32>,
    u: Vec<f32>,
    d: Vec<f32>,
}

impl Ctx {
    /// cond(生の条件 [0 = c0n, 1..25 = 包絡, 25 = log f0, 26 = 有声, 27..31 = 周期性, 31..31+K = 事後])の包絡の行にずれを足す。
    fn apply(&mut self, cond: &mut [f32]) {
        let k = self.k;
        self.x.clear();
        self.x.extend_from_slice(&cond[31..31 + k]);
        self.x.push(cond[0]);
        self.x.push(cond[25]);
        self.x.push(cond[26]);
        self.x.extend_from_slice(&cond[1..25]);
        self.h.resize(self.inp.co_len(), 0.0);
        self.inp.run(&self.x, 1, &mut self.h);
        for b in self.blocks.iter_mut() {
            self.t.clear();
            self.t.extend(self.h.iter().map(|&v| if v < 0.0 { v * 0.1 } else { v }));
            self.u.resize(self.h.len(), 0.0);
            b.run(&self.t, 1, &mut self.u);
            for (a, &v) in self.h.iter_mut().zip(self.u.iter()) {
                *a += v;
            }
        }
        self.t.clear();
        self.t.extend(self.h.iter().map(|&v| if v < 0.0 { v * 0.1 } else { v }));
        self.d.resize(24, 0.0);
        self.out.run(&self.t, 1, &mut self.d);
        for (c, &v) in cond[1..25].iter_mut().zip(self.d.iter()) {
            *c += v;
        }
    }
}

const GAMMA64: u64 = 0x9E3779B97F4A7C15;

pub(crate) fn splitmix_u(seed: u64, q: u64) -> f64 {
    let mut z = seed.wrapping_add(q.wrapping_mul(GAMMA64));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

/// 遅い音色の揺らぎ(training/train_ppgvoc.py の slow_noise_prng・diag_ppgvoc21): 共通の乱数(splitmix64 + Box-Muller・フレームごとに k 成分)を
/// 成分ごとの最小位相 FIR で因果に濾過 × 利得 → 基底 [24, k] × σ を包絡の行に足す。先読み 0・発話の統計なし・起動時に L−1 フレームで FIR を満たす(定常の開始)。
struct Slow {
    k: usize,
    l: usize,
    b: Vec<f64>,
    fir: Vec<f64>,
    gain: Vec<f64>,
    sigma: f64,
    seed: u64,
    q: u64,
    ring: Vec<f64>,
    pos: usize,
    e: Vec<f64>,
    w: Vec<f64>,
}

impl Slow {
    fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.q = 0;
        self.pos = 0;
        self.ring.fill(0.0);
        for _ in 0..self.l - 1 {
            self.push();
        }
    }

    fn push(&mut self) {
        for i in (0..self.k).step_by(2) {
            let u1 = splitmix_u(self.seed, self.q + 1);
            let u2 = splitmix_u(self.seed, self.q + 2);
            self.q += 2;
            let r = (-2.0 * u1.ln()).sqrt();
            let th = 2.0 * std::f64::consts::PI * u2;
            self.e[i] = r * th.cos();
            self.e[i + 1] = r * th.sin();
        }
        self.pos = (self.pos + 1) % self.l;
        let o = self.pos * self.k;
        self.ring[o..o + self.k].copy_from_slice(&self.e);
    }

    fn apply(&mut self, cond: &mut [f32]) {
        self.push();
        for j in 0..self.k {
            let h = &self.fir[j * self.l..(j + 1) * self.l];
            let mut acc = 0.0f64;
            let mut s = self.pos;
            for &hm in h {
                acc += hm * self.ring[s * self.k + j];
                s = if s == 0 { self.l - 1 } else { s - 1 };
            }
            self.w[j] = acc * self.gain[j];
        }
        for r in 0..24 {
            let d: f64 = (0..self.k).map(|j| self.b[r * self.k + j] * self.w[j]).sum();
            cond[1 + r] += (self.sigma * d) as f32;
        }
    }
}

pub struct RVoc {
    pre: CConv,
    ctx: Option<Ctx>,
    slow: Option<Slow>,
    /// 外から与える遅い揺らぎ [24](操作者の揺らぎ・artic_vc::OpSlow)。文脈の符号器のずれの後・乱数の揺らぎの後に包絡の行へ足す(TabVoc.make と同じ順)。
    ext_slow: Option<Vec<f32>>,
    cbuf: Vec<f32>,
    pub d_cond: usize,
    /// 音量の条件を低い帯域の水準に(diag_ppgvoc47・cfg.c0low): 行 0 = a·(行 0 + Σ_{r=1..24} v_r·行 r)。None = 従来。
    pub c0low: Option<(Vec<f32>, f32)>,
    stages: Vec<Stage>,
    par: Option<RbWorker>,
    post: CConv,
    h0: Vec<f32>,
    out_in: Vec<f32>,
    pub ch: usize,
    /// 診断: Some のとき毎フレーム、生成器に入る実効の条件の先頭 27 行 [c0・包絡 24・log f0・有声] を積む(LIGHTVC_DUMP_VCOND)
    pub vdump: Option<Vec<f32>>,
    /// 診断: Some のとき毎フレーム、文脈の符号器の前の生の条件 [d_cond] と符号器の後の包絡 24 行を積む(LIGHTVC_DUMP_RCOND)
    pub rdump: Option<Vec<f32>>,
    /// 構音の描画(生成器なし・current/artic_render.md)。Some なら生成器の段を通らず、生の条件と遅い揺らぎ + R の ΔS から描く。
    pub render: Option<crate::artic_render::ArticRender>,
}

fn src_geom(tot: usize) -> (usize, usize, usize) {
    let s = HOP / tot;
    if s > 1 {
        (2 * s, s, s)
    } else {
        (3, 1, 2)
    }
}

impl RVoc {
    fn assemble(ch: usize, d_cond: usize, kernels: &[usize], dils: &[usize], mut conv: impl FnMut(&str, usize, usize, usize, usize) -> Result<CConv>,
                mut sconv: impl FnMut(&str, usize, usize, usize, usize) -> Result<SConv>) -> Result<Self> {
        let pre = conv("pre", d_cond, ch, 3, 1)?;
        let mut stages = Vec::new();
        let mut tot = 1;
        for (i, &r) in UPS.iter().enumerate() {
            let (ci, co) = (ch >> i, ch >> (i + 1));
            tot *= r;
            let up = conv(&format!("up.{i}"), ci, co, 2 * r + 1, 1)?;
            let (k, s, lp) = src_geom(tot);
            let src = sconv(&format!("src.{i}"), co, k, s, lp)?;
            let mut res = Vec::new();
            for (j, &k) in kernels.iter().enumerate() {
                let mut c1 = Vec::new();
                let mut c2 = Vec::new();
                for (m, &d) in dils.iter().enumerate() {
                    c1.push(conv(&format!("res.{i}.{j}.c1.{m}"), co, co, k, d)?);
                    c2.push(conv(&format!("res.{i}.{j}.c2.{m}"), co, co, k, 1)?);
                }
                res.push(Rb::new(c1, c2));
            }
            let n_res = res.len();
            stages.push(Stage { r, ci, co, prev: vec![0.0; ci], up, src, res, n_res, xi: vec![], x: vec![], xr: vec![], acc: vec![], es: vec![] });
        }
        let post = conv("post", ch >> UPS.len(), 1, 7, 1)?;
        Ok(Self { pre, ctx: None, slow: None, ext_slow: None, cbuf: vec![], d_cond, c0low: None, stages, par: None, post, h0: vec![0.0; ch], out_in: vec![], ch, vdump: None, rdump: None, render: None })
    }

    pub fn random(ch: usize, seed: u64) -> Self {
        let state = std::cell::Cell::new(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
        let rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let s = state.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    state.set(s);
                    (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
                })
                .collect()
        };
        Self::assemble(
            ch,
            D_COND,
            &[3, 7, 11],
            &[1, 3, 5],
            |_, ci, co, k, d| Ok(CConv::from_raw(rnd(co * ci * k, 1.0 / ((ci * k) as f32).sqrt()), rnd(co, 0.01), co, ci, k, d)),
            |_, co, k, st, lp| Ok(SConv::from_raw(rnd(co * 2 * k, 1.0 / ((2 * k) as f32).sqrt()), rnd(co, 0.01), co, 2, k, st, lp)),
        )
        .expect("random weights")
    }

    /// training/export_rvoc.py の書き出し(manifest.json・weights.bin)から作る。キー名 = PyTorch のモジュールの経路 + .w / .b。
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let cfg = &man["cfg"];
        let ch = cfg["ch"].as_u64().context("cfg.ch")? as usize;
        let d_cond = cfg["d_cond"].as_u64().context("cfg.d_cond")? as usize;
        let ppg_k = cfg["ppg_k"].as_u64().unwrap_or(0) as usize;
        let cv_d = cfg["cv_d"].as_u64().unwrap_or(0) as usize;
        anyhow::ensure!(d_cond == D_COND || d_cond == D_COND + ppg_k + cv_d, "d_cond {d_cond} != {D_COND} (+ ppg_k {ppg_k} + cv_d {cv_d})");
        let vec_of = |k: &str| -> Result<Vec<usize>> {
            Ok(cfg[k].as_array().with_context(|| format!("cfg.{k}"))?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
        };
        let (kernels, dils, ups) = (vec_of("kernels")?, vec_of("dils")?, vec_of("ups")?);
        anyhow::ensure!(ups == UPS, "ups {ups:?} != {UPS:?}");
        let get = |name: &str, want: [usize; 3]| -> Result<(Vec<f32>, Vec<f32>)> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (b, _) = ts.get(&format!("{name}.b"))?;
            anyhow::ensure!(sh == want, "{name}: shape {sh:?} != {want:?}");
            Ok((w, b))
        };
        let mut m = Self::assemble(
            ch,
            d_cond,
            &kernels,
            &dils,
            |name, ci, co, k, d| {
                let (w, b) = get(name, [co, ci, k])?;
                Ok(CConv::from_raw(w, b, co, ci, k, d))
            },
            |name, co, k, st, lp| {
                let (w, b) = get(name, [co, 2, k])?;
                Ok(SConv::from_raw(w, b, co, 2, k, st, lp))
            },
        )?;
        if let Some(cc) = cfg.get("ctx").filter(|v| !v.is_null()) {
            let cch = cc["ch"].as_u64().context("cfg.ctx.ch")? as usize;
            let cd: Vec<usize> = cc["dils"].as_array().context("cfg.ctx.dils")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
            let mk = |name: &str, co: usize, ci: usize, k: usize, d: usize| -> Result<CConv> {
                let (w, b) = get(name, [co, ci, k])?;
                Ok(CConv::from_raw(w, b, co, ci, k, d))
            };
            let inp = mk("ctx.inp", cch, ppg_k + 3 + 24, 1, 1)?;
            let blocks = cd.iter().enumerate().map(|(i, &d)| mk(&format!("ctx.blocks.{i}"), cch, cch, 3, d)).collect::<Result<Vec<_>>>()?;
            let out = mk("ctx.out", 24, cch, 1, 1)?;
            m.ctx = Some(Ctx { k: ppg_k, inp, blocks, out, x: vec![], h: vec![], t: vec![], u: vec![], d: vec![] });
        }
        if let Some(cl) = cfg.get("c0low").filter(|v| !v.is_null()) {
            let v: Vec<f32> = cl["v"].as_array().context("cfg.c0low.v")?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            anyhow::ensure!(v.len() == 25, "cfg.c0low.v の長さ {} != 25", v.len());
            m.c0low = Some((v, cl["a"].as_f64().context("cfg.c0low.a")? as f32));
        }
        if let Some(sc) = cfg.get("slow").filter(|v| !v.is_null()) {
            let k = sc["k"].as_u64().context("cfg.slow.k")? as usize;
            let l = sc["L"].as_u64().context("cfg.slow.L")? as usize;
            anyhow::ensure!(k % 2 == 0 && l >= 2, "cfg.slow: k {k} は偶数・L {l} ≥ 2");
            let ld = |name: &str, want: Vec<usize>| -> Result<Vec<f64>> {
                let (v, sh) = ts.get(name)?;
                anyhow::ensure!(sh == want, "{name}: shape {sh:?} != {want:?}");
                Ok(v.into_iter().map(|x| x as f64).collect())
            };
            let mut sl = Slow { k, l, b: ld("slow.B", vec![24, k])?, fir: ld("slow.fir", vec![k, l])?, gain: ld("slow.gain", vec![k])?,
                                sigma: sc["sigma"].as_f64().context("cfg.slow.sigma")?, seed: 0, q: 0, ring: vec![0.0; k * l], pos: 0, e: vec![0.0; k], w: vec![0.0; k] };
            sl.reset(sc["seed"].as_u64().unwrap_or(0));
            m.slow = Some(sl);
        }
        Ok(m)
    }

    /// 各段の最後の残差ブロック(最も重い k)を常駐のワーカースレッドへ移す。出力は逐次と同じ(加算の順序だけ違う)。
    pub fn enable_parallel(&mut self) {
        if self.par.is_some() {
            return;
        }
        let rbs: Vec<Rb> = self.stages.iter_mut().map(|st| st.res.pop().expect("残差ブロック")).collect();
        self.par = Some(RbWorker::spawn(rbs));
    }

    /// 遅い音色の揺らぎの乱数を種 seed から始め直す(揺らぎが無ければ何もしない)。
    pub fn reset_slow(&mut self, seed: u64) {
        if let Some(sl) = self.slow.as_mut() {
            sl.reset(seed);
        }
    }

    /// 遅い音色の揺らぎの基底 [24, k](行優先)と k(揺らぎが無ければ None)。操作者の揺らぎ(artic_vc::OpSlow)が同じ基底へ射影する。
    pub fn slow_basis(&self) -> Option<(Vec<f64>, usize)> {
        self.slow.as_ref().map(|sl| (sl.b.clone(), sl.k))
    }

    /// 外から与える遅い揺らぎ [24] を次のブロックから使う(None で止める)。
    pub fn set_ext_slow(&mut self, z: Option<&[f32]>) {
        match z {
            Some(v) => self.ext_slow.get_or_insert_with(|| vec![0.0; 24]).copy_from_slice(v),
            None => self.ext_slow = None,
        }
    }

    /// 遅い音色の揺らぎの倍率(揺らぎが無ければ None)。
    pub fn slow_sigma(&mut self, sigma: Option<f64>) -> Option<f64> {
        let sl = self.slow.as_mut()?;
        if let Some(v) = sigma {
            sl.sigma = v;
        }
        Some(sl.sigma)
    }

    /// cond [d_cond](文脈の符号器があれば生の条件・内部で包絡のずれを足す・遅い揺らぎがあればその後に足す)・exc [2][HOP] → out [HOP]
    pub fn process_block(&mut self, cond: &[f32], exc: &[f32], out: &mut [f32]) {
        if let Some(rd) = self.render.as_mut() {
            self.cbuf.clear();
            self.cbuf.extend_from_slice(cond);
            if let Some(sl) = self.slow.as_mut() {
                sl.apply(&mut self.cbuf);
            }
            if let Some(e) = self.ext_slow.as_ref() {
                for (c, &v) in self.cbuf[1..25].iter_mut().zip(e.iter()) {
                    *c += v;
                }
            }
            let mut post = [0.0f32; 24];
            for r in 0..24 {
                post[r] = self.cbuf[1 + r] - cond[1 + r];
            }
            if let Some(d) = self.rdump.as_mut() {
                d.extend_from_slice(cond);
                d.extend_from_slice(&self.cbuf[1..25]);
            }
            rd.process(cond, &post, out);
            return;
        }
        if self.ctx.is_some() || self.slow.is_some() || self.ext_slow.is_some() {
            self.cbuf.clear();
            self.cbuf.extend_from_slice(cond);
            if let Some(d) = self.rdump.as_mut() {
                d.extend_from_slice(cond);
            }
            if let Some(cx) = self.ctx.as_mut() {
                cx.apply(&mut self.cbuf);
            }
            if let Some(d) = self.rdump.as_mut() {
                d.extend_from_slice(&self.cbuf[1..25]);
            }
            if let Some(sl) = self.slow.as_mut() {
                sl.apply(&mut self.cbuf);
            }
            if let Some(e) = self.ext_slow.as_ref() {
                for (c, &v) in self.cbuf[1..25].iter_mut().zip(e.iter()) {
                    *c += v;
                }
            }
            self.pre.run(&self.cbuf, 1, &mut self.h0);
            if let Some(d) = self.vdump.as_mut() {
                d.extend_from_slice(&self.cbuf[..27]);
            }
        } else {
            self.pre.run(cond, 1, &mut self.h0);
            if let Some(d) = self.vdump.as_mut() {
                d.extend_from_slice(&cond[..27]);
            }
            if let Some(d) = self.rdump.as_mut() {
                d.extend_from_slice(cond);
                d.extend_from_slice(&cond[1..25]);
            }
        }
        let mut cur = self.h0.clone();
        let mut n = 1usize;
        for (i, st) in self.stages.iter_mut().enumerate() {
            n = st.run(&cur, n, exc, i, self.par.as_ref());
            cur.clear();
            cur.extend_from_slice(&st.acc);
        }
        self.out_in.resize(cur.len(), 0.0);
        for (a, &b) in self.out_in.iter_mut().zip(cur.iter()) {
            *a = if b < 0.0 { b * 0.01 } else { b };
        }
        self.post.run(&self.out_in, n, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(p: &Path) -> Vec<f32> {
        let raw = std::fs::read(p).unwrap();
        raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// RVOC_EXPORT=<export_rvoc.py の出力> cargo test --release -p lightvc-core rvoc_parity -- --nocapture
    #[test]
    fn rvoc_parity_and_rtf() {
        let Ok(dir) = std::env::var("RVOC_EXPORT") else {
            eprintln!("RVOC_EXPORT 未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let mut m = RVoc::load(dir).unwrap();
        let fx = dir.join("fixture");
        let cond = read_f32(&fx.join("cond.f32"));
        let exc = read_f32(&fx.join("exc.f32"));
        let y = read_f32(&fx.join("y.f32"));
        let dc = m.d_cond;
        let t = cond.len() / dc;
        assert_eq!(exc.len(), t * 2 * HOP);
        assert_eq!(y.len(), t * HOP);
        let mut out = vec![0.0f32; t * HOP];
        let mut times = Vec::with_capacity(t);
        let t_all = std::time::Instant::now();
        for i in 0..t {
            let t0 = std::time::Instant::now();
            m.process_block(&cond[i * dc..(i + 1) * dc], &exc[i * 2 * HOP..(i + 1) * 2 * HOP], &mut out[i * HOP..(i + 1) * HOP]);
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let el = t_all.elapsed().as_secs_f64();
        let md = out.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
        println!("parity: max|Rust − Python| {md:.3e}  (出力 RMS {rms:.3e})");
        println!("RTF (1 thread, streaming 240-sample blocks) {:.3}  per-block ms p50 {:.3} p95 {:.3} p99 {:.3} (予算 5 ms)",
                 el / (t as f64 * HOP as f64 / 48000.0), q(0.5), q(0.95), q(0.99));
        assert!(md < 1e-3 * rms.max(1e-3) + 1e-5, "parity FAIL {md}");
        let mut mp = RVoc::load(dir).unwrap();
        mp.enable_parallel();
        let mut outp = vec![0.0f32; t * HOP];
        let tp = std::time::Instant::now();
        for i in 0..t {
            mp.process_block(&cond[i * dc..(i + 1) * dc], &exc[i * 2 * HOP..(i + 1) * 2 * HOP], &mut outp[i * HOP..(i + 1) * HOP]);
        }
        let elp = tp.elapsed().as_secs_f64();
        let dp = out.iter().zip(outp.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("並列(最も重い残差ブロックをワーカーへ): 逐次との max|差| {dp:.3e}・RTF {:.3}", elp / (t as f64 * HOP as f64 / 48000.0));
        assert!(dp == 0.0, "parallel != sequential {dp}");
        let f0 = read_f32(&fx.join("f0.f32"));
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        let c = &man["const"];
        let mut ex = Excite::new(c["H_MAX"].as_f64().unwrap() as f32, c["H_ROLL"].as_f64().unwrap() as f32, c["K_MAX"].as_u64().unwrap() as usize, 1);
        let mut e = vec![0.0f32; 2 * HOP];
        let (mut mh, mut hr, mut nr) = (0.0f32, 0.0f64, 0.0f64);
        for i in 0..t {
            ex.block(f0[i], &mut e);
            for j in 0..HOP {
                mh = mh.max((e[j] - exc[i * 2 * HOP + j]).abs());
                hr += (exc[i * 2 * HOP + j] as f64).powi(2);
                nr += (e[HOP + j] as f64).powi(2);
            }
        }
        let n_s = (t * HOP) as f64;
        println!("excitation: パルス列 max|Rust − Python| {mh:.3e}(RMS {:.3e})・雑音 RMS {:.3e}(期待 0.01)", (hr / n_s).sqrt(), (nr / n_s).sqrt());
        assert!(mh < 1e-4, "excitation parity FAIL {mh}");
    }
}
