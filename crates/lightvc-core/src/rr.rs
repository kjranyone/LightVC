//! R(参照を読む・読み 2 層・training/train_rr.py RR2)と R̄ の近似 G(training/d587_rbar_distill.py)の 1 フレームずつの因果な推論。
//! 重み = training/export_rr.py。製品の S 形 = R(問い, 表, 目標の記憶) − G(問い, 表)。問いの幹(qin・qb)は共有。
//! 先読み 0: 幹と枝は因果の畳み込み・記憶(鍵と値)は登録で固定。

use std::path::Path;

use anyhow::{Context, Result};

use crate::nvoc::{CConv, Tensors};

struct Blk {
    a: CConv,
    b: CConv,
}

struct Branch {
    qp: Vec<CConv>,
    op: Vec<CConv>,
    lb: Vec<Blk>,
    din: CConv,
    db: Vec<Blk>,
    dout: CConv,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    m: usize,
}

/// 記憶の符号器の 1 つの畳み込み(登録時・非因果・中央揃え)。
struct Raw {
    w: Vec<f32>,
    b: Vec<f32>,
    co: usize,
    ci: usize,
    k: usize,
    d: usize,
}

impl Raw {
    /// x [ci][m] → y [co][m](中央揃えの非因果・端は 0 詰め)。
    fn run(&self, x: &[f32], m: usize) -> Vec<f32> {
        let pad = (self.k - 1) * self.d / 2;
        let mut y = vec![0.0f32; self.co * m];
        for o in 0..self.co {
            let row = &mut y[o * m..(o + 1) * m];
            row.iter_mut().for_each(|v| *v = self.b[o]);
            for c in 0..self.ci {
                let xr = &x[c * m..(c + 1) * m];
                for j in 0..self.k {
                    let coef = self.w[(o * self.ci + c) * self.k + j];
                    let s = (j * self.d) as isize - pad as isize;
                    let (t0, t1) = ((-s).max(0) as usize, (m as isize - s).min(m as isize).max(0) as usize);
                    if t1 > t0 {
                        crate::simd::axpy(&mut row[t0..t1], &xr[(t0 as isize + s) as usize..(t1 as isize + s) as usize], coef);
                    }
                }
            }
        }
        y
    }
}

struct MemEnc {
    min: Raw,
    mb: Vec<(Raw, Raw)>,
    kp: Vec<Raw>,
    vp: Vec<Raw>,
}

pub struct Rr {
    mem: Option<MemEnc>,
    qin: CConv,
    qb: Vec<Blk>,
    r: Branch,
    g: Branch,
    h: usize,
    dk: usize,
    ch: usize,
    pub nq: usize,
    trunk: Vec<f32>,
    x: Vec<f32>,
    t1: Vec<f32>,
    t2: Vec<f32>,
    t3: Vec<f32>,
    qq: Vec<f32>,
    rd: Vec<f32>,
    sc: Vec<f32>,
    cat: Vec<f32>,
    o_r: Vec<f32>,
    o_g: Vec<f32>,
}

fn lrelu(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x *= 0.1;
        }
    }
}

/// [頭][dk][m] → [頭][m][dk](1 列の鍵・値を連続に)。
fn to_hmd(a: &[f32], h: usize, dk: usize, m: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; a.len()];
    for hh in 0..h {
        for d in 0..dk {
            for j in 0..m {
                o[(hh * m + j) * dk + d] = a[(hh * dk + d) * m + j];
            }
        }
    }
    o
}

fn attn_scalar(q: &[f32], k: &[f32], v: &[f32], m: usize, dk: usize, sc: &mut [f32], r: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for j in 0..m {
        let s: f32 = q.iter().zip(&k[j * dk..(j + 1) * dk]).map(|(&a, &b)| a * b).sum();
        sc[j] = s;
        mx = mx.max(s);
    }
    r.iter_mut().for_each(|x| *x = 0.0);
    let mut z = 0.0f32;
    for j in 0..m {
        let w = (sc[j] - mx).exp();
        z += w;
        for (o, &x) in r.iter_mut().zip(&v[j * dk..(j + 1) * dk]) {
            *o += w * x;
        }
    }
    let iz = 1.0 / z;
    r.iter_mut().for_each(|x| *x *= iz);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        let x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0)), _mm256_set1_ps(-87.0));
        let n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_145_75), x);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428_606_8e-6), r);
        let mut p = _mm256_set1_ps(1.987_569_1e-4);
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_9e-3));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.166_666_66));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
        p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0)));
        let e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
        _mm256_mul_ps(p, _mm256_castsi256_ps(e))
    }
}

/// dk = 32 の 1 頭の注意(鍵・値 [m][32])。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_avx2(q: &[f32], k: &[f32], v: &[f32], m: usize, sc: &mut [f32], r: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let q0 = _mm256_loadu_ps(q.as_ptr());
        let q1 = _mm256_loadu_ps(q.as_ptr().add(8));
        let q2 = _mm256_loadu_ps(q.as_ptr().add(16));
        let q3 = _mm256_loadu_ps(q.as_ptr().add(24));
        let mut mx = f32::NEG_INFINITY;
        for j in 0..m {
            let kp = k.as_ptr().add(j * 32);
            let mut a = _mm256_mul_ps(q0, _mm256_loadu_ps(kp));
            a = _mm256_fmadd_ps(q1, _mm256_loadu_ps(kp.add(8)), a);
            a = _mm256_fmadd_ps(q2, _mm256_loadu_ps(kp.add(16)), a);
            a = _mm256_fmadd_ps(q3, _mm256_loadu_ps(kp.add(24)), a);
            let hi = _mm256_extractf128_ps(a, 1);
            let s4 = _mm_add_ps(_mm256_castps256_ps128(a), hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            let s = _mm_cvtss_f32(s1);
            *sc.get_unchecked_mut(j) = s;
            if s > mx {
                mx = s;
            }
        }
        let vmx = _mm256_set1_ps(mx);
        let mut j = 0;
        while j + 8 <= m {
            let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(sc.as_ptr().add(j)), vmx));
            _mm256_storeu_ps(sc.as_mut_ptr().add(j), e);
            j += 8;
        }
        while j < m {
            *sc.get_unchecked_mut(j) = (*sc.get_unchecked(j) - mx).exp();
            j += 1;
        }
        let (mut r0, mut r1, mut r2, mut r3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut z = 0.0f32;
        for j in 0..m {
            let w = *sc.get_unchecked(j);
            z += w;
            let vw = _mm256_set1_ps(w);
            let vp = v.as_ptr().add(j * 32);
            r0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vp), r0);
            r1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vp.add(8)), r1);
            r2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vp.add(16)), r2);
            r3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(vp.add(24)), r3);
        }
        let iz = _mm256_set1_ps(1.0 / z);
        _mm256_storeu_ps(r.as_mut_ptr(), _mm256_mul_ps(r0, iz));
        _mm256_storeu_ps(r.as_mut_ptr().add(8), _mm256_mul_ps(r1, iz));
        _mm256_storeu_ps(r.as_mut_ptr().add(16), _mm256_mul_ps(r2, iz));
        _mm256_storeu_ps(r.as_mut_ptr().add(24), _mm256_mul_ps(r3, iz));
    }
}

fn attn(q: &[f32], k: &[f32], v: &[f32], m: usize, dk: usize, sc: &mut [f32], r: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if dk == 32 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { attn_avx2(q, k, v, m, sc, r) };
            return;
        }
    }
    attn_scalar(q, k, v, m, dk, sc, r);
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let raw = std::fs::read(p).with_context(|| format!("{}", p.display()))?;
    Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

impl Blk {
    fn step(&mut self, x: &mut [f32], t1: &mut Vec<f32>, t2: &mut Vec<f32>) {
        let ch = x.len();
        t1.clear();
        t1.extend_from_slice(x);
        lrelu(t1);
        t2.resize(ch, 0.0);
        self.a.run(t1, 1, t2);
        lrelu(t2);
        t1.resize(ch, 0.0);
        self.b.run(t2, 1, t1);
        for (o, &d) in x.iter_mut().zip(t1.iter()) {
            *o += d;
        }
    }

    fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
    }
}

impl Branch {
    fn reset(&mut self) {
        self.qp.iter_mut().chain(self.op.iter_mut()).for_each(|c| c.reset());
        self.lb.iter_mut().chain(self.db.iter_mut()).for_each(|b| b.reset());
        self.din.reset();
        self.dout.reset();
    }
}

impl Rr {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let cfg = &man["cfg"];
        let gu = |k: &str| -> Result<usize> { cfg[k].as_u64().map(|v| v as usize).with_context(|| format!("cfg.{k}")) };
        let gv = |k: &str| -> Result<Vec<usize>> {
            Ok(cfg[k].as_array().with_context(|| format!("cfg.{k}"))?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect())
        };
        let (nq, ch, h, dk, layers) = (gu("nq")?, gu("ch")?, gu("heads")?, gu("dk")?, gu("layers")?);
        let (qd, lbd, dbd) = (gv("qd")?, gv("lb_dils")?, gv("db_dils")?);
        let conv = |name: &str, d: usize| -> Result<CConv> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (b, _) = ts.get(&format!("{name}.b"))?;
            Ok(CConv::from_raw(w, b, sh[0], sh[1], sh[2], d))
        };
        let blk = |name: &str, d: usize| -> Result<Blk> { Ok(Blk { a: conv(&format!("{name}.a"), d)?, b: conv(&format!("{name}.b"), 1)? }) };
        let branch = |p: &str| -> Result<Branch> {
            let mut qp = vec![];
            let mut op = vec![];
            let mut lb = vec![];
            for l in 0..layers {
                qp.push(conv(&format!("{p}.qp.{l}"), 1)?);
                op.push(conv(&format!("{p}.op.{l}"), 1)?);
                lb.push(blk(&format!("{p}.lb.{l}"), lbd[l])?);
            }
            let db = dbd.iter().enumerate().map(|(i, &d)| blk(&format!("{p}.db.{i}"), d)).collect::<Result<Vec<_>>>()?;
            Ok(Branch { qp, op, lb, din: conv(&format!("{p}.din"), 1)?, db, dout: conv(&format!("{p}.dout"), 1)?, k: vec![], v: vec![], m: 0 })
        };
        let qb = qd.iter().enumerate().map(|(i, &d)| blk(&format!("qb.{i}"), d)).collect::<Result<Vec<_>>>()?;
        let r = branch("r")?;
        let mut g = branch("g")?;
        for l in 0..layers {
            let (k, sk) = ts.get(&format!("g.k.{l}"))?;
            let (v, _) = ts.get(&format!("g.v.{l}"))?;
            anyhow::ensure!(sk[0] == h && sk[1] == dk, "g.k.{l} の形");
            g.m = sk[2];
            g.k.push(to_hmd(&k, h, dk, sk[2]));
            g.v.push(to_hmd(&v, h, dk, sk[2]));
        }
        let raw = |name: &str, d: usize| -> Result<Raw> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (b, _) = ts.get(&format!("{name}.b"))?;
            Ok(Raw { w, b, co: sh[0], ci: sh[1], k: sh[2], d })
        };
        let mem = if ts.get("mem.min.w").is_ok() {
            let kd = gv("kd")?;
            Some(MemEnc {
                min: raw("mem.min", 1)?,
                mb: kd.iter().enumerate().map(|(i, &d)| Ok((raw(&format!("mem.mb.{i}.a"), d)?, raw(&format!("mem.mb.{i}.b"), 1)?))).collect::<Result<Vec<_>>>()?,
                kp: (0..layers).map(|l| raw(&format!("mem.kp.{l}"), 1)).collect::<Result<Vec<_>>>()?,
                vp: (0..layers).map(|l| raw(&format!("mem.vp.{l}"), 1)).collect::<Result<Vec<_>>>()?,
            })
        } else {
            None
        };
        Ok(Self { mem, qin: conv("qin", 1)?, qb, r, g, h, dk, ch, nq, trunk: vec![0.0; ch], x: vec![0.0; ch], t1: vec![], t2: vec![], t3: vec![], qq: vec![0.0; h * dk], rd: vec![0.0; h * dk], sc: vec![], cat: vec![0.0; ch + 24], o_r: vec![0.0; 25], o_g: vec![0.0; 25] })
    }

    /// 目標の記憶(登録で固定): 層ごとの鍵・値 [頭][dk][m]。
    pub fn set_memory(&mut self, k: Vec<Vec<f32>>, v: Vec<Vec<f32>>, m: usize) -> Result<()> {
        anyhow::ensure!(k.len() == self.r.qp.len() && v.len() == k.len(), "記憶の層の数");
        for (a, b) in k.iter().zip(v.iter()) {
            anyhow::ensure!(a.len() == self.h * self.dk * m && b.len() == a.len(), "記憶の大きさ");
        }
        self.r.k = k.iter().map(|a| to_hmd(a, self.h, self.dk, m)).collect();
        self.r.v = v.iter().map(|a| to_hmd(a, self.h, self.dk, m)).collect();
        self.r.m = m;
        Ok(())
    }

    /// 書き出しの記憶ファイル(<dir>/mem_k{l}.f32・mem_v{l}.f32・[頭][dk][m])を読む。stride > 1 なら記憶の列を等間隔に間引く(計算量 1/stride・stride 2 で R の出力の相対誤差 約 3%)。
    pub fn load_memory(&mut self, dir: &Path, stride: usize) -> Result<()> {
        let layers = self.r.qp.len();
        let st = stride.max(1);
        let mut k = vec![];
        let mut v = vec![];
        let mut m = 0;
        for l in 0..layers {
            let (a, b) = (read_f32(&dir.join(format!("mem_k{l}.f32")))?, read_f32(&dir.join(format!("mem_v{l}.f32")))?);
            let m0 = a.len() / (self.h * self.dk);
            let sel: Vec<usize> = (0..m0).step_by(st).collect();
            m = sel.len();
            let pick = |x: &[f32]| -> Vec<f32> {
                let mut o = Vec::with_capacity(self.h * self.dk * m);
                for r in 0..self.h * self.dk {
                    o.extend(sel.iter().map(|&j| x[r * m0 + j]));
                }
                o
            };
            k.push(pick(&a));
            v.push(pick(&b));
        }
        self.set_memory(k, v, m)
    }

    /// 登録: 参照のフレームごとの rp [K][m](C1 の事後の上位 16)・rv [25][m](CheapTrick の包絡 c1..c24 / 10・相対 c0)→ 層ごとの鍵・値 [頭][dk][m](training/train_rr.py RR2.memory)。
    pub fn encode_memory(&self, rp: &[f32], rv: &[f32], m: usize) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let me = self.mem.as_ref().context("書き出しに記憶の符号器(mem.*)が無い(export_rr.py で書き直す)")?;
        anyhow::ensure!(rp.len() + rv.len() == me.min.ci * m, "rp・rv の大きさ");
        let mut x = Vec::with_capacity(me.min.ci * m);
        x.extend_from_slice(rp);
        x.extend_from_slice(rv);
        let mut h = me.min.run(&x, m);
        for (a, b) in me.mb.iter() {
            let mut t = h.clone();
            lrelu(&mut t);
            let mut u = a.run(&t, m);
            lrelu(&mut u);
            let dlt = b.run(&u, m);
            for (o, &dv) in h.iter_mut().zip(dlt.iter()) {
                *o += dv;
            }
        }
        Ok((me.kp.iter().map(|c| c.run(&h, m)).collect(), me.vp.iter().map(|c| c.run(&h, m)).collect()))
    }

    pub fn has_memory(&self) -> bool {
        self.r.m > 0
    }

    pub fn reset(&mut self) {
        self.qin.reset();
        self.qb.iter_mut().for_each(|b| b.reset());
        self.r.reset();
        self.g.reset();
    }

    fn branch_step(br: &mut Branch, trunk: &[f32], tab: &[f32], h: usize, dk: usize, ch: usize, x: &mut Vec<f32>, t1: &mut Vec<f32>, t2: &mut Vec<f32>, qq: &mut Vec<f32>, rd: &mut Vec<f32>, sc: &mut Vec<f32>, cat: &mut Vec<f32>, t3: &mut Vec<f32>, o: &mut [f32]) {
        x.clear();
        x.extend_from_slice(trunk);
        let m = br.m;
        let s = 1.0 / (dk as f32).sqrt();
        for l in 0..br.qp.len() {
            br.qp[l].run(x, 1, qq);
            let (kk, vv) = (&br.k[l], &br.v[l]);
            sc.resize(m, 0.0);
            qq.iter_mut().for_each(|x| *x *= s);
            for hh in 0..h {
                attn(&qq[hh * dk..(hh + 1) * dk], &kk[hh * m * dk..(hh + 1) * m * dk], &vv[hh * m * dk..(hh + 1) * m * dk], m, dk, sc, &mut rd[hh * dk..(hh + 1) * dk]);
            }
            t3.resize(ch, 0.0);
            br.op[l].run(rd, 1, t3);
            for (a, &b) in x.iter_mut().zip(t3.iter()) {
                *a += b;
            }
            br.lb[l].step(x, t1, t2);
        }
        cat[..ch].copy_from_slice(x);
        for (c, &t) in cat[ch..].iter_mut().zip(tab) {
            *c = t / 10.0;
        }
        x.resize(ch, 0.0);
        br.din.run(cat, 1, x);
        for b in br.db.iter_mut() {
            b.step(x, t1, t2);
        }
        br.dout.run(x, 1, o);
    }

    /// 1 フレーム: 問い q [nq]・表の引き tab [24](包絡の生の単位)→ ds [24](R − G の包絡の差・生の単位)・戻り値 = R − G の c0 の差。
    pub fn step(&mut self, q: &[f32], tab: &[f32], ds: &mut [f32]) -> f32 {
        assert!(self.has_memory(), "R の目標の記憶が未設定");
        assert_eq!(q.len(), self.nq);
        assert_eq!(tab.len(), 24);
        self.qin.run(q, 1, &mut self.trunk);
        for b in self.qb.iter_mut() {
            b.step(&mut self.trunk, &mut self.t1, &mut self.t2);
        }
        let (h, dk, ch) = (self.h, self.dk, self.ch);
        Self::branch_step(&mut self.r, &self.trunk, tab, h, dk, ch, &mut self.x, &mut self.t1, &mut self.t2, &mut self.qq, &mut self.rd, &mut self.sc, &mut self.cat, &mut self.t3, &mut self.o_r);
        Self::branch_step(&mut self.g, &self.trunk, tab, h, dk, ch, &mut self.x, &mut self.t1, &mut self.t2, &mut self.qq, &mut self.rd, &mut self.sc, &mut self.cat, &mut self.t3, &mut self.o_g);
        for i in 0..24 {
            ds[i] = 10.0 * (self.o_r[i] - self.o_g[i]);
        }
        self.o_r[24] - self.o_g[24]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RR_EXPORT=<export_rr.py の出力> cargo test --release -p lightvc-core rr_memory_parity -- --nocapture
    #[test]
    fn rr_memory_parity() {
        let Ok(dir) = std::env::var("RR_EXPORT") else {
            eprintln!("RR_EXPORT が未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let fx = dir.join("fixture");
        let m = Rr::load(dir).unwrap();
        let rp = read_f32(&fx.join("rp.f32")).unwrap();
        let rv = read_f32(&fx.join("rv.f32")).unwrap();
        let mm = rv.len() / 25;
        let t0 = std::time::Instant::now();
        let (k, v) = m.encode_memory(&rp, &rv, mm).unwrap();
        let el = t0.elapsed().as_secs_f64();
        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for l in 0..k.len() {
            for (a, name) in [(&k[l], format!("mem_k{l}.f32")), (&v[l], format!("mem_v{l}.f32"))] {
                let e = read_f32(&fx.join(name)).unwrap();
                assert_eq!(a.len(), e.len());
                for (x, y) in a.iter().zip(e.iter()) {
                    worst = worst.max((x - y).abs());
                    scale = scale.max(y.abs());
                }
            }
        }
        eprintln!("rr_memory_parity M {mm} max|Δ| {worst:.2e}(最大値 {scale:.2})・{el:.2} s");
        assert!(worst < 1e-3 * scale.max(1.0), "記憶が一致しない");
    }

    /// RR_EXPORT=<export_rr.py の出力> cargo test --release -p lightvc-core rr_parity -- --nocapture
    #[test]
    fn rr_parity() {
        let Ok(dir) = std::env::var("RR_EXPORT") else {
            eprintln!("RR_EXPORT が未設定のためスキップ");
            return;
        };
        let dir = Path::new(&dir);
        let fx = dir.join("fixture");
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx.join("meta.json")).unwrap()).unwrap();
        let t = meta["T"].as_u64().unwrap() as usize;
        let mut m = Rr::load(dir).unwrap();
        m.load_memory(&fx, 1).unwrap();
        let q = read_f32(&fx.join("q.f32")).unwrap();
        let tab = read_f32(&fx.join("tab.f32")).unwrap();
        let ds = read_f32(&fx.join("ds.f32")).unwrap();
        let dc = read_f32(&fx.join("dc.f32")).unwrap();
        let nq = m.nq;
        let (mut qf, mut tf, mut o) = (vec![0.0f32; nq], vec![0.0f32; 24], vec![0.0f32; 24]);
        let (mut worst, mut worst_c, mut rms) = (0.0f32, 0.0f32, 0.0f64);
        let t0 = std::time::Instant::now();
        for i in 0..t {
            for c in 0..nq {
                qf[c] = q[c * t + i];
            }
            for c in 0..24 {
                tf[c] = tab[c * t + i];
            }
            let c0 = m.step(&qf, &tf, &mut o);
            for c in 0..24 {
                let e = (o[c] - ds[c * t + i]).abs();
                worst = worst.max(e);
                rms += (ds[c * t + i] as f64).powi(2);
            }
            worst_c = worst_c.max((c0 - dc[i]).abs());
        }
        let el = t0.elapsed().as_secs_f64();
        eprintln!("rr_parity T {t} max|Δds| {worst:.2e} max|Δdc| {worst_c:.2e} ds rms {:.3} ・ {:.3} ms/フレーム(RTF {:.3})", (rms / (24 * t) as f64).sqrt(), el * 1e3 / t as f64, el / (t as f64 * 0.005));
        assert!(worst < 2e-3 && worst_c < 2e-4, "一致しない");
    }
}
