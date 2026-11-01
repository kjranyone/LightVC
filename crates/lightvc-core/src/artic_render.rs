//! 構音の描画(生成器なし・current/artic_render.md の基準 PR4)。参照の実装 = training/causal_synth.py・training/d663_shape_power.py・training/artic_render_ref.py。
//! 生の条件 [31 + K + C2](文脈の符号器の前)と遅い揺らぎ + R の ΔS(包絡の行 24)→ 出力 [HOP]。先読み 0。重みは training/export_artic_render.py。

use std::f64::consts::PI;
use std::path::Path;

use anyhow::{Context, Result};

use crate::nvoc::{CConv, Tensors, HOP, SR};
use crate::rvoc::splitmix_u;
use crate::ship_front::fft;

const N: usize = 2048;
const NB: usize = N / 2 + 1;
const NM: usize = 128;

fn leaky(v: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.extend(v.iter().map(|&x| if x < 0.0 { x * 0.1 } else { x }));
}

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let m = (lo + hi) / 2;
        if xs[m] <= x {
            lo = m;
        } else {
            hi = m;
        }
    }
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

/// 因果なパラメトリック合成(training/causal_synth.py と同じ式・雑音 = splitmix64 + Box-Muller の共通の乱数)。
struct Synth {
    buf: Vec<f64>,
    phase: f64,
    prev: Option<(f64, Vec<f64>, Vec<f64>)>,
    seed: u64,
    q: u64,
    re: Vec<f64>,
    im: Vec<f64>,
    hre: Vec<f64>,
    him: Vec<f64>,
    nre: Vec<f64>,
    nim: Vec<f64>,
    gre: Vec<f64>,
    gim: Vec<f64>,
    s_: Vec<f64>,
    amp: Vec<f64>,
}

impl Synth {
    fn new(seed: u64) -> Self {
        Self {
            buf: vec![0.0; N + 2 * HOP],
            phase: 0.0,
            prev: None,
            seed,
            q: 0,
            re: vec![0.0; N],
            im: vec![0.0; N],
            hre: vec![0.0; NB],
            him: vec![0.0; NB],
            nre: vec![0.0; 2 * N],
            nim: vec![0.0; 2 * N],
            gre: vec![0.0; 2 * N],
            gim: vec![0.0; 2 * N],
            s_: vec![0.0; NB],
            amp: vec![0.0; NB],
        }
    }

    /// amp [NB] → 最小位相の複素スペクトル (hre, him) [NB](ケプストラムの折り返し)。
    fn minphase(&mut self) {
        for k in 0..N {
            let kk = if k <= N / 2 { k } else { N - k };
            self.re[k] = self.amp[kk].max(1e-12).ln();
            self.im[k] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        let s = 1.0 / N as f64;
        for i in 0..N {
            let c = self.re[i] * s;
            self.re[i] = if i == 0 || i == N / 2 {
                c
            } else if i < N / 2 {
                2.0 * c
            } else {
                0.0
            };
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        for k in 0..NB {
            let e = self.re[k].exp();
            self.hre[k] = e * self.im[k].cos();
            self.him[k] = e * self.im[k].sin();
        }
    }

    /// (hre, him)·e^{−j2πkd/N} の実の逆変換 → re[0..N](np.fft.irfft と同じ: 直流とナイキストの虚部は捨てる)。
    fn irfft_delay(&mut self, d: f64) {
        for k in 0..NB {
            let ph = -2.0 * PI * k as f64 * d / N as f64;
            let (c, s) = (ph.cos(), ph.sin());
            let (a, b) = (self.hre[k] * c - self.him[k] * s, self.hre[k] * s + self.him[k] * c);
            self.re[k] = a;
            self.im[k] = if k == 0 || k == N / 2 { 0.0 } else { -b };
        }
        for k in NB..N {
            self.re[k] = self.re[N - k];
            self.im[k] = -self.im[N - k];
        }
        fft(&mut self.re, &mut self.im);
        let s = 1.0 / N as f64;
        for i in 0..N {
            self.re[i] *= s;
        }
    }

    fn block(&mut self, f0: f64, sp: &[f64], ap: &[f64], out: &mut [f32]) {
        if self.prev.is_none() {
            self.prev = Some((f0, sp.to_vec(), ap.to_vec()));
        }
        let (f0p, spp, app) = self.prev.take().unwrap();
        for i in 0..HOP {
            let w = (i + 1) as f64 / HOP as f64;
            if f0 <= 0.0 && f0p <= 0.0 {
                continue;
            }
            let f = if f0p <= 0.0 {
                f0
            } else if f0 <= 0.0 {
                f0p
            } else {
                (1.0 - w) * f0p + w * f0
            };
            if (f0 <= 0.0 && w > 0.5) || (f0p <= 0.0 && w <= 0.5) {
                continue;
            }
            let step = f / SR as f64;
            self.phase += step;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                let d = 1.0 - self.phase / step;
                let p = SR as f64 / f;
                for k in 0..NB {
                    let s_ = (1.0 - w) * spp[k] + w * sp[k];
                    let a_ = ((1.0 - w) * app[k] + w * ap[k]).clamp(0.0, 1.0);
                    self.amp[k] = ((s_ * (1.0 - a_ * a_)).max(0.0) * p).sqrt();
                }
                self.minphase();
                self.irfft_delay(d);
                for j in 0..N {
                    self.buf[i + j] += self.re[j];
                }
            }
        }
        for k in 0..NB {
            let s_m = 0.5 * (spp[k] + sp[k]);
            let a_m = (0.5 * (app[k] + ap[k])).clamp(0.0, 1.0);
            self.amp[k] = (s_m * a_m * a_m).max(0.0).sqrt();
        }
        self.minphase();
        self.irfft_delay(0.0);
        for i in 0..2 * N {
            self.gre[i] = if i < N { self.re[i] } else { 0.0 };
            self.gim[i] = 0.0;
        }
        fft(&mut self.gre, &mut self.gim);
        for i in (0..HOP).step_by(2) {
            let u1 = splitmix_u(self.seed, self.q + 1);
            let u2 = splitmix_u(self.seed, self.q + 2);
            self.q += 2;
            let r = (-2.0 * u1.ln()).sqrt();
            let th = 2.0 * PI * u2;
            self.nre[i] = r * th.cos();
            self.nre[i + 1] = r * th.sin();
        }
        for i in HOP..2 * N {
            self.nre[i] = 0.0;
        }
        self.nim.iter_mut().for_each(|v| *v = 0.0);
        fft(&mut self.nre, &mut self.nim);
        for k in 0..2 * N {
            let (a, b) = (self.nre[k] * self.gre[k] - self.nim[k] * self.gim[k], self.nre[k] * self.gim[k] + self.nim[k] * self.gre[k]);
            self.nre[k] = a;
            self.nim[k] = -b;
        }
        fft(&mut self.nre, &mut self.nim);
        let s = 1.0 / (2 * N) as f64;
        for j in 0..HOP + N {
            self.buf[j] += self.nre[j] * s;
        }
        for (o, &v) in out.iter_mut().zip(self.buf[..HOP].iter()) {
            *o = v as f32;
        }
        self.buf.copy_within(HOP.., 0);
        let l = self.buf.len();
        self.buf[l - HOP..].iter_mut().for_each(|v| *v = 0.0);
        self.prev = Some((f0, sp.to_vec(), ap.to_vec()));
    }
}

/// 構音の描画: 内容の網(EnvNet)+ 網の遅い補正を外す 1 極 + 大きさ / f0 の慣性 + 形のパワーの係数の慣性 + 4 帯の周期性の非周期性 + 較正の曲線 + 因果な合成。
pub struct ArticRender {
    k: usize,
    cv: usize,
    inp: CConv,
    blocks: Vec<CConv>,
    out: CConv,
    x: Vec<f32>,
    h: Vec<f32>,
    t: Vec<f32>,
    u: Vec<f32>,
    d: Vec<f32>,
    ema: Option<Vec<f64>>,
    a_hp: f64,
    c0s: Option<f64>,
    a_c0: f64,
    lv_s: Option<f64>,
    a_lv: f64,
    lv_thr: f32,
    /// 大きさの音節の動きのつまみ γ(発声の走る平均 2 Hz からの差を γ 倍・1.0 = 操作者のまま)。
    pub level_gamma: f64,
    lfs: f64,
    prev_v: bool,
    a_lf: f64,
    lsb: Option<f64>,
    a_s: f64,
    c0_sil: f64,
    dct: Vec<f64>,
    fc: Vec<f64>,
    lfq: Vec<f64>,
    band_lc: [f64; 4],
    curve: Vec<f64>,
    curve1: Vec<f64>,
    /// 声道長のつまみ(帯の中心を α 倍・1.0 = 中立)。
    pub alpha: f64,
    gain: f32,
    lm: Vec<f64>,
    fcs: Vec<f64>,
    sp: Vec<f64>,
    ap: Vec<f64>,
    synth: Synth,
}

impl ArticRender {
    pub fn load(dir: &Path, seed: u64) -> Result<Self> {
        let (ts, man) = Tensors::load(dir)?;
        let cfg = &man["cfg"];
        let k = cfg["k"].as_u64().context("cfg.k")? as usize;
        let cv = cfg["cv"].as_u64().context("cfg.cv")? as usize;
        let ch = cfg["ch"].as_u64().context("cfg.ch")? as usize;
        let dils: Vec<usize> = cfg["dils"].as_array().context("cfg.dils")?.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect();
        let f = |n: &str| cfg[n].as_f64().with_context(|| format!("cfg.{n}"));
        anyhow::ensure!(cfg["hop"].as_u64() == Some(HOP as u64) && cfg["nfft"].as_u64() == Some(N as u64), "hop / nfft が違う");
        let mk = |name: &str, co: usize, ci: usize, kk: usize, d: usize| -> Result<CConv> {
            let (w, sh) = ts.get(&format!("{name}.w"))?;
            let (b, _) = ts.get(&format!("{name}.b"))?;
            anyhow::ensure!(sh == [co, ci, kk], "{name}: shape {sh:?} != {:?}", [co, ci, kk]);
            Ok(CConv::from_raw(w, b, co, ci, kk, d))
        };
        let inp = mk("env.inp", ch, k + 3 + 24 + cv, 1, 1)?;
        let blocks = dils.iter().enumerate().map(|(i, &d)| mk(&format!("env.blocks.{i}"), ch, ch, 3, d)).collect::<Result<Vec<_>>>()?;
        let out = mk("env.out", 24, ch, 1, 1)?;
        let ld = |name: &str, want: Vec<usize>| -> Result<Vec<f64>> {
            let (v, sh) = ts.get(name)?;
            anyhow::ensure!(sh == want, "{name}: shape {sh:?} != {want:?}");
            Ok(v.into_iter().map(|x| x as f64).collect())
        };
        let a = |hz: f64| (-2.0 * PI * hz * HOP as f64 / SR as f64).exp();
        let bands: Vec<(f64, f64)> = cfg["per_bands"].as_array().context("cfg.per_bands")?.iter().map(|b| (b[0].as_f64().unwrap_or(0.0), b[1].as_f64().unwrap_or(0.0))).collect();
        anyhow::ensure!(bands.len() == 4, "per_bands は 4 帯");
        let mut band_lc = [0.0; 4];
        for (i, (lo, hi)) in bands.iter().enumerate() {
            band_lc[i] = (lo * hi).sqrt().ln();
        }
        let lfq = (0..NB).map(|kk| (kk as f64 * SR as f64 / N as f64).max(1.0).ln()).collect();
        Ok(Self {
            k,
            cv,
            inp,
            blocks,
            out,
            x: vec![],
            h: vec![],
            t: vec![],
            u: vec![],
            d: vec![],
            ema: None,
            a_hp: (-(HOP as f64) / SR as f64 / f("hp_tau_s")?).exp(),
            c0s: None,
            a_c0: a(f("c0_hz")?),
            lv_s: None,
            a_lv: a(cfg["level_hz"].as_f64().unwrap_or(2.0)),
            lv_thr: cfg["level_thr"].as_f64().unwrap_or(0.05) as f32,
            level_gamma: cfg["level_gamma"].as_f64().unwrap_or(1.0),
            lfs: 0.0,
            prev_v: false,
            a_lf: a(f("lf_hz")?),
            lsb: None,
            a_s: a(f("shape_hz")?),
            c0_sil: f("c0_sil")?,
            dct: ld("dct", vec![25, NM])?,
            fc: ld("fc", vec![NM])?,
            lfq,
            band_lc,
            curve: ld("curve", vec![NB])?,
            curve1: ld("curve1", vec![NB])?,
            alpha: 1.0,
            gain: 10f64.powf(cfg["gain_db"].as_f64().unwrap_or(0.0) / 20.0) as f32,
            lm: vec![0.0; NM],
            fcs: vec![0.0; NM],
            sp: vec![0.0; NB],
            ap: vec![0.0; NB],
            synth: Synth::new(seed),
        })
    }

    /// 目標ごとの表の引きのパワーの較正の曲線 [NB](dB・登録の時に target_curve で作る)を使う: 曲線 = 描画の較正 1 × 10^(dB/10)(書き出しの全体の曲線 2 の代わり)。
    pub fn set_target_curve(&mut self, db: &[f32]) {
        for k in 0..NB {
            self.curve[k] = self.curve1[k] * 10f64.powf(db[k] as f64 / 10.0);
        }
    }

    /// 登録の時の目標ごとの曲線(training/d666_target_cal.py と同じ式): 参照の実の行のパワー(c0 = 対数の平均・正規化なし)対 表の引き(事後 × 表)を描画と同じ約束で比べ、
    /// 周波数ごとの比 → 1/6 oct で平滑 → 500 Hz を 0 dB → [−6, 12] dB。env = Analyzer.envelope の [T][25]・logits = C1 のロジット [T][K]・table = 登録の表 [24][K]。
    pub fn target_curve(&self, env: &[[f32; 25]], logits: &[Vec<f32>], table: &[f32]) -> Vec<f32> {
        let t = env.len().min(logits.len());
        let k = self.k;
        let fq: Vec<f64> = (0..NB).map(|i| i as f64 * SR as f64 / N as f64).collect();
        let shape = |lm: &[f64]| -> Vec<f64> { fq.iter().map(|&f| (2.0 * interp(f, &self.fc, lm)).exp()).collect() };
        let mut er = vec![vec![0.0f64; NB]; t];
        let mut eo = vec![vec![0.0f64; NB]; t];
        let mut lsb: Option<f64> = None;
        let mut q = vec![0.0f64; k];
        for i in 0..t {
            let c0 = env[i][0] as f64;
            let mut lm = vec![0.0f64; NM];
            for b in 0..NM {
                let mut acc = c0 * self.dct[b];
                for j in 1..25 {
                    acc += env[i][j] as f64 * self.dct[j * NM + b];
                }
                lm[b] = acc;
            }
            er[i] = shape(&lm);
            let mx = logits[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let mut z = 0.0;
            for u in 0..k {
                q[u] = (logits[i][u] as f64 - mx).exp();
                z += q[u];
            }
            for b in 0..NM {
                let mut acc = 0.0;
                for j in 1..25 {
                    let mut tr = 0.0;
                    for u in 0..k {
                        tr += table[(j - 1) * k + u] as f64 * q[u];
                    }
                    acc += tr / z * self.dct[j * NM + b];
                }
                lm[b] = acc;
            }
            let mut e = shape(&lm);
            let s = e.iter().sum::<f64>() / NB as f64;
            let l = match lsb {
                Some(p) => self.a_s * p + (1.0 - self.a_s) * s.ln(),
                None => s.ln(),
            };
            lsb = Some(l);
            let g = l.exp() / s * (2.0 * c0 / (NM as f64).sqrt()).exp();
            e.iter_mut().for_each(|v| *v *= g);
            eo[i] = e;
        }
        let pw: Vec<f64> = er.iter().map(|e| 10.0 * e.iter().sum::<f64>().max(1e-300).log10()).collect();
        let mx = pw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (mut pr, mut po) = (vec![0.0f64; NB], vec![0.0f64; NB]);
        for i in 0..t {
            if pw[i] > mx - 40.0 {
                for kk in 0..NB {
                    pr[kk] += er[i][kk];
                    po[kk] += eo[i][kk];
                }
            }
        }
        let (sr_, so_) = (pr.iter().sum::<f64>(), po.iter().sum::<f64>());
        let db: Vec<f64> = (0..NB).map(|kk| 10.0 * ((pr[kk] / sr_ + 1e-30) / (po[kk] / so_ + 1e-30)).log10()).collect();
        let lf: Vec<f64> = fq.iter().map(|&f| f.max(20.0).log2()).collect();
        let mut sm: Vec<f64> = (0..NB)
            .map(|i| {
                let (mut a, mut n) = (0.0, 0usize);
                for j in 0..NB {
                    if (lf[j] - lf[i]).abs() <= 1.0 / 12.0 {
                        a += db[j];
                        n += 1;
                    }
                }
                a / n as f64
            })
            .collect();
        let r500 = interp(500.0, &fq, &sm);
        sm.iter_mut().for_each(|v| *v = (*v - r500).clamp(-6.0, 12.0));
        let i60 = fq.iter().position(|&f| f >= 60.0).unwrap_or(0);
        let v60 = sm[i60];
        for i in 0..i60 {
            sm[i] = v60;
        }
        sm.into_iter().map(|v| v as f32).collect()
    }

    /// cond = 生の条件 [0 = c0n, 1..25 = 表の引き(+ H1 EQ), 25 = log f0, 26 = 有声, 27..31 = 周期性, 31..31+K = 事後, 31+K.. = C2]・post = 遅い揺らぎ + R の ΔS [24] → out [HOP]。
    pub fn process(&mut self, cond: &[f32], post: &[f32], out: &mut [f32]) {
        let (k, cv) = (self.k, self.cv);
        self.x.clear();
        self.x.extend_from_slice(&cond[31..31 + k]);
        self.x.push(cond[0]);
        self.x.push(cond[25]);
        self.x.push(cond[26]);
        self.x.extend_from_slice(&cond[1..25]);
        self.x.extend_from_slice(&cond[31 + k..31 + k + cv]);
        self.h.resize(self.inp.co_len(), 0.0);
        self.inp.run(&self.x, 1, &mut self.h);
        for b in self.blocks.iter_mut() {
            leaky(&self.h, &mut self.t);
            self.u.resize(self.h.len(), 0.0);
            b.run(&self.t, 1, &mut self.u);
            for (a, &v) in self.h.iter_mut().zip(self.u.iter()) {
                *a += v;
            }
        }
        leaky(&self.h, &mut self.t);
        self.d.resize(24, 0.0);
        self.out.run(&self.t, 1, &mut self.d);
        let a_hp = self.a_hp;
        let ema = self.ema.get_or_insert_with(|| self.d.iter().map(|&v| v as f64).collect());
        let mut rows = [0.0f64; 24];
        for r in 0..24 {
            let dv = self.d[r] as f64;
            ema[r] = a_hp * ema[r] + (1.0 - a_hp) * dv;
            rows[r] = cond[1 + r] as f64 + (dv - ema[r]) + post[r] as f64;
        }
        let c0 = 10.0 * cond[0] as f64 + self.c0_sil;
        let c0s = match self.c0s {
            Some(p) => self.a_c0 * p + (1.0 - self.a_c0) * c0,
            None => c0,
        };
        self.c0s = Some(c0s);
        let c0o = if cond[0] > self.lv_thr {
            let s = match self.lv_s {
                Some(p) => self.a_lv * p + (1.0 - self.a_lv) * c0s,
                None => c0s,
            };
            self.lv_s = Some(s);
            if self.level_gamma >= 1.0 { c0s } else { s + self.level_gamma * (c0s - s) }
        } else {
            c0s
        };
        let v = cond[26] > 0.5;
        let lf = if v { cond[25] as f64 } else { 0.0 };
        self.lfs = if v && self.prev_v { self.a_lf * self.lfs + (1.0 - self.a_lf) * lf } else { lf };
        self.prev_v = v;
        let f0 = if v { 200.0 * self.lfs.exp() } else { 0.0 };
        for b in 0..NM {
            let mut acc = 0.0;
            for j in 1..25 {
                acc += rows[j - 1] * 10.0 * self.dct[j * NM + b];
            }
            self.lm[b] = acc;
            self.fcs[b] = self.fc[b] * self.alpha;
        }
        let mut s = 0.0;
        for kk in 0..NB {
            let fq = kk as f64 * SR as f64 / N as f64;
            let e = (2.0 * interp(fq, &self.fcs, &self.lm)).exp();
            self.sp[kk] = e;
            s += e;
        }
        s /= NB as f64;
        let ls = s.ln();
        let lsb = match self.lsb {
            Some(p) => self.a_s * p + (1.0 - self.a_s) * ls,
            None => ls,
        };
        self.lsb = Some(lsb);
        let g = lsb.exp() / s * (2.0 * c0o / (NM as f64).sqrt()).exp();
        for kk in 0..NB {
            self.sp[kk] *= g * self.curve[kk];
        }
        let mut a2 = [0.0f64; 4];
        for i in 0..4 {
            let hnr = 30.0 * (cond[27 + i] as f64).clamp(0.0, 1.5);
            a2[i] = 1.0 / (1.0 + 10f64.powf(hnr / 10.0));
        }
        for kk in 0..NB {
            self.ap[kk] = if f0 <= 0.0 { 1.0 } else { interp(self.lfq[kk], &self.band_lc, &a2).sqrt().clamp(1e-3, 1.0) };
        }
        let (sp, ap) = (std::mem::take(&mut self.sp), std::mem::take(&mut self.ap));
        self.synth.block(f0, &sp, &ap, out);
        out.iter_mut().for_each(|v| *v *= self.gain);
        self.sp = sp;
        self.ap = ap;
    }
}
