//! 段 P の前段の門(研究の形 snv_p30_c+mask:<門>・training/train_snv_mask.py の Gated と features)のストリーミング版。製品の既定の前段 v3 と選べる別の前段。
//! 門の網と重みの網(train_snv_mask.MaskNet = train_snv_sep.Front → inp → LayerNorm と因果の膨張畳み込みと GELU の段 → LayerNorm → head)を 5 ms ずつ流す。
//! 門の確率(f64 の sigmoid)≥ thr が on フレーム続いたら開き、< thr が off フレーム続いたら閉じる(train_snv_mask.hysteresis)。
//! 開いたフレームだけ重み √sigmoid(重みのロジット)を p_front::process_block_masked に渡す(閉じたフレームは前段 v3 と同じ計算)。重みと一致検査 = training/export_snv_gate.py。

use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::affinity::Team;
use crate::nvoc::{Tensors, HOP};
use crate::p30_model::{at, CConv};
use crate::prof::prof;
use crate::ship_front::fft;

const CH: usize = 256;
const NFFT: usize = 2048;
const NFFT2: usize = 512;
const D_IN: usize = 256;
pub const NB: usize = NFFT / 2 + 1;

fn layer_norm(h: &[f32], w: &[f32], b: &[f32], eps: f64, out: &mut [f32]) {
    let n = h.len() as f64;
    let m = h.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = h.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n;
    let r = 1.0 / (var + eps).sqrt();
    for (((o, &x), &wi), &bi) in out.iter_mut().zip(h).zip(w).zip(b) {
        *o = ((x as f64 - m) * r * wi as f64 + bi as f64) as f32;
    }
}

fn gelu(v: &mut [f32]) {
    for x in v.iter_mut() {
        let z = *x as f64;
        *x = (0.5 * z * (1.0 + libm::erf(z * std::f64::consts::FRAC_1_SQRT_2))) as f32;
    }
}

fn sigmoid(z: f32) -> f64 {
    1.0 / (1.0 + (-(z as f64)).exp())
}

/// 帯の行列 fb [nb][bands](PyTorch の Pw @ fb)を [bands][nb] と各帯の非 0 の範囲に。
fn bands(fb: &[f32], nb: usize) -> (Vec<f32>, Vec<(usize, usize)>) {
    let nbd = fb.len() / nb;
    let mut t = vec![0.0f32; fb.len()];
    for k in 0..nb {
        for b in 0..nbd {
            t[b * nb + k] = fb[k * nbd + b];
        }
    }
    let rng = (0..nbd)
        .map(|b| {
            let row = &t[b * nb..(b + 1) * nb];
            let lo = row.iter().position(|&w| w != 0.0).unwrap_or(0);
            (lo, row.iter().rposition(|&w| w != 0.0).map_or(lo, |h| h + 1))
        })
        .collect();
    (t, rng)
}

/// train_snv_sep.Front: 左寄せの Hann 2048 と 512 のパワー → 帯(線形 64 + mel 160 | mel 32)→ log(· + 1e-6) / 4。
struct NetFront {
    win: Vec<f64>,
    win2: Vec<f64>,
    fb: Vec<f32>,
    rng: Vec<(usize, usize)>,
    fb2: Vec<f32>,
    rng2: Vec<(usize, usize)>,
    eps: f64,
}

impl NetFront {
    fn new(ts: &Tensors, p: &str, eps: f64) -> Result<Self> {
        let g = |n: &str| ts.get(&format!("{p}.front.{n}"));
        let ((win, sw), (win2, sw2), (fb, sf), (fb2, sf2)) = (g("win")?, g("win2")?, g("fb")?, g("fb2")?);
        ensure!(sw == [NFFT] && sw2 == [NFFT2] && sf[0] == NB && sf2[0] == NFFT2 / 2 + 1 && sf[1] + sf2[1] == D_IN, "{p}.front の形");
        let ((fb, rng), (fb2, rng2)) = (bands(&fb, NB), bands(&fb2, NFFT2 / 2 + 1));
        Ok(Self { win: win.iter().map(|&v| v as f64).collect(), win2: win2.iter().map(|&v| v as f64).collect(), fb, rng, fb2, rng2, eps })
    }

    fn same(&self, o: &Self) -> bool {
        self.win == o.win && self.win2 == o.win2 && self.fb == o.fb && self.fb2 == o.fb2
    }

    /// 直近 NFFT サンプル → 特徴 [D_IN]。
    fn run(&self, raw: &[f32], out: &mut [f32], re: &mut [f64], im: &mut [f64]) {
        let mut o = 0;
        for (n, win, fb, rng) in [(NFFT, &self.win, &self.fb, &self.rng), (NFFT2, &self.win2, &self.fb2, &self.rng2)] {
            let nb = n / 2 + 1;
            let x = &raw[raw.len() - n..];
            for i in 0..n {
                re[i] = x[i] as f64 * win[i];
                im[i] = 0.0;
            }
            fft(&mut re[..n], &mut im[..n]);
            for k in 0..nb {
                re[k] = re[k] * re[k] + im[k] * im[k];
            }
            for (b, &(lo, hi)) in rng.iter().enumerate() {
                let s: f64 = fb[b * nb + lo..b * nb + hi].iter().zip(&re[lo..hi]).map(|(&w, &q)| w as f64 * q).sum();
                out[o + b] = ((s + self.eps).ln() / 4.0) as f32;
            }
            o += rng.len();
        }
    }
}

/// train_snv_mask.MaskNet の inp 以降(段 = LayerNorm → 因果の膨張畳み込み k3 → GELU → 1×1 → 残差)。状態 = 段ごとの LayerNorm の出力の環。
struct MaskNet {
    inp: CConv,
    ln: Vec<(Vec<f32>, Vec<f32>)>,
    conv: Vec<CConv>,
    out: Vec<CConv>,
    lnf: (Vec<f32>, Vec<f32>),
    head: CConv,
    eps: f64,
    ring: Vec<Vec<f32>>,
    h: Vec<f32>,
    z: Vec<f32>,
    tmp: Vec<f32>,
    t: usize,
}

impl MaskNet {
    fn new(ts: &Tensors, p: &str, dils: &[usize], eps: f64) -> Result<Self> {
        let cv = |n: &str, d: usize| CConv::new(ts, &format!("{p}.{n}"), d, 0);
        let ln = |n: &str| -> Result<(Vec<f32>, Vec<f32>)> { Ok((ts.get(&format!("{p}.{n}.weight"))?.0, ts.get(&format!("{p}.{n}.bias"))?.0)) };
        let (_, si) = ts.get(&format!("{p}.inp.weight"))?;
        ensure!(si == [CH, D_IN, 1], "{p}.inp {si:?}");
        let conv = dils.iter().enumerate().map(|(i, &d)| cv(&format!("blocks.{i}.conv"), d)).collect::<Result<Vec<_>>>()?;
        let ring = conv.iter().map(CConv::ring).collect();
        Ok(Self {
            inp: cv("inp", 1)?, ln: (0..dils.len()).map(|i| ln(&format!("blocks.{i}.ln"))).collect::<Result<Vec<_>>>()?, conv,
            out: (0..dils.len()).map(|i| cv(&format!("blocks.{i}.out"), 1)).collect::<Result<Vec<_>>>()?, lnf: ln("ln")?, head: cv("head", 1)?, eps, ring,
            h: vec![0.0; CH], z: vec![0.0; CH], tmp: vec![0.0; CH], t: 0,
        })
    }

    /// 1 フレーム: 特徴 [D_IN] → out があれば head の出力(head は状態を持たないので要るときだけ)。
    fn step(&mut self, x: &[f32], out: Option<&mut [f32]>) {
        let t = self.t;
        self.inp.one(x, &mut self.h);
        for i in 0..self.conv.len() {
            let r = at(self.ring[i].len(), CH, t);
            layer_norm(&self.h, &self.ln[i].0, &self.ln[i].1, self.eps, &mut self.ring[i][r]);
            self.conv[i].run(&self.ring[i], t, &mut self.z);
            gelu(&mut self.z);
            self.out[i].one(&self.z, &mut self.tmp);
            for (h, d) in self.h.iter_mut().zip(&self.tmp) {
                *h += d;
            }
        }
        if let Some(o) = out {
            layer_norm(&self.h, &self.lnf.0, &self.lnf.1, self.eps, &mut self.z);
            self.head.one(&self.z, o);
        }
        self.t += 1;
    }
}

/// 前段の門(門の網・重みの網・ヒステリシス)。
pub struct Gate {
    front: NetFront,
    g: MaskNet,
    m: MaskNet,
    on: usize,
    off: usize,
    thr: f64,
    gamma: f64,
    raw: Vec<f32>,
    feat: Vec<f32>,
    zg: Vec<f32>,
    zm: Vec<f32>,
    w: Vec<f64>,
    open: bool,
    c_on: usize,
    c_off: usize,
    re: Vec<f64>,
    im: Vec<f64>,
    team: Option<Team>,
}

/// 門の網と重みの網を 2 つのスレッドで並べるときの生のポインタ(互いに素な網と出力・Team::run の前後で同期)。
struct Pair(*mut MaskNet, *mut MaskNet, *const f32, *mut f32, *mut f32);

unsafe impl Sync for Pair {}

impl Pair {
    unsafe fn run(&self, r: usize) {
        unsafe {
            let x = std::slice::from_raw_parts(self.2, D_IN);
            if r == 0 { (*self.0).step(x, Some(std::slice::from_raw_parts_mut(self.3, 1))) } else { (*self.1).step(x, Some(std::slice::from_raw_parts_mut(self.4, NB))) }
        }
    }
}

impl Gate {
    /// 前段の書き出しの manifest の cfg.kind = "gate" なら門を読む(kind なし = 前段 v3 だけ = None)。
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).context("前段の manifest")?)?;
        match man["cfg"]["kind"].as_str() {
            None => return Ok(None),
            Some("gate") => {}
            Some(k) => anyhow::bail!("{}: 未知の前段の型 {k}", dir.display()),
        }
        let (ts, man) = Tensors::load(dir).context("門の重み")?;
        let cfg = &man["cfg"];
        let num = |k: &str| cfg[k].as_f64().with_context(|| format!("cfg {k}"));
        let dils = |k: &str| -> Result<Vec<usize>> { cfg[k].as_array().with_context(|| format!("cfg {k}"))?.iter().map(|d| d.as_u64().map(|d| d as usize).context("dil")).collect() };
        let hy = dils("hyst")?;
        ensure!(cfg["dec"].as_str() == Some("myin") && hy.len() == 2, "門の cfg(dec = myin・hyst = [on, off])");
        ensure!(num("ch")? as usize == CH && num("nfft")? as usize == NFFT && num("nfft2")? as usize == NFFT2 && num("d_in")? as usize == D_IN && num("nb_gate")? as usize == 1 && num("nb_mask")? as usize == NB, "門の網の形");
        let (eps, leps) = (num("ln_eps")?, num("log_eps")?);
        let front = NetFront::new(&ts, "gate", leps)?;
        ensure!(front.same(&NetFront::new(&ts, "mask", leps)?), "門の網と重みの網の前段の窓と帯が違う");
        Ok(Some(Self {
            front, g: MaskNet::new(&ts, "gate", &dils("gate_dil")?, eps)?, m: MaskNet::new(&ts, "mask", &dils("mask_dil")?, eps)?, on: hy[0], off: hy[1], thr: num("thr")?, gamma: num("gamma")?,
            raw: vec![0.0; NFFT], feat: vec![0.0; D_IN], zg: vec![0.0; 1], zm: vec![0.0; NB], w: vec![1.0; NB], open: false, c_on: 0, c_off: 0, re: vec![0.0; NFFT], im: vec![0.0; NFFT], team: None,
        }))
    }

    /// 1 ブロック(HOP サンプル)→ 門が開いていれば重み m [NB](振幅)・閉じていれば None(未来の入力を見ない)。
    pub fn step(&mut self, x: &[f32]) -> Option<&[f64]> {
        assert_eq!(x.len(), HOP, "ブロックは HOP サンプル");
        self.raw.copy_within(HOP.., 0);
        self.raw[NFFT - HOP..].copy_from_slice(x);
        prof!(GATE_FEAT, self.front.run(&self.raw, &mut self.feat, &mut self.re, &mut self.im));
        let both = self.team.is_some();
        if let Some(team) = &self.team {
            let p = Pair(&mut self.g, &mut self.m, self.feat.as_ptr(), self.zg.as_mut_ptr(), self.zm.as_mut_ptr());
            prof!(GATE_NET, team.run(&|r| unsafe { p.run(r) }));
        } else {
            prof!(GATE_NET, self.g.step(&self.feat, Some(&mut self.zg)));
        }
        (self.c_on, self.c_off) = if sigmoid(self.zg[0]) >= self.thr { (self.c_on + 1, 0) } else { (0, self.c_off + 1) };
        if !self.open && self.c_on >= self.on {
            self.open = true;
        } else if self.open && self.c_off >= self.off {
            self.open = false;
        }
        if !both {
            prof!(MASK_NET, self.m.step(&self.feat, if self.open { Some(&mut self.zm) } else { None }));
        }
        if !self.open {
            return None;
        }
        prof!(MASK_HEAD, for (w, &z) in self.w.iter_mut().zip(&self.zm) {
            let s = sigmoid(z).sqrt();
            *w = if self.gamma != 1.0 { s.powf(self.gamma) } else { s };
        });
        Some(&self.w)
    }

    /// 門の網と重みの網を 2 つのスレッドで並べる(重みの網の頭は門が閉じていても計算する・状態を持たないので出力は 1 bit 同じ)。
    pub fn enable_parallel(&mut self) {
        self.team = Some(Team::new(2));
    }

    /// 直前のブロックの門のロジット・開いているか。
    pub fn last(&self) -> (f32, bool) {
        (self.zg[0], self.open)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::p_front::PFront;
    use crate::snv_stream::to_y;
    use std::path::PathBuf;

    fn rd(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// results/ の門の書き出し(manifest の cfg.kind = gate・training/export_snv_gate.py)。
    pub(crate) fn gate_exports() -> Vec<PathBuf> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results");
        let mut out: Vec<PathBuf> = std::fs::read_dir(&root)
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        out.retain(|p| {
            p.file_name().is_some_and(|n| n.to_string_lossy().ends_with("_export"))
                && std::fs::read_to_string(p.join("manifest.json")).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()).is_some_and(|m| m["cfg"]["kind"] == "gate")
        });
        out.sort();
        if out.is_empty() {
            eprintln!("skip: results/*_export に門の書き出しが無い(training/export_snv_gate.py)");
        }
        out
    }

    /// 段ごとの一致(門のロジット・開閉・重み・myin の高さ・y / lf / v)。開閉は 1 フレームも割れてはいけない。
    #[test]
    fn parity_with_python_gate() {
        for dir in gate_exports() {
            let fx = dir.join("fixture");
            let sh: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx.join("shapes.json")).unwrap()).unwrap();
            let t_n = sh["T"].as_u64().unwrap() as usize;
            let x = rd(&fx.join("x.f32"));
            let (zg_p, st_p, zm_p, f0_p, y_p, lf_p, v_p) = (rd(&fx.join("gate_logit.f32")), rd(&fx.join("state.f32")), rd(&fx.join("mask_logit.f32")), rd(&fx.join("f0.f32")), rd(&fx.join("y.f32")), rd(&fx.join("lf.f32")), rd(&fx.join("v.f32")));
            let mut gt = Gate::load(&dir).unwrap().unwrap();
            let mut fr = PFront::load(&dir).unwrap();
            let thr = gt.thr;
            let (mut zg_max, mut zg_ref, mut zm_max, mut zm_ref) = (0f32, 0f32, 0f32, 0f32);
            let (mut flips, mut bad, mut f0_same, mut y_max, mut lf_max, mut v_bad, mut n_open) = (Vec::new(), Vec::new(), 0usize, 0f32, 0f32, 0usize, 0usize);
            let mut yy = vec![0.0f32; 29];
            let tm = std::time::Instant::now();
            for t in 0..t_n {
                let blk = &x[t * HOP..(t + 1) * HOP];
                let m = gt.step(blk).map(|m| m.to_vec());
                let (zg, open) = gt.last();
                zg_max = zg_max.max((zg - zg_p[t]).abs());
                zg_ref = zg_ref.max(zg_p[t].abs());
                if (sigmoid(zg) >= thr) != (sigmoid(zg_p[t]) >= thr) {
                    flips.push(t);
                }
                if open != (st_p[t] > 0.5) {
                    bad.push(t);
                }
                if open && st_p[t] > 0.5 {
                    n_open += 1;
                    for (a, b) in gt.zm.iter().zip(&zm_p[t * NB..(t + 1) * NB]) {
                        zm_max = zm_max.max((a - b).abs());
                        zm_ref = zm_ref.max(b.abs());
                    }
                }
                let o = fr.process_block_masked(blk, m.as_deref());
                let (v, lf) = to_y(&o, &mut yy);
                let fp = f0_p[t];
                if (o.f0 > 0.0) == (fp > 0.0) && (o.f0 - fp).abs() <= 1e-3 * fp.max(1.0) {
                    f0_same += 1;
                    for k in 0..29 {
                        y_max = y_max.max((yy[k] - y_p[t * 29 + k]).abs());
                    }
                    lf_max = lf_max.max((lf - lf_p[t]).abs());
                }
                if v != v_p[t] {
                    v_bad += 1;
                }
            }
            let rtf = tm.elapsed().as_secs_f64() / (t_n as f64 * HOP as f64 / 48000.0);
            let agree = f0_same as f64 / t_n as f64;
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            eprintln!("{name}: 門のロジットの相対の最大差 {:.2e}・閾 {thr} の判定が割れたフレーム {} {:?}・開閉が割れたフレーム {} {:?}(開いて一致 {n_open} フレーム)", zg_max / zg_ref, flips.len(), &flips[..flips.len().min(8)], bad.len(), &bad[..bad.len().min(8)]);
            eprintln!("{name}: 重みのロジットの相対の最大差 {:.2e}・myin の高さの一致 {agree:.4}・有声の割れ {v_bad}・y の最大差 {y_max:.2e}・lf の最大差 {lf_max:.2e}(高さが一致したフレーム)・RTF {rtf:.3}(比較込み・{t_n} フレーム)", zm_max / zm_ref);
            assert!(bad.is_empty(), "{name}: 開閉が {} フレームで割れた(最初 {:?})", bad.len(), &bad[..bad.len().min(8)]);
            assert!(zg_max / zg_ref < 2e-3 && zm_max / zm_ref < 2e-3, "{name}: 網の出力の差");
            assert!(agree >= 0.995, "{name}: myin の高さの一致 {agree}");
            assert!(y_max < 1e-3, "{name}: y の差 {y_max}");
        }
    }

    /// 音楽を足さない入力: 門は開かず、前段 v3 と 1 bit 同じ。
    #[test]
    fn clean_input_equals_v3() {
        for dir in gate_exports() {
            let fx = dir.join("fixture");
            let (x, st_p, zg_p) = (rd(&fx.join("clean_x.f32")), rd(&fx.join("clean_state.f32")), rd(&fx.join("clean_gate_logit.f32")));
            let mut gt = Gate::load(&dir).unwrap().unwrap();
            let (mut fg, mut fv) = (PFront::load(&dir).unwrap(), PFront::load(&dir).unwrap());
            let (mut opened, mut diff, mut zg_max) = (0usize, 0usize, 0f32);
            for (t, blk) in x.chunks_exact(HOP).enumerate() {
                let m = gt.step(blk).map(|m| m.to_vec());
                opened += usize::from(m.is_some());
                zg_max = zg_max.max((gt.last().0 - zg_p[t]).abs());
                let (a, b) = (fg.process_block_masked(blk, m.as_deref()), fv.process_block(blk));
                if a.f0.to_bits() != b.f0.to_bits() || a.env.iter().zip(&b.env).any(|(p, q)| p.to_bits() != q.to_bits()) || a.per.iter().zip(&b.per).any(|(p, q)| p.to_bits() != q.to_bits()) {
                    diff += 1;
                }
            }
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            eprintln!("{name}: 音楽なし {} フレーム・開いたフレーム {opened}(Python {})・前段 v3 と違うフレーム {diff}・門のロジットの最大差 {zg_max:.2e}", x.len() / HOP, st_p.iter().filter(|&&v| v > 0.5).count());
            assert_eq!(opened, 0);
            assert_eq!(diff, 0);
        }
    }
}
