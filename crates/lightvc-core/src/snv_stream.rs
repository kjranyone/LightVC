//! paper_snv の製品経路(段 P): 48 kHz・240 サンプルのブロック → 前段 v3(p_front)→ 模型 → rvoc3hi(rvoc)→ 240 サンプル。
//! 模型は書き出しの manifest の cfg.kind で選ぶ: kind なし = 製品の既定 E / C(p_model・P11)・"p30" = 研究の基準の型 snv_p30_c(p30_model)・
//! "k3g" = 研究の候補 K3G(k3g_model = p30 の型の組の平均 + 門が開いたフレームの錨・固定の窓の包絡 p_fixenv を読む組は源と登録の波形からその包絡も作る)。
//! 前段も書き出しの manifest の cfg.kind で選ぶ: kind なし = 前段 v3・"gate" = 門と重みつきの前段(p_gate・研究の形 +mask:<門>)。門は源と操作者の登録だけに掛け、目標の参照は前段 v3(snv_p1_eval.System.prep)。
//! 特徴の変換は training/train_snv_p1.to_y と snv_p1_eval.convert と同じ: Y = [(c0 − C0_SIL)/10, 包絡 1..25 / 10, 周期性 ×有声]・lf = ln(f0 / 200)・C の高さと励起は k 半音上。
//! 登録(操作者・目標の参照)は波形を同じ前段に通した特徴から(発話ごとに新しい前段)。端から端の一致検査の固定の入出力 = training/export_snv_stream.py(p30 は export_snv_p30.py・門は export_snv_gate.py・K3G は export_snv_k3g.py)。
//! アルゴリズム遅延 = ブロック HOP + rvoc の DELAY(前段と模型の先読み 0)。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::k3g_model::K3gModel;
use crate::nvoc::{DELAY, HOP, N_MEL};
use crate::p_fixenv::{FixEnv, D_ENV};
use crate::p_front::{PFrame, PFront};
use crate::p_gate::Gate;
use crate::p30_model::P30Model;
use crate::p_model::{PModel, D_A, D_Y};
use crate::prof::prof;
use crate::rvoc::{Excite, RVoc, D_COND};

const LN2_12: f32 = std::f32::consts::LN_2 / 12.0;

fn c0_sil() -> f32 {
    (1e-5f32).ln() * (N_MEL as f32).sqrt()
}

/// 前段の 1 フレーム → (Y [D_Y]・有声・lf)。
pub(crate) fn to_y(fr: &PFrame, y: &mut [f32]) -> (f32, f32) {
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

/// 固定の窓の包絡 e と前段の y → 符号の特徴 yc [D_Y](包絡 = e を y の尺度に・周期性 = y のまま・snv_p1_eval.System.prep の yc)。
fn to_yc(e: &[f32; D_ENV], y: &[f32], yc: &mut [f32]) {
    yc[0] = (e[0] - c0_sil()) / 10.0;
    for k in 1..D_ENV {
        yc[k] = e[k] / 10.0;
    }
    yc[D_ENV..D_Y].copy_from_slice(&y[D_ENV..D_Y]);
}

/// 登録の発話の特徴(チャンネル優先): Y [D_Y][t]・固定の窓の包絡つきの yc [D_Y][t](その組のある模型だけ)・有声 [t]。
struct Utt {
    y: Vec<f32>,
    yc: Option<Vec<f32>>,
    v: Vec<f32>,
    t: usize,
}

/// 模型の型(manifest の cfg.kind)。
enum Model {
    P11(Box<PModel>),
    P30(Box<P30Model>),
    K3g(Box<K3gModel>),
}

impl Model {
    fn load(dir: &Path) -> Result<Self> {
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).context("模型の manifest")?)?;
        match man["cfg"]["kind"].as_str() {
            None => Ok(Self::P11(Box::new(PModel::load(dir)?))),
            Some("p30") => Ok(Self::P30(Box::new(P30Model::load(dir)?))),
            Some("k3g") => Ok(Self::K3g(Box::new(K3gModel::load(dir)?))),
            Some(k) => anyhow::bail!("{}: 未知の模型の型 {k}", dir.display()),
        }
    }

    fn enable_parallel(&mut self, n: usize) {
        match self {
            Self::P11(m) => m.enable_parallel(n),
            Self::P30(m) => m.enable_parallel(n),
            Self::K3g(m) => m.enable_parallel(n),
        }
    }

    fn fixenv(&self) -> Option<FixEnv> {
        match self {
            Self::K3g(m) => m.fixenv(),
            _ => None,
        }
    }

    fn enroll_operator(&mut self, u: &Utt) {
        match self {
            Self::P11(m) => {
                m.enroll_operator(&u.y, &u.v, u.t);
            }
            Self::P30(m) => m.enroll_operator(&u.y, &u.v, u.t),
            Self::K3g(m) => m.enroll_operator(&u.y, u.yc.as_deref(), &u.v, u.t),
        }
    }

    fn enroll_target(&mut self, u: &Utt) {
        match self {
            Self::P11(m) => {
                m.enroll_target(&u.y, &u.v, u.t);
            }
            Self::P30(m) => {
                m.enroll_target(&u.y, &u.v, u.t);
            }
            Self::K3g(m) => {
                m.enroll_target(&u.y, u.yc.as_deref(), &u.v, u.t);
            }
        }
    }

    /// yc = 固定の窓の包絡つきの符号の特徴・open = 門が開いているフレーム(どちらも K3G だけが読む)。
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, y: &[f32], yc: Option<&[f32]>, v: f32, lf: f32, open: bool, a_out: &mut [f32], y_out: &mut [f32]) {
        match self {
            Self::P11(m) => m.step(y, v, lf, a_out, y_out),
            Self::P30(m) => m.step(y, v, lf, a_out, y_out),
            Self::K3g(m) => m.step(y, yc, v, lf, open, a_out, y_out),
        }
    }
}

pub struct SnvStream {
    front_dir: PathBuf,
    front: PFront,
    gate: Option<Gate>,
    fix: Option<FixEnv>,
    model: Model,
    exc: Excite,
    voc: RVoc,
    k: f32,
    y: Vec<f32>,
    yc: Vec<f32>,
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
        let model = Model::load(model)?;
        Ok(Self {
            front_dir: front.to_path_buf(), front: PFront::load(front)?, gate: Gate::load(front)?, fix: model.fixenv(), model, exc, voc: RVoc::load(voc)?, k: 0.0,
            y: vec![0.0; D_Y], yc: vec![0.0; D_Y], a: vec![0.0; D_A], yo: vec![0.0; D_Y], cond: vec![0.0; D_COND], e: vec![0.0; 2 * HOP], noise: None,
        })
    }

    /// 模型の網(P11 = E / C・p30 = C)を n スレッドに分け(n = 1・2・4・K3G は組を min(n, 組の数) スレッドに分ける)、描画器の最も重い残差ブロックをワーカーへ移す。
    pub fn enable_parallel(&mut self, n: usize) {
        self.model.enable_parallel(n);
        self.voc.enable_parallel();
        if let (true, Some(g)) = (n >= 2, self.gate.as_mut()) {
            g.enable_parallel();
        }
    }

    /// 模型の型("p11" = 製品の既定・"p30" = snv_p30_c・"k3g" = K3G)。
    pub fn model_kind(&self) -> &'static str {
        match self.model {
            Model::P11(_) => "p11",
            Model::P30(_) => "p30",
            Model::K3g(_) => "k3g",
        }
    }

    /// 前段の型("v3" = 製品の既定・"gate" = 門と重みつき)。
    pub fn front_kind(&self) -> &'static str {
        if self.gate.is_some() { "gate" } else { "v3" }
    }

    /// 直前のブロックで門が開いていたか(門の無い前段は false)。
    pub fn gate_open(&self) -> bool {
        self.gate.as_ref().is_some_and(|g| g.last().1)
    }

    /// 固定の半音(運用 = +0 / +5 / +7 / +12)。
    pub fn set_semitones(&mut self, k: f32) {
        self.k = k;
    }

    /// 発話の波形(48 kHz)を新しい前段(gated = 門も新しく通す)に通した特徴。固定の窓の包絡は門に依らず元の波形から(発話ごとに新しい流し)。
    fn analyze(&self, pcm: &[f32], gated: bool) -> Result<Utt> {
        let mut fr = PFront::load(&self.front_dir)?;
        let mut gt = if gated { Gate::load(&self.front_dir)? } else { None };
        let mut fx = self.fix.as_ref().map(FixEnv::fresh);
        let t = pcm.len() / HOP;
        let (mut y, mut v) = (vec![0.0f32; D_Y * t], vec![0.0f32; t]);
        let mut yc = fx.as_ref().map(|_| vec![0.0f32; D_Y * t]);
        let (mut yy, mut yk) = (vec![0.0f32; D_Y], vec![0.0f32; D_Y]);
        for tt in 0..t {
            let blk = &pcm[tt * HOP..(tt + 1) * HOP];
            let m = gt.as_mut().and_then(|g| g.step(blk));
            let f = fr.process_block_masked(blk, m);
            v[tt] = to_y(&f, &mut yy).0;
            for c in 0..D_Y {
                y[c * t + tt] = yy[c];
            }
            if let (Some(fx), Some(yc)) = (fx.as_mut(), yc.as_mut()) {
                to_yc(&fx.step(blk), &yy, &mut yk);
                for c in 0..D_Y {
                    yc[c * t + tt] = yk[c];
                }
            }
        }
        Ok(Utt { y, yc, v, t })
    }

    fn concat(parts: &[Utt]) -> Utt {
        let t: usize = parts.iter().map(|p| p.t).sum();
        let cat = |f: &dyn Fn(&Utt) -> &[f32]| {
            let mut y = vec![0.0f32; D_Y * t];
            let mut o = 0;
            for p in parts {
                for c in 0..D_Y {
                    y[c * t + o..c * t + o + p.t].copy_from_slice(&f(p)[c * p.t..(c + 1) * p.t]);
                }
                o += p.t;
            }
            y
        };
        let yc = parts.iter().all(|p| p.yc.is_some()).then(|| cat(&|p| p.yc.as_deref().unwrap_or(&[])));
        Utt { y: cat(&|p| &p.y), yc, v: parts.iter().flat_map(|p| p.v.iter().copied()).collect(), t }
    }

    /// 操作者の登録(発話の波形の組・48 kHz・門のある前段なら門も通す)。
    pub fn enroll_operator(&mut self, utts: &[&[f32]]) -> Result<()> {
        let parts = utts.iter().map(|u| self.analyze(u, true)).collect::<Result<Vec<_>>>()?;
        self.model.enroll_operator(&Self::concat(&parts));
        Ok(())
    }

    /// 目標の参照(発話の波形の組・48 kHz・前段 v3 のまま)。
    pub fn enroll_target(&mut self, utts: &[&[f32]]) -> Result<()> {
        let parts = utts.iter().map(|u| self.analyze(u, false)).collect::<Result<Vec<_>>>()?;
        self.model.enroll_target(&Self::concat(&parts));
        Ok(())
    }

    /// 励起の雑音を外から与える(一致検査だけ・[n] を順に使う)。
    pub fn set_noise_fixture(&mut self, v: Vec<f32>) {
        self.noise = Some((v, 0));
    }

    /// 入力 [HOP] → 出力 [HOP](未来の入力を見ない)。出力 RVoc の時間 = 入力 − DELAY。
    pub fn process_block(&mut self, x: &[f32], out: &mut [f32]) {
        prof!(TOTAL, self.block(x, out))
    }

    fn block(&mut self, x: &[f32], out: &mut [f32]) {
        let m = prof!(GATE, self.gate.as_mut().and_then(|g| g.step(x)));
        let open = m.is_some();
        let fr = prof!(FRONT, self.front.process_block_masked(x, m));
        let (v, lf) = to_y(&fr, &mut self.y);
        let lfk = lf + self.k * LN2_12 * v;
        if let Some(f) = self.fix.as_mut() {
            prof!(FIXENV, to_yc(&f.step(x), &self.y, &mut self.yc));
        }
        let yc = self.fix.is_some().then_some(self.yc.as_slice());
        prof!(MODEL, self.model.step(&self.y, yc, v, lfk, open, &mut self.a, &mut self.yo));
        self.cond[..25].copy_from_slice(&self.yo[..25]);
        self.cond[25] = lfk;
        self.cond[26] = v;
        for k in 0..4 {
            self.cond[27 + k] = self.yo[25 + k].clamp(0.0, 1.5) * v;
        }
        let f0k = if fr.f0 > 0.0 { fr.f0 * (self.k / 12.0).exp2() } else { 0.0 };
        prof!(EXC, self.exc.block(f0k, &mut self.e));
        if let Some((nz, i)) = self.noise.as_mut() {
            self.e[HOP..].copy_from_slice(&nz[*i..*i + HOP]);
            *i += HOP;
        }
        prof!(VOC, self.voc.process_block(&self.cond, &self.e, out));
    }

    /// 直前のブロックの C の出力 Y [D_Y]。
    pub fn last_y(&self) -> &[f32] {
        &self.yo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p_gate::tests::gate_exports;

    const P11_FX: &str = "results/snv_stream_export/fixture";
    const P30_FX: &str = "results/snv_p30_export/fixture_stream";
    const K3G: &str = "results/snv_k3g_export";
    const K3G16: &str = "results/snv_k3g16_export";
    const K3G16KV: &str = "results/snv_k3g16kv_export";
    const V3: &str = "results/pfront_export";

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn rd(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn open(fx: &Path) -> Option<serde_json::Value> {
        if !fx.join("shapes.json").exists() {
            eprintln!("skip: {} が無い(training/export_snv_stream.py・export_snv_p30.py・export_snv_gate.py・export_snv_k3g.py)", fx.display());
            return None;
        }
        Some(serde_json::from_str(&std::fs::read_to_string(fx.join("shapes.json")).unwrap()).unwrap())
    }

    /// shapes.json の前段(front を与えればその前段)・模型・描画器で組み、fixture の操作者の登録と目標の参照を通す。
    fn build(fx: &Path, sh: &serde_json::Value, front: Option<&Path>) -> SnvStream {
        let r = root();
        let p = |k: &str| r.join(sh[k].as_str().unwrap());
        let mut s = SnvStream::load(&front.map_or_else(|| p("front"), Path::to_path_buf), &p("model"), &p("voc")).unwrap();
        s.set_semitones(sh["k"].as_f64().unwrap() as f32);
        s.enroll_operator(&[&rd(&fx.join("op_x.f32"))]).unwrap();
        let refs: Vec<Vec<f32>> = (0..sh["refs"].as_u64().unwrap()).map(|j| rd(&fx.join(format!("ref{j}_x.f32")))).collect();
        s.enroll_target(&refs.iter().map(|r| r.as_slice()).collect::<Vec<_>>()).unwrap();
        s
    }

    /// (出力・ブロックあたり ms・門が開いていたブロック)。
    fn run(s: &mut SnvStream, x: &[f32], ys: Option<&mut Vec<f32>>) -> (Vec<f32>, Vec<f64>, Vec<bool>) {
        let t = x.len() / HOP;
        let mut out = vec![0.0f32; t * HOP];
        let (mut ms, mut op) = (Vec::with_capacity(t), Vec::with_capacity(t));
        let mut yv = Vec::new();
        for i in 0..t {
            let t0 = std::time::Instant::now();
            s.process_block(&x[i * HOP..(i + 1) * HOP], &mut out[i * HOP..(i + 1) * HOP]);
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            op.push(s.gate_open());
            yv.extend_from_slice(s.last_y());
        }
        if let Some(y) = ys {
            *y = yv;
        }
        (out, ms, op)
    }

    fn quant(ms: &[f64]) -> (f64, f64, f64, f64) {
        let mut v = ms.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
        (v.iter().sum::<f64>() / (v.len() as f64 * 5.0), q(0.5), q(0.95), v[v.len() - 1])
    }

    /// 振幅のスペクトログラムの相対の差 ‖|A| − |B|‖ / ‖|B|‖(STFT: Hann 1024・hop 240・位相に依らない比べ)。
    fn spec_conv(a: &[f32], b: &[f32]) -> f64 {
        let n = 1024;
        let win: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()).collect();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let (mut ra, mut ia, mut rb, mut ib) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut st = 0;
        while st + n <= a.len().min(b.len()) {
            for i in 0..n {
                (ra[i], ia[i], rb[i], ib[i]) = (a[st + i] as f64 * win[i], 0.0, b[st + i] as f64 * win[i], 0.0);
            }
            crate::fft::fft(&mut ra, &mut ia);
            crate::fft::fft(&mut rb, &mut ib);
            for k in 0..=n / 2 {
                let (ma, mb) = ((ra[k] * ra[k] + ia[k] * ia[k]).sqrt(), (rb[k] * rb[k] + ib[k] * ib[k]).sqrt());
                num += (ma - mb) * (ma - mb);
                den += mb * mb;
            }
            st += HOP;
        }
        (num / den.max(1e-30)).sqrt()
    }

    /// 端から端の一致(Python の前段 → 模型 → rvoc3hi と同じ波形)。strict = 波形を標本ごとの線で比べる範囲の終わり(None = 全部)。
    /// 門つきは最初に門が開くまでを標本ごとの線で、開いた後を Y の線と振幅のスペクトログラムの相対の差 < 1e-3 で比べる:
    /// 開いた後は重みの網の f32 の丸めで myin の高さが相対で最大 5.5e-6 ずれ、励起の位相に積もって波形が違う(RMS の 1.1 %・振幅のスペクトログラムは 2.0e-4)= 標本ごとの線は掛けない。
    fn parity(fx: &Path, kind: &str, front: &str, strict: Option<usize>) {
        let Some(sh) = open(fx) else { return };
        let mut s = build(fx, &sh, None);
        assert_eq!((s.model_kind(), s.front_kind()), (kind, front));
        s.set_noise_fixture(rd(&fx.join("noise.f32")));
        let x = rd(&fx.join("src_x.f32"));
        let mut ys = Vec::new();
        let (out, _, op) = run(&mut s, &x, Some(&mut ys));
        let (w, yo) = (rd(&fx.join("w.f32")), rd(&fx.join("yo.f32")));
        let ry = ys.iter().zip(&yo).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / yo.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let n = strict.unwrap_or(out.len()).min(out.len());
        let md = out[..n].iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rms = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
        eprintln!("端から端({kind}・前段 {front}): C の出力 Y の相対の最大差 {ry:.2e}・波形 max|Rust − Python| {md:.3e}(先頭 {n} 標本・RMS {rms:.3e})・{} フレーム(門が開いたフレーム {})", x.len() / HOP, op.iter().filter(|&&o| o).count());
        let sc = (n < out.len()).then(|| {
            let ma = out[n..].iter().zip(&w[n..]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let sc = spec_conv(&out[n..], &w[n..]);
            eprintln!("端から端({kind}・前段 {front}): 門が開いた後({n} 標本〜)の振幅のスペクトログラムの相対の差 {sc:.2e}(線 1e-3)・波形 max 差 {ma:.3e}(RMS の {:.2} %・位相のずれ = 標本ごとの線は掛けない)", 100.0 * ma / rms);
            sc
        });
        assert!(ry < 2e-3, "Y の差 {ry}");
        assert!(md < 1e-3 * rms + 1e-5, "波形の差 {md}");
        assert!(sc.is_none_or(|sc| sc < 1e-3), "門が開いた後の振幅のスペクトログラムの差 {sc:?}");
    }

    /// 門の書き出しの fixture の最初に門が開くフレーム。
    fn first_open(d: &Path) -> usize {
        let sh: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("fixture/shapes.json")).unwrap()).unwrap();
        sh["transitions"].as_array().unwrap().iter().find(|v| v[1].as_u64() == Some(1)).and_then(|v| v[0].as_u64()).expect("開く遷移") as usize
    }

    /// 出荷の関門: 未来不変性(実音声・入力の n 以降を別の実音声に差し替えて n より前の出力が 1 bit も動かない・n 以降が動く)・静的な遅延の台帳・RTF。
    fn gate(fx: &Path, kind: &str, front: &str, extra: &[usize]) {
        let Some(sh) = open(fx) else { return };
        let (x, alt) = (rd(&fx.join("src_x.f32")), rd(&fx.join("alt_x.f32")));
        let t = (x.len() / HOP).min(alt.len() / HOP);
        let (o1, _, op) = run(&mut build(fx, &sh, None), &x[..t * HOP], None);
        let mut worst = 0usize;
        for &cut in [t / 4, t / 2, 3 * t / 4].iter().chain(extra) {
            let n = cut * HOP;
            let mut xe = x[..t * HOP].to_vec();
            xe[n..].copy_from_slice(&alt[n..t * HOP]);
            let (o2, _, _) = run(&mut build(fx, &sh, None), &xe, None);
            let first = o1.iter().zip(&o2).position(|(a, b)| a.to_bits() != b.to_bits());
            let Some(first) = first else { panic!("INCONCLUSIVE: 差し替えで出力が動かない(n = {n})") };
            let la = n.saturating_sub(first);
            worst = worst.max(la);
            eprintln!("未来不変性({kind}・前段 {front}): 差し替え {n} サンプル目(門 {})→ 最初に動く出力 {first}(先読み {la} サンプル)", if op[cut.saturating_sub(1)] { "開" } else { "閉" });
        }
        assert_eq!(worst, 0, "先読み {worst} サンプル");
        let lat_ms = SnvStream::LATENCY_SAMPLES as f64 / 48.0;
        let ledger = lat_ms + 10.0;
        eprintln!("静的な遅延の台帳({kind}・前段 {front}): ブロック {HOP} + rvoc の DELAY {DELAY} = {lat_ms:.1} ms + 内容の符号器の予備 10 ms = {ledger:.1} ms(< 30 ms)");
        assert!(ledger < 30.0);
        if front == "v3" {
            rtf(fx, &sh, kind, "");
        }
    }

    /// RTF とブロックあたりの時間(逐次と並列 2・4)。
    fn rtf(fx: &Path, sh: &serde_json::Value, kind: &str, note: &str) {
        let x = rd(&fx.join("src_x.f32"));
        for par in [0usize, 2, 4] {
            let mut s = build(fx, sh, None);
            assert_eq!(s.model_kind(), kind);
            if par > 0 {
                s.enable_parallel(par);
            }
            let (_, ms, op) = run(&mut s, &x, None);
            let (rtf, p50, p95, mx) = quant(&ms);
            eprintln!("RTF({kind}{note}・並列 {par}・0 = 逐次): {rtf:.3}・ブロックあたり ms p50 {p50:.2} p95 {p95:.2} 最大 {mx:.2}(予算 5 ms)");
            let (open, shut): (Vec<_>, Vec<_>) = ms.iter().zip(&op).partition(|(_, o)| **o);
            if !open.is_empty() {
                let f = |v: &[(&f64, &bool)]| quant(&v.iter().map(|(m, _)| **m).collect::<Vec<_>>());
                let ((_, o50, o95, _), (_, c50, c95, _)) = (f(&open), f(&shut));
                eprintln!("  門が開いたブロック {}: p50 {o50:.2} p95 {o95:.2} ms・閉じたブロック {}: p50 {c50:.2} p95 {c95:.2} ms", open.len(), shut.len());
            }
        }
    }

    /// 門つきと前段 v3 の同じ模型を同じ入力でブロックごとに交互に回した RTF(混んだ機械での比べ)。
    fn rtf_pair(fx: &Path, sh: &serde_json::Value) {
        let x = rd(&fx.join("src_x.f32"));
        let v3 = root().join(V3);
        for par in [0usize, 2, 4] {
            let (mut a, mut b) = (build(fx, sh, None), build(fx, sh, Some(&v3)));
            if par > 0 {
                a.enable_parallel(par);
                b.enable_parallel(par);
            }
            let (mut oa, mut ob) = (vec![0.0f32; HOP], vec![0.0f32; HOP]);
            let (mut ma, mut mb) = (Vec::new(), Vec::new());
            for (i, blk) in x.chunks_exact(HOP).enumerate() {
                for k in 0..2 {
                    let (s, o, m) = if (i + k) % 2 == 0 { (&mut a, &mut oa, &mut ma) } else { (&mut b, &mut ob, &mut mb) };
                    let t0 = std::time::Instant::now();
                    s.process_block(blk, o);
                    m.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let ((ra, a50, a95, amx), (rb, b50, b95, bmx)) = (quant(&ma), quant(&mb));
            eprintln!("RTF(並列 {par}・同じ走行で交互): 門つき {ra:.3}(p50 {a50:.2} p95 {a95:.2} 最大 {amx:.2} ms)・前段 v3 {rb:.3}(p50 {b50:.2} p95 {b95:.2} 最大 {bmx:.2} ms)・比 {:.3}", ra / rb);
        }
    }

    #[test]
    fn parity_end_to_end() {
        parity(&root().join(P11_FX), "p11", "v3", None);
    }

    #[test]
    fn parity_end_to_end_p30() {
        parity(&root().join(P30_FX), "p30", "v3", None);
    }

    #[test]
    fn parity_end_to_end_gate() {
        for d in gate_exports() {
            parity(&d.join("fixture_stream"), "p30", "gate", Some(first_open(&d) * HOP));
        }
    }

    #[test]
    fn ship_gate() {
        gate(&root().join(P11_FX), "p11", "v3", &[]);
    }

    #[test]
    fn ship_gate_p30() {
        gate(&root().join(P30_FX), "p30", "v3", &[]);
    }

    /// 門つきの前段の関門: 門が開いている区間の中の差し替えを足す(fixture の最初の開いた区間の真ん中)。RTF は前段 v3 との交互の比べ。
    #[test]
    fn ship_gate_gate() {
        for d in gate_exports() {
            let fs = d.join("fixture_stream");
            let Some(sh) = open(&fs) else { continue };
            let tr: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("fixture/shapes.json")).unwrap()).unwrap();
            let tr: Vec<(usize, bool)> = tr["transitions"].as_array().unwrap().iter().map(|v| (v[0].as_u64().unwrap() as usize, v[1].as_u64().unwrap() == 1)).collect();
            let on = tr.iter().find(|v| v.1).map(|v| v.0).expect("開く遷移");
            let off = tr.iter().find(|v| !v.1 && v.0 > on).map(|v| v.0).expect("閉じる遷移");
            gate(&fs, "p30", "gate", &[(on + off) / 2, on + 1]);
            rtf_pair(&fs, &sh);
        }
    }

    /// K3G の端から端: 音楽なし(門が開かない = 全部を標本ごとの線)と音楽つき(門が開く = 錨の経路・開くまでを標本ごとの線、開いた後を Y の線と振幅のスペクトログラム)。
    #[test]
    fn parity_end_to_end_k3g() {
        k3g_parity(&root().join(K3G));
    }

    /// 同じ試験を、回路と CPC の重みを f16 で持つ K3G(段 P94・training/export_snv_k3g.py を SNV_WF16=1 で書き出したもの)で。
    #[test]
    fn parity_end_to_end_k3g16() {
        k3g_parity(&root().join(K3G16));
    }

    /// 同じ試験を、重みに加えて注意の鍵と値も f16 で持つ K3G(段 P94 step c・SNV_WF16=1 SNV_KV16=1 の書き出し)で。
    #[test]
    fn parity_end_to_end_k3g16kv() {
        k3g_parity(&root().join(K3G16KV));
    }

    /// 並べた経路(模型の組・描画器のワーカー・門の網と重みの網)の出力が、逐次と 1 bit 同じ(K3G の f16・f32 と門つきの p30・音楽つきと音楽なし)。
    #[test]
    fn parallel_bit_identical() {
        let mut fxs: Vec<PathBuf> = [K3G16KV, K3G16, K3G].iter().flat_map(|d| ["fixture_stream", "fixture_stream_music"].map(|n| root().join(d).join(n))).collect();
        fxs.extend(gate_exports().into_iter().map(|d| d.join("fixture_stream")));
        for fx in fxs {
            let Some(sh) = open(&fx) else { continue };
            let x = rd(&fx.join("src_x.f32"));
            let mut outs = Vec::new();
            for par in [0usize, 2, 4] {
                let mut s = build(&fx, &sh, None);
                if par > 0 {
                    s.enable_parallel(par);
                }
                s.set_noise_fixture(rd(&fx.join("noise.f32")));
                let mut ys = Vec::new();
                let (out, _, _) = run(&mut s, &x, Some(&mut ys));
                outs.push((par, out, ys));
            }
            for (par, out, ys) in &outs[1..] {
                let same = out.iter().zip(&outs[0].1).all(|(a, b)| a.to_bits() == b.to_bits()) && ys.iter().zip(&outs[0].2).all(|(a, b)| a.to_bits() == b.to_bits());
                eprintln!("並べた経路({}・並列 {par}): 逐次と 1 bit 同じ {same}", fx.display());
                assert!(same && out.len() == outs[0].1.len(), "{}: 並列 {par} が逐次と違う", fx.display());
            }
        }
    }

    /// 関門 S(1 ブロックの p99 < 5 ms・4 並列)を、K3G(f16)と対照(門つきの p30)を同じ時間帯に交互に LVC_S_N 回(既定 6)測る(段 P94)。
    /// cargo test --release -p lightvc-core gate_s_k3g16 -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore]
    fn gate_s_k3g16() {
        let n: usize = std::env::var("LVC_S_N").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let k3g = std::env::var("LVC_K3G").unwrap_or_else(|_| K3G16.to_string());
        let mut fxs = vec![(format!("{k3g} 音楽つき"), root().join(&k3g).join("fixture_stream_music")), (format!("{k3g} 音楽なし"), root().join(&k3g).join("fixture_stream"))];
        fxs.extend(gate_exports().into_iter().take(1).map(|d| ("p30 + 門(対照)".to_string(), d.join("fixture_stream"))));
        let mut worst: Vec<(f64, f64)> = vec![(0.0, 0.0); fxs.len()];
        for rep in 0..n {
            for (i, (name, fx)) in fxs.iter().enumerate() {
                let Some(sh) = open(fx) else { continue };
                let x = rd(&fx.join("src_x.f32"));
                let mut s = build(fx, &sh, None);
                s.enable_parallel(4);
                let (_, ms, _) = run(&mut s, &x, None);
                let mut v = ms.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
                let rtf = v.iter().sum::<f64>() / (v.len() as f64 * 5.0);
                eprintln!("関門 S({name}・回 {rep}・並列 4): RTF {rtf:.3}・p50 {:.2} p95 {:.2} p99 {:.2} 最大 {:.2} ms", q(0.5), q(0.95), q(0.99), v[v.len() - 1]);
                worst[i] = (worst[i].0.max(q(0.99)), worst[i].1.max(rtf));
            }
        }
        for ((name, _), (p99, rtf)) in fxs.iter().zip(&worst) {
            eprintln!("関門 S の最悪({name}・{n} 回): p99 {p99:.2} ms(線 < 5 ms)・RTF {rtf:.3}");
        }
    }

    fn k3g_parity(d: &Path) {
        if open(&d.join("fixture_stream")).is_none() {
            return;
        }
        parity(&d.join("fixture_stream"), "k3g", "gate", None);
        if open(&d.join("fixture_stream_music")).is_some() {
            parity(&d.join("fixture_stream_music"), "k3g", "gate", Some(first_open(d) * HOP));
        }
    }

    /// K3G の関門(音楽つきの源 = 門が閉じた区間と開いた区間の両方): 未来不変性(既定の 3 点 = 開く前・開いている間・閉じた後に、開いた区間の真ん中・開いた直後・閉じた直後を足す)・遅延の台帳・RTF(逐次と並列)。
    #[test]
    fn ship_gate_k3g() {
        k3g_gate(&root().join(K3G));
    }

    /// 同じ関門を、重みを f16 で持つ K3G(段 P94)で。
    #[test]
    fn ship_gate_k3g16() {
        k3g_gate(&root().join(K3G16));
    }

    /// 同じ関門を、鍵と値も f16 で持つ K3G(段 P94 step c)で。
    #[test]
    fn ship_gate_k3g16kv() {
        k3g_gate(&root().join(K3G16KV));
    }

    fn k3g_gate(d: &Path) {
        let fm = d.join("fixture_stream_music");
        let Some(sm) = open(&fm) else { return };
        let tr: Vec<(usize, bool)> = sm["transitions"].as_array().unwrap().iter().map(|v| (v[0].as_u64().unwrap() as usize, v[1].as_u64().unwrap() == 1)).collect();
        let on = tr.iter().find(|v| v.1).map(|v| v.0).expect("開く遷移");
        let off = tr.iter().find(|v| !v.1 && v.0 > on).map(|v| v.0).expect("閉じる遷移");
        gate(&fm, "k3g", "gate", &[(on + off) / 2, on + 1, off + 1]);
        rtf(&fm, &sm, "k3g", "・前段 gate・音楽つき");
    }

    /// 全ての fixture の Rust の出力(波形と Y・逐次と 4 並列)を LVC_DUMP のフォルダへ書く(最適化の前後で 1 bit 同じかを cmp で確かめる):
    /// LVC_DUMP=<dir> cargo test --release -p lightvc-core dump_outputs -- --ignored
    #[test]
    #[ignore]
    fn dump_outputs() {
        let Some(dir) = std::env::var_os("LVC_DUMP").map(PathBuf::from) else { return };
        std::fs::create_dir_all(&dir).unwrap();
        let mut fxs = vec![("p11".to_string(), root().join(P11_FX)), ("p30".to_string(), root().join(P30_FX)), ("k3g".to_string(), root().join(K3G).join("fixture_stream")), ("k3g_music".to_string(), root().join(K3G).join("fixture_stream_music"))];
        fxs.extend(gate_exports().into_iter().map(|d| (format!("gate_{}", d.file_name().unwrap().to_string_lossy()), d.join("fixture_stream"))));
        for (name, fx) in fxs {
            let Some(sh) = open(&fx) else { continue };
            let x = rd(&fx.join("src_x.f32"));
            for par in [0usize, 4] {
                let mut s = build(&fx, &sh, None);
                if par > 0 {
                    s.enable_parallel(par);
                }
                s.set_noise_fixture(rd(&fx.join("noise.f32")));
                let mut ys = Vec::new();
                let (out, _, _) = run(&mut s, &x, Some(&mut ys));
                let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
                std::fs::write(dir.join(format!("{name}_par{par}_w.f32")), bytes(&out)).unwrap();
                std::fs::write(dir.join(format!("{name}_par{par}_y.f32")), bytes(&ys)).unwrap();
            }
        }
    }

    /// K3G の段ごとの時間の内訳(逐次・2・4 並列 × 音楽つき・音楽なし・ブロックあたりの平均 ms。組の中の段はスレッドごとの時間の和)。時間がかかるので明示して回す:
    /// cargo test --release -p lightvc-core profile_k3g -- --ignored --nocapture
    #[test]
    #[ignore]
    fn profile_k3g() {
        use crate::prof as pf;
        let k3g = std::env::var("LVC_K3G").unwrap_or_else(|_| K3G.to_string());
        for name in ["fixture_stream_music", "fixture_stream"] {
            let fx = root().join(&k3g).join(name);
            let Some(sh) = open(&fx) else { return };
            let x = rd(&fx.join("src_x.f32"));
            let t = (x.len() / HOP) as f64;
            for par in [0usize, 2, 4] {
                let mut s = build(&fx, &sh, None);
                if par > 0 {
                    s.enable_parallel(par);
                }
                pf::reset();
                pf::enable(true);
                let (_, ms, _) = run(&mut s, &x, None);
                pf::enable(false);
                let ns = pf::snapshot();
                let v = |i: usize| ns[i] as f64 / t / 1e6;
                let (rtf, p50, p95, mx) = quant(&ms);
                eprintln!("内訳({name}・並列 {par}): RTF {rtf:.3}・p50 {p50:.2} p95 {p95:.2} 最大 {mx:.2} ms・段の平均 ms / ブロック:");
                eprintln!("  全体 {:.3} = 門 {:.3}(特徴 {:.3}・門の網 {:.3}・重みの網 {:.3}・重み {:.3})+ 前段 {:.3}(詰め替え {:.3}・CMNDF {:.3}・低域 {:.3}・候補と DP {:.3}・パワー {:.3}・包絡 {:.3}・周期性 {:.3})+ 固定の窓 {:.3}",
                          v(pf::TOTAL), v(pf::GATE), v(pf::GATE_FEAT), v(pf::GATE_NET), v(pf::MASK_NET), v(pf::MASK_HEAD), v(pf::FRONT), v(pf::F_BUF), v(pf::F_CMNDF), v(pf::F_LOWBAND), v(pf::F_DP), v(pf::F_POWER), v(pf::F_ENV), v(pf::F_PER), v(pf::FIXENV));
                eprintln!("  + 模型 {:.3}(平均と錨 {:.3})+ 励起 {:.3} + 描画器 {:.3}(前 {:.3}・補間と励起の畳み込み {:.3}・残差 {:.3}・ワーカー待ち {:.3}・後 {:.3})",
                          v(pf::MODEL), v(pf::MIX), v(pf::EXC), v(pf::VOC), v(pf::VOC_PRE), v(pf::VOC_UP), v(pf::VOC_RB), v(pf::VOC_WAIT), v(pf::VOC_POST));
                for i in 0..3 {
                    let m = |k: usize| v(pf::MEMBER + i * pf::PER_MEMBER + k);
                    eprintln!("  組 {i}: {:.3} = CPC {:.3} + 演奏の読み {:.3} + 回路の畳み込み {:.3} + 回路の表の読み {:.3}", m(pf::M_ALL), m(pf::M_CPC), m(pf::M_PLAY), m(pf::M_CONV), m(pf::M_ATTN));
                }
            }
        }
    }

    /// 音楽を足さない入力(R1 の fixture_stream)では、門つきの前段の出力が前段 v3 の p30 と 1 bit 同じ(登録の門も開かない)。
    #[test]
    fn clean_gate_equals_p30() {
        let fx = root().join(P30_FX);
        let Some(sh) = open(&fx) else { return };
        let x = rd(&fx.join("src_x.f32"));
        let w = rd(&fx.join("w.f32"));
        for d in gate_exports() {
            let (mut a, mut b) = (build(&fx, &sh, Some(&d)), build(&fx, &sh, None));
            assert_eq!((a.front_kind(), b.front_kind()), ("gate", "v3"));
            a.set_noise_fixture(rd(&fx.join("noise.f32")));
            b.set_noise_fixture(rd(&fx.join("noise.f32")));
            let ((oa, _, op), (ob, _, _)) = (run(&mut a, &x, None), run(&mut b, &x, None));
            let diff = oa.iter().zip(&ob).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            let md = oa.iter().zip(&w).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
            eprintln!("{}: 音楽なしの源 {} フレーム・門が開いたフレーム {}・前段 v3 の p30 と違う標本 {diff}・R1 の Python の波形との max 差 {md:.3e}", d.file_name().unwrap().to_string_lossy(), x.len() / HOP, op.iter().filter(|&&o| o).count());
            assert_eq!(diff, 0);
        }
    }
}
