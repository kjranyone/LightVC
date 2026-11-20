//! paper_snv の模型(演奏の取り出し E = training/train_snv_p11.EncS・回路 C = training/train_snv_p1.Circ)のストリーミング推論。
//! 登録(オフライン): 参照 → 構造(大域の要約 z → 各段の FiLM・注意の鍵と値)。E の構造は操作者の登録から、C の構造は目標の参照から(鍵 = 目標の参照に E を通した演奏)。
//! 流し(フレームごと・因果): E([Y, 有声]; 操作者の構造)→ 演奏 a → C([a, lf + k 半音, 有声]; 目標の構造)→ Y(29)。重みと一致検査 = training/export_snv_model.py。

use std::path::Path;

use anyhow::{Context, Result};

use crate::affinity::Team;
use crate::nvoc::Tensors;
use crate::prof::prof;

const CH: usize = 256;
const HEADS: usize = 4;
const DK: usize = 64;
const DV: usize = 16;
const DILS: [usize; 12] = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32];
const ATTN_AFTER: usize = 3;
pub const D_Y: usize = 29;
pub const D_A: usize = 64;

pub(crate) struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
    i: usize,
    o: usize,
}

impl Lin {
    pub(crate) fn new(ts: &Tensors, name: &str) -> Result<Self> {
        let (w, sh) = ts.get(&format!("{name}.weight"))?;
        let (b, _) = ts.get(&format!("{name}.bias"))?;
        Ok(Self { w, b, o: sh[0], i: sh[1] })
    }

    /// 重みを f16 に丸めて f32 で持つ(段 P94・小さい層は帯域に効かないので f32 のまま読む)。
    pub(crate) fn half(&mut self) {
        self.w.iter_mut().for_each(|v| *v = f32_of(f16_of(*v)));
    }

    pub(crate) fn run(&self, x: &[f32], y: &mut [f32]) {
        for o in 0..self.o {
            let row = &self.w[o * self.i..(o + 1) * self.i];
            y[o] = self.b[o] + dot8(row, x);
        }
    }
}

/// 非因果の Conv1d(登録だけ・パディング (k−1)/2): x [ci][t] → y [co][t]。
struct OConv {
    w: Vec<f32>,
    b: Vec<f32>,
    ci: usize,
    co: usize,
    k: usize,
}

impl OConv {
    fn new(ts: &Tensors, name: &str) -> Result<Self> {
        let (w, sh) = ts.get(&format!("{name}.weight"))?;
        let (b, _) = ts.get(&format!("{name}.bias"))?;
        Ok(Self { w, b, co: sh[0], ci: sh[1], k: sh[2] })
    }

    fn run(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (ci, co, k) = (self.ci, self.co, self.k);
        let p = (k - 1) / 2;
        let mut y = vec![0.0f32; co * t];
        for o in 0..co {
            let yo = &mut y[o * t..(o + 1) * t];
            yo.iter_mut().for_each(|v| *v = self.b[o]);
            for c in 0..ci {
                let xc = &x[c * t..(c + 1) * t];
                for j in 0..k {
                    let w = self.w[(o * ci + c) * k + j];
                    let sh = j as isize - p as isize;
                    let (lo, hi) = ((-sh).max(0) as usize, (t as isize - sh).min(t as isize).max(0) as usize);
                    for (yv, xv) in yo[lo..hi].iter_mut().zip(&xc[(lo as isize + sh) as usize..(hi as isize + sh) as usize]) {
                        *yv += w * xv;
                    }
                }
            }
        }
        y
    }
}

#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += ca[i] * cb[i];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    let n = a.len() / 8 * 8;
    for i in n..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn relu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = x.max(0.0));
}
/// 構造(登録の定数): 鍵 k [heads][tr][dk]・値 v [heads][tr][dv]。
pub struct Memory {
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    tr: usize,
    kh: Vec<u16>,
    vh: Vec<u16>,
}

impl Memory {
    /// 鍵と値を f16 でも持ち、フレームごとの表の読みはそちらを読む(段 P94 step c・training/snv_p1_eval の SNV_KV16 と同じ値)。
    pub(crate) fn half(&mut self) {
        self.kh = self.k.iter().map(|&x| f16_of(x)).collect();
        self.vh = self.v.iter().map(|&x| f16_of(x)).collect();
    }
}

/// f32 → f16(最も近い偶数への丸め・非正規化数と無限大を含む・torch の .half() と同じ値)。
pub(crate) fn f16_of(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let sh = (14 - e) as u32;
        let (mut r, rem, half) = (m >> sh, m & ((1 << sh) - 1), 1u32 << (sh - 1));
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

/// f16 → f32(厳密)。
pub(crate) fn f32_of(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let (exp, man) = (((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            let (mut m, mut e) = (man, 113u32);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// 重みの行列 [rows][n](f32 か f16・段 P94)。f16 は読むときに f32 へ広げ、f32 の経路と同じ順で足す(= f16 に丸めた f32 の重みと 1 bit 同じ)。
pub(crate) struct Wm {
    f: Vec<f32>,
    h: Vec<u16>,
}

impl Wm {
    pub(crate) fn new(f: Vec<f32>, half: bool) -> Self {
        if half { Self { h: f.iter().map(|&v| f16_of(v)).collect(), f: Vec::new() } } else { Self { f, h: Vec::new() } }
    }

    pub(crate) fn mv(&self, segs: &[&[f32]], b: Option<&[f32]>, out: &mut [f32]) {
        if self.h.is_empty() { mv(&self.f, segs, b, out) } else { mv16(&self.h, segs, b, out) }
    }

    pub(crate) fn into_f32(self) -> Vec<f32> {
        if self.h.is_empty() { self.f } else { self.h.iter().map(|&v| f32_of(v)).collect() }
    }
}

/// mv の f16 の重みの版(同じ足し順)。
pub(crate) fn mv16(w: &[u16], segs: &[&[f32]], b: Option<&[f32]>, out: &mut [f32]) {
    let n: usize = segs.iter().map(|s| s.len()).sum();
    debug_assert_eq!(w.len(), out.len() * n);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") && segs.iter().all(|s| s.len() % 8 == 0) {
        unsafe { mv16_avx2(w, segs, n, b, out) };
        return;
    }
    let mut row = vec![0.0f32; n];
    for (r, y) in out.iter_mut().enumerate() {
        for (d, &h) in row.iter_mut().zip(&w[r * n..(r + 1) * n]) {
            *d = f32_of(h);
        }
        let mut s = b.map_or(0.0, |b| b[r]);
        let mut o = 0;
        for seg in segs {
            s += dot8(&row[o..o + seg.len()], seg);
            o += seg.len();
        }
        *y = s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mv16_avx2(w: &[u16], segs: &[&[f32]], n: usize, b: Option<&[f32]>, out: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
    let rows = out.len();
    let wp = w.as_ptr();
    let ld = |p: *const u16| _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i));
    let mut r = 0;
    while r + 4 <= rows {
        let mut a = [_mm256_setzero_ps(); 4];
        let mut o = 0;
        for seg in segs {
            let xp = seg.as_ptr();
            let mut c = 0;
            while c < seg.len() {
                let xv = _mm256_loadu_ps(xp.add(c));
                for (q, aq) in a.iter_mut().enumerate() {
                    *aq = _mm256_fmadd_ps(ld(wp.add((r + q) * n + o + c)), xv, *aq);
                }
                c += 8;
            }
            o += seg.len();
        }
        for (q, aq) in a.iter().enumerate() {
            out[r + q] = hsum(*aq) + b.map_or(0.0, |b| b[r + q]);
        }
        r += 4;
    }
    while r < rows {
        let mut a = _mm256_setzero_ps();
        let mut o = 0;
        for seg in segs {
            let mut c = 0;
            while c < seg.len() {
                a = _mm256_fmadd_ps(ld(wp.add(r * n + o + c)), _mm256_loadu_ps(seg.as_ptr().add(c)), a);
                c += 8;
            }
            o += seg.len();
        }
        out[r] = hsum(a) + b.map_or(0.0, |b| b[r]);
        r += 1;
    }
    }
}

/// 行優先の行列 w [rows][n] × x(segs の連結・n = Σ 長さ)+ b → out [rows]。
pub(crate) fn mv(w: &[f32], segs: &[&[f32]], b: Option<&[f32]>, out: &mut [f32]) {
    let n: usize = segs.iter().map(|s| s.len()).sum();
    debug_assert_eq!(w.len(), out.len() * n);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && segs.iter().all(|s| s.len() % 8 == 0) {
        unsafe { mv_avx2(w, segs, n, b, out) };
        return;
    }
    for (r, y) in out.iter_mut().enumerate() {
        let row = &w[r * n..(r + 1) * n];
        let mut s = b.map_or(0.0, |b| b[r]);
        let mut o = 0;
        for seg in segs {
            s += dot8(&row[o..o + seg.len()], seg);
            o += seg.len();
        }
        *y = s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mv_avx2(w: &[f32], segs: &[&[f32]], n: usize, b: Option<&[f32]>, out: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
    let rows = out.len();
    let wp = w.as_ptr();
    let mut r = 0;
    while r + 4 <= rows {
        let mut a = [_mm256_setzero_ps(); 4];
        let mut o = 0;
        for seg in segs {
            let xp = seg.as_ptr();
            let mut c = 0;
            while c < seg.len() {
                let xv = _mm256_loadu_ps(xp.add(c));
                for (q, aq) in a.iter_mut().enumerate() {
                    *aq = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add((r + q) * n + o + c)), xv, *aq);
                }
                c += 8;
            }
            o += seg.len();
        }
        for (q, aq) in a.iter().enumerate() {
            out[r + q] = hsum(*aq) + b.map_or(0.0, |b| b[r + q]);
        }
        r += 4;
    }
    while r < rows {
        let mut a = _mm256_setzero_ps();
        let mut o = 0;
        for seg in segs {
            let mut c = 0;
            while c < seg.len() {
                a = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(r * n + o + c)), _mm256_loadu_ps(seg.as_ptr().add(c)), a);
                c += 8;
            }
            o += seg.len();
        }
        out[r] = hsum(a) + b.map_or(0.0, |b| b[r]);
        r += 1;
    }
    }
}

/// wsum の f16 の値の版(同じ順の積と和)。
unsafe fn wsum16(p: &[f32], v: &[u16], acc: &mut [f32; DV]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
        unsafe { wsum16_avx2(p, v, acc) };
        return;
    }
    for (pt, vr) in p.iter().zip(v.chunks_exact(DV)) {
        for d in 0..DV {
            acc[d] += pt * f32_of(vr[d]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn wsum16_avx2(p: &[f32], v: &[u16], acc: &mut [f32; DV]) {
    use core::arch::x86_64::*;
    unsafe {
        let (mut a0, mut a1) = (_mm256_loadu_ps(acc.as_ptr()), _mm256_loadu_ps(acc.as_ptr().add(8)));
        for (t, &pt) in p.iter().enumerate() {
            let s = _mm256_set1_ps(pt);
            let vp = v.as_ptr().add(t * DV);
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(s, _mm256_cvtph_ps(_mm_loadu_si128(vp as *const __m128i))));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(s, _mm256_cvtph_ps(_mm_loadu_si128(vp.add(8) as *const __m128i))));
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), a0);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), a1);
    }
}

/// Σ_t p[t]·v[t][..DV](注意の値の重み付きの和)。
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
unsafe fn wsum(p: &[f32], v: &[f32], acc: &mut [f32; DV]) {
    for (pt, vr) in p.iter().zip(v.chunks_exact(DV)) {
        for d in 0..DV {
            acc[d] += pt * vr[d];
        }
    }
}

/// 1 段の重み(全チャンネル): 畳み込み wc [2CH][k·CH](列 = j·CH + c・タップ j = 時刻 t − (k−1−j)·d)・出力 wo [CH][CH]・FiLM。
struct BlockW {
    d: usize,
    k: usize,
    wc: Vec<f32>,
    bc: Vec<f32>,
    wo: Vec<f32>,
    bo: Vec<f32>,
    film: Lin,
}

/// rank の持ち分: 出力チャンネル [lo, hi)(畳み込みは a の行 lo..hi とゲートの行 CH+lo..CH+hi)・注意の頭 [h0, h1)。
struct Shard {
    lo: usize,
    hi: usize,
    h0: usize,
    h1: usize,
    wc: Vec<Wm>,
    bc: Vec<Vec<f32>>,
    wo: Vec<Wm>,
    bo: Vec<Vec<f32>>,
    g: Vec<Vec<f32>>,
    s: Vec<Vec<f32>>,
    wq: Wm,
    bq: Vec<f32>,
    wa: Wm,
    ba: Vec<f32>,
    ab: Vec<f32>,
    tmp: Vec<f32>,
    qv: Vec<f32>,
    logit: Vec<f32>,
}

/// 段ごとの並列の手順が触る生のポインタ(書き込みは rank ごとに互いに素な区間・Team::run の前後で同期)。
struct Raw {
    sh: *mut Shard,
    ring: [*mut f32; 12],
    rmask: [usize; 12],
    dil: [usize; 12],
    k: [usize; 12],
    gated: *mut f32,
    ha: *mut f32,
    hf: *mut f32,
    ctx: *mut f32,
    t: usize,
    mk: *const f32,
    mvv: *const f32,
    mk16: *const u16,
    mv16p: *const u16,
    kv16: bool,
    tr: usize,
}

unsafe impl Sync for Raw {}

impl Raw {
    unsafe fn slot(&self, i: usize, lag: usize) -> *mut f32 {
        unsafe { self.ring[i].add((self.t.wrapping_sub(lag) & self.rmask[i]) * CH) }
    }

    unsafe fn phase_a(&self, r: usize, i: usize) {
        unsafe {
            let s = &mut *self.sh.add(r);
            let n = s.hi - s.lo;
            let (d, k) = (self.dil[i], self.k[i]);
            let mut taps: [&[f32]; 4] = [&[]; 4];
            for (j, tp) in taps.iter_mut().enumerate().take(k) {
                *tp = std::slice::from_raw_parts(self.slot(i, (k - 1 - j) * d), CH);
            }
            s.wc[i].mv(&taps[..k], Some(&s.bc[i]), &mut s.ab[..2 * n]);
            let gt = std::slice::from_raw_parts_mut(self.gated.add(s.lo), n);
            for c in 0..n {
                let a = s.ab[c] * (1.0 + s.g[i][c]) + s.s[i][c];
                gt[c] = a.tanh() * (1.0 / (1.0 + (-s.ab[n + c]).exp()));
            }
        }
    }

    unsafe fn phase_b(&self, r: usize, i: usize) {
        unsafe {
            let s = &mut *self.sh.add(r);
            let n = s.hi - s.lo;
            s.wo[i].mv(&[std::slice::from_raw_parts(self.gated, CH)], Some(&s.bo[i]), &mut s.tmp[..n]);
            let src = self.slot(i, 0).add(s.lo);
            let dst = if i == ATTN_AFTER { self.ha.add(s.lo) } else if i == 11 { self.hf.add(s.lo) } else { self.slot(i + 1, 0).add(s.lo) };
            for c in 0..n {
                *dst.add(c) = *src.add(c) + s.tmp[c];
            }
        }
    }

    unsafe fn phase_q(&self, r: usize) {
        unsafe {
            let s = &mut *self.sh.add(r);
            let nq = (s.h1 - s.h0) * DK;
            s.wq.mv(&[std::slice::from_raw_parts(self.ha, CH)], Some(&s.bq), &mut s.qv[..nq]);
            if s.logit.len() < self.tr {
                s.logit.resize(self.tr, 0.0);
            }
            let sc = 1.0 / (DK as f32).sqrt();
            for (li, hd) in (s.h0..s.h1).enumerate() {
                let lg = &mut s.logit[..self.tr];
                if self.kv16 {
                    mv16(std::slice::from_raw_parts(self.mk16.add(hd * self.tr * DK), self.tr * DK), &[&s.qv[li * DK..(li + 1) * DK]], None, lg);
                } else {
                    mv(std::slice::from_raw_parts(self.mk.add(hd * self.tr * DK), self.tr * DK), &[&s.qv[li * DK..(li + 1) * DK]], None, lg);
                }
                let mut mx = f32::NEG_INFINITY;
                for l in lg.iter_mut() {
                    *l *= sc;
                    mx = mx.max(*l);
                }
                let mut z = 0.0f32;
                for l in lg.iter_mut() {
                    *l = (*l - mx).exp();
                    z += *l;
                }
                let mut acc = [0f32; DV];
                if self.kv16 {
                    wsum16(lg, std::slice::from_raw_parts(self.mv16p.add(hd * self.tr * DV), self.tr * DV), &mut acc);
                } else {
                    wsum(lg, std::slice::from_raw_parts(self.mvv.add(hd * self.tr * DV), self.tr * DV), &mut acc);
                }
                let ch = std::slice::from_raw_parts_mut(self.ctx.add(hd * DV), DV);
                let iz = 1.0 / z;
                for d in 0..DV {
                    ch[d] = acc[d] * iz;
                }
            }
        }
    }

    unsafe fn phase_o(&self, r: usize) {
        unsafe {
            let s = &mut *self.sh.add(r);
            let n = s.hi - s.lo;
            s.wa.mv(&[std::slice::from_raw_parts(self.ctx, HEADS * DV)], Some(&s.ba), &mut s.tmp[..n]);
            let (src, dst) = (self.ha.add(s.lo), self.slot(ATTN_AFTER + 1, 0).add(s.lo));
            for c in 0..n {
                *dst.add(c) = *src.add(c) + s.tmp[c];
            }
        }
    }
}

/// 因果の網(E と C で共通の形): inp → 12 段の FiLM つき膨張畳み込み(3 段目の後に参照を読む注意)→ head。
/// 段の出力チャンネルを Team の rank に分ける(rank 数 = 1・2・4)。段 i の入力の履歴 = 環 ring[i] [R][CH](R = 2 の冪 > (k−1)·d)。
pub(crate) struct Net {
    w: Vec<BlockW>,
    inp: Lin,
    head: Lin,
    q: Lin,
    o: Lin,
    shards: Vec<Shard>,
    ring: Vec<Vec<f32>>,
    gated: Vec<f32>,
    ha: Vec<f32>,
    hf: Vec<f32>,
    ctx: Vec<f32>,
    t: usize,
    z: Option<Vec<f32>>,
    half: bool,
}

impl Net {
    pub(crate) fn new(ts: &Tensors, p: &str) -> Result<Self> {
        let mut w = Vec::new();
        for (i, &d) in DILS.iter().enumerate() {
            let (cw, sh) = ts.get(&format!("{p}.blocks.{i}.conv.weight"))?;
            let (bc, _) = ts.get(&format!("{p}.blocks.{i}.conv.bias"))?;
            let (co, ci, k) = (sh[0], sh[1], sh[2]);
            anyhow::ensure!(co == 2 * CH && ci == CH && k <= 4, "blocks.{i}.conv {sh:?}");
            let mut wc = vec![0.0f32; co * k * ci];
            for o in 0..co {
                for c in 0..ci {
                    for j in 0..k {
                        wc[(o * k + j) * ci + c] = cw[(o * ci + c) * k + j];
                    }
                }
            }
            let (wo, _) = ts.get(&format!("{p}.blocks.{i}.out.weight"))?;
            let (bo, _) = ts.get(&format!("{p}.blocks.{i}.out.bias"))?;
            w.push(BlockW { d, k, wc, bc, wo, bo, film: Lin::new(ts, &format!("{p}.blocks.{i}.film"))? });
        }
        let ring = w.iter().map(|b| vec![0.0f32; ((b.k - 1) * b.d + 1).next_power_of_two() * CH]).collect();
        let mut net = Self {
            w, inp: Lin::new(ts, &format!("{p}.inp"))?, head: Lin::new(ts, &format!("{p}.head"))?, q: Lin::new(ts, &format!("{p}.q"))?, o: Lin::new(ts, &format!("{p}.o"))?,
            shards: Vec::new(), ring, gated: vec![0.0; CH], ha: vec![0.0; CH], hf: vec![0.0; CH], ctx: vec![0.0; HEADS * DV], t: 0, z: None, half: false,
        };
        net.split(1);
        Ok(net)
    }

    pub(crate) fn split(&mut self, p: usize) {
        assert!(CH.is_multiple_of(p) && HEADS.is_multiple_of(p), "rank 数 {p}");
        let (n, nh) = (CH / p, HEADS / p);
        let half = self.half;
        let rows = |w: &[f32], width: usize, r: std::ops::Range<usize>| w[r.start * width..r.end * width].to_vec();
        let wm = |v: Vec<f32>| Wm::new(v, half);
        self.shards = (0..p)
            .map(|r| {
                let (lo, hi, h0, h1) = (r * n, (r + 1) * n, r * nh, (r + 1) * nh);
                let kc = |b: &BlockW| b.k * CH;
                Shard {
                    lo, hi, h0, h1,
                    wc: self.w.iter().map(|b| wm([rows(&b.wc, kc(b), lo..hi), rows(&b.wc, kc(b), CH + lo..CH + hi)].concat())).collect(),
                    bc: self.w.iter().map(|b| [&b.bc[lo..hi], &b.bc[CH + lo..CH + hi]].concat()).collect(),
                    wo: self.w.iter().map(|b| wm(rows(&b.wo, CH, lo..hi))).collect(),
                    bo: self.w.iter().map(|b| b.bo[lo..hi].to_vec()).collect(),
                    g: vec![vec![0.0; n]; 12], s: vec![vec![0.0; n]; 12],
                    wq: wm(rows(&self.q.w, CH, h0 * DK..h1 * DK)), bq: self.q.b[h0 * DK..h1 * DK].to_vec(),
                    wa: wm(rows(&self.o.w, HEADS * DV, lo..hi)), ba: self.o.b[lo..hi].to_vec(),
                    ab: vec![0.0; 2 * n], tmp: vec![0.0; n], qv: vec![0.0; nh * DK], logit: Vec::new(),
                }
            })
            .collect();
        if let Some(z) = self.z.clone() {
            self.set_z(&z);
        }
    }

    /// フレームごとに読む重み(段の畳み込みと出力・注意の q と o・inp・head)を f16 にする(段 P94・training/snv_p1_eval.half_weights と同じ値)。
    pub(crate) fn set_half(&mut self) {
        self.half = true;
        self.inp.half();
        self.head.half();
        self.split(self.shards.len().max(1));
    }

    pub(crate) fn set_z(&mut self, z: &[f32]) {
        let mut gs = vec![0.0; 2 * CH];
        for (i, b) in self.w.iter().enumerate() {
            b.film.run(z, &mut gs);
            for s in self.shards.iter_mut() {
                s.g[i].copy_from_slice(&gs[s.lo..s.hi]);
                s.s[i].copy_from_slice(&gs[CH + s.lo..CH + s.hi]);
            }
        }
        self.z = Some(z.to_vec());
    }

    pub(crate) fn reset(&mut self) {
        self.ring.iter_mut().for_each(|r| r.iter_mut().for_each(|v| *v = 0.0));
        self.t = 0;
    }

    pub(crate) fn step(&mut self, team: &Team, x: &[f32], mem: &Memory, out: &mut [f32]) {
        assert_eq!(team.size(), self.shards.len(), "Team と分け方の rank 数");
        let m0 = self.ring[0].len() / CH - 1;
        let s0 = self.t & m0;
        self.inp.run(x, &mut self.ring[0][s0 * CH..(s0 + 1) * CH]);
        let raw = Raw {
            sh: self.shards.as_mut_ptr(),
            ring: std::array::from_fn(|i| self.ring[i].as_mut_ptr()),
            rmask: std::array::from_fn(|i| self.ring[i].len() / CH - 1),
            dil: std::array::from_fn(|i| self.w[i].d),
            k: std::array::from_fn(|i| self.w[i].k),
            gated: self.gated.as_mut_ptr(), ha: self.ha.as_mut_ptr(), hf: self.hf.as_mut_ptr(), ctx: self.ctx.as_mut_ptr(),
            t: self.t, mk: mem.k.as_ptr(), mvv: mem.v.as_ptr(), mk16: mem.kh.as_ptr(), mv16p: mem.vh.as_ptr(), kv16: !mem.kh.is_empty(), tr: mem.tr,
        };
        for i in 0..12 {
            prof!(member M_CONV, {
                team.run(&|r| unsafe { raw.phase_a(r, i) });
                team.run(&|r| unsafe { raw.phase_b(r, i) });
            });
            if i == ATTN_AFTER {
                prof!(member M_ATTN, {
                    team.run(&|r| unsafe { raw.phase_q(r) });
                    team.run(&|r| unsafe { raw.phase_o(r) });
                });
            }
        }
        self.head.run(&self.hf, out);
        self.t += 1;
    }
}

/// 構造の推定(登録・オフライン): genc → 平均 → gz = z・鍵と値。
pub(crate) struct StructEnc {
    genc: [OConv; 3],
    gz: Lin,
    kenc: Option<[OConv; 2]>,
    kmlp: Option<[Lin; 2]>,
    venc: [OConv; 2],
}

impl StructEnc {
    pub(crate) fn new(ts: &Tensors, p: &str, from_frames: bool) -> Result<Self> {
        let oc = |n: &str| OConv::new(ts, &format!("{p}.{n}"));
        Ok(Self {
            genc: [oc("genc.0")?, oc("genc.2")?, oc("genc.4")?],
            gz: Lin::new(ts, &format!("{p}.gz"))?,
            kenc: if from_frames { Some([oc("kenc.0")?, oc("kenc.2")?]) } else { None },
            kmlp: if from_frames { None } else { Some([Lin::new(ts, &format!("{p}.k.0"))?, Lin::new(ts, &format!("{p}.k.2"))?]) },
            venc: [oc("venc.0")?, oc("venc.2")?],
        })
    }

    /// rx [ci][t](構造の入力)・ra [D_A][t](C の鍵の入力 = 参照の演奏・E は None)→ (z, 記憶)。
    pub(crate) fn run(&self, rx: &[f32], ra: Option<&[f32]>, t: usize) -> (Vec<f32>, Memory) {
        let mut g = self.genc[0].run(rx, t);
        relu(&mut g);
        let mut g = self.genc[1].run(&g, t);
        relu(&mut g);
        let g = self.genc[2].run(&g, t);
        let mean: Vec<f32> = (0..CH).map(|c| g[c * t..(c + 1) * t].iter().sum::<f32>() / t as f32).collect();
        let mut z = vec![0.0; self.gz.o];
        self.gz.run(&mean, &mut z);
        let mut kf = vec![0.0f32; HEADS * DK * t];
        if let Some(ke) = &self.kenc {
            let mut h = ke[0].run(rx, t);
            relu(&mut h);
            let kk = ke[1].run(&h, t);
            for hd in 0..HEADS {
                for tt in 0..t {
                    for d in 0..DK {
                        kf[(hd * t + tt) * DK + d] = kk[(hd * DK + d) * t + tt];
                    }
                }
            }
        } else if let (Some(km), Some(ra)) = (&self.kmlp, ra) {
            let (mut xi, mut h1, mut h2) = (vec![0.0; D_A], vec![0.0; km[0].o], vec![0.0; km[1].o]);
            for tt in 0..t {
                for c in 0..D_A {
                    xi[c] = ra[c * t + tt];
                }
                km[0].run(&xi, &mut h1);
                relu(&mut h1);
                km[1].run(&h1, &mut h2);
                for hd in 0..HEADS {
                    kf[(hd * t + tt) * DK..(hd * t + tt + 1) * DK].copy_from_slice(&h2[hd * DK..(hd + 1) * DK]);
                }
            }
        }
        let mut h = self.venc[0].run(rx, t);
        relu(&mut h);
        let vv = self.venc[1].run(&h, t);
        let mut vf = vec![0.0f32; HEADS * DV * t];
        for hd in 0..HEADS {
            for tt in 0..t {
                for d in 0..DV {
                    vf[(hd * t + tt) * DV + d] = vv[(hd * DV + d) * t + tt];
                }
            }
        }
        (z, Memory { k: kf, v: vf, tr: t, kh: Vec::new(), vh: Vec::new() })
    }
}


/// 変換器: 操作者の登録と目標の参照で構造を作り、フレームごとに Y(29)を出す。
pub struct PModel {
    e: Net,
    c: Net,
    se: StructEnc,
    sc: StructEnc,
    team: Team,
    mem_e: Option<Memory>,
    mem_c: Option<Memory>,
    xe: Vec<f32>,
    eo: Vec<f32>,
    xc: Vec<f32>,
}

impl PModel {
    pub fn load(dir: &Path) -> Result<Self> {
        let (ts, _man) = Tensors::load(dir).context("p_model の重み")?;
        Ok(Self {
            e: Net::new(&ts, "E")?, c: Net::new(&ts, "C")?, se: StructEnc::new(&ts, "E", true)?, sc: StructEnc::new(&ts, "C", false)?, team: Team::new(1),
            mem_e: None, mem_c: None, xe: vec![0.0; D_Y + 1], eo: vec![0.0; 2 * D_A], xc: vec![0.0; D_A + 2],
        })
    }

    /// E と C の各段の出力チャンネルを n 個のスレッドに分ける(n = 1・2・4・状態は保つ)。
    pub fn enable_parallel(&mut self, n: usize) {
        self.e.split(n);
        self.c.split(n);
        self.team = Team::new(n);
    }

    fn xin(y: &[f32], v: &[f32], t: usize) -> Vec<f32> {
        let mut x = y.to_vec();
        x.extend_from_slice(&v[..t]);
        x
    }

    /// 操作者の登録: y [D_Y][t]・有声 [t](チャンネル優先)。
    pub fn enroll_operator(&mut self, y: &[f32], v: &[f32], t: usize) -> Vec<f32> {
        let (z, mem) = self.se.run(&Self::xin(y, v, t), None, t);
        self.e.set_z(&z);
        self.e.reset();
        self.mem_e = Some(mem);
        z
    }

    /// 目標の参照: y [D_Y][t]・有声 [t] → 参照の演奏 a_ref [D_A][t](目標自身の E の構造で E を通す)→ C の構造。
    pub fn enroll_target(&mut self, y: &[f32], v: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let rx = Self::xin(y, v, t);
        let (ze, mem_t) = self.se.run(&rx, None, t);
        let zo = self.e.z.take();
        self.e.set_z(&ze);
        self.e.reset();
        let mut a_ref = vec![0.0f32; D_A * t];
        let (mut xi, mut o) = (vec![0.0; D_Y + 1], vec![0.0; 2 * D_A]);
        for tt in 0..t {
            for c in 0..D_Y + 1 {
                xi[c] = rx[c * t + tt];
            }
            self.e.step(&self.team, &xi, &mem_t, &mut o);
            for c in 0..D_A {
                a_ref[c * t + tt] = o[c];
            }
        }
        match zo {
            Some(z) => self.e.set_z(&z),
            None => self.e.z = None,
        }
        self.e.reset();
        let (zc, mem_c) = self.sc.run(y, Some(&a_ref), t);
        self.c.set_z(&zc);
        self.c.reset();
        self.mem_c = Some(mem_c);
        (zc, a_ref)
    }

    /// 1 フレーム: 操作者の Y(29)・有声・C に渡す高さ lf(= log(f0 / 200) + k 半音)→ (演奏 a [D_A], 出力 Y [D_Y])。
    pub fn step(&mut self, y: &[f32], v: f32, lf: f32, a_out: &mut [f32], y_out: &mut [f32]) {
        let (me, mc) = (self.mem_e.as_ref().expect("enroll_operator が先"), self.mem_c.as_ref().expect("enroll_target が先"));
        self.xe[..D_Y].copy_from_slice(y);
        self.xe[D_Y] = v;
        self.e.step(&self.team, &self.xe, me, &mut self.eo);
        a_out.copy_from_slice(&self.eo[..D_A]);
        self.xc[..D_A].copy_from_slice(&self.eo[..D_A]);
        self.xc[D_A] = lf;
        self.xc[D_A + 1] = v;
        self.c.step(&self.team, &self.xc, mc, y_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let m = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let s = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        m / s
    }

    #[test]
    fn f16_matches_hardware_and_rounded_f32() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut xs: Vec<f32> = (0..200_000)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                f32::from_bits(st as u32)
            })
            .filter(|v| v.is_finite())
            .collect();
        xs.extend([0.0, -0.0, 65504.0, 65520.0, 65519.99, 6.1e-5, 5.96e-8, 2.98e-8, 2.99e-8, 1e-10, -1e-10, f32::INFINITY, f32::NEG_INFINITY]);
        for (k, scale) in [1.0f32, 1e-3, 1e-6, 1e3].iter().enumerate() {
            for &x in xs.iter().take(50_000) {
                let v = x * scale;
                let h = f16_of(v);
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("f16c") {
                    let hw = unsafe {
                        use core::arch::x86_64::*;
                        _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(v), _MM_FROUND_TO_NEAREST_INT), 0) as u16
                    };
                    if !v.is_nan() {
                        assert_eq!(h, hw, "f16_of({v:e}) 尺度 {k}");
                    }
                    let back = unsafe {
                        use core::arch::x86_64::*;
                        _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(h as i16)))
                    };
                    assert_eq!(f32_of(h).to_bits(), back.to_bits(), "f32_of({h:#x})");
                }
            }
        }
        let (rows, n) = (37, 256 * 3);
        let w: Vec<f32> = xs.iter().take(rows * n).map(|v| (v.to_bits() % 2000) as f32 / 1000.0 - 1.0).collect();
        let x: Vec<f32> = (0..n).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
        let b: Vec<f32> = (0..rows).map(|i| i as f32 * 0.01).collect();
        let segs: [&[f32]; 3] = [&x[..256], &x[256..512], &x[512..]];
        let wr: Vec<f32> = w.iter().map(|&v| f32_of(f16_of(v))).collect();
        let (mut o1, mut o2) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        mv(&wr, &segs, Some(&b), &mut o1);
        Wm::new(w, true).mv(&segs, Some(&b), &mut o2);
        assert!(o1.iter().zip(&o2).all(|(a, c)| a.to_bits() == c.to_bits()), "f16 の経路と丸めた f32 の重みの経路が 1 bit 同じでない");
    }

    #[test]
    fn parity_with_python_model() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/p_model_export");
        if !dir.join("manifest.json").exists() {
            eprintln!("skip: {} が無い(training/export_snv_model.py)", dir.display());
            return;
        }
        let fx = dir.join("fixture");
        let sh: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx.join("shapes.json")).unwrap()).unwrap();
        let (t, tro, trt) = (sh["T"].as_u64().unwrap() as usize, sh["Tr_op"].as_u64().unwrap() as usize, sh["Tr_t"].as_u64().unwrap() as usize);
        let (sy, sv, slf) = (rd(&fx.join("src_y.f32")), rd(&fx.join("src_v.f32")), rd(&fx.join("src_lf_k.f32")));
        let (pa, pyo) = (rd(&fx.join("a.f32")), rd(&fx.join("yo.f32")));
        let mut seq: Option<Vec<f32>> = None;
        for par in [1usize, 2, 4] {
            let mut m = PModel::load(&dir).unwrap();
            m.enable_parallel(par);
            let zop = m.enroll_operator(&tpose(&rd(&fx.join("op_y.f32")), tro, D_Y), &rd(&fx.join("op_v.f32")), tro);
            let r_zop = rel(&zop, &rd(&fx.join("sE_op_z.f32")));
            let r_kop = rel(&m.mem_e.as_ref().unwrap().k, &rd(&fx.join("sE_op_k.f32")));
            let r_vop = rel(&m.mem_e.as_ref().unwrap().v, &rd(&fx.join("sE_op_v.f32")));
            let (zc, a_ref) = m.enroll_target(&tpose(&rd(&fx.join("ref_y.f32")), trt, D_Y), &rd(&fx.join("ref_v.f32")), trt);
            let r_aref = rel(&a_ref, &tpose(&rd(&fx.join("a_ref.f32")), trt, D_A));
            let r_zc = rel(&zc, &rd(&fx.join("sC_z.f32")));
            let r_kc = rel(&m.mem_c.as_ref().unwrap().k, &rd(&fx.join("sC_k.f32")));
            let r_vc = rel(&m.mem_c.as_ref().unwrap().v, &rd(&fx.join("sC_v.f32")));
            let (mut a_all, mut y_all) = (vec![0.0f32; t * D_A], vec![0.0f32; t * D_Y]);
            let (mut a, mut yo) = (vec![0.0; D_A], vec![0.0; D_Y]);
            let mut ms = Vec::with_capacity(t);
            for tt in 0..t {
                let t0 = std::time::Instant::now();
                m.step(&sy[tt * D_Y..(tt + 1) * D_Y], sv[tt], slf[tt], &mut a, &mut yo);
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
                a_all[tt * D_A..(tt + 1) * D_A].copy_from_slice(&a);
                y_all[tt * D_Y..(tt + 1) * D_Y].copy_from_slice(&yo);
            }
            let (r_a, r_y) = (rel(&a_all, &pa), rel(&y_all, &pyo));
            let same = seq.as_ref().map(|s| s.iter().zip(&y_all).all(|(p, q)| p.to_bits() == q.to_bits()));
            let tot: f64 = ms.iter().sum();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| ms[((ms.len() - 1) as f64 * p) as usize];
            eprintln!("スレッド {par}: 構造 E(操作者) z {r_zop:.2e} 鍵 {r_kop:.2e} 値 {r_vop:.2e} / 参照の演奏 {r_aref:.2e} / 構造 C z {r_zc:.2e} 鍵 {r_kc:.2e} 値 {r_vc:.2e}");
            eprintln!("スレッド {par}: 流し 演奏 a {r_a:.2e} 出力 Y {r_y:.2e}(相対の最大差)・1 スレッドと 1 bit 一致 {same:?}・RTF {:.3}・フレームあたり ms p50 {:.2} p95 {:.2} p99 {:.2}(参照 {trt}・登録 {tro} フレーム)", tot / (t as f64 * 5.0), q(0.5), q(0.95), q(0.99));
            for (name, r) in [("zop", r_zop), ("kop", r_kop), ("vop", r_vop), ("aref", r_aref), ("zc", r_zc), ("kc", r_kc), ("vc", r_vc), ("a", r_a), ("y", r_y)] {
                assert!(r < 2e-3, "{name}: 相対の差 {r}");
            }
            if let Some(s) = same {
                assert!(s, "スレッド {par} の出力が 1 スレッドと違う");
            }
            seq.get_or_insert(y_all);
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod bench {
    use super::mv;
    use core::arch::x86_64::*;

    const ROWS: usize = 512;
    const N: usize = 768;
    const MATS: usize = 22;

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn mv_f16(w: &[u16], x: &[f32], out: &mut [f32]) {
        unsafe {
            for (r, o) in out.chunks_exact_mut(4).enumerate() {
                let mut a = [_mm256_setzero_ps(); 4];
                let mut c = 0;
                while c < N {
                    let xv = _mm256_loadu_ps(x.as_ptr().add(c));
                    for (q, aq) in a.iter_mut().enumerate() {
                        let wv = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add((r * 4 + q) * N + c) as *const __m128i));
                        *aq = _mm256_fmadd_ps(wv, xv, *aq);
                    }
                    c += 8;
                }
                for (q, aq) in a.iter().enumerate() {
                    let mut t = [0f32; 8];
                    _mm256_storeu_ps(t.as_mut_ptr(), *aq);
                    o[q] = t.iter().sum();
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn mv_i8(w: &[i8], x: &[f32], out: &mut [f32]) {
        unsafe {
            for (r, o) in out.chunks_exact_mut(4).enumerate() {
                let mut a = [_mm256_setzero_ps(); 4];
                let mut c = 0;
                while c < N {
                    let xv = _mm256_loadu_ps(x.as_ptr().add(c));
                    for (q, aq) in a.iter_mut().enumerate() {
                        let wi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(w.as_ptr().add((r * 4 + q) * N + c) as *const __m128i));
                        *aq = _mm256_fmadd_ps(_mm256_cvtepi32_ps(wi), xv, *aq);
                    }
                    c += 8;
                }
                for (q, aq) in a.iter().enumerate() {
                    let mut t = [0f32; 8];
                    _mm256_storeu_ps(t.as_mut_ptr(), *aq);
                    o[q] = t.iter().sum();
                }
            }
        }
    }

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed | 1;
        (0..n)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
            })
            .collect()
    }

    /// 1 組ぶんの重み(22 枚 × [512][768] = 34.6 MB の f32)を threads 本のスレッドがそれぞれ別の写しで読む 1 回の通しの時間 ms(最小)。same = 1 枚目だけを 22 回(キャッシュに載る = 計算だけの時間)。
    fn pass(threads: usize, kind: &str, same: bool) -> f64 {
        let x = rnd(N, 7);
        let bar = std::sync::Barrier::new(threads + 1);
        let best = std::sync::Mutex::new(vec![f64::INFINITY; threads]);
        let reps = 30;
        std::thread::scope(|sc| {
            for th in 0..threads {
                let (x, bar, best) = (&x, &bar, &best);
                sc.spawn(move || {
                    crate::affinity::pin_current_to_perf();
                    let f: Vec<f32> = rnd(MATS * ROWS * N, 11 + th as u64);
                    let h: Vec<u16> = if kind == "f16" { f.iter().map(|&v| (v * 32768.0) as i16 as u16).collect() } else { Vec::new() };
                    let b: Vec<i8> = if kind == "i8" { f.iter().map(|&v| (v * 1270.0) as i8).collect() } else { Vec::new() };
                    let mut out = vec![0.0f32; ROWS];
                    let mut sink = 0.0f32;
                    for _ in 0..reps {
                        bar.wait();
                        let t0 = std::time::Instant::now();
                        for m in 0..MATS {
                            let o = if same { 0 } else { m * ROWS * N };
                            match kind {
                                "f16" => unsafe { mv_f16(&h[o..o + ROWS * N], x, &mut out) },
                                "i8" => unsafe { mv_i8(&b[o..o + ROWS * N], x, &mut out) },
                                _ => mv(&f[o..o + ROWS * N], &[x], None, &mut out),
                            }
                            sink += out[m];
                        }
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        let mut g = best.lock().unwrap();
                        g[th] = g[th].min(ms);
                    }
                    std::hint::black_box(sink);
                });
            }
            for _ in 0..reps {
                std::thread::sleep(std::time::Duration::from_millis(3));
                bar.wait();
            }
        });
        let g = best.lock().unwrap();
        g.iter().sum::<f64>() / threads as f64
    }

    /// メモリ帯域が律速かを確かめる: 同じ計算で、重みがキャッシュに載る場合と載らない場合・スレッド数・重みの型(f32 / f16 / int8)を振る。
    /// cargo test --release -p lightvc-core bench_weight_stream -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_weight_stream() {
        let mb = (MATS * ROWS * N * 4) as f64 / 1e6;
        for kind in ["f32", "f16", "i8"] {
            let bytes = mb * match kind { "f16" => 0.5, "i8" => 0.25, _ => 1.0 };
            let c = pass(1, kind, true);
            eprintln!("{kind}: 計算だけ(1 枚を 22 回・1 スレッド){c:.3} ms");
            for th in [1usize, 2, 3, 4, 6] {
                let (s, cc) = (pass(th, kind, false), pass(th, kind, true));
                eprintln!("{kind}: {th} スレッド・組 1 つぶん {bytes:.1} MB の通し {s:.3} ms / スレッド(合計 {:.1} GB/s)・同じ形でキャッシュに載る場合 {cc:.3} ms・比 {:.2}", th as f64 * bytes / s, s / cc);
            }
        }
    }
}
