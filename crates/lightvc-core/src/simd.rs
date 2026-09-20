//! Hand-written SIMD kernels for the frequency-structured trunk.
//!
//! Candle's CPU path costs ~10x what these tensors need: measured V1D at
//! dim256 ran **slower** in candle (p95 0.43) than in PyTorch (0.24), and the
//! frequency-convolutional trunk needs 106 GFLOPS to hit the RTF budget. That
//! is ~40% of this machine's AVX2+FMA peak, so it is reachable — but only with
//! kernels written for these exact shapes.
//!
//! Layout is `[C][T][F]` with **F contiguous**: F is 257 and T is 2, so the
//! vector axis has to be F. Every inner loop is then a broadcast-scalar times
//! an F-vector FMA, which is the shape AVX2 is best at.
//!
//! MIT-only project: no external SIMD crate, just `core::arch`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `[C][T][F]` tensor with F contiguous.
#[derive(Clone)]
pub struct Ten {
    pub c: usize,
    pub t: usize,
    pub f: usize,
    pub d: Vec<f32>,
}

impl Ten {
    pub fn zeros(c: usize, t: usize, f: usize) -> Self {
        Self { c, t, f, d: vec![0.0; c * t * f] }
    }
    #[inline]
    pub fn at(&self, c: usize, t: usize) -> &[f32] {
        let o = (c * self.t + t) * self.f;
        &self.d[o..o + self.f]
    }
    #[inline]
    pub fn at_mut(&mut self, c: usize, t: usize) -> &mut [f32] {
        let o = (c * self.t + t) * self.f;
        &mut self.d[o..o + self.f]
    }
}

/// `dst[i] += s * src[i]` over `n` floats.
#[inline]
fn axpy_scalar(dst: &mut [f32], src: &[f32], s: f32) {
    for (d, x) in dst.iter_mut().zip(src) {
        *d += s * x;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(dst: &mut [f32], src: &[f32], s: f32) {
    let n = dst.len();
    let vs = _mm256_set1_ps(s);
    let mut i = 0;
    while i + 8 <= n {
        let a = _mm256_loadu_ps(dst.as_ptr().add(i));
        let b = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(vs, b, a));
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) += s * *src.get_unchecked(i);
        i += 1;
    }
}

#[inline]
pub fn axpy(dst: &mut [f32], src: &[f32], s: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { axpy_avx2(dst, src, s) };
            return;
        }
    }
    axpy_scalar(dst, src, s);
}

/// Conv over (freq, time). `w` is `[co][ci][kf][kt]`, dilated in freq only.
///
/// Time padding is **left-only** (causal): the input already carries `kt-1`
/// frames of left context, so `out.t == inp.t - (kt-1)`. Freq is zero-padded
/// by `pf` on both sides, matching `F.pad(h, (0,0,pf,pf))` in `v2f.py`.
///
/// **Register-blocked over output channels.** The naive form (one `axpy` per
/// (co, ci, df, dt)) reads the source once per FMA and read-modify-writes the
/// destination — 1 FLOP per 2 loads plus a store, which measured 24 GFLOPS
/// against a 106 GFLOPS requirement. Holding `CO_BLK` accumulators in
/// registers lets one source load feed `CO_BLK` FMAs.
const CO_BLK: usize = 4;
/// AVX2 版が 1 回で埋める周波数幅（2 タイル ＝ 16 float）。
const F_STEP: usize = 16;

fn pad_freq(inp: &Ten, pf: usize) -> Ten {
    let fp = inp.f + 2 * pf;
    let mut o = Ten::zeros(inp.c, inp.t, fp);
    for c in 0..inp.c {
        for t in 0..inp.t {
            let src = inp.at(c, t);
            o.at_mut(c, t)[pf..pf + inp.f].copy_from_slice(src);
        }
    }
    o
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_tile_avx2(pad: &Ten, w: &[f32], b: &[f32], out: &mut Ten,
                         ci: usize, kf: usize, kt: usize, dil: usize,
                         o0: usize, nb: usize, t: usize, f0: usize) {
    // ⚠ 4co × 2F が実測最良。4co × 3F（アキュムレータ 12）はレジスタ逼迫で
    //   スピルし **22% 遅くなった**（0.244 → 0.322）。広げない。
    let mut a = [_mm256_setzero_ps(); 8];
    for j in 0..nb {
        let v = _mm256_set1_ps(b[o0 + j]);
        a[j] = v;
        a[j + 4] = v;
    }
    for i in 0..ci {
        for df in 0..kf {
            let off = f0 + df * dil;
            for dt in 0..kt {
                let src = pad.at(i, t + dt);
                let v0 = _mm256_loadu_ps(src.as_ptr().add(off));
                let v1 = _mm256_loadu_ps(src.as_ptr().add(off + 8));
                for j in 0..nb {
                    let s = _mm256_set1_ps(
                        *w.get_unchecked((((o0 + j) * ci + i) * kf + df) * kt + dt));
                    a[j] = _mm256_fmadd_ps(s, v0, a[j]);
                    a[j + 4] = _mm256_fmadd_ps(s, v1, a[j + 4]);
                }
            }
        }
    }
    for j in 0..nb {
        let dst = out.at_mut(o0 + j, t);
        _mm256_storeu_ps(dst.as_mut_ptr().add(f0), a[j]);
        _mm256_storeu_ps(dst.as_mut_ptr().add(f0 + 8), a[j + 4]);
    }
}

fn conv_tile_scalar(pad: &Ten, w: &[f32], b: &[f32], out: &mut Ten,
                    ci: usize, kf: usize, kt: usize, dil: usize,
                    o0: usize, nb: usize, t: usize, f0: usize, n: usize) {
    let mut acc = [[0.0f32; 8]; CO_BLK];
    for j in 0..nb {
        for a in acc[j].iter_mut().take(n) {
            *a = b[o0 + j];
        }
    }
    for i in 0..ci {
        for df in 0..kf {
            let off = f0 + df * dil;
            for dt in 0..kt {
                let src = pad.at(i, t + dt);
                for j in 0..nb {
                    let s = w[(((o0 + j) * ci + i) * kf + df) * kt + dt];
                    for k in 0..n {
                        acc[j][k] += s * src[off + k];
                    }
                }
            }
        }
    }
    for j in 0..nb {
        let dst = out.at_mut(o0 + j, t);
        dst[f0..f0 + n].copy_from_slice(&acc[j][..n]);
    }
}

pub fn conv_ft(inp: &Ten, w: &[f32], b: &[f32], co: usize,
               kf: usize, kt: usize, dil: usize) -> Ten {
    let ot = inp.t + 1 - kt;
    let mut out = Ten::zeros(co, ot, inp.f);
    let mut pad = Ten::zeros(inp.c, inp.t, inp.f + 2 * (((kf - 1) * dil) / 2));
    conv_ft_into(inp, w, b, co, kf, kt, dil, &mut out, &mut pad);
    out
}

/// 確保しない版。`out` と `pad` は呼び出し側が使い回す。
#[allow(clippy::too_many_arguments)]
pub fn conv_ft_into(inp: &Ten, w: &[f32], b: &[f32], co: usize,
                    kf: usize, kt: usize, dil: usize,
                    out: &mut Ten, pad: &mut Ten) {
    let ci = inp.c;
    let f = inp.f;
    let ot = inp.t + 1 - kt;
    let pf = ((kf - 1) * dil) / 2;
    // ⚠ fill(0.0) を毎回しない——縁は一度 0 になったら書かれない（interior だけ上書き）。
    //   旧版は 103 KB の fill ＋ 行ごとの to_vec()（ヒープ確保 96 回/呼び出し）で
    //   層あたり ~30 us を食っていた。pad と inp は別テンソルなので直接コピーできる。
    debug_assert_eq!(pad.f, f + 2 * pf);
    for c in 0..inp.c {
        for t in 0..inp.t {
            let o = (c * pad.t + t) * pad.f + pf;
            let i0 = (c * inp.t + t) * inp.f;
            let (dst, src) = (&mut pad.d[o..o + f] as *mut [f32], &inp.d[i0..i0 + f]);
            unsafe { (*dst).copy_from_slice(src) };
        }
    }
    let pad = &*pad;
    #[cfg(target_arch = "x86_64")]
    let use_avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx = false;
    for t in 0..ot {
        let mut o0 = 0;
        while o0 < co {
            let nb = CO_BLK.min(co - o0);
            let mut f0 = 0;
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                while f0 + F_STEP <= f {
                    unsafe { conv_tile_avx2(pad, w, b, out, ci, kf, kt, dil, o0, nb, t, f0) };
                    f0 += F_STEP;
                }
            }
            while f0 + 8 <= f {
                conv_tile_scalar(pad, w, b, out, ci, kf, kt, dil, o0, nb, t, f0, 8);
                f0 += 8;
            }
            if f0 < f {
                conv_tile_scalar(pad, w, b, out, ci, kf, kt, dil, o0, nb, t, f0, f - f0);
            }
            o0 += CO_BLK;
        }
    }
}

/// 1x1 conv over channels: `out[o][t][f] = b[o] + sum_i w[o][i] * inp[i][t][f]`.
///
/// conv_ft と同じ理由で出力チャネルをレジスタにブロックする。素の `axpy`
/// 版は 16 GFLOPS しか出ていなかった（入力を FMA ごとに読み直すため）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pw_tile_avx2(inp: &Ten, w: &[f32], b: &[f32], out: &mut Ten,
                       ci: usize, o0: usize, nb: usize, t: usize, f0: usize) {
    let mut acc = [_mm256_setzero_ps(); CO_BLK];
    for (j, a) in acc.iter_mut().enumerate().take(nb) {
        *a = _mm256_set1_ps(b[o0 + j]);
    }
    for i in 0..ci {
        let v = _mm256_loadu_ps(inp.at(i, t).as_ptr().add(f0));
        for j in 0..nb {
            let s = *w.get_unchecked((o0 + j) * ci + i);
            acc[j] = _mm256_fmadd_ps(_mm256_set1_ps(s), v, acc[j]);
        }
    }
    for j in 0..nb {
        _mm256_storeu_ps(out.at_mut(o0 + j, t).as_mut_ptr().add(f0), acc[j]);
    }
}

pub fn pointwise(inp: &Ten, w: &[f32], b: &[f32], co: usize) -> Ten {
    let mut out = Ten::zeros(co, inp.t, inp.f);
    pointwise_into(inp, w, b, co, &mut out);
    out
}

/// 確保しない版。
pub fn pointwise_into(inp: &Ten, w: &[f32], b: &[f32], co: usize, out: &mut Ten) {
    let ci = inp.c;
    let f = inp.f;
    #[cfg(target_arch = "x86_64")]
    let use_avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx = false;
    for t in 0..inp.t {
        let mut o0 = 0;
        while o0 < co {
            let nb = CO_BLK.min(co - o0);
            let mut f0 = 0;
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                while f0 + 8 <= f {
                    unsafe { pw_tile_avx2(inp, w, b, out, ci, o0, nb, t, f0) };
                    f0 += 8;
                }
            }
            while f0 < f {
                for j in 0..nb {
                    let mut a = b[o0 + j];
                    for i in 0..ci {
                        a += w[(o0 + j) * ci + i] * inp.at(i, t)[f0];
                    }
                    out.at_mut(o0 + j, t)[f0] = a;
                }
                f0 += 1;
            }
            o0 += CO_BLK;
        }
    }
}

/// GELU. `F.gelu(x)`（erf 版）に対し最大 1e-3。
///
/// ⚠ スカラー `tanh()` を 8 回呼ぶ実装は **1 層あたり 0.486 ms** かかり、
/// 畳み込み本体（0.224 ms）の 2 倍という最大の律速だった。`tanh` は
/// libm 呼び出しでベクトル化されない。有理近似に置き換えて AVX2 に載せる。
#[inline]
fn tanh_approx(x: f32) -> f32 {
    // Padé 型。|x| <= 4.6 で誤差 < 2e-4、外側は飽和させる。
    let x = x.clamp(-4.6, 4.6);
    let x2 = x * x;
    let n = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let d = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    n / d
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_avx2(x: &mut [f32]) {
    let c = _mm256_set1_ps(0.797_884_56);
    let k = _mm256_set1_ps(0.044715);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let lim = _mm256_set1_ps(4.6);
    let nlim = _mm256_set1_ps(-4.6);
    let (c1, c2, c3) = (_mm256_set1_ps(135135.0), _mm256_set1_ps(17325.0), _mm256_set1_ps(378.0));
    let (d1, d2, d3, d4) = (_mm256_set1_ps(135135.0), _mm256_set1_ps(62370.0),
                            _mm256_set1_ps(3150.0), _mm256_set1_ps(28.0));
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let u = _mm256_loadu_ps(x.as_ptr().add(i));
        let u3 = _mm256_mul_ps(_mm256_mul_ps(u, u), u);
        let a = _mm256_mul_ps(c, _mm256_fmadd_ps(k, u3, u));
        let a = _mm256_min_ps(_mm256_max_ps(a, nlim), lim);
        let a2 = _mm256_mul_ps(a, a);
        let nu = _mm256_mul_ps(a, _mm256_fmadd_ps(a2, _mm256_fmadd_ps(a2, _mm256_add_ps(c3, a2), c2), c1));
        let de = _mm256_fmadd_ps(a2, _mm256_fmadd_ps(a2, _mm256_fmadd_ps(a2, d4, d3), d2), d1);
        let th = _mm256_div_ps(nu, de);
        let r = _mm256_mul_ps(_mm256_mul_ps(half, u), _mm256_add_ps(one, th));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < n {
        let u = *x.get_unchecked(i);
        *x.get_unchecked_mut(i) = 0.5 * u * (1.0 + tanh_approx(0.797_884_56 * (u + 0.044715 * u * u * u)));
        i += 1;
    }
}

#[inline]
pub fn gelu_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { gelu_avx2(x) };
            return;
        }
    }
    const C: f32 = 0.797_884_56;
    for v in x.iter_mut() {
        let u = *v;
        *v = 0.5 * u * (1.0 + tanh_approx(C * (u + 0.044715 * u * u * u)));
    }
}

/// cummean 正規化の走行状態（層ごと）。ストリームでも一括でも同じ値になる。
#[derive(Clone, Copy, Default)]
pub struct CumNorm {
    pub k: f64,
    pub sum_m: f64,
    pub sum_v: f64,
}

/// cummean: 各フレームの (C,F) 平均 m_t・分散 v_t を取り、
/// 累積平均 m̄ = Σm/k、v̄ = Σ(v+m²)/k − m̄² で正規化する（`v2f.py` と同一）。
pub fn freq_norm_cummean(x: &mut Ten, g: &[f32], b: &[f32], eps: f32, st: &mut CumNorm) {
    let n = (x.c * x.f) as f64;
    for t in 0..x.t {
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for c in 0..x.c {
            for &v in x.at(c, t) {
                s1 += v as f64;
                s2 += (v as f64) * (v as f64);
            }
        }
        let m = s1 / n;
        let v = s2 / n - m * m;
        st.k += 1.0;
        st.sum_m += m;
        st.sum_v += v + m * m;
        let mb = st.sum_m / st.k;
        let vb = (st.sum_v / st.k - mb * mb).max(0.0);
        let inv = 1.0 / ((vb + eps as f64).sqrt());
        for c in 0..x.c {
            let (gc, bc) = (g[c], b[c]);
            for w in x.at_mut(c, t) {
                *w = (((*w as f64 - mb) * inv) as f32) * gc + bc;
            }
        }
    }
}

/// FreqNorm: normalise over (C, F) per time frame. **T never mixes.**
pub fn freq_norm(x: &mut Ten, g: &[f32], b: &[f32], eps: f32) {
    let n = (x.c * x.f) as f32;
    for t in 0..x.t {
        let (mut s, mut q) = (0.0f64, 0.0f64);
        for c in 0..x.c {
            for &v in x.at(c, t) {
                s += v as f64;
                q += (v as f64) * (v as f64);
            }
        }
        let m = (s / n as f64) as f32;
        let var = (q / n as f64) as f32 - m * m;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..x.c {
            let (gc, bc) = (g[c], b[c]);
            for v in x.at_mut(c, t) {
                *v = (*v - m) * inv * gc + bc;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// スレッド並列。**出力チャネル方向で割る**（時間は K=2 しかなく、
// 周波数は dilation の halo が 90 bin に達して割れない——実測で否定済み）。
// 外部クレートを足さない（MIT 維持・依存監査を増やさない）。
// ---------------------------------------------------------------------------

/// conv の並列仕事が読むコンテキスト。`Scratch` が持ち、毎回フィールドだけ更新する。
#[derive(Clone, Copy, Default)]
pub struct ConvCtx {
    inp: usize, w: usize, wl: usize, b: usize, bl: usize,
    lo: usize, hi: usize, kf: usize, kt: usize, dil: usize,
    out: usize, pad: usize,
}

fn conv_job(ctx: usize) {
    unsafe {
        let c = &*(ctx as *const ConvCtx);
        let inp = &*(c.inp as *const Ten);
        let w = std::slice::from_raw_parts(c.w as *const f32, c.wl);
        let b = std::slice::from_raw_parts(c.b as *const f32, c.bl);
        let out = &mut *(c.out as *mut Ten);
        let pad = &mut *(c.pad as *mut Ten);
        conv_ft_range(inp, w, b, c.lo, c.hi, c.kf, c.kt, c.dil, out, pad);
    }
}

/// `conv_ft_into` を出力チャネルで分割して常駐プールに投げる。
///
/// # Safety
/// 各ワーカが `out` の自分の出力チャネル範囲にしか書かないことに依存する。
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv_ft_into_par(pool: &Pool, inp: &Ten, w: &[f32], b: &[f32], co: usize,
                               kf: usize, kt: usize, dil: usize,
                               out: &mut Ten, pads: &mut [Ten],
                               ctxs: &mut [ConvCtx]) {
    let n = (pool.n + 1).min(pads.len()).max(1);
    if n < 2 || co < 2 * CO_BLK {
        conv_ft_into(inp, w, b, co, kf, kt, dil, out, &mut pads[0]);
        return;
    }
    let per = ((co / n) / CO_BLK).max(1) * CO_BLK;
    let mut jobs = [Job { f: conv_job, ctx: 0 }; 4];
    let mut nj = 0usize;
    let mut lo = 0usize;
    for k in 0..n {
        let hi = if k == n - 1 { co } else { (lo + per).min(co) };
        if lo >= hi { break; }
        ctxs[k] = ConvCtx {
            inp: inp as *const Ten as usize,
            w: w.as_ptr() as usize, wl: w.len(),
            b: b.as_ptr() as usize, bl: b.len(),
            lo, hi, kf, kt, dil,
            out: out as *mut Ten as usize,
            pad: &mut pads[k] as *mut Ten as usize,
        };
        jobs[nj] = Job { f: conv_job, ctx: &ctxs[k] as *const ConvCtx as usize };
        nj += 1;
        lo = hi;
    }
    pool.run_with_self(&jobs[..nj]);
}

/// `pointwise_into` の出力チャネル範囲版。
pub fn pointwise_range(inp: &Ten, w: &[f32], b: &[f32], lo: usize, hi: usize, out: &mut Ten) {
    let ci = inp.c;
    let f = inp.f;
    #[cfg(target_arch = "x86_64")]
    let use_avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx = false;
    for t in 0..inp.t {
        let mut o0 = lo;
        while o0 < hi {
            let nb = CO_BLK.min(hi - o0);
            let mut f0 = 0;
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                while f0 + 8 <= f {
                    unsafe { pw_tile_avx2(inp, w, b, out, ci, o0, nb, t, f0) };
                    f0 += 8;
                }
            }
            while f0 < f {
                for j in 0..nb {
                    let mut a = b[o0 + j];
                    for i in 0..ci {
                        a += w[(o0 + j) * ci + i] * inp.at(i, t)[f0];
                    }
                    out.at_mut(o0 + j, t)[f0] = a;
                }
                f0 += 1;
            }
            o0 += CO_BLK;
        }
    }
}

/// mlp の並列仕事のコンテキスト。
#[derive(Clone, Copy, Default)]
pub struct MlpCtx {
    inp: usize, w: usize, wl: usize, b: usize, bl: usize,
    lo: usize, hi: usize, out: usize, gelu: bool,
}

fn mlp_job(ctx: usize) {
    unsafe {
        let c = &*(ctx as *const MlpCtx);
        let inp = &*(c.inp as *const Ten);
        let w = std::slice::from_raw_parts(c.w as *const f32, c.wl);
        let b = std::slice::from_raw_parts(c.b as *const f32, c.bl);
        let out = &mut *(c.out as *mut Ten);
        pointwise_range(inp, w, b, c.lo, c.hi, out);
        if c.gelu {
            let o = (c.lo * out.t) * out.f;
            let e = (c.hi * out.t) * out.f;
            gelu_inplace(&mut out.d[o..e]);
        }
    }
}

/// pw1 + GELU + pw2 を出力チャネル分割で並列化する（バリア 2 回）。
///
/// # Safety
/// 各範囲が互いに素であること。
#[allow(clippy::too_many_arguments)]
pub unsafe fn mlp_par(pool: &Pool, inp: &Ten, w1: &[f32], b1: &[f32],
                      w2: &[f32], b2: &[f32], ch: usize,
                      u: &mut Ten, out: &mut Ten, ctxs: &mut [MlpCtx]) {
    let n = pool.n + 1;
    if n < 2 {
        pointwise_into(inp, w1, b1, 3 * ch, u);
        gelu_inplace(&mut u.d);
        pointwise_into(u, w2, b2, ch, out);
        return;
    }
    let mut jobs = [Job { f: mlp_job, ctx: 0 }; 4];
    // 段 1: pw1 + gelu
    let co1 = 3 * ch;
    let per1 = ((co1 / n) / CO_BLK).max(1) * CO_BLK;
    let mut nj = 0usize;
    let mut lo = 0usize;
    for k in 0..n {
        let hi = if k == n - 1 { co1 } else { (lo + per1).min(co1) };
        if lo >= hi { break; }
        ctxs[nj] = MlpCtx {
            inp: inp as *const Ten as usize,
            w: w1.as_ptr() as usize, wl: w1.len(),
            b: b1.as_ptr() as usize, bl: b1.len(),
            lo, hi, out: u as *mut Ten as usize, gelu: true,
        };
        jobs[nj] = Job { f: mlp_job, ctx: &ctxs[nj] as *const MlpCtx as usize };
        nj += 1;
        lo = hi;
    }
    pool.run_with_self(&jobs[..nj]);
    // 段 2: pw2（u は読み取り専用）
    let per2 = ((ch / n) / CO_BLK).max(1) * CO_BLK;
    let mut nj = 0usize;
    let mut lo = 0usize;
    for k in 0..n {
        let hi = if k == n - 1 { ch } else { (lo + per2).min(ch) };
        if lo >= hi { break; }
        ctxs[nj] = MlpCtx {
            inp: u as *const Ten as usize,
            w: w2.as_ptr() as usize, wl: w2.len(),
            b: b2.as_ptr() as usize, bl: b2.len(),
            lo, hi, out: out as *mut Ten as usize, gelu: false,
        };
        jobs[nj] = Job { f: mlp_job, ctx: &ctxs[nj] as *const MlpCtx as usize };
        nj += 1;
        lo = hi;
    }
    pool.run_with_self(&jobs[..nj]);
}

/// `conv_ft_into` の出力チャネル範囲版。
#[allow(clippy::too_many_arguments)]
pub fn conv_ft_range(inp: &Ten, w: &[f32], b: &[f32], lo: usize, hi: usize,
                     kf: usize, kt: usize, dil: usize, out: &mut Ten, pad: &mut Ten) {
    let ci = inp.c;
    let f = inp.f;
    let ot = inp.t + 1 - kt;
    let pf = ((kf - 1) * dil) / 2;
    // ⚠ fill(0.0) を毎回しない——縁は一度 0 になったら書かれない（interior だけ上書き）。
    //   旧版は 103 KB の fill ＋ 行ごとの to_vec()（ヒープ確保 96 回/呼び出し）で
    //   層あたり ~30 us を食っていた。pad と inp は別テンソルなので直接コピーできる。
    debug_assert_eq!(pad.f, f + 2 * pf);
    for c in 0..inp.c {
        for t in 0..inp.t {
            let o = (c * pad.t + t) * pad.f + pf;
            let i0 = (c * inp.t + t) * inp.f;
            let (dst, src) = (&mut pad.d[o..o + f] as *mut [f32], &inp.d[i0..i0 + f]);
            unsafe { (*dst).copy_from_slice(src) };
        }
    }
    let pad = &*pad;
    #[cfg(target_arch = "x86_64")]
    let use_avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx = false;
    for t in 0..ot {
        let mut o0 = lo;
        while o0 < hi {
            let nb = CO_BLK.min(hi - o0);
            let mut f0 = 0;
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                while f0 + F_STEP <= f {
                    unsafe { conv_tile_avx2(pad, w, b, out, ci, kf, kt, dil, o0, nb, t, f0) };
                    f0 += F_STEP;
                }
            }
            while f0 + 8 <= f {
                conv_tile_scalar(pad, w, b, out, ci, kf, kt, dil, o0, nb, t, f0, 8);
                f0 += 8;
            }
            if f0 < f {
                conv_tile_scalar(pad, w, b, out, ci, kf, kt, dil, o0, nb, t, f0, f - f0);
            }
            o0 += CO_BLK;
        }
    }
}

/// `conv_ft` + `freq_norm` + `pw1` + `gelu` + `pw2` を 1 ブロック分、
/// 周波数を `nt` 分割して並列に走らせる。
///
/// ⚠ `freq_norm` は (C, F) 全体の平均・分散を使うので**分割できない**。
/// 先に単スレッドで統計を取り、正規化そのものだけ分割する。
pub struct Layer {
    /// 正規化: 0 = freq（per-frame）、1 = **cummean**（減衰なし累積平均 ＝
    /// GroupNorm の因果版。出荷モデルはこれ。`v2f.py` の `FreqNorm mode="cummean"`）。
    pub norm_mode: u8,
    pub ch: usize,
    pub kf: usize,
    pub kt: usize,
    pub dil: usize,
    pub w_c: Vec<f32>,
    pub b_c: Vec<f32>,
    pub g_n: Vec<f32>,
    pub b_n: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl Layer {
    pub fn new(ch: usize, kf: usize, kt: usize, dil: usize) -> Self {
        Self {
            norm_mode: 0,
            ch, kf, kt, dil,
            w_c: vec![0.01; ch * ch * kf * kt], b_c: vec![0.0; ch],
            g_n: vec![1.0; ch], b_n: vec![0.0; ch],
            w1: vec![0.01; 3 * ch * ch], b1: vec![0.0; 3 * ch],
            w2: vec![0.01; ch * 3 * ch], b2: vec![0.0; ch],
        }
    }

    /// 残差ブロック 1 段。`inp` は左文脈 `kt-1` 込み `[ch][kt-1+ot][f]`。
    pub fn forward(&self, inp: &Ten) -> Ten {
        let mut sc = Scratch::new(self, inp.f, inp.t + 1 - self.kt);
        self.forward_into(inp, &mut sc);
        sc.o2.clone()
    }

    /// 並列版。conv を出力チャネルで分ける（実測 2 スレッドで 1.8 倍・ビット一致）。
    ///
    /// # Safety
    /// `sc.o1` の出力チャネル範囲が互いに素であることに依存する。
    pub unsafe fn forward_into_par(&self, inp: &Ten, sc: &mut Scratch, pool: &Pool) {
        let (o1, pads, cctx) = (&mut sc.o1, &mut sc.pads, &mut sc.cctx);
        conv_ft_into_par(pool, inp, &self.w_c, &self.b_c, self.ch, self.kf, self.kt,
                         self.dil, o1, pads, cctx);
        if self.norm_mode == 1 {
            freq_norm_cummean(&mut sc.o1, &self.g_n, &self.b_n, 1e-5, &mut sc.cum);
        } else {
            freq_norm(&mut sc.o1, &self.g_n, &self.b_n, 1e-5);
        }
        mlp_par(pool, &sc.o1, &self.w1, &self.b1, &self.w2, &self.b2, self.ch,
                &mut sc.u, &mut sc.o2, &mut sc.mctx);
    }

    /// **確保しない版。** 5.8 ms ごとに呼ばれるので割り当てを毎回やらない。
    pub fn forward_into(&self, inp: &Ten, sc: &mut Scratch) {
        conv_ft_into(inp, &self.w_c, &self.b_c, self.ch, self.kf, self.kt, self.dil,
                     &mut sc.o1, &mut sc.pad);
        if self.norm_mode == 1 {
            freq_norm_cummean(&mut sc.o1, &self.g_n, &self.b_n, 1e-5, &mut sc.cum);
        } else {
            freq_norm(&mut sc.o1, &self.g_n, &self.b_n, 1e-5);
        }
        pointwise_into(&sc.o1, &self.w1, &self.b1, 3 * self.ch, &mut sc.u);
        gelu_inplace(&mut sc.u.d);
        pointwise_into(&sc.u, &self.w2, &self.b2, self.ch, &mut sc.o2);
    }
}

/// 1 層ぶんの作業領域。`TrunkState` が層ごとに持って使い回す。
pub struct Scratch {
    /// cummean の走行状態。**リセットはストリーム開始時のみ**（発話内で持ち越す）。
    pub cum: CumNorm,
    pub pad: Ten,
    /// 並列時のスレッドごとの pad。単スレッドでは `pads[0]` を使う。
    pub pads: Vec<Ten>,
    pub cctx: Vec<ConvCtx>,
    pub mctx: Vec<MlpCtx>,
    pub o1: Ten,
    pub u: Ten,
    pub o2: Ten,
}

impl Scratch {
    pub fn new(l: &Layer, f: usize, ot: usize) -> Self {
        let pf = ((l.kf - 1) * l.dil) / 2;
        Self {
            cum: CumNorm::default(),
            pad: Ten::zeros(l.ch, ot + l.kt - 1, f + 2 * pf),
            pads: (0..2).map(|_| Ten::zeros(l.ch, ot + l.kt - 1, f + 2 * pf)).collect(),
            cctx: vec![ConvCtx::default(); 4],
            mctx: vec![MlpCtx::default(); 4],
            o1: Ten::zeros(l.ch, ot, f),
            u: Ten::zeros(3 * l.ch, ot, f),
            o2: Ten::zeros(l.ch, ot, f),
        }
    }
}

/// 幹のブロック実行。**層ごとに時間方向の左文脈 `kt-1` を保持する。**
///
/// ⚠ 保持しないと層 1 が 16 フレーム、層 2 が 14…と全 span を計算してしまう
/// （実測 16.4 ms＝キャッシュ版の 4.5 倍）。各層はちょうど `emit` フレームだけ出す。
///
/// ⚠ **周波数帯での分割はしない。** dilation が 1,1,2,2,4,4,8,8 と伸びるので
/// halo が `sum((kf-1)*dil/2) = 90` bin に達し、4 分割しても各帯が 245/257 bin を
/// 読むことになって並列の意味が消える（実測: 2 スレッドで 1.17 倍、4 で悪化）。
/// 並列化するなら**出力チャネル軸**（halo 不要）。
pub struct TrunkState {
    /// 層ごとの `[左文脈 kt-1 ++ emit]` 連結バッファ。**使い回す。**
    cat: Vec<Ten>,
    sc: Vec<Scratch>,
}

impl TrunkState {
    pub fn new(layers: &[Layer], f: usize) -> Self {
        Self::with_emit(layers, f, 2)
    }
    pub fn with_emit(layers: &[Layer], f: usize, emit: usize) -> Self {
        Self {
            cat: layers.iter().map(|l| Ten::zeros(l.ch, l.kt - 1 + emit, f)).collect(),
            sc: layers.iter().map(|l| Scratch::new(l, f, emit)).collect(),
        }
    }

    /// `x`: `[ch][emit][f]` -> `[ch][emit][f]`。状態を進める。
    ///
    /// ⚠ ここは 5.8 ms ごとに呼ばれる。**確保とコピーを毎回やらない**——
    /// `cat` バッファは層ごとに使い回し、キャッシュ更新は `copy_within` で済ませる。
    /// 並列版。
    ///
    /// # Safety
    /// `forward_into_par` と同じ。
    pub unsafe fn step_par(&mut self, layers: &[Layer], x: &Ten, pool: &Pool) -> Ten {
        self.run(layers, x, Some(pool))
    }

    pub fn step(&mut self, layers: &[Layer], x: &Ten) -> Ten {
        self.run(layers, x, None)
    }

    fn run(&mut self, layers: &[Layer], x: &Ten, pool: Option<&Pool>) -> Ten {
        let mut h = x.clone();
        for (li, l) in layers.iter().enumerate() {
            let kc = l.kt - 1;
            let cat = &mut self.cat[li];
            // cat = [左文脈 kc][新 h]
            for c in 0..l.ch {
                let co = c * cat.t * cat.f;
                let ho = c * h.t * h.f;
                cat.d[co + kc * cat.f..co + (kc + h.t) * cat.f]
                    .copy_from_slice(&h.d[ho..ho + h.t * h.f]);
            }
            match pool {
                Some(p) => unsafe { l.forward_into_par(cat, &mut self.sc[li], p) },
                None => l.forward_into(cat, &mut self.sc[li]),
            }
            let y = &self.sc[li].o2;
            // 次回の左文脈＝今回の cat の末尾 kc フレーム
            for c in 0..l.ch {
                let co = c * cat.t * cat.f;
                cat.d.copy_within(co + h.t * cat.f..co + (h.t + kc) * cat.f, co);
            }
            for c in 0..l.ch {
                let ho = c * h.t * h.f;
                let yo = c * y.t * y.f;
                for k in 0..h.t * h.f {
                    h.d[ho + k] += y.d[yo + k];
                }
            }
        }
        h
    }
}


/// 出力チャネル軸での 2 スレッド実行（`thread_budget = 2`）。
///
/// ⚠ **周波数軸では割らない**——dilation 1..8 で halo が 90 bin に達し、
/// 4 分割しても各帯が 245/257 bin を読むので並列の意味が消える（実測済み）。
/// 出力チャネル軸なら halo 不要で、各スレッドは全入力チャネルを読むだけ。
///
/// ⚠ スレッドは**毎ブロック起こさない**。`std::thread::scope` の spawn は
/// 5.8 ms 予算に対して重すぎる（実測: 4 スレッドで逆に悪化した）。
/// 常駐ワーカに仕事を渡す（往復 3.2 us ＝ 1 層 142 us の 2%）。
pub struct Pool {
    tx: Vec<std::sync::mpsc::Sender<Job>>,
    done: std::sync::mpsc::Receiver<()>,
    pub n: usize,
}

/// ワーカに渡す仕事。**クロージャを使わない**——Box のヒープ確保が
/// 層あたり 3 バリア × 毎ブロックで積もり、実測 ~28 us/層の食い込みになった。
/// 生の関数ポインタ + コンテキストポインタで運ぶ。
///
/// 呼び出し側が「同じ出力に 2 スレッドが書かない」ことを保証する（出力チャネルで割る）。
#[derive(Clone, Copy)]
struct Job {
    f: fn(usize),
    ctx: usize,
}
unsafe impl Send for Job {}

impl Pool {
    pub fn new(n: usize) -> Self {
        let (dtx, done) = std::sync::mpsc::channel();
        let mut tx = Vec::new();
        for _ in 0..n {
            let (jtx, jrx) = std::sync::mpsc::channel::<Job>();
            let d = dtx.clone();
            std::thread::spawn(move || {
                while let Ok(j) = jrx.recv() {
                    (j.f)(j.ctx);
                    let _ = d.send(());
                }
            });
            tx.push(jtx);
        }
        Self { tx, done, n }
    }

    /// `fs` の各要素を 1 スレッドずつに配る。**要素数 <= n** であること。
    ///
    /// # Safety
    /// 各クロージャが**互いに素な領域だけ**を書くこと。
    /// **呼び出しスレッド自身が最初の仕事をやる。** `jobs.len() <= n + 1`。
    ///
    /// ワーカへの往復が 1 本減り、`jobs[0]` の書き込み先が呼び出しコアの
    /// キャッシュに残る。ヒープ確保ゼロ。
    ///
    /// # Safety
    /// 各仕事が互いに素な領域だけを書くこと。ctx が指す先が呼び出し中生きていること。
    pub unsafe fn run_with_self(&self, jobs: &[Job]) {
        assert!(jobs.len() <= self.n + 1);
        let k = jobs.len() - 1;
        for (i, j) in jobs[1..].iter().enumerate() {
            let _ = self.tx[i].send(*j);
        }
        (jobs[0].f)(jobs[0].ctx);
        for _ in 0..k {
            let _ = self.done.recv();
        }
    }

    /// 常駐スレッドの往復コスト（仕事ゼロ）。並列化の損益分岐を測るため。
    pub fn roundtrip(&self) -> std::time::Duration {
        fn nop(_: usize) {}
        let t0 = std::time::Instant::now();
        let jobs: Vec<Job> = (0..self.n + 1).map(|_| Job { f: nop, ctx: 0 }).collect();
        unsafe { self.run_with_self(&jobs) };
        t0.elapsed()
    }
}


// ---------------------------------------------------------------------------
// PyTorch (`training/v2f.py`) との一致検査
// ⚠ 36 巡目の教訓の再演: インデックス指定の一括書き換えでこの節を丸ごと
//   消してしまい、テスト数だけ見て「合格」と誤読した。**テストは名前で数える。**
// ---------------------------------------------------------------------------

/// `v2f.py` の `state_dict` を平坦 f32 で書き出したものを読む
/// （`training/export_v2f.py` と並び順が契約）。
pub struct V2fWeights {
    pub inp_w: Vec<f32>,
    pub inp_b: Vec<f32>,
    pub layers: Vec<Layer>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

impl V2fWeights {
    pub fn from_flat(d: &[f32], cin: usize, ch: usize, nl: usize,
                     kf: usize, kt: usize) -> Self {
        let mut o = 0usize;
        let mut take = |n: usize| { let r = d[o..o + n].to_vec(); o += n; r };
        let inp_w = take(ch * cin);
        let inp_b = take(ch);
        let mut layers = Vec::with_capacity(nl);
        for i in 0..nl {
            let mut l = Layer::new(ch, kf, kt, 1 << (i / 2));
            l.w_c = take(ch * ch * kf * kt);
            l.b_c = take(ch);
            l.g_n = take(ch);
            l.b_n = take(ch);
            l.w1 = take(3 * ch * ch);
            l.b1 = take(3 * ch);
            l.w2 = take(ch * 3 * ch);
            l.b2 = take(ch);
            layers.push(l);
        }
        let out_w = take(2 * ch);
        let out_b = take(2);
        Self { inp_w, inp_b, layers, out_w, out_b }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(name: &str) -> Vec<f32> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name);
        std::fs::read(p).expect("testdata").chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn to_ten(v: &[f32], c: usize, f: usize, t: usize) -> Ten {
        let mut o = Ten::zeros(c, t, f);
        for ci in 0..c {
            for fi in 0..f {
                for ti in 0..t {
                    o.at_mut(ci, ti)[fi] = v[(ci * f + fi) * t + ti];
                }
            }
        }
        o
    }

    fn offline(w: &V2fWeights, x: &Ten, ch: usize) -> Ten {
        let mut cur = pointwise(x, &w.inp_w, &w.inp_b, ch);
        for l in &w.layers {
            let y = l.forward(&cur);
            let ot = y.t;
            let mut n = Ten::zeros(ch, ot, cur.f);
            for c in 0..ch {
                for t in 0..ot {
                    let src = cur.at(c, t + (cur.t - ot));
                    let add = y.at(c, t);
                    let dst = n.at_mut(c, t);
                    for k in 0..cur.f { dst[k] = src[k] + add[k]; }
                }
            }
            cur = n;
        }
        cur
    }

    #[test]
    fn v2f_matches_pytorch() {
        let (cin, ch, nl, kf, kt, f, emit, ctx) =
            (4usize, 16usize, 8usize, 7usize, 3usize, 257usize, 2usize, 16usize);
        let w = V2fWeights::from_flat(&td("v2f_w.bin"), cin, ch, nl, kf, kt);
        let x = to_ten(&td("v2f_x.bin"), cin, f, emit + ctx);
        let cur = offline(&w, &x, ch);
        let o = pointwise(&cur, &w.out_w, &w.out_b, 2);
        let want = td("v2f_y.bin");
        let mut e = 0.0f32;
        for c in 0..2 {
            for fi in 0..f {
                for t in 0..emit {
                    let g = o.at(c, o.t - emit + t)[fi];
                    e = e.max((g - want[(c * f + fi) * emit + t]).abs());
                }
            }
        }
        assert!(e <= 2e-3, "v2f torch vs simd 最大絶対誤差 {e}");
    }

    #[test]
    fn v2f_streaming_equals_offline() {
        let (cin, ch, nl, kf, kt, f, emit, ctx) =
            (4usize, 16usize, 8usize, 7usize, 3usize, 257usize, 2usize, 16usize);
        let w = V2fWeights::from_flat(&td("v2f_w.bin"), cin, ch, nl, kf, kt);
        let x = to_ten(&td("v2f_x.bin"), cin, f, emit + ctx);
        let h = pointwise(&x, &w.inp_w, &w.inp_b, ch);
        let full = offline(&w, &x, ch);
        let mut st = TrunkState::with_emit(&w.layers, f, emit);
        let nblk = (emit + ctx) / emit;
        let mut last = Ten::zeros(ch, emit, f);
        for b in 0..nblk {
            let mut xb = Ten::zeros(ch, emit, f);
            for c in 0..ch {
                for t in 0..emit {
                    xb.at_mut(c, t).copy_from_slice(h.at(c, b * emit + t));
                }
            }
            last = st.step(&w.layers, &xb);
        }
        let mut e = 0.0f32;
        for c in 0..ch {
            for t in 0..emit {
                for k in 0..f {
                    e = e.max((last.at(c, t)[k] - full.at(c, full.t - emit + t)[k]).abs());
                }
            }
        }
        assert!(e <= 1e-3, "streaming != offline: 最大絶対誤差 {e}");
    }

    // conv_f_stride / conv_f_transpose の検査は V2P（周波数プーリング）打ち切りに
    // 伴い削除（対象の関数ごと撤去済み。RESEARCH.md 切り分け 9）。
}
