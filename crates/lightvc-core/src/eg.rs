//! E1/G1 (変換段) の推論。重み順序は training/export_eg.py と完全一致。
//!
//! 構造 (両ネット共通・k=3):
//!   inp Conv1d(cin->dim, 3, 左pad2) -> L x CausalBlock -> out Linear(dim->cout)
//!   CausalBlock: r + pw2(gelu_erf(pw1(LayerNorm(dw(x)))))
//!     dw = Conv1d(dim,dim,3, dilation=2^(i/2), 左padのみ)  ※depthwiseではない
//!
//! データは [T][C] フレーム主。重みは load 時に [i][o] 転置 (broadcast-FMA で
//! W を 1 パス連続ストリーム＝matvec の帯域最適形)。GELU は erf 厳密版
//! (PyTorch F.gelu 既定と一致)。素朴スカラー実装は RTF E 0.48/G 0.40 で
//! 予算 (0.10/0.05) を割れなかった実測があるため、この形が必須。

const LN_EPS: f32 = 1e-5;

fn erf64(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7 = f32 の丸め水準)
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

#[inline]
fn gelu_erf(x: f32) -> f32 {
    (0.5 * x as f64 * (1.0 + erf64(x as f64 * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// y[o] += Σ_i x[i] * w[i*co + o]。w は [ci][co] 転置済み・連続 1 パス。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_avx2(w: &[f32], x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let co = y.len();
    let cv = co / 8 * 8;
    unsafe {
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let bx = _mm256_set1_ps(xi);
            let wr = w.as_ptr().add(i * co);
            let yp = y.as_mut_ptr();
            let mut o = 0usize;
            while o < cv {
                let acc = _mm256_fmadd_ps(bx, _mm256_loadu_ps(wr.add(o)), _mm256_loadu_ps(yp.add(o)));
                _mm256_storeu_ps(yp.add(o), acc);
                o += 8;
            }
            while o < co {
                *y.get_unchecked_mut(o) += xi * *w.get_unchecked(i * co + o);
                o += 1;
            }
        }
    }
}

fn matvec_scalar(w: &[f32], x: &[f32], y: &mut [f32]) {
    let co = y.len();
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 {
            continue;
        }
        let wr = &w[i * co..(i + 1) * co];
        for o in 0..co {
            y[o] += xi * wr[o];
        }
    }
}

/// o0..o1 の出力範囲だけを計算する範囲版（行 stride = co）。
/// 各 y[o] は単一スレッドが同じ i 順で積むので、分割してもビット一致。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_avx2_range(w: *const f32, co: usize, x: *const f32, xn: usize,
                            y: *mut f32, o0: usize, o1: usize) {
    use std::arch::x86_64::*;
    let cv = o0 + (o1 - o0) / 8 * 8;
    unsafe {
        for i in 0..xn {
            let xi = *x.add(i);
            if xi == 0.0 {
                continue;
            }
            let bx = _mm256_set1_ps(xi);
            let wr = w.add(i * co);
            let mut o = o0;
            while o < cv {
                let acc = _mm256_fmadd_ps(bx, _mm256_loadu_ps(wr.add(o)),
                                          _mm256_loadu_ps(y.add(o)));
                _mm256_storeu_ps(y.add(o), acc);
                o += 8;
            }
            while o < o1 {
                *y.add(o) += xi * *wr.add(o);
                o += 1;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct EgJob {
    w: *const f32,
    co: usize,
    x: *const f32,
    xn: usize,
    y: *mut f32,
    o0: usize,
    o1: usize,
}
unsafe impl Send for EgJob {}

/// 常駐 1 ワーカ・spin 同期。matvec 半分 (~9us) に mpsc 往復は重すぎて
/// 2 倍遅くなった実測がある（E 0.14→0.28）ので、atomic セマフォで手渡す。
/// ワーカは busy spin（1 ハイパースレッドを占有）——realtime 音声の設計判断。
pub struct EgPool {
    sh: std::sync::Arc<EgShared>,
}

struct EgShared {
    seq: std::sync::atomic::AtomicUsize,
    done: std::sync::atomic::AtomicUsize,
    quit: std::sync::atomic::AtomicBool,
    job: std::cell::UnsafeCell<EgJob>,
}
unsafe impl Sync for EgShared {}

impl EgPool {
    pub fn new() -> Self {
        let sh = std::sync::Arc::new(EgShared {
            seq: std::sync::atomic::AtomicUsize::new(0),
            done: std::sync::atomic::AtomicUsize::new(0),
            quit: std::sync::atomic::AtomicBool::new(false),
            job: std::cell::UnsafeCell::new(EgJob {
                w: std::ptr::null(), co: 0, x: std::ptr::null(), xn: 0,
                y: std::ptr::null_mut(), o0: 0, o1: 0,
            }),
        });
        let sh2 = sh.clone();
        std::thread::spawn(move || {
            use std::sync::atomic::Ordering::*;
            let mut seen = 0usize;
            loop {
                let s = sh2.seq.load(Acquire);
                if s == seen {
                    if sh2.quit.load(Relaxed) {
                        return;
                    }
                    std::hint::spin_loop();
                    continue;
                }
                seen = s;
                let j = unsafe { *sh2.job.get() };
                unsafe { matvec_avx2_range(j.w, j.co, j.x, j.xn, j.y, j.o0, j.o1) };
                sh2.done.store(seen, Release);
            }
        });
        EgPool { sh }
    }
}

impl Drop for EgPool {
    fn drop(&mut self) {
        self.sh.quit.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for EgPool {
    fn default() -> Self {
        Self::new()
    }
}

fn matvec_par(pool: Option<&EgPool>, w: &[f32], x: &[f32], y: &mut [f32]) {
    let co = y.len();
    #[cfg(target_arch = "x86_64")]
    if let Some(pl) = pool {
        if co >= 128
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            use std::sync::atomic::Ordering::*;
            let mid = (co / 2) & !7;
            unsafe {
                *pl.sh.job.get() = EgJob {
                    w: w.as_ptr(), co, x: x.as_ptr(), xn: x.len(),
                    y: y.as_mut_ptr(), o0: mid, o1: co,
                };
            }
            let s = pl.sh.seq.load(Relaxed) + 1;
            pl.sh.seq.store(s, Release);
            unsafe {
                matvec_avx2_range(w.as_ptr(), co, x.as_ptr(), x.len(),
                                  y.as_mut_ptr(), 0, mid)
            };
            while pl.sh.done.load(Acquire) != s {
                std::hint::spin_loop();
            }
            return;
        }
    }
    matvec(w, x, y);
}

#[inline]
fn matvec(w: &[f32], x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe { matvec_avx2(w, x, y) };
            return;
        }
    }
    matvec_scalar(w, x, y);
}

struct Blk {
    dw_w: [Vec<f32>; 3], // tap j: [dim][dim] ([i][o] 転置)
    dw_b: Vec<f32>,
    nw: Vec<f32>,
    nb: Vec<f32>,
    pw1_w: Vec<f32>, // [dim][3dim] ([i][o])
    pw1_b: Vec<f32>,
    pw2_w: Vec<f32>, // [3dim][dim] ([i][o])
    pw2_b: Vec<f32>,
    dil: usize,
}

pub struct Eg1d {
    pub dim: usize,
    pub layers: usize,
    pub cin: usize,
    pub cout: usize,
    inp_w: [Vec<f32>; 3], // tap j: [cin][dim]
    inp_b: Vec<f32>,
    blk: Vec<Blk>,
    out_w: Vec<f32>, // [dim][cout]
    out_b: Vec<f32>,
}

/// [o][i] (PyTorch Linear) -> [i][o]
fn tr2(w: &[f32], o: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0f32; w.len()];
    for oo in 0..o {
        for ii in 0..i {
            v[ii * o + oo] = w[oo * i + ii];
        }
    }
    v
}

/// [o][i][3] (PyTorch Conv1d) -> tap 別 [i][o]
fn tr3(w: &[f32], o: usize, i: usize) -> [Vec<f32>; 3] {
    let mut v = [vec![0f32; o * i], vec![0f32; o * i], vec![0f32; o * i]];
    for oo in 0..o {
        for ii in 0..i {
            for j in 0..3 {
                v[j][ii * o + oo] = w[(oo * i + ii) * 3 + j];
            }
        }
    }
    v
}

impl Eg1d {
    pub fn from_flat(w: &[f32], cin: usize, cout: usize, dim: usize, layers: usize) -> Self {
        let mut p = 0usize;
        let mut take = |n: usize| {
            let v = w[p..p + n].to_vec();
            p += n;
            v
        };
        let inp_w = tr3(&take(dim * cin * 3), dim, cin);
        let inp_b = take(dim);
        let mut blk = Vec::with_capacity(layers);
        for i in 0..layers {
            blk.push(Blk {
                dw_w: tr3(&take(dim * dim * 3), dim, dim),
                dw_b: take(dim),
                nw: take(dim),
                nb: take(dim),
                pw1_w: tr2(&take(3 * dim * dim), 3 * dim, dim),
                pw1_b: take(3 * dim),
                pw2_w: tr2(&take(dim * 3 * dim), dim, 3 * dim),
                pw2_b: take(dim),
                dil: 1usize << (i / 2),
            });
        }
        let out_w = tr2(&take(cout * dim), cout, dim);
        let out_b = take(cout);
        assert_eq!(p, w.len(), "重みサイズ不一致 (export_eg.py と突き合わせる)");
        Eg1d { dim, layers, cin, cout, inp_w, inp_b, blk, out_w, out_b }
    }

    /// バッチ実行。x: [T][cin] フレーム主 (長さ t*cin)。返り: [T][cout]。
    /// 左パディングのみ = 先読み 0。ストリームとフレーム毎に同一の演算順。
    pub fn process(&self, x: &[f32], t: usize) -> Vec<f32> {
        let d = self.dim;
        let mut h = vec![0f32; t * d];
        for ti in 0..t {
            let dst = &mut h[ti * d..(ti + 1) * d];
            dst.copy_from_slice(&self.inp_b);
            for j in 0..3usize {
                let src_t = ti as isize - 2 + j as isize;
                if src_t < 0 {
                    continue;
                }
                let xr = &x[src_t as usize * self.cin..(src_t as usize + 1) * self.cin];
                matvec(&self.inp_w[j], xr, dst);
            }
        }
        let mut tmp = vec![0f32; d];
        let mut u = vec![0f32; 3 * d];
        for b in &self.blk {
            let prev = h.clone();
            for ti in 0..t {
                tmp.copy_from_slice(&b.dw_b);
                for j in 0..3usize {
                    let src_t = ti as isize - ((2 - j) * b.dil) as isize;
                    if src_t < 0 {
                        continue;
                    }
                    let xr = &prev[src_t as usize * d..(src_t as usize + 1) * d];
                    matvec(&b.dw_w[j], xr, &mut tmp);
                }
                mlp(b, &mut tmp, &mut u, &mut h[ti * d..(ti + 1) * d]);
            }
        }
        let mut y = vec![0f32; t * self.cout];
        for ti in 0..t {
            let dst = &mut y[ti * self.cout..(ti + 1) * self.cout];
            dst.copy_from_slice(&self.out_b);
            matvec(&self.out_w, &h[ti * d..(ti + 1) * d], dst);
        }
        y
    }
}

/// LayerNorm -> pw1 -> gelu -> pw2 -> 残差 (dst += mlp)。tmp は dw 出力。
fn mlp(b: &Blk, tmp: &mut [f32], u: &mut [f32], dst: &mut [f32]) {
    mlp_pool(None, b, tmp, u, dst)
}

fn mlp_pool(pool: Option<&EgPool>, b: &Blk, tmp: &mut [f32], u: &mut [f32], dst: &mut [f32]) {
    let d = tmp.len();
    let mu = tmp.iter().sum::<f32>() / d as f32;
    let var = tmp.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for o in 0..d {
        tmp[o] = (tmp[o] - mu) * inv * b.nw[o] + b.nb[o];
    }
    u.copy_from_slice(&b.pw1_b);
    matvec_par(pool, &b.pw1_w, tmp, u);
    for v in u.iter_mut() {
        *v = gelu_erf(*v);
    }
    matvec_par(pool, &b.pw2_w, u, dst);
    // pw2_b は matvec が加算しないのでここで足す (残差 dst には既に r が入っている)
    for o in 0..d {
        dst[o] += b.pw2_b[o];
    }
}

/// ストリーム実行: 1 フレームずつ。各層が左文脈 2*dil フレームのリングを保持。
pub struct EgStream {
    inp_hist: Vec<f32>, // [2][cin]
    blk_hist: Vec<Vec<f32>>, // 層 i: [2*dil][dim] リング
    pos: Vec<usize>,
    n_seen: usize,
}

impl EgStream {
    pub fn new(net: &Eg1d) -> Self {
        EgStream {
            inp_hist: vec![0f32; 2 * net.cin],
            blk_hist: net.blk.iter().map(|b| vec![0f32; 2 * b.dil * net.dim]).collect(),
            pos: vec![0; net.layers],
            n_seen: 0,
        }
    }

    pub fn reset(&mut self) {
        self.inp_hist.iter_mut().for_each(|v| *v = 0.0);
        for h in &mut self.blk_hist {
            h.iter_mut().for_each(|v| *v = 0.0);
        }
        self.pos.iter_mut().for_each(|v| *v = 0);
        self.n_seen = 0;
    }

    /// x: [cin] 1 フレーム。返り: [cout]。バッチ実行と同一の演算順。
    pub fn step(&mut self, net: &Eg1d, x: &[f32]) -> Vec<f32> {
        self.step_par(net, x, None)
    }

    /// pool 付き。分割は出力チャネルで各 y[o] は単一スレッド＝ビット一致。
    pub fn step_par(&mut self, net: &Eg1d, x: &[f32], pool: Option<&EgPool>) -> Vec<f32> {
        let d = net.dim;
        let cin = net.cin;
        let mut h = net.inp_b.clone();
        for j in 0..3usize {
            let xr: &[f32] = match j {
                0 if self.n_seen >= 2 => &self.inp_hist[0..cin],
                1 if self.n_seen >= 1 => &self.inp_hist[cin..2 * cin],
                2 => x,
                _ => &[],
            };
            if xr.is_empty() {
                continue;
            }
            matvec_par(pool, &net.inp_w[j], xr, &mut h);
        }
        self.inp_hist.copy_within(cin..2 * cin, 0);
        self.inp_hist[cin..2 * cin].copy_from_slice(x);

        let mut tmp = vec![0f32; d];
        let mut u = vec![0f32; 3 * d];
        for (li, b) in net.blk.iter().enumerate() {
            let cap = 2 * b.dil;
            tmp.copy_from_slice(&b.dw_b);
            for j in 0..2usize {
                let back = (2 - j) * b.dil;
                if self.n_seen < back {
                    continue;
                }
                let idx = (self.pos[li] + cap - back) % cap;
                let xr = &self.blk_hist[li][idx * d..(idx + 1) * d];
                matvec_par(pool, &b.dw_w[j], xr, &mut tmp);
            }
            matvec_par(pool, &b.dw_w[2], &h, &mut tmp);
            let idx = self.pos[li];
            self.blk_hist[li][idx * d..(idx + 1) * d].copy_from_slice(&h);
            self.pos[li] = (idx + 1) % cap;
            mlp_pool(pool, b, &mut tmp, &mut u, &mut h);
        }
        self.n_seen += 1;
        let mut y = net.out_b.clone();
        matvec_par(pool, &net.out_w, &h, &mut y);
        y
    }
}


// ---------------------------------------------------------------------------
// f16 重み版。E+G=36MB/フレームの帯域律速に対し重みを半減する
// （kernel は F16C で 8 要素ずつ f32 へ展開して FMA。x/y/正規化は f32 のまま）。
// step のロジックは EgStream::step_par と 1:1 対応——変更は必ず両方に入れる。
// ---------------------------------------------------------------------------

fn to_f16_bits(v: f32) -> u16 {
    // IEEE 754 half へ round-to-nearest-even で変換（load 時のみ・速度不問）
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        if (m & (half - 1)) > 0 || ((m >> (shift - 1)) & 3) == 3 {
            r += (m & half != 0) as u32 * 0;
        }
        // round to nearest even
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

fn q16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| to_f16_bits(x)).collect()
}

/// y[o] += Σ_i x[i] * f16(w[i*co+o])。要 avx2+fma+f16c。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matvec_avx2_h(w: &[u16], x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let co = y.len();
    let cv = co / 8 * 8;
    unsafe {
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let bx = _mm256_set1_ps(xi);
            let wr = w.as_ptr().add(i * co);
            let yp = y.as_mut_ptr();
            let mut o = 0usize;
            while o < cv {
                let wh = _mm_loadu_si128(wr.add(o) as *const __m128i);
                let wf = _mm256_cvtph_ps(wh);
                let acc = _mm256_fmadd_ps(bx, wf, _mm256_loadu_ps(yp.add(o)));
                _mm256_storeu_ps(yp.add(o), acc);
                o += 8;
            }
            while o < co {
                let wf = f16_to_f32(*w.get_unchecked(i * co + o));
                *y.get_unchecked_mut(o) += xi * wf;
                o += 1;
            }
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let b = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 - 10;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 10 + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(b)
}

fn matvec_h(w: &[u16], x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            unsafe { matvec_avx2_h(w, x, y) };
            return;
        }
    }
    let co = y.len();
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 {
            continue;
        }
        for o in 0..co {
            y[o] += xi * f16_to_f32(w[i * co + o]);
        }
    }
}

struct BlkH {
    dw_w: [Vec<u16>; 3],
    dw_b: Vec<f32>,
    nw: Vec<f32>,
    nb: Vec<f32>,
    pw1_w: Vec<u16>,
    pw1_b: Vec<f32>,
    pw2_w: Vec<u16>,
    pw2_b: Vec<f32>,
    dil: usize,
}

pub struct Eg1dH {
    pub dim: usize,
    pub layers: usize,
    pub cin: usize,
    pub cout: usize,
    inp_w: [Vec<u16>; 3],
    inp_b: Vec<f32>,
    blk: Vec<BlkH>,
    out_w: Vec<u16>,
    out_b: Vec<f32>,
}

impl Eg1dH {
    pub fn from_net(n: &Eg1d) -> Self {
        Eg1dH {
            dim: n.dim,
            layers: n.layers,
            cin: n.cin,
            cout: n.cout,
            inp_w: [q16(&n.inp_w[0]), q16(&n.inp_w[1]), q16(&n.inp_w[2])],
            inp_b: n.inp_b.clone(),
            blk: n
                .blk
                .iter()
                .map(|b| BlkH {
                    dw_w: [q16(&b.dw_w[0]), q16(&b.dw_w[1]), q16(&b.dw_w[2])],
                    dw_b: b.dw_b.clone(),
                    nw: b.nw.clone(),
                    nb: b.nb.clone(),
                    pw1_w: q16(&b.pw1_w),
                    pw1_b: b.pw1_b.clone(),
                    pw2_w: q16(&b.pw2_w),
                    pw2_b: b.pw2_b.clone(),
                    dil: b.dil,
                })
                .collect(),
            out_w: q16(&n.out_w),
            out_b: n.out_b.clone(),
        }
    }
}

fn mlp_h(b: &BlkH, tmp: &mut [f32], u: &mut [f32], dst: &mut [f32]) {
    let d = tmp.len();
    let mu = tmp.iter().sum::<f32>() / d as f32;
    let var = tmp.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for o in 0..d {
        tmp[o] = (tmp[o] - mu) * inv * b.nw[o] + b.nb[o];
    }
    u.copy_from_slice(&b.pw1_b);
    matvec_h(&b.pw1_w, tmp, u);
    for v in u.iter_mut() {
        *v = gelu_erf(*v);
    }
    matvec_h(&b.pw2_w, u, dst);
    for o in 0..d {
        dst[o] += b.pw2_b[o];
    }
}

impl EgStream {
    pub fn new_h(net: &Eg1dH) -> Self {
        EgStream {
            inp_hist: vec![0f32; 2 * net.cin],
            blk_hist: net.blk.iter().map(|b| vec![0f32; 2 * b.dil * net.dim]).collect(),
            pos: vec![0; net.layers],
            n_seen: 0,
        }
    }

    /// f16 版 step。ロジックは step_par と 1:1（変更は両方へ）。
    pub fn step_h(&mut self, net: &Eg1dH, x: &[f32]) -> Vec<f32> {
        let d = net.dim;
        let cin = net.cin;
        let mut h = net.inp_b.clone();
        for j in 0..3usize {
            let xr: &[f32] = match j {
                0 if self.n_seen >= 2 => &self.inp_hist[0..cin],
                1 if self.n_seen >= 1 => &self.inp_hist[cin..2 * cin],
                2 => x,
                _ => &[],
            };
            if xr.is_empty() {
                continue;
            }
            matvec_h(&net.inp_w[j], xr, &mut h);
        }
        self.inp_hist.copy_within(cin..2 * cin, 0);
        self.inp_hist[cin..2 * cin].copy_from_slice(x);

        let mut tmp = vec![0f32; d];
        let mut u = vec![0f32; 3 * d];
        for (li, b) in net.blk.iter().enumerate() {
            let cap = 2 * b.dil;
            tmp.copy_from_slice(&b.dw_b);
            for j in 0..2usize {
                let back = (2 - j) * b.dil;
                if self.n_seen < back {
                    continue;
                }
                let idx = (self.pos[li] + cap - back) % cap;
                let xr = &self.blk_hist[li][idx * d..(idx + 1) * d];
                matvec_h(&b.dw_w[j], xr, &mut tmp);
            }
            matvec_h(&b.dw_w[2], &h, &mut tmp);
            let idx = self.pos[li];
            self.blk_hist[li][idx * d..(idx + 1) * d].copy_from_slice(&h);
            self.pos[li] = (idx + 1) % cap;
            mlp_h(b, &mut tmp, &mut u, &mut h);
        }
        self.n_seen += 1;
        let mut y = net.out_b.clone();
        matvec_h(&net.out_w, &h, &mut y);
        y
    }
}
