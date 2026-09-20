//! Y-S1 causal codec decoder（training/causal_codec.py `CausalDecoderStream` と parity）。
//!
//! c32 構成: pre Conv1d(32→512,k7) → 4×[ConvTranspose(2r,r) overlap-add +
//! 3×ResUnit(k7 dil 1,3,9)] → SnakeBeta → post Conv1d(32→1,k7) → tanh。
//! strides 実行順 (3,4,5,8)。channels (512→256→128→64→32)。hop=480。
//!
//! streaming: `decode_step(z: &[f32;32]) -> [f32; 480]`。
//! ConvTranspose は frame が生む 2r サンプルの**後半 r を state に保持**し、
//! 次 frame の前半 r と overlap-add、確定した先頭 r を下流へ渡す
//! （Python `UpStream` と同一規則・bias は emit に一度だけ加算）。
//! ResUnit の conv1 は左pad (k-1)*dil を state 保持。Snake の alpha/beta は
//! **log-domain** で保存され、ここで exp() する。
//! 重み layout は training/export_ys1.py の manifest と完全一致。

use std::fs;

pub const LATENT_DIM: usize = 32;
pub const HOP: usize = 480;

pub struct ResUnit {
    pub c: usize,
    pub hidden: usize,
    pub dil: usize,
    pub a1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w1: Vec<u16>,
    pub bias1: Vec<f32>,
    pub a2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w2: Vec<u16>,
    pub bias2: Vec<f32>,
    pub st1: Vec<f32>, // pad (=6*dil) x c … conv1 への左文脳
}

pub struct UpStage {
    pub cout: usize,
    pub cin: usize,
    pub stride: usize,
    pub up_w: Vec<f32>,
    pub up_wt: Vec<u16>,   // [ci][k*cout + o]（broadcast-FMA 用）
    pub up_wt2: Vec<u16>,  // [o2*cin + ci]（出力側 dot 用）
    pub up_b: Vec<f32>,
    pub res: Vec<ResUnit>,
    pub tail: Vec<f32>,
    pub scr: Vec<Vec<f32>>,
}

pub struct CodecDecoder {
    pub pre_w: Vec<u16>,
    worker: Option<std::sync::Arc<Wk>>,
    pub pre_b: Vec<f32>,
    pub stages: Vec<UpStage>,
    post_a: Vec<f32>,
    post_b: Vec<f32>,
    post_w: Vec<f32>, // [32*7] (1 出力なので [7][32] を flatten: w[k*32+c])
    post_bias: f32,
    pre_state: Vec<f32>,
    post_state: Vec<f32>,
    pub post_ea: Vec<f32>,
    pub post_eb: Vec<f32>,
}

struct Wk {
    seq: std::sync::atomic::AtomicUsize,
    done: std::sync::atomic::AtomicUsize,
    quit: std::sync::atomic::AtomicBool,
    job: std::cell::UnsafeCell<SnJob>,
}
unsafe impl Sync for Wk {}
unsafe impl Send for SnJob {}
unsafe impl Send for Wk {}

#[derive(Clone, Copy)]
struct SnJob {
    cur: *const f32,
    ea: *const f32,
    eb: *const f32,
    sn: *mut f32,
    t0: usize,
    t1: usize,
    n: usize,
}

impl Drop for CodecDecoder {
    fn drop(&mut self) {
        if let Some(w) = &self.worker {
            w.quit.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    map: std::collections::HashMap<String, (usize, usize)>,
}

impl<'a> Reader<'a> {
    fn get(&self, key: &str) -> Vec<f32> {
        let (off, n) = self.map[key];  // off はバイト単位 (export_ys1.py)
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let b = &self.data[off + i * 4..off + (i + 1) * 4];
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        out
    }
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    f32::from_bits(((h as u32 & 0x8000) << 16)
        | (((h as u32 & 0x7c00) + 0x1c000) << 13)
        | ((h as u32 & 0x03ff) << 13))
}

#[inline]
fn f32_to_f16(x: f32) -> u16 {
    // RTNE (デフォルト). 0/subnormal は簡易(本重み範囲で問題なし)
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = (b & 0x007f_ffff) as i32;
    if exp == 0xff {
        return sign | 0x7c00;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7bff;
    }
    if e <= 0 {
        return sign;
    }
    let m = (man + 0x1000) >> 13;
    let mut out = sign | ((e as u32) << 10) as u16 | m as u16;
    if m > 0x3ff {
        out = sign | (((e + 1) as u32) << 10) as u16;
    }
    out
}

#[inline]
pub fn snake_pub(x: f32, a_log: f32, b_log: f32) -> f32 {
    snake(x, a_log, b_log)
}

fn snake(x: f32, a_log: f32, b_log: f32) -> f32 {
    snake_p(x, a_log.exp(), b_log.exp())
}



#[inline]
fn snake_p(x: f32, a: f32, b: f32) -> f32 {
    x + (a * x).sin().powi(2) / (b + 1e-9)
}

fn up_stage_fma(st: &mut UpStage, x: &[f32]) -> Vec<f32> {
    let r = st.stride;
    let n = x.len() / st.cin;
    let w2 = st.cout * 2 * r;
    let mut raw = vec![0f32; (n * r + r) * st.cout];
    let mut buf = vec![0f32; w2];
    for i in 0..n {
        let xin = &x[i * st.cin..(i + 1) * st.cin];
        unsafe { dot_f16_out(&st.up_wt2, xin, st.cin, &mut buf) };
        let base = i * r * st.cout;
        for (o, v) in buf.iter().enumerate() {
            raw[base + o] += v;
        }
    }
    let mut emit: Vec<f32> = Vec::with_capacity(n * r * st.cout);
    for (o, v) in raw[..r * st.cout].iter_mut().enumerate() {
        *v += st.tail[o];
    }
    emit.extend_from_slice(&raw[..n * r * st.cout]);
    st.tail.copy_from_slice(&raw[n * r * st.cout..]);
    for (o, v) in emit.iter_mut().enumerate() {
        *v += st.up_b[o % st.cout];
    }
    let mut out: Vec<f32> = Vec::with_capacity(emit.len());
    for t in 0..(emit.len() / st.cout) {
        let mut xt: Vec<f32> = emit[t * st.cout..(t + 1) * st.cout].to_vec();
        for ru in st.res.iter_mut() {
            xt = res_unit(ru, &xt);
        }
        out.extend_from_slice(&xt);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f16_out(w: &[u16], x: &[f32], n: usize, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    // タイル化: 8 出力を同時進行し x ブロックをレジスタ共有
    let cv = n / 8 * 8;
    let no = dst.len();
    let ob = no / 8 * 8;
    let xp = x.as_ptr();
    let mut o0 = 0;
    while o0 < ob {
        let mut accs: [_; 8] = std::mem::zeroed();
        std::ptr::write(&mut accs as *mut [_] as *mut [__m256; 8], {
            let mut a: [__m256; 8] = std::mem::zeroed();
            for v in a.iter_mut() { *v = _mm256_set1_ps(0.0); }
            a
        });
        let accs = &mut *(accs.as_mut_ptr() as *mut [__m256; 8]);
        let mut i = 0;
        while i < cv {
            let xv = _mm256_loadu_ps(xp.add(i));
            for b in 0..8 {
                let wr = w.as_ptr().add((o0 + b) * n + i);
                let wh = _mm_loadu_si128(wr as *const __m128i);
                let wf = _mm256_cvtph_ps(wh);
                accs[b] = _mm256_fmadd_ps(wf, xv, accs[b]);
            }
            i += 8;
        }
        for b in 0..8 {
            let mut m = [0f32; 8];
            _mm256_storeu_ps(m.as_mut_ptr(), accs[b]);
            let mut s = 0f32;
            for v in m { s += v; }
            let mut j = cv;
            while j < n {
                s += f16_to_f32(*w.get_unchecked((o0 + b) * n + j))
                    * *x.get_unchecked(j);
                j += 1;
            }
            dst[o0 + b] = s;
        }
        o0 += 8;
    }
    while o0 < no {
        let wr = w.as_ptr().add(o0 * n);
        let mut acc = _mm256_set1_ps(0.0);
        let mut i = 0;
        while i < cv {
            let wh = _mm_loadu_si128(wr.add(i) as *const __m128i);
            let wf = _mm256_cvtph_ps(wh);
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xp.add(i)), acc);
            i += 8;
        }
        let mut m = [0f32; 8];
        _mm256_storeu_ps(m.as_mut_ptr(), acc);
        let mut s = 0f32;
        for v in m { s += v; }
        while i < n {
            s += f16_to_f32(*w.get_unchecked(o0 * n + i)) * *x.get_unchecked(i);
            i += 1;
        }
        dst[o0] = s;
        o0 += 1;
    }
}

fn make_worker() -> std::sync::Arc<Wk> {
    let wk = std::sync::Arc::new(Wk {
        seq: std::sync::atomic::AtomicUsize::new(0),
        done: std::sync::atomic::AtomicUsize::new(0),
        quit: std::sync::atomic::AtomicBool::new(false),
        job: std::cell::UnsafeCell::new(SnJob {
            cur: std::ptr::null(), ea: std::ptr::null(), eb: std::ptr::null(),
            sn: std::ptr::null_mut(), t0: 0, t1: 0, n: 0,
        }),
    });
    let w2 = wk.clone();
    std::thread::spawn(move || {
        use std::sync::atomic::Ordering::*;
        let mut seen = 0usize;
        loop {
            let s = w2.seq.load(Acquire);
            if s == seen {
                if w2.quit.load(Relaxed) { return; }
                std::hint::spin_loop();
                continue;
            }
            seen = s;
            let j = unsafe { *w2.job.get() };
            unsafe {
                for t in j.t0..j.t1 {
                    for c in 0..32usize {
                        *j.sn.add(t * 32 + c) = snake_p(
                            *j.cur.add(t * 32 + c),
                            *j.ea.add(c), *j.eb.add(c));
                    }
                }
            }
            w2.done.store(seen, Release);
        }
    });
    wk
}

impl CodecDecoder {
    pub fn load(bin: &str, json: &str) -> anyhow::Result<Self> {
        let data = fs::read(bin)?;
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(json)?)?;
        let mut map = std::collections::HashMap::new();
        for t in meta["tensors"].as_array().unwrap() {
            map.insert(
                t["key"].as_str().unwrap().to_string(),
                (t["offset"].as_u64().unwrap() as usize,
                 t["count"].as_u64().unwrap() as usize),
            );
        }
        let r = Reader { data: &data, map };
        let st_cfg = [(512usize, 256usize, 3usize), (256, 128, 4), (128, 64, 5), (64, 32, 8)];
        let mut stages = Vec::new();
        for (i, &(cin, cout, stride)) in st_cfg.iter().enumerate() {
            let mut res = Vec::new();
            for (j, &dil) in [1usize, 3, 9].iter().enumerate() {
                let w1raw = r.get(&format!("decoder.stages.{i}.res.{j}.conv1.weight"));
                let hidden = cout / 2;
                // torch [out, in, k] -> [out][in*k] は contiguous で同一 layout
                res.push(ResUnit {
                    c: cout,
                    hidden,
                    dil,
                    a1: r.get(&format!("decoder.stages.{i}.res.{j}.act1.alpha")),
                    b1: r.get(&format!("decoder.stages.{i}.res.{j}.act1.beta")),
                    w1: w1raw.iter().map(|v| f32_to_f16(*v)).collect(),
                    bias1: r.get(&format!("decoder.stages.{i}.res.{j}.conv1.bias")),
                    a2: r.get(&format!("decoder.stages.{i}.res.{j}.act2.alpha")),
                    b2: r.get(&format!("decoder.stages.{i}.res.{j}.act2.beta")),
                    w2: r.get(&format!("decoder.stages.{i}.res.{j}.conv2.weight"))
                        .iter().map(|v| f32_to_f16(*v)).collect(),
                    bias2: r.get(&format!("decoder.stages.{i}.res.{j}.conv2.bias")),
                    st1: vec![0.0; 6 * dil * cout],
                });
            }
            let upw = r.get(&format!("decoder.stages.{i}.up.weight"));
            // torch [cin, cout, 2r] -> [ci][k*cout+o]
            let k2 = 2 * stride;
            let mut up_wt = vec![0u16; cin * cout * k2];
            for ci in 0..cin {
                for o in 0..cout {
                    for k in 0..k2 {
                        up_wt[ci * k2 * cout + k * cout + o] =
                            f32_to_f16(upw[ci * cout * k2 + o * k2 + k]);
                    }
                }
            }
            // 出力側 dot 用転置: wt2[o2*cin + ci]
            let k2 = 2 * stride;
            let mut up_wt2 = vec![0u16; cin * cout * k2];
            for ci in 0..cin {
                for idx in 0..(cout * k2) {
                    up_wt2[idx * cin + ci] = up_wt[ci * cout * k2 + idx];
                }
            }
            stages.push(UpStage {
                cin,
                cout,
                stride,
                up_w: upw,
                up_wt,
                up_wt2,
                up_b: r.get(&format!("decoder.stages.{i}.up.bias")),
                res,
                tail: vec![0.0; stride * cout],
                scr: (0..3).map(|_| vec![0.0; cout * 7 + cout / 2 * 3 + cout]).collect(),
            });
        }
        let pa = r.get("decoder.post_act.alpha");
        let pb = r.get("decoder.post_act.beta");
        let net_ea: Vec<f32> = pa.iter().map(|v| v.exp()).collect();
        let net_eb: Vec<f32> = pb.iter().map(|v| v.exp()).collect();
        Ok(Self {
            pre_w: r.get("decoder.pre.weight").iter().map(|v| f32_to_f16(*v)).collect(),
            pre_b: r.get("decoder.pre.bias"),
            stages,
            post_a: pa,
            post_b: pb,
            post_w: r.get("decoder.post.weight"),
            post_bias: r.get("decoder.post.bias")[0],
            pre_state: vec![0.0; 6 * LATENT_DIM],
            post_state: vec![0.0; 6 * 32],
            worker: Some(make_worker()),
            post_ea: net_ea,
            post_eb: net_eb,
        })
    }

    pub fn reset(&mut self) {
        self.pre_state.fill(0.0);
        self.post_state.fill(0.0);
        for st in &mut self.stages {
            st.tail.fill(0.0);
            for r in &mut st.res {
                r.st1.fill(0.0);
            }
        }
    }

    /// z: [32]（1 latent frame）→ 480 samples。
    pub fn decode_step(&mut self, z: &[f32]) -> Vec<f32> {
        // --- pre: Conv1d(32->512, k7, 左pad6) ---
        // channel ごとの直近 7 時刻列を組み、weight flat 順 [c*7+k] と突き合わせる
        // hist: [c][6]（過去）→ cm[c][k=0..6] = (hist[c][..], z[c])
        let hist = &self.pre_state;          // 6*32, layout [c*6 + t]
        let mut cm = [0f32; LATENT_DIM * 7]; // [c*7 + k]
        for c in 0..LATENT_DIM {
            for k in 0..6 {
                cm[c * 7 + k] = hist[c * 6 + k];
            }
            cm[c * 7 + 6] = z[c];
        }
        let mut h = vec![0f32; 512];
        fma_h(&self.pre_w, &cm, &self.pre_b, &mut h, LATENT_DIM * 7);
        for c in 0..LATENT_DIM {
            for k in 0..6 {
                self.pre_state[c * 6 + k] = cm[c * 7 + k + 1];
            }
        }

        // --- stages ---
        let mut cur: Vec<f32> = h; // 長さ r_prev x cout_prev（stage ごとに変化）
        for st in &mut self.stages {
            cur = up_stage_fma(st, &cur);
        }

        // --- post: SnakeBeta + Conv1d(32->1, k7 左pad6) ---
        // 入力: cur = 時刻 major [t][32]。snake 適用後の系列 sn[t*32+c]。
        let n = cur.len() / 32;
        let mut sn = vec![0f32; n * 32];
        for t in 0..n {
            for c in 0..32 {
                sn[t * 32 + c] = snake_p(cur[t * 32 + c], self.post_ea[c], self.post_eb[c]);
            }
        }
        // 各 channel の系列（左pad=post_state[c*7+k]… hist は channel-major [c][6]）
        // post_state layout: [c*6 + t]
        let mut y = vec![0f32; HOP];
        for t in 0..n.min(HOP) {
            // 窓ベクトル [c*7+k] を組み立て 1 dot
            let mut win = [0f32; 32 * 7];
            for c in 0..32 {
                for k in 0..7 {
                    let pos = t as isize - (6 - k) as isize;
                    win[c * 7 + k] = if pos < 0 {
                        self.post_state[c * 6 + (6 + pos as usize)]
                    } else {
                        sn[pos as usize * 32 + c]
                    };
                }
            }
            let mut acc = self.post_bias;
            for (idx, &xv) in win.iter().enumerate() {
                acc += self.post_w[idx] * xv;
            }
            y[t] = acc.tanh();
        }
        // 状態更新: 各 ch の sn 末尾 6 時刻
        if n >= 1 {
            for c in 0..32 {
                for k in 0..6 {
                    let pos = n as isize - 6 + k as isize;
                    self.post_state[c * 6 + k] = if pos < 0 { 0.0 }
                        else { sn[pos as usize * 32 + c] };
                }
            }
        }
        y
    }
}

fn res_unit(ru: &mut ResUnit, x: &[f32]) -> Vec<f32> {
    let c = ru.c;
    // conv1 入力 = snake1(x)。各 channel の時刻系列（左pad = st1 + 現在 1 点）
    // st1 layout: [c][pad]
    let pad = 6 * ru.dil;
    // 窓ベクトル win[ch*7+k] を組み立ててから各 o を連続 FMA
    let mut win = vec![0f32; c * 7];
    {
        let ea: Vec<f32> = ru.a1.iter().map(|v| v.exp()).collect();
        let eb: Vec<f32> = ru.b1.iter().map(|v| v.exp()).collect();
        for ch in 0..c {
            let s_new = snake_p(x[ch], ea[ch], eb[ch]);
            for k in 0..7 {
                let off = (6 - k) as usize * ru.dil;
                win[ch * 7 + k] = if k == 6 { s_new }
                    else if off <= pad { ru.st1[ch * pad + pad - off] } else { 0.0 };
            }
        }
    }
    let mut y1 = vec![0f32; ru.hidden];
    fma_h(&ru.w1, &win, &ru.bias1, &mut y1, c * 7);
    // 状態更新: 各 ch の履歴を 1 要素シフト(alloc なし)
    for ch in 0..c {
        let base = ch * pad;
        ru.st1.copy_within(base + 1..base + pad, base);
        ru.st1[base + pad - 1] = win[ch * 7 + 6];
    }
    // conv2 (k1): snake2 適用後に flat FMA
    let mut s2 = vec![0f32; ru.hidden];
    {
        let ea: Vec<f32> = ru.a2.iter().map(|v| v.exp()).collect();
        let eb: Vec<f32> = ru.b2.iter().map(|v| v.exp()).collect();
        for (h, v) in y1.iter().enumerate() {
            s2[h] = snake_p(*v, ea[h], eb[h]);
        }
    }
    let mut add = vec![0f32; c];
    fma_h(&ru.w2, &s2, &ru.bias2, &mut add, ru.hidden);
    let mut out = x.to_vec();
    for o in 0..c { out[o] += add[o]; }
    out
}

/// f16 重み版: y[o] = bias[o] + Σ x[i]*f16(w[o*n+i])
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn fma_flat_h(w: &[u16], x: &[f32], bias: &[f32], y: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let cv = n / 8 * 8;
    for o in 0..y.len() {
        let wr = w.as_ptr().add(o * n);
        let xp = x.as_ptr();
        let mut acc = _mm256_set1_ps(0.0);
        let mut i = 0;
        while i < cv {
            let wh = _mm_loadu_si128(wr.add(i) as *const __m128i);
            let wf = _mm256_cvtph_ps(wh);
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xp.add(i)), acc);
            i += 8;
        }
        let mut m = [0f32; 8];
        _mm256_storeu_ps(m.as_mut_ptr(), acc);
        let mut s = bias[o];
        for v in m { s += v; }
        while i < n {
            s += f16_to_f32(*w.get_unchecked(o * n + i)) * *x.get_unchecked(i);
            i += 1;
        }
        y[o] = s;
    }
}

pub fn fma_h_pub(w: &[u16], x: &[f32], bias: &[f32], y: &mut [f32], n: usize) {
    fma_h(w, x, bias, y, n)
}

pub fn fma_h(w: &[u16], x: &[f32], bias: &[f32], y: &mut [f32], n: usize) {
    unsafe { fma_flat_h(w, x, bias, y, n) }
}

/// x[0..n] と w[o*n..] の dot を SIMD FMA で y[o] = bias[o] + Σ に書き込む。
fn up_stage_fma_flat(w: &[f32], x: &[f32], bias: &[f32], y: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let cv = n / 8 * 8;
    unsafe {
        for o in 0..y.len() {
            let wr = w.as_ptr().add(o * n);
            let xp = x.as_ptr();
            let mut acc = _mm256_set1_ps(0.0);
            let mut i = 0;
            while i < cv {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(wr.add(i)),
                                      _mm256_loadu_ps(xp.add(i)), acc);
                i += 8;
            }
            let mut m = [0f32; 8];
            _mm256_storeu_ps(m.as_mut_ptr(), acc);
            let mut s = bias[o];
            for v in m { s += v; }
            while i < n {
                s += w[o * n + i] * x[i];
                i += 1;
            }
            y[o] = s;
        }
    }
}
