//! C.4 full graph（v2f ch24/L8 SIMD 幹 ＋ 出荷 front-end）。
//! 判定基準は rtf_basis.json のとおり **ここの実測**（合格線 0.35 − 凍結許容差）。
use lightvc_core::ship_front as SF;
use lightvc_core::simd::*;
use std::time::Instant;

fn main() {
    let blocks: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let (ch, nl, kf, kt, emit, f) = (24usize, 8usize, 7usize, 3usize, 2usize, 257usize);
    let layers: Vec<Layer> = (0..nl).map(|i| {
        let mut l = Layer::new(ch, kf, kt, 1 << (i / 2));
        for (k, v) in l.w_c.iter_mut().enumerate() { *v = ((k % 17) as f32 - 8.0) * 0.002; }
        l
    }).collect();
    let pool = Pool::new(1);
    let mut st = TrunkState::new(&layers, f);
    let mut fs = SF::FrontState::new();
    let td = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let fb: Vec<f32> = std::fs::read(td.join("mel_fb_1024_80.bin")).unwrap()
        .chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let k = SF::HOP_S * emit;
    let ring = SF::NFFT_A + 3 * SF::HOP_A;
    let mut buf = vec![0.0f32; ring + k];
    for (i, v) in buf.iter_mut().enumerate() { *v = ((i as f32) * 0.01).sin() * 0.1; }
    let noise: Vec<f32> = (0..k).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let mut x = Ten::zeros(ch, emit, f);
    for (i, v) in x.d.iter_mut().enumerate() { *v = ((i % 23) as f32 - 11.0) * 0.05; }
    let mut f0v = vec![0.0f32; emit];

    let mut run = |b: &[f32]| {
        let w0 = b.len() - SF::NFFT_A;
        let (_m, f0, _voi) = fs.step(&b[w0..], &fb);
        f0v[0] = f0;
        f0v[1] = f0;
        let _e = SF::excitation(&f0v, k, &noise, 0.3);
        let _ = unsafe { st.step_par(&layers, &x, &pool) };
    };
    for _ in 0..30 { run(&buf); }
    let audio = k as f64 / SF::SR as f64;
    let mut v = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        let t0 = Instant::now();
        run(&buf);
        v.push(t0.elapsed().as_secs_f64() / audio);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{{\"impl\":\"simd v2f ch24/L8 + front\",\"blocks\":{blocks},\
\"rtf_p50\":{:.4},\"rtf_p95\":{:.4},\"rtf_worst\":{:.4}}}",
        v[blocks / 2], v[blocks * 95 / 100], v[blocks - 1]);
}
