use lightvc_core::ship_front as SF;
use std::time::Instant;
fn bench<F: FnMut()>(n: usize, mut f: F) -> f64 {
    for _ in 0..20 { f(); }
    let t0 = Instant::now();
    for _ in 0..n { f(); }
    t0.elapsed().as_secs_f64() / n as f64
}
fn main() {
    let td = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let fb: Vec<f32> = std::fs::read(td.join("mel_fb_1024_80.bin")).unwrap()
        .chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let k = SF::HOP_S * 2;
    let mut buf = vec![0.0f32; SF::NFFT_A];
    for (i, v) in buf.iter_mut().enumerate() { *v = ((i as f32) * 0.013).sin() * 0.1; }
    let noise: Vec<f32> = (0..k).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let blk = k as f64 / SF::SR as f64;
    let mut fs = SF::FrontState::new();
    let e1 = bench(300, || { let _ = fs.step(&buf, &fb); });
    // 有声時の励起（f0 = 110 Hz -> 200 倍音）
    let f0v = vec![110.0f32; 2];
    let e2 = bench(300, || { let _ = SF::excitation(&f0v, k, &noise, 0.3); });
    // 高め f0（f0 = 440 -> 50 倍音）
    let f0h = vec![440.0f32; 2];
    let e3 = bench(300, || { let _ = SF::excitation(&f0h, k, &noise, 0.3); });
    println!("front.step       {:8.4} ms  RTF {:.4}", e1 * 1e3, e1 / blk);
    println!("excitation f0=110 {:7.4} ms  RTF {:.4}", e2 * 1e3, e2 / blk);
    println!("excitation f0=440 {:7.4} ms  RTF {:.4}", e3 * 1e3, e3 / blk);
}
