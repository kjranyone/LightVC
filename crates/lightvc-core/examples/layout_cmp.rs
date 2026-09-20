use lightvc_core::simd::*;
use std::time::Instant;
fn main() {
    let (kf, kt, emit) = (7usize, 3usize, 2usize);
    let blk = emit as f64 * 128.0 / 44100.0;
    println!("{:>4} {:>4} {:>10} {:>10} {:>9}  （conv 単体）", "ch", "F", "ms", "GFLOPS", "x8層RTF");
    for (ch, f) in [(16usize, 257usize), (24, 257), (32, 257), (48, 257), (64, 257)] {
        let w = vec![0.01f32; ch * ch * kf * kt];
        let b = vec![0.0f32; ch];
        let a = Ten { c: ch, t: emit + kt - 1, f, d: vec![0.1; ch * (emit + kt - 1) * f] };
        let c = TenC { t: emit + kt - 1, f, c: ch, d: vec![0.1; ch * (emit + kt - 1) * f] };
        let n = 300;
        for _ in 0..30 { let _ = conv_ft(&a, &w, &b, ch, kf, kt, 1); }
        let t0 = Instant::now();
        for _ in 0..n { let _ = conv_ft(&a, &w, &b, ch, kf, kt, 1); }
        let e1 = t0.elapsed().as_secs_f64() / n as f64;
        let _ = &c;
        let e2 = f64::NAN;
        let mac = ch * ch * kf * kt * f * emit;
        println!("{:>4} {:>4} {:>10.4} {:>10.1} {:>9.4}", ch, f, e1 * 1e3,
                 2.0 * mac as f64 / e1 / 1e9, e1 * 8.0 / blk);
    }
}
