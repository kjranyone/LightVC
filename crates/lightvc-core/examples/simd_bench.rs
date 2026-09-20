//! 自前 SIMD カーネルの到達スループット。必要値は 106 GFLOPS（V2F ch24/L8）。
use lightvc_core::simd::*;
use std::time::Instant;

fn main() {
    let (f, t, kf, kt) = (257usize, 2usize, 7usize, 3usize);
    println!("{:>4} {:>4} {:>7} {:>10} {:>10}", "ch", "dil", "iters", "ms/blk", "GFLOPS");
    for ch in [16usize, 24, 32, 48] {
        let inp = Ten { c: ch, t: t + kt - 1, f, d: vec![0.1; ch * (t + kt - 1) * f] };
        let w = vec![0.01f32; ch * ch * kf * kt];
        let b = vec![0.0f32; ch];
        let w1 = vec![0.01f32; 3 * ch * ch];
        let b1 = vec![0.0f32; 3 * ch];
        let w2 = vec![0.01f32; ch * 3 * ch];
        let b2 = vec![0.0f32; ch];
        let n = 200;
        for _ in 0..20 {
            let h = conv_ft(&inp, &w, &b, ch, kf, kt, 1);
            let mut h = pointwise(&h, &w1, &b1, 3 * ch);
            gelu_inplace(&mut h.d);
            let _ = pointwise(&h, &w2, &b2, ch);
        }
        let t0 = Instant::now();
        for _ in 0..n {
            let mut h = conv_ft(&inp, &w, &b, ch, kf, kt, 1);
            freq_norm(&mut h, &b2, &b2, 1e-5);
            let mut h = pointwise(&h, &w1, &b1, 3 * ch);
            gelu_inplace(&mut h.d);
            let _ = pointwise(&h, &w2, &b2, ch);
        }
        let el = t0.elapsed().as_secs_f64() / n as f64;
        // 1 層あたりの MAC
        let mac = (ch * ch * kf * kt + ch * 3 * ch + 3 * ch * ch) * f * t;
        println!("{:>4} {:>4} {:>7} {:>10.4} {:>10.1}",
                 ch, 1, n, el * 1e3, 2.0 * mac as f64 / el / 1e9);
    }
}
