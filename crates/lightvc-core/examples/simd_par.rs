use lightvc_core::simd::*;
use std::time::Instant;

fn mk(ch: usize, nl: usize, kf: usize, kt: usize) -> Vec<Layer> {
    (0..nl).map(|i| {
        let mut l = Layer::new(ch, kf, kt, 1 << (i / 2));
        for (k, v) in l.w_c.iter_mut().enumerate() { *v = ((k % 17) as f32 - 8.0) * 0.002; }
        for (k, v) in l.w1.iter_mut().enumerate() { *v = ((k % 13) as f32 - 6.0) * 0.003; }
        for (k, v) in l.w2.iter_mut().enumerate() { *v = ((k % 11) as f32 - 5.0) * 0.004; }
        l
    }).collect()
}

fn main() {
    let (kf, kt, emit) = (7usize, 3usize, 2usize);
    let blk = emit as f64 * 128.0 / 44100.0;
    println!("{:>5} {:>3} {:>5} {:>9} {:>9} {:>8}  予算 0.25", "ch", "L", "F", "RTF-1t", "RTF-2t", "速度比");
    for (ch, nl, f) in [(16usize, 8usize, 257usize), (24, 8, 257), (24, 6, 257), (32, 8, 257)] {
        let layers = mk(ch, nl, kf, kt);
        let mut x = Ten::zeros(ch, emit, f);
        for (k, v) in x.d.iter_mut().enumerate() { *v = ((k % 23) as f32 - 11.0) * 0.05; }
        let mut s1 = TrunkState::new(&layers, f);
        for _ in 0..20 { let _ = s1.step(&layers, &x); }
        let n = 200;
        let t0 = Instant::now();
        for _ in 0..n { let _ = s1.step(&layers, &x); }
        let e1 = t0.elapsed().as_secs_f64() / n as f64;

        let pool = Pool::new(1);  // ワーカ 1 + 呼び出し側 = 2 スレッド
        let mut s2 = TrunkState::new(&layers, f);
        for _ in 0..20 { let _ = unsafe { s2.step_par(&layers, &x, &pool) }; }
        let t0 = Instant::now();
        for _ in 0..n { let _ = unsafe { s2.step_par(&layers, &x, &pool) }; }
        let e2 = t0.elapsed().as_secs_f64() / n as f64;

        let mut a = TrunkState::new(&layers, f);
        let mut b = TrunkState::new(&layers, f);
        let ra = a.step(&layers, &x);
        let rb = unsafe { b.step_par(&layers, &x, &pool) };
        let err = ra.d.iter().zip(&rb.d).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        println!("{:>5} {:>3} {:>5} {:>9.4} {:>9.4} {:>8.4}{}  誤差 {:.0e}",
                 ch, nl, f, e1 / blk, e2 / blk, e1 / e2,
                 if e2 / blk < 0.25 { "  PASS" } else { "" }, err);
    }
}
