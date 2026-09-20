use lightvc_core::simd::*;
use std::time::Instant;
fn main() {
    let (kf, kt, emit, f) = (7usize, 3usize, 2usize, 257usize);
    let blk = emit as f64 * 128.0 / 44100.0;
    println!("{:>4} {:>10} {:>10} {:>8} {:>10}", "ch", "1t[ms]", "2t[ms]", "速度比", "一致誤差");
    for ch in [16usize, 24, 32] {
        let mut w = vec![0.0f32; ch * ch * kf * kt];
        for (k, v) in w.iter_mut().enumerate() { *v = ((k % 19) as f32 - 9.0) * 0.003; }
        let mut b = vec![0.0f32; ch];
        for (k, v) in b.iter_mut().enumerate() { *v = (k as f32) * 0.01; }
        let mut inp = Ten::zeros(ch, emit + kt - 1, f);
        for (k, v) in inp.d.iter_mut().enumerate() { *v = ((k % 23) as f32 - 11.0) * 0.05; }
        let pf = ((kf - 1) * 1) / 2;
        let mut o1 = Ten::zeros(ch, emit, f);
        let mut p1 = Ten::zeros(ch, emit + kt - 1, f + 2 * pf);
        conv_ft_into(&inp, &w, &b, ch, kf, kt, 1, &mut o1, &mut p1);

        let pool = Pool::new(2);
        let mut o2 = Ten::zeros(ch, emit, f);
        let mut pads: Vec<Ten> = (0..2).map(|_| Ten::zeros(ch, emit + kt - 1, f + 2 * pf)).collect();
        unsafe { conv_ft_into_par(&pool, &inp, &w, &b, ch, kf, kt, 1, &mut o2, &mut pads) };
        let e = o1.d.iter().zip(&o2.d).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        let n = 400;
        for _ in 0..40 { conv_ft_into(&inp, &w, &b, ch, kf, kt, 1, &mut o1, &mut p1); }
        let t0 = Instant::now();
        for _ in 0..n { conv_ft_into(&inp, &w, &b, ch, kf, kt, 1, &mut o1, &mut p1); }
        let e1 = t0.elapsed().as_secs_f64() / n as f64;
        for _ in 0..40 { unsafe { conv_ft_into_par(&pool, &inp, &w, &b, ch, kf, kt, 1, &mut o2, &mut pads) }; }
        let t0 = Instant::now();
        for _ in 0..n { unsafe { conv_ft_into_par(&pool, &inp, &w, &b, ch, kf, kt, 1, &mut o2, &mut pads) }; }
        let e2 = t0.elapsed().as_secs_f64() / n as f64;
        println!("{:>4} {:>10.4} {:>10.4} {:>8.2} {:>10.1e}   (blk {:.2} ms)",
                 ch, e1 * 1e3, e2 * 1e3, e1 / e2, e, blk * 1e3);
    }
}
