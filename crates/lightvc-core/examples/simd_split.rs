use lightvc_core::simd::*;
use std::time::Instant;
fn bench<F: FnMut()>(n: usize, mut f: F) -> f64 {
    for _ in 0..30 { f(); }
    let t0 = Instant::now();
    for _ in 0..n { f(); }
    t0.elapsed().as_secs_f64() / n as f64
}
fn main() {
    let (f, kf, kt, emit, ch) = (257usize, 7usize, 3usize, 2usize, 24usize);
    let l = Layer::new(ch, kf, kt, 1);
    let inp = Ten { c: ch, t: emit + kt - 1, f, d: vec![0.1; ch * (emit + kt - 1) * f] };
    let mut sc = Scratch::new(&l, f, emit);
    let mut pad = Ten::zeros(ch, emit + kt - 1, f + 2 * ((kf - 1) / 2));
    let mut o1 = Ten::zeros(ch, emit, f);
    let mut u = Ten::zeros(3 * ch, emit, f);
    let mut o2 = Ten::zeros(ch, emit, f);
    let n = 3000;
    let e_all = bench(n, || l.forward_into(&inp, &mut sc));
    let e_c = bench(n, || conv_ft_into(&inp, &l.w_c, &l.b_c, ch, kf, kt, 1, &mut o1, &mut pad));
    let e_n = bench(n, || freq_norm(&mut o1, &l.g_n, &l.b_n, 1e-5));
    let e_p1 = bench(n, || pointwise_into(&o1, &l.w1, &l.b1, 3 * ch, &mut u));
    let e_g = bench(n, || gelu_inplace(&mut u.d));
    let e_p2 = bench(n, || pointwise_into(&u, &l.w2, &l.b2, ch, &mut o2));
    let mac_c = ch * ch * kf * kt * f * emit;
    println!("conv      {:8.4} ms  {:6.1} GFLOPS", e_c * 1e3, 2.0 * mac_c as f64 / e_c / 1e9);
    println!("norm      {:8.4} ms", e_n * 1e3);
    println!("pw1       {:8.4} ms", e_p1 * 1e3);
    println!("gelu      {:8.4} ms", e_g * 1e3);
    println!("pw2       {:8.4} ms", e_p2 * 1e3);
    println!("部品和    {:8.4} ms / forward_into {:8.4} ms", (e_c+e_n+e_p1+e_g+e_p2)*1e3, e_all*1e3);
}
