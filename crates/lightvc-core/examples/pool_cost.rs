use lightvc_core::simd::Pool;
fn main() {
    for n in [1usize, 2] {
        let p = Pool::new(n);
        for _ in 0..200 { p.roundtrip(); }
        let mut v = Vec::new();
        for _ in 0..2000 { v.push(p.roundtrip().as_secs_f64()); }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{}スレッド 往復 p50 {:.1} us / p95 {:.1} us  （ブロック 5805 us）",
                 n, v[1000] * 1e6, v[1900] * 1e6);
    }
}
