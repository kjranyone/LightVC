//! A2 回路の CPU 1 スレッド RTF（乱数の重み・ブロック 240 サンプルのストリーミング）。
//!     cargo run --release -p lightvc-core --example a2vc_bench -- 24 32 40 48

use std::time::Instant;

use lightvc_core::a2vc::A2Circuit;
use lightvc_core::nvoc::HOP;

fn main() {
    let widths: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let widths = if widths.is_empty() { vec![24, 32, 40, 48] } else { widths };
    let d_in = 132 + 256;
    let secs = 20.0;
    let nb = (secs * 48000.0 / HOP as f64) as usize;
    let car: Vec<f32> = (0..3 * HOP).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let ctl: Vec<f32> = (0..d_in).map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5).collect();
    let mut out = vec![0f32; HOP];
    for ch in widths {
        let mut m = A2Circuit::random(ch, d_in, 7);
        for _ in 0..200 {
            m.process_block(&car, &ctl, &mut out);
        }
        let mut lat = Vec::with_capacity(nb);
        let t0 = Instant::now();
        let mut acc = 0f32;
        for _ in 0..nb {
            let t = Instant::now();
            m.process_block(&car, &ctl, &mut out);
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
            acc += out[0];
        }
        let el = t0.elapsed().as_secs_f64();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| lat[((lat.len() - 1) as f64 * q) as usize];
        println!("ch {ch:3}  {:.2} GMAC/s  RTF {:.3}  block(5ms) p50 {:.3} ms  p95 {:.3} ms  p99 {:.3} ms  max {:.3} ms  (chk {acc:.3})",
                 m.macs_per_second() / 1e9, el / secs, p(0.5), p(0.95), p(0.99), lat[lat.len() - 1]);
    }
}
