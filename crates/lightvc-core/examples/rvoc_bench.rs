//! rvoc の CPU 1 スレッド RTF(乱数の重み・240 サンプルのブロック)。
//!     cargo run --release -p lightvc-core --example rvoc_bench -- 192 256

use std::time::Instant;

use lightvc_core::nvoc::HOP;
use lightvc_core::rvoc::{RVoc, D_COND};

fn main() {
    let widths: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let widths = if widths.is_empty() { vec![256] } else { widths };
    let secs = 20.0;
    let nb = (secs * 48000.0 / HOP as f64) as usize;
    let cond: Vec<f32> = (0..D_COND).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let exc: Vec<f32> = (0..2 * HOP).map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5).collect();
    let mut out = vec![0f32; HOP];
    for ch in widths {
        let mut m = RVoc::random(ch, 7);
        for _ in 0..100 {
            m.process_block(&cond, &exc, &mut out);
        }
        let mut lat = Vec::with_capacity(nb);
        let t0 = Instant::now();
        let mut chk = 0f32;
        for _ in 0..nb {
            let t = Instant::now();
            m.process_block(&cond, &exc, &mut out);
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
            chk += out[0];
        }
        let el = t0.elapsed().as_secs_f64();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| lat[((lat.len() - 1) as f64 * q) as usize];
        println!("ch {ch:3}  RTF {:.3}  block p50 {:.3} ms  p95 {:.3} ms  p99 {:.3} ms  max {:.3} ms  (chk {chk:.3})",
                 el / secs, p(0.5), p(0.95), p(0.99), lat[lat.len() - 1]);
    }
}
