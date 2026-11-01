//! 構音 VC の製品経路の部品ごとの時間(1 スレッド)。
//!     cargo run --release -p lightvc-core --example artic_vc_bench -- results/rvoc3hi/export results/f2_2/export results/f0est3/export results/c1_1/export

use std::path::Path;
use std::time::Instant;

use lightvc_core::c1::C1;
use lightvc_core::f0est::F0Est;
use lightvc_core::lvl::Lvl;
use lightvc_core::nvoc::HOP;
use lightvc_core::rvoc::{Excite, RVoc, D_COND};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n = 1000;
    let x: Vec<f32> = (0..n * HOP).map(|i| ((i as f32) * 0.0371).sin() * 0.1 + ((i * 7919) % 1000) as f32 * 1e-5).collect();
    let rt = |name: &str, el: f64| println!("{name:8} RTF {:.3}", el / (n as f64 * HOP as f64 / 48000.0));
    let mut c1 = C1::load(Path::new(&a[3])).unwrap();
    let t = Instant::now();
    for i in 0..n {
        c1.process_block(&x[i * HOP..(i + 1) * HOP]);
    }
    rt("C1", t.elapsed().as_secs_f64());
    let mut f1 = F0Est::load(Path::new(&a[2])).unwrap();
    let t = Instant::now();
    for i in 0..n {
        f1.process_block(&x[i * HOP..(i + 1) * HOP]);
    }
    rt("F1", t.elapsed().as_secs_f64());
    let mut f2 = Lvl::load(Path::new(&a[1])).unwrap();
    let t = Instant::now();
    for i in 0..n {
        f2.process_block(&x[i * HOP..(i + 1) * HOP]);
    }
    rt("F2", t.elapsed().as_secs_f64());
    let mut v = RVoc::load(Path::new(&a[0])).unwrap();
    let mut ex = Excite::new(22000.0, 2000.0, 440, 1);
    let cond = vec![0.3f32; D_COND];
    let mut e = vec![0.0f32; 2 * HOP];
    let mut out = vec![0.0f32; HOP];
    let t = Instant::now();
    for _ in 0..n {
        ex.block(220.0, &mut e);
        v.process_block(&cond, &e, &mut out);
    }
    rt("rvoc+exc", t.elapsed().as_secs_f64());
}
