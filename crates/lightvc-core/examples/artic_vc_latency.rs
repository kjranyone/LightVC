//! 構音 VC の製品経路の 1 ブロック(240 サンプル = 5 ms)の処理時間の分布(柔らかい引き・つまみ 0 / 0.35・伸縮つき話者差 α 1.2・製品候補 cand = 出力部 rvtab_full 360k・α 1.3・γ 0.7・色づけ・登録に res_cov が要る)。
//!     cargo run --release -p lightvc-core --example artic_vc_latency -- <目標の登録> <操作者の登録>

use std::path::Path;
use std::time::Instant;

use lightvc_core::artic_vc::{ArticVc, Enroll};
use lightvc_core::enroll::Analyzer;
use lightvc_core::nvoc::HOP;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let p = |s: &str| Path::new(s).to_path_buf();
    let (rvoc, lvl, f0e, c1, f3, consts) = (p("results/rvoc3hi/export"), p("results/f2_2/export"), p("results/f0est3/export"), p("results/c1_1/export"), p("results/f3_1/export"), p("results/enroll_export"));
    let n = 4000;
    let x: Vec<f32> = (0..n * HOP).map(|i| ((i as f32) * 0.0371).sin() * 0.1 + ((i * 7919) % 1000) as f32 * 1e-5).collect();
    let rvoc_cand = p("results/rvtab_full/export_360k");
    for cfg in ["mix0", "mix035", "warp120", "cand"] {
        let rv = if cfg == "cand" { &rvoc_cand } else { &rvoc };
        let mut m = ArticVc::load(rv, &lvl, &f0e, &c1, Path::new(&a[0]), 1).unwrap();
        m.soft = Some(1.0);
        m.w_sm = 1;
        if cfg != "mix0" {
            m.load_f3(&f3).unwrap();
        }
        if cfg == "mix035" {
            m.mix = 0.35;
        }
        if cfg == "warp120" || cfg == "cand" {
            let op = Enroll::load(Path::new(&a[1])).unwrap();
            let an = Analyzer::load(&consts).unwrap();
            m.set_operator_warp(&op, an.warp_matrix(if cfg == "cand" { 1.3 } else { 1.2 })).unwrap();
            if cfg == "cand" {
                m.set_color(&op).unwrap();
                m.resid = 0.7;
                m.per_p75 = true;
            }
        }
        let mut out = vec![0.0f32; HOP];
        let mut ts = Vec::with_capacity(n);
        for i in 0..n {
            let t = Instant::now();
            m.process_block(&x[i * HOP..(i + 1) * HOP], &mut out);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut s = ts[200..].to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| s[((s.len() as f64 - 1.0) * f) as usize];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        println!("{cfg:8} ブロック ms: 平均 {mean:.3}・中央 {:.3}・p95 {:.3}・p99 {:.3}・最大 {:.3}(予算 5.000)・RTF {:.3}", q(0.5), q(0.95), q(0.99), q(1.0), mean / 5.0);
    }
}
