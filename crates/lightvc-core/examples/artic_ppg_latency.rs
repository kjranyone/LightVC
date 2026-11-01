//! 製品の経路(--causal-ct・--per-p75・高さは操作者の登録の中央値から)の 1 ブロック(240 サンプル = 5 ms)の処理時間の分布を、出力部ごとに比べる
//! (今の rvoc3hi 対 内容 → 波形の生成器の書き出し)。1 スレッド。合否は最悪値(最大)で読む。
//!     cargo run --release -p lightvc-core --example artic_ppg_latency -- <目標の登録> <操作者の登録> <出力部の書き出し>[:<C2 の書き出し>]...
//! 内容 → 波形の生成器には製品の設定の操作者の遅い揺らぎ(--opslow 4,1.5)をかける。C2 は `出力部:C2` で指定(diag_ppgvoc52)。

use std::path::Path;
use std::time::Instant;

use lightvc_core::artic_vc::{ArticVc, Enroll};
use lightvc_core::enroll::{Analyzer, CausalCt};
use lightvc_core::nvoc::HOP;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let p = |s: &str| Path::new(s).to_path_buf();
    let (lvl, f0e, c1, consts) = (p("results/f2_2/export"), p("results/f0est3/export"), p("results/c1_1/export"), p("results/enroll_export"));
    let n = 4000;
    let x: Vec<f32> = (0..n * HOP).map(|i| ((i as f32) * 0.0371).sin() * 0.1 + ((i * 7919) % 1000) as f32 * 1e-5).collect();
    let op = Enroll::load(Path::new(&a[1])).unwrap();
    for spec in &a[2..] {
        let (rv, c2) = match spec.split_once(':') {
            Some((r, c)) => (r.to_string(), Some(c.to_string())),
            None => (spec.clone(), None),
        };
        let mut m = ArticVc::load(Path::new(&rv), &lvl, &f0e, &c1, Path::new(&a[0]), 1).unwrap();
        m.per_p75 = true;
        m.cct = Some(CausalCt::new(&Analyzer::load(&consts).unwrap()));
        m.set_fixed_med(op.mu);
        if let Some(c) = &c2 {
            m.set_c2(Path::new(c)).unwrap();
        }
        if m.is_ppg() {
            m.set_opslow(&op, 4.0, 1.5).unwrap();
        }
        let rv = spec;
        let mut out = vec![0.0f32; HOP];
        let mut ts = Vec::with_capacity(n);
        for i in 0..n {
            let t = Instant::now();
            m.process_block(&x[i * HOP..(i + 1) * HOP], &mut out);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let warm = &mut ts[50..].to_vec();
        warm.sort_by(|u, v| u.partial_cmp(v).unwrap());
        let q = |f: f64| warm[((warm.len() - 1) as f64 * f) as usize];
        let mean = warm.iter().sum::<f64>() / warm.len() as f64;
        println!("{rv}: 1 ブロック ms 平均 {mean:.3} p50 {:.3} p95 {:.3} p99 {:.3} 最大 {:.3}(枠 5 ms)・RTF {:.3}", q(0.5), q(0.95), q(0.99), warm[warm.len() - 1], mean / 5.0);
    }
}
