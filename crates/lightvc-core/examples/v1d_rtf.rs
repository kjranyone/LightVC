//! C.4 速度。**RTF = wall time ÷ 音声実時間**（CPU 時間の合計ではない）。
//!
//! `results/z0/rtf_basis.json` が「最終判定はここ」と宣言した実測。
//! PyTorch の e2e は op 起動のインタプリタ費で膨らむので、製品の値はこちら。
//!
//!   cargo run --release --example v1d_rtf -p lightvc-core -- [blocks] [threads]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lightvc_core::v1d::{V1d, V1dCfg, V1dStream};
use std::time::Instant;

const SR: f64 = 44100.0;
const HOP_S: usize = 128;
// K は引数で変える（既定 2 ＝ 2.4a の決定）。固定費と限界費を切り分けるため。

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let blocks: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let dev = Device::Cpu;
    let dim: usize = std::env::var("V1D_DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let layers: usize = std::env::var("V1D_L").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let wf = std::env::var("V1D_W").unwrap_or_else(|_| "v1d_test.safetensors".into());
    let cfg = V1dCfg { dim, layers, ..V1dCfg::DEFAULT };

    let td = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[td.join(&wf)], DType::F32, &dev)?
    };
    let net = V1d::load(cfg, vb)?;
    let mut st = V1dStream::new(&net, &dev)?;

    let x = Tensor::randn(0f32, 1f32, (1, cfg.cin(), k), &dev)?;
    let pre = Tensor::randn(0f32, 1f32, (1, cfg.nbin, k), &dev)?;
    let pim = Tensor::randn(0f32, 1f32, (1, cfg.nbin, k), &dev)?;

    for _ in 0..50 {
        st.step(&net, &x, &pre, &pim)?;
    }

    // ⚠ 最悪値も採る。中央値だけで読むと合否がマシン負荷で反転する。
    let audio = k as f64 * HOP_S as f64 / SR;
    let mut v = Vec::with_capacity(blocks);
    let t0 = Instant::now();
    for _ in 0..blocks {
        let s = Instant::now();
        st.step(&net, &x, &pre, &pim)?;
        v.push(s.elapsed().as_secs_f64() / audio);
    }
    let mean = t0.elapsed().as_secs_f64() / blocks as f64 / audio;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (p50, p95, worst) = (v[blocks / 2], v[blocks * 95 / 100], v[blocks - 1]);

    println!("{{\"impl\":\"candle-cpu\",\"K\":{k},\"blocks\":{blocks},\
\"block_ms\":{:.3},\"net_rtf_mean\":{:.4},\"net_rtf_p50\":{:.4},\
\"net_rtf_p95\":{:.4},\"net_rtf_worst\":{:.4},\
\"dim\":{},\"L\":{},\"k_in\":{},\"k\":{},\"nbin\":{},\"ctx\":{}}}",
        audio * 1000.0, mean, p50, p95, worst,
        cfg.dim, cfg.layers, cfg.k_in, cfg.k, cfg.nbin, cfg.ctx());
    Ok(())
}
