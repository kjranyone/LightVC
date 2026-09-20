//! C.4 full graph RTF。**V 単体ではなく front-end 込み**で測る。
//!   front-end（causal_f0 / mel / 調波発振 / cstft / cistft 相当）＋ V1D
//!
//!   cargo run --release --example full_graph_rtf -p lightvc-core -- [blocks]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lightvc_core::ship_front as SF;
use lightvc_core::v1d::{V1d, V1dCfg, V1dStream};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let blocks: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let k = SF::HOP_S * 2; // K=2 ブロック = 256 サンプル
    let dev = Device::Cpu;
    let cfg = V1dCfg { dim: 128, ..V1dCfg::DEFAULT };
    let td = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[td.join("v1d_d128.safetensors")], DType::F32, &dev)?
    };
    let net = V1d::load(cfg, vb)?;
    let mut st = V1dStream::new(&net, &dev)?;

    let fb: Vec<f32> = std::fs::read(td.join("mel_fb_1024_80.bin"))?
        .chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // 入力リング（1792）＋ 1 ブロック。毎ブロックこの窓を解析する。
    let ring = SF::NFFT_A + 3 * SF::HOP_A;
    let mut buf = vec![0.0f32; ring + k];
    for (i, v) in buf.iter_mut().enumerate() {
        *v = ((i as f32) * 0.01).sin() * 0.1;
    }
    let noise: Vec<f32> = (0..k).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();

    let x = Tensor::randn(0f32, 1f32, (1, cfg.cin(), 2), &dev)?;
    let pre = Tensor::randn(0f32, 1f32, (1, cfg.nbin, 2), &dev)?;
    let pim = Tensor::randn(0f32, 1f32, (1, cfg.nbin, 2), &dev)?;

    // ⚠ **1 ブロック = 解析 1 フレーム**。リング全体を測り直すと 5 倍の無駄。
    let mut fs = SF::FrontState::new();
    let mut f0v = vec![0.0f32; 2];
    let mut run = |b: &[f32]| -> anyhow::Result<()> {
        let w0 = b.len() - SF::NFFT_A;
        let (_m, f0, _voi) = fs.step(&b[w0..], &fb);
        f0v[0] = f0;
        f0v[1] = f0;
        let _e = SF::excitation(&f0v, k, &noise, 0.3);
        st.step(&net, &x, &pre, &pim)?;
        Ok(())
    };
    for _ in 0..20 { run(&buf)?; }

    let audio = k as f64 / SF::SR as f64;
    let mut v = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        let s = Instant::now();
        run(&buf)?;
        v.push(s.elapsed().as_secs_f64() / audio);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{{\"impl\":\"candle-cpu full-graph\",\"blocks\":{blocks},\
\"block_ms\":{:.3},\"rtf_p50\":{:.4},\"rtf_p95\":{:.4},\"rtf_worst\":{:.4}}}",
        audio * 1000.0, v[blocks / 2], v[blocks * 95 / 100], v[blocks - 1]);
    Ok(())
}
