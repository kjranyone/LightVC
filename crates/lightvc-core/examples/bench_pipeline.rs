use anyhow::Result;
use candle_core::{Device, Tensor};
use lightvc_core::soft_rvq::load_soft_rvq;
use lightvc_core::utte_adapter::load_adapter;
use std::path::PathBuf;
use std::time::Instant;

fn bench<F: FnMut() -> Result<()>>(label: &str, n: usize, mut f: F) -> Result<()> {
    for _ in 0..3 {
        f()?;
    }
    let t0 = Instant::now();
    for _ in 0..n {
        f()?;
    }
    let us = t0.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  {:<30}  {:>8.1} µs  ({:.3} ms)",
        label, us, us / 1000.0
    );
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");

    let dac_q = base.join("models/dac_quantizer.safetensors");
    let adapter_w = base.join("models/utte_adapter_b1.safetensors");

    let soft_rvq = load_soft_rvq(&dac_q, &device)?;
    let adapter = load_adapter(&adapter_w, &device)?;
    println!("=== Component Benchmark (CPU) ===\n");

    let tau = 5.0f64;
    let timbre = Tensor::randn(0f32, 1f32, (1, 192), &device)?;

    for &frames in &[1usize, 8, 32, 128, 256] {
        let z_s = Tensor::randn(0f32, 1f32, (1, 1024, frames), &device)?;
        let q0_s = Tensor::randn(0f32, 1f32, (1, 1024, frames), &device)?;

        println!("frames={}:", frames);

        bench("soft RVQ only", 30, || {
            soft_rvq.soft_requantize(&q0_s, &z_s, tau)?;
            Ok(())
        })?;

        let z_q = soft_rvq.soft_requantize(&q0_s, &z_s, tau)?;

        bench("UTTE adapter only", 50, || {
            adapter.forward(&z_q, &timbre)?;
            Ok(())
        })?;

        bench("soft RVQ + adapter", 30, || {
            let zq = soft_rvq.soft_requantize(&q0_s, &z_s, tau)?;
            adapter.forward(&zq, &timbre)?;
            Ok(())
        })?;

        println!();
    }

    println!("Done.");
    Ok(())
}
