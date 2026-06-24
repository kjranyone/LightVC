use anyhow::Result;
use candle_core::{Device, Tensor};
use lightvc_core::utte_adapter::load_adapter;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let weights = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/utte_adapter_b1.safetensors");

    if !weights.exists() {
        eprintln!("Weights not found: {}", weights.display());
        std::process::exit(1);
    }

    let adapter = load_adapter(&weights, &device)?;
    println!("UTTE adapter loaded");
    println!("device: {:?}", device);
    println!();

    let timbre = Tensor::randn(0f32, 1f32, (1, 192), &device)?;

    for &frames in &[1usize, 2, 8, 32, 128, 256] {
        let z_q = Tensor::randn(0f32, 1f32, (1, 1024, frames), &device)?;

        for _ in 0..3 {
            let _ = adapter.forward(&z_q, &timbre)?;
        }

        let n_runs = 50;
        let t0 = Instant::now();
        for _ in 0..n_runs {
            let _ = adapter.forward(&z_q, &timbre)?;
        }
        let elapsed = t0.elapsed();
        let mean_us = elapsed.as_micros() as f64 / n_runs as f64;

        println!(
            "frames={:>4}  mean={:>8.1} µs  ({:.3} ms)",
            frames,
            mean_us,
            mean_us / 1000.0
        );
    }

    println!("\nDone.");
    Ok(())
}
