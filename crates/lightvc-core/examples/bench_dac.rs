use anyhow::Result;
use candle_core::{Device, Tensor};
use lightvc_core::codec::{DacCodec, DacConfig};
use std::path::PathBuf;
use std::time::Instant;

fn bench<F: FnMut() -> Result<()>>(label: &str, n: usize, mut f: F) -> Result<f64> {
    for _ in 0..3 {
        f()?;
    }
    let t0 = Instant::now();
    for _ in 0..n {
        f()?;
    }
    let us = t0.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  {:<20} {:>8.1} µs  ({:.3} ms)",
        label,
        us,
        us / 1000.0
    );
    Ok(us / 1000.0)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");
    let dac_weights = base.join("models/dac_44khz.safetensors");
    if !dac_weights.exists() {
        anyhow::bail!("DAC weights not found: {}", dac_weights.display());
    }

    let codec = DacCodec::from_file(&dac_weights, &DacConfig::default(), device.clone())?;
    println!("=== DAC Encode/Decode Benchmark (CPU) ===\n");

    println!("Encode:");
    for &frames in &[1usize, 2, 8, 32, 128] {
        let samples = frames * lightvc_core::DAC_HOP_LENGTH;
        let pcm = Tensor::randn(0f32, 0.1f32, (1, 1, samples), &device)?;

        println!("frames={} samples={}:", frames, samples);
        bench("encode", 10, || {
            codec.encode(&pcm)?;
            Ok(())
        })?;
        println!();
    }

    println!("Decode:");
    for &frames in &[8usize, 16, 32, 64, 128] {
        let samples = frames * lightvc_core::DAC_HOP_LENGTH;
        let pcm = Tensor::randn(0f32, 0.1f32, (1, 1, samples), &device)?;
        let latent = Tensor::randn(0f32, 1f32, (1, lightvc_core::DAC_LATENT_DIM, frames), &device)?;

        println!("frames={} samples={}:", frames, samples);
        bench("decode", 10, || {
            codec.decode(&latent)?;
            Ok(())
        })?;
        bench("encode+decode", 5, || {
            let z = codec.encode(&pcm)?;
            codec.decode(&z)?;
            Ok(())
        })?;
        println!();
    }

    println!("Done.");
    Ok(())
}
