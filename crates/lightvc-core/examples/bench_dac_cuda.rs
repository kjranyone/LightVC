use anyhow::Result;
use candle_core::{Device, Tensor};
use lightvc_core::codec::{DacCodec, DacConfig};
use std::path::PathBuf;
use std::time::Instant;

fn bench<F: FnMut() -> Result<Tensor>>(label: &str, n: usize, mut f: F) -> Result<f64> {
    for _ in 0..5 {
        let r = f()?;
        let _ = r.sum_all()?.to_scalar::<f32>()?;
    }
    let t0 = Instant::now();
    for _ in 0..n {
        let r = f()?;
        let _ = r.sum_all()?.to_scalar::<f32>()?;
    }
    let ms = t0.elapsed().as_micros() as f64 / n as f64 / 1000.0;
    println!("  {:<20} {:>8.3} ms", label, ms);
    Ok(ms)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("=== DAC Encode/Decode Benchmark (CUDA) ===\n");

    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");
    let dac_weights = base.join("models/dac_44khz.safetensors");
    if !dac_weights.exists() {
        anyhow::bail!("DAC weights not found: {}", dac_weights.display());
    }

    let codec = DacCodec::from_file(&dac_weights, &DacConfig::default(), device.clone())?;
    println!("DAC loaded on CUDA\n");

    println!("Encode:");
    for &frames in &[1usize, 2, 4, 8, 32, 128] {
        let samples = frames * lightvc_core::DAC_HOP_LENGTH;
        let pcm = Tensor::randn(0f32, 0.1f32, (1, 1, samples), &device)?;
        println!("frames={} ({} samples):", frames, samples);
        bench("encode", 20, || codec.encode(&pcm))?;
    }

    println!("\nDecode:");
    for &frames in &[1usize, 2, 4, 8, 32, 128] {
        let latent = Tensor::randn(0f32, 1f32, (1, lightvc_core::DAC_LATENT_DIM, frames), &device)?;
        println!("frames={}:", frames);
        bench("decode", 20, || codec.decode(&latent))?;
    }

    println!("\nEncode + Decode:");
    for &frames in &[1usize, 4, 8, 32] {
        let samples = frames * lightvc_core::DAC_HOP_LENGTH;
        let pcm = Tensor::randn(0f32, 0.1f32, (1, 1, samples), &device)?;
        println!("frames={}:", frames);
        bench("encode+decode", 10, || {
            let z = codec.encode(&pcm)?;
            codec.decode(&z)
        })?;
    }

    println!("\nDone.");
    Ok(())
}
