use anyhow::Result;
use candle_core::{DType, Device, Tensor, Module};
use lightvc_core::dac_model::DacModel;
use lightvc_core::DacConfig;
use candle_nn::VarBuilder;
use std::path::PathBuf;

fn mse(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = (a - b)?;
    let s = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok(s as f64)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");

    let dac_finetuned = base.join("models/dac_44khz_finetuned.safetensors");
    let ref_file = base.join("results/decoder_finetuned_parity.safetensors");

    for (name, path) in [
        ("Fine-tuned DAC", &dac_finetuned),
        ("Parity reference", &ref_file),
    ] {
        if !path.exists() {
            anyhow::bail!("{} not found: {}", name, path.display());
        }
    }

    println!("=== Decoder Fine-Tune Rust/Candle Parity ===\n");

    let config = DacConfig::default();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&dac_finetuned], DType::F32, &device)?
    };
    let model = DacModel::new(&(&config).into(), vb)?;

    let ref_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&ref_file], DType::F32, &device)?
    };
    let z = ref_vb.get((1, 1024, 32), "z_input")?;
    let audio_ref = ref_vb.get((1, 1, 16384), "audio_ref")?;

    println!("Input latent: {:?}", z.shape());

    let audio = model.decoder.forward(&z)?;
    println!("Output audio: {:?}", audio.shape());

    let mse_val = mse(&audio, &audio_ref)?;
    let rmse = mse_val.sqrt();
    println!("\nParity MSE:  {:.10}", mse_val);
    println!("Parity RMSE: {:.10}", rmse);

    if rmse < 1e-4 {
        println!("PASS: Parity within tolerance (RMSE < 1e-4)");
    } else if rmse < 1e-2 {
        println!("WARN: Moderate difference (RMSE < 1e-2)");
    } else {
        println!("FAIL: Large difference (RMSE >= 1e-2)");
    }

    println!("\n--- Latency benchmark (CPU) ---");
    for _ in 0..3 {
        let _ = model.decoder.forward(&z)?;
    }

    let z4 = Tensor::randn(0f32, 1f32, (1, 1024, 4), &device)?;
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        let _ = model.decoder.forward(&z4)?;
    }
    let elapsed = t0.elapsed();
    println!("Decode T=4:   {:.2}ms/iter", elapsed.as_secs_f64() * 1000.0 / 20.0);

    let z32 = Tensor::randn(0f32, 1f32, (1, 1024, 32), &device)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        let _ = model.decoder.forward(&z32)?;
    }
    let elapsed = t0.elapsed();
    println!("Decode T=32:  {:.2}ms/iter", elapsed.as_secs_f64() * 1000.0 / 10.0);

    let z256 = Tensor::randn(0f32, 1f32, (1, 1024, 256), &device)?;
    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        let _ = model.decoder.forward(&z256)?;
    }
    let elapsed = t0.elapsed();
    println!("Decode T=256: {:.2}ms/iter", elapsed.as_secs_f64() * 1000.0 / 5.0);

    Ok(())
}
