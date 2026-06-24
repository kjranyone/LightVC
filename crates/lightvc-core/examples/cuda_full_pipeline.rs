use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use lightvc_core::codec::{DacCodec, DacConfig};
use lightvc_core::soft_rvq::load_soft_rvq;
use lightvc_core::utte_adapter::load_adapter;
use std::path::PathBuf;
use std::time::Instant;

fn mse(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = (a - b)?;
    Ok(diff.sqr()?.mean_all()?.to_scalar::<f32>()? as f64)
}

fn sync_ms<F: FnMut() -> Result<Tensor>>(n: usize, mut f: F) -> Result<f64> {
    for _ in 0..5 {
        let r = f()?;
        let _ = r.sum_all()?.to_scalar::<f32>()?;
    }
    let t0 = Instant::now();
    for _ in 0..n {
        let r = f()?;
        let _ = r.sum_all()?.to_scalar::<f32>()?;
    }
    Ok(t0.elapsed().as_micros() as f64 / n as f64 / 1000.0)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");

    let dac_weights = base.join("models/dac_44khz.safetensors");
    let dac_q_weights = base.join("models/dac_quantizer.safetensors");
    let adapter_weights = base.join("models/utte_adapter_b1.safetensors");
    let ref_weights = base.join("models/rust_parity_ref.safetensors");

    println!("=== CUDA Full Pipeline Parity + Benchmark ===\n");

    let codec = DacCodec::from_file(&dac_weights, &DacConfig::default(), device.clone())?;
    println!("DAC codec loaded (CUDA)");
    let soft_rvq = load_soft_rvq(&dac_q_weights, &device)?;
    println!("Soft RVQ loaded (CUDA)");
    let adapter = load_adapter(&adapter_weights, &device)?;
    println!("UTTE adapter loaded (CUDA)");

    let rvb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[&ref_weights], DType::F32, &device)?
    };
    let t = 298;
    let z_s = rvb.get((1, 1024, t), "z_s")?;
    let q0_s = rvb.get((1, 1024, t), "q0_s")?;
    let timbre = rvb.get((1, 192), "timbre")?;
    let z_q_ref = rvb.get((1, 1024, t), "z_q_ref")?;
    let z_q_adapted_ref = rvb.get((1, 1024, t), "z_q_adapted_ref")?;
    let audio_ref = rvb.get((1, 1, t * 512), "audio_ref")?;
    println!("Reference loaded (T={})\n", t);

    let tau = 5.0f64;

    println!("--- Parity ---");
    let z_q = soft_rvq.soft_requantize(&q0_s, &z_s, tau)?;
    let rvq_mse = mse(&z_q, &z_q_ref)?;
    println!("Soft RVQ MSE:     {:.8}  {}", rvq_mse, if rvq_mse < 1e-4 { "PASS" } else { "FAIL" });

    let z_q_adapted = adapter.forward(&z_q, &timbre)?;
    let ad_mse = mse(&z_q_adapted, &z_q_adapted_ref)?;
    println!("Adapter MSE:      {:.8}  {}", ad_mse, if ad_mse < 1e-4 { "PASS" } else { "FAIL" });

    let audio = codec.decode(&z_q_adapted)?;
    let au_mse = mse(&audio, &audio_ref)?;
    println!("Decode MSE:       {:.8}  {}", au_mse, if au_mse < 1e-3 { "PASS" } else { "CHECK" });

    println!("\n--- Component Latency (CUDA, T={}) ---", t);
    let ms_rvq = sync_ms(30, || soft_rvq.soft_requantize(&q0_s, &z_s, tau))?;
    println!("Soft RVQ:          {:>7.2} ms", ms_rvq);

    let ms_ad = sync_ms(50, || adapter.forward(&z_q, &timbre))?;
    println!("UTTE adapter:      {:>7.2} ms", ms_ad);

    let ms_dec = sync_ms(30, || codec.decode(&z_q_adapted))?;
    println!("DAC decode:        {:>7.2} ms", ms_dec);

    let samples = t * 512;
    let pcm = rvb.get((1, 1, samples), "audio_ref")?;
    let ms_enc = sync_ms(30, || codec.encode(&pcm))?;
    println!("DAC encode:        {:>7.2} ms", ms_enc);

    println!("\n--- Full Pipeline (CUDA) ---");
    let ms_full = sync_ms(20, || {
        let z_q = soft_rvq.soft_requantize(&q0_s, &z_s, tau)?;
        let z_qa = adapter.forward(&z_q, &timbre)?;
        codec.decode(&z_qa)
    })?;
    println!("RVQ+adapter+decode: {:>7.2} ms", ms_full);

    let ms_e2e = sync_ms(20, || {
        let z = codec.encode(&pcm)?;
        let zq = soft_rvq.soft_requantize(&q0_s, &z, tau)?;
        let zqa = adapter.forward(&zq, &timbre)?;
        codec.decode(&zqa)
    })?;
    println!("encode+RVQ+ad+dec:  {:>7.2} ms", ms_e2e);

    println!("\n--- Latency Budget (T={}, {:.0}ms audio) ---", t, t as f32 / 86.13 * 1000.0);
    println!("VC processing:     {:>7.2} ms", ms_e2e);
    println!("Resampling:        ~6 ms");
    println!("Audio I/O (ASIO):  ~6 ms");
    let total = ms_e2e + 12.0;
    println!("TOTAL estimate:    {:>7.2} ms  {}", total, if total < 50.0 { "✅ <50ms" } else { "✗ >50ms" });

    println!("\nDone.");
    Ok(())
}
