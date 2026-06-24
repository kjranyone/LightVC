use anyhow::Result;
use candle_core::{Device, Tensor};
use lightvc_core::b1_pipeline::{B1Offline, B1Streaming};
use lightvc_core::streaming::ChunkMode;
use std::path::PathBuf;

#[cfg(feature = "cuda")]
fn get_device() -> Result<Device> {
    Ok(Device::new_cuda(0)?)
}
#[cfg(not(feature = "cuda"))]
fn get_device() -> Result<Device> {
    Ok(Device::Cpu)
}

fn main() -> Result<()> {
    let device = get_device()?;
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");
    let dac_w = base.join("models/dac_44khz.safetensors");
    let q_w = base.join("models/dac_quantizer.safetensors");
    let ad_w = base.join("models/utte_adapter_b1.safetensors");
    let ref_w = base.join("models/rust_parity_ref.safetensors");

    println!("=== B1 Streaming Pipeline Test ===\n");

    let rvb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[&ref_w], candle_core::DType::F32, &device)?
    };
    let timbre = rvb.get((1, 192), "timbre")?;
    let z_s = rvb.get((1, 1024, 298), "z_s")?;
    println!("Loaded timbre + z_s reference (T=298)");

    let codec = lightvc_core::codec::DacCodec::from_file(&dac_w, &Default::default(), device.clone())?;
    let pcm_offline = codec.decode(&z_s)?;
    let pcm_vec = pcm_offline.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
    println!("Decoded z_s → {} PCM samples ({:.1}s)\n", pcm_vec.len(), pcm_vec.len() as f32 / 44100.0);

    println!("--- Offline reference ---");
    let mut offline = B1Offline::new(&dac_w, &q_w, &ad_w, device.clone())?;
    offline.set_timbre(timbre.clone());
    let pcm_tensor = Tensor::from_vec(pcm_vec.clone(), (1, 1, pcm_vec.len()), &device)?;
    let out_offline = offline.process(&pcm_tensor)?;
    let out_offline_vec = out_offline.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
    println!("Offline output: {} samples\n", out_offline_vec.len());

    for (mode_name, chunk_mode) in [("Strict", ChunkMode::Strict), ("Balanced", ChunkMode::Balanced)] {
        println!("--- {} mode ---", mode_name);
        let chunk_sz = chunk_mode.samples_per_chunk();
        let lookahead = chunk_mode.lookahead_samples();
        println!(
            "  chunk={} samples ({:.1}ms), lookahead={} samples ({:.1}ms), algo latency={:.1}ms",
            chunk_sz,
            chunk_sz as f32 / 44.1,
            lookahead,
            lookahead as f32 / 44.1,
            chunk_mode.algorithmic_latency_samples() as f32 / 44.1,
        );

        let mut stream = B1Streaming::new(&dac_w, &q_w, &ad_w, chunk_mode, device.clone())?;
        stream.set_timbre(timbre.clone());

        let out_stream = stream.process_full(&pcm_vec)?;
        println!("  Streaming output: {} samples", out_stream.len());

        let min_len = out_offline_vec.len().min(out_stream.len());
        if min_len > 0 {
            let mut diff_sum = 0.0f64;
            for i in 0..min_len {
                let d = out_offline_vec[i] - out_stream[i];
                diff_sum += (d * d) as f64;
            }
            let mse = diff_sum / min_len as f64;
            let sig_power: f64 = out_offline_vec[..min_len].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / min_len as f64;
            let snr = if mse > 1e-12 {
                10.0 * (sig_power / mse).log10()
            } else {
                f64::INFINITY
            };
            println!("  vs offline: MSE={:.2e}, SNR={:.1} dB", mse, snr);
        }

        print!("  Latency:");
        stream.timings.summary();
        println!();
    }

    println!("Done.");
    Ok(())
}
