use anyhow::Result;
use candle_core::{Device, Tensor};
use lightvc_core::codec::{DacCodec, DacConfig};
use lightvc_core::soft_rvq::load_soft_rvq;
use lightvc_core::streaming::{ChunkMode, StreamingCodec};
use lightvc_core::utte_adapter::load_adapter;
use std::path::PathBuf;

fn raw_snr(ref_sig: &[f32], test_sig: &[f32]) -> (f64, f64) {
    let n = ref_sig.len().min(test_sig.len());
    if n == 0 { return (f64::MIN, f64::INFINITY); }
    let mut sig = 0.0f64;
    let mut diff = 0.0f64;
    for i in 0..n {
        sig += (ref_sig[i] as f64) * (ref_sig[i] as f64);
        let d = ref_sig[i] as f64 - test_sig[i] as f64;
        diff += d * d;
    }
    let mse = diff / n as f64;
    let snr = if mse > 1e-15 { 10.0 * (sig / n as f64 / mse).log10() } else { f64::INFINITY };
    (snr, mse)
}

fn aligned_snr(ref_sig: &[f32], test_sig: &[f32], max_lag: i32) -> (f64, i32) {
    let n = ref_sig.len().min(test_sig.len()) as i32;
    let mut best = (f64::MIN, 0i32);
    for lag in -max_lag..=max_lag {
        let len = n - lag.abs();
        if len <= 0 { continue; }
        let (rs, ts) = if lag >= 0 { (0, lag as usize) } else { ((-lag) as usize, 0) };
        let (snr, _) = raw_snr(&ref_sig[rs..rs+len as usize], &test_sig[ts..ts+len as usize]);
        if snr > best.0 { best = (snr, lag); }
    }
    best
}

fn tensor_snr(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
    let n = a.dim(2)?.min(b.dim(2)?);
    let a_n = a.narrow(2, 0, n)?;
    let b_n = b.narrow(2, 0, n)?;
    let diff = (&a_n - &b_n)?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
    let a_pow = a_n.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
    let snr = if mse > 1e-15 { 10.0 * (a_pow / mse).log10() } else { f64::INFINITY };
    Ok((snr, mse))
}

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");
    let dac_w = base.join("models/dac_44khz.safetensors");
    let q_w = base.join("models/dac_quantizer.safetensors");
    let ad_w = base.join("models/utte_adapter_b1.safetensors");
    let ref_w = base.join("models/rust_parity_ref.safetensors");

    println!("=== SNR Diagnostic ===\n");

    let rvb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[&ref_w], candle_core::DType::F32, &device)?
    };
    let z_s = rvb.get((1, 1024, 298), "z_s")?;
    let timbre = rvb.get((1, 192), "timbre")?;
    let z_q_adapted_ref = rvb.get((1, 1024, 298), "z_q_adapted_ref")?;
    let audio_ref = rvb.get((1, 1, 298 * 512), "audio_ref")?;

    let codec = DacCodec::from_file(&dac_w, &DacConfig::default(), device.clone())?;
    let soft_rvq = load_soft_rvq(&q_w, &device)?;
    let adapter = load_adapter(&ad_w, &device)?;

    let pcm = codec.decode(&z_s)?;
    let pcm_vec = pcm.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;

    let audio_ref_vec = audio_ref.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;

    for &frames_per_chunk in &[1usize, 4] {
        let mode = if frames_per_chunk == 1 { ChunkMode::Strict } else { ChunkMode::Balanced };
        let chunk_sz = mode.samples_per_chunk();
        let label = if frames_per_chunk == 1 { "Strict" } else { "Balanced" };
        println!("===== {} ({} frames/chunk) =====\n", label, frames_per_chunk);

        // --- 1. Encode-only parity ---
        let mut enc = StreamingCodec::new(&dac_w, &DacConfig::default(), mode, device.clone())?;
        let mut latents: Vec<Tensor> = Vec::new();
        for pos in (0..pcm_vec.len()).step_by(chunk_sz) {
            let end = (pos + chunk_sz).min(pcm_vec.len());
            let mut chunk = pcm_vec[pos..end].to_vec();
            chunk.resize(chunk_sz, 0.0);
            let lat = enc.encode_step(&chunk)?;
            if lat.dim(2)? > 0 { latents.push(lat); }
        }
        let lat_stream = Tensor::cat(&latents.iter().collect::<Vec<_>>(), 2)?;
        let (enc_snr, enc_mse) = tensor_snr(&z_s, &lat_stream)?;
        println!("1. Encode latent parity:  SNR={:.1} dB  MSE={:.2e}  (offline T={} vs stream T={})",
                 enc_snr, enc_mse, z_s.dim(2)?, lat_stream.dim(2)?);

        // --- 2. Decode-only parity ---
        let mut dec = StreamingCodec::new(&dac_w, &DacConfig::default(), mode, device.clone())?;
        let dec_chunk = if frames_per_chunk == 1 { 1 } else { 4 };
        let mut dec_out: Vec<f32> = Vec::new();
        let t_total = z_q_adapted_ref.dim(2)?;
        for start in (0..t_total).step_by(dec_chunk) {
            let end = (start + dec_chunk).min(t_total);
            let n = end - start;
            let lat_chunk = z_q_adapted_ref.narrow(2, start, n)?;
            let out = dec.decode_step(&lat_chunk)?;
            dec_out.extend_from_slice(&out);
        }
        let (dec_snr, dec_mse) = raw_snr(&audio_ref_vec, &dec_out);
        let (dec_asnr, dec_lag) = aligned_snr(&audio_ref_vec, &dec_out, 200);
        println!("2. Decode-only parity:    raw SNR={:.1} dB  aligned SNR={:.1} dB (lag={})  MSE={:.2e}",
                 dec_snr, dec_asnr, dec_lag, dec_mse);

        // --- 3. Full streaming pipeline ---
        let mut full = StreamingCodec::new(&dac_w, &DacConfig::default(), mode, device.clone())?;
        let mut full_out: Vec<f32> = Vec::new();
        for pos in (0..pcm_vec.len()).step_by(chunk_sz) {
            let end = (pos + chunk_sz).min(pcm_vec.len());
            let mut chunk = pcm_vec[pos..end].to_vec();
            chunk.resize(chunk_sz, 0.0);

            let lat = full.encode_step(&chunk)?;
            let frames = lat.dim(2)?;
            if frames == 0 { continue; }

            let q0 = soft_rvq.quantize_q0(&lat)?;
            let zq = soft_rvq.soft_requantize(&q0, &lat, 5.0)?;
            let zqa = adapter.forward(&zq, &timbre)?;
            let out = full.decode_step(&zqa)?;
            full_out.extend_from_slice(&out);
        }
        let (full_snr, full_mse) = raw_snr(&audio_ref_vec, &full_out);
        let (full_asnr, full_lag) = aligned_snr(&audio_ref_vec, &full_out, 200);
        println!("3. Full pipeline:         raw SNR={:.1} dB  aligned SNR={:.1} dB (lag={})  MSE={:.2e}",
                 full_snr, full_asnr, full_lag, full_mse);

        // --- 4. Pipeline split: streaming encode + offline process + offline decode ---
        let zq_stream = {
            let q0_s = soft_rvq.quantize_q0(&lat_stream)?;
            let zq_s = soft_rvq.soft_requantize(&q0_s, &lat_stream, 5.0)?;
            adapter.forward(&zq_s, &timbre)?
        };
        let audio_split_dec = codec.decode(&zq_stream)?;
        let audio_split_vec = audio_split_dec.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        let (split_snr, split_mse) = raw_snr(&audio_ref_vec, &audio_split_vec);
        let (split_asnr, split_lag) = aligned_snr(&audio_ref_vec, &audio_split_vec, 200);
        println!("4. Stream-enc+offline-rest: SNR={:.1} dB  aligned={:.1} dB (lag={})  MSE={:.2e}",
                 split_snr, split_asnr, split_lag, split_mse);

        // --- 5. Pipeline split: offline encode + streaming process + streaming decode ---
        let mut split2_dec = StreamingCodec::new(&dac_w, &DacConfig::default(), mode, device.clone())?;
        let mut split2_out: Vec<f32> = Vec::new();
        for start in (0..t_total).step_by(dec_chunk) {
            let end = (start + dec_chunk).min(t_total);
            let n = end - start;
            let lat_chunk = z_s.narrow(2, start, n)?;
            let q0 = soft_rvq.quantize_q0(&lat_chunk)?;
            let zq = soft_rvq.soft_requantize(&q0, &lat_chunk, 5.0)?;
            let zqa = adapter.forward(&zq, &timbre)?;
            let out = split2_dec.decode_step(&zqa)?;
            split2_out.extend_from_slice(&out);
        }
        let (split2_snr, split2_mse) = raw_snr(&audio_ref_vec, &split2_out);
        let (split2_asnr, split2_lag) = aligned_snr(&audio_ref_vec, &split2_out, 200);
        println!("5. Offline-enc+stream-rest:  SNR={:.1} dB  aligned={:.1} dB (lag={})  MSE={:.2e}",
                 split2_snr, split2_asnr, split2_lag, split2_mse);

        println!();
    }

    println!("=== Summary ===");
    println!("If test 1 (encode) SNR is low → encoder context mismatch");
    println!("If test 2 (decode) SNR is low → decode OLA artifacts");
    println!("If test 4 SNR > test 3 → encode is the bottleneck");
    println!("If test 5 SNR > test 3 → decode is the bottleneck");
    println!("If aligned SNR >> raw SNR → timing shift, not quality loss");

    Ok(())
}
