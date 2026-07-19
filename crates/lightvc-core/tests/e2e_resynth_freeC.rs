//! M3 gates for the FreeVocoder resynthesis path (Rust mel → freeC vocoder).
//!
//! `e2e_resynth_freeC`  — offline: `audio → Rust mel → FreeVocoder.forward` vs
//!   the PyTorch end-to-end reference wave (`e2e_freeC.safetensors`). Reports
//!   best-lag SNR (expect ≈60 dB near lag 0; the only delta vs the 88 dB
//!   `parity_freeC` gate is the Rust-vs-Python mel).
//!
//! `streaming_e2e_recovery_freeC` — RECOVERY gate for the realtime path. The
//!   lookahead-centered streaming mel (`stream_push`, 1088-sample lookahead)
//!   reproduces the offline centered framing exactly, so the realtime path
//!   (streaming mel → causal streaming vocoder) now matches the PyTorch E2E
//!   reference at the causal `nfft/2` lag. This asserts the recovery from the
//!   old left-aligned mel's ~1.3 dB collapse back to ~60 dB, and audits the
//!   streaming-mel-vs-offline-mel identity and the streaming-vs-offline E2E lag.

use std::path::PathBuf;

use candle_core::{DType, Device};
use lightvc_core::free_resynth::FreeResynth;
use lightvc_core::free_vocoder::{FreeVocoder, Grid};
use lightvc_core::mel::MelExtractor;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity").join(name)
}

fn fixtures_ready() -> bool {
    ["freeC.safetensors", "e2e_freeC.safetensors", "mel_basis_44k_2048_128.safetensors"]
        .iter()
        .all(|n| fixture(n).exists())
}

#[test]
#[allow(non_snake_case)]
fn e2e_resynth_freeC() {
    let device = Device::Cpu;
    if !fixtures_ready() {
        eprintln!("skip e2e_resynth_freeC: fixtures absent (gitignored).");
        return;
    }
    let ext = MelExtractor::from_safetensors(
        &fixture("mel_basis_44k_2048_128.safetensors"),
        &device,
    )
    .unwrap();
    let voc =
        FreeVocoder::from_safetensors_with_grid(&fixture("freeC.safetensors"), Grid::FREEC, &device)
            .unwrap();

    let t = candle_core::safetensors::load(fixture("e2e_freeC.safetensors"), &device).unwrap();
    let audio = t.get("audio").unwrap().to_dtype(DType::F32).unwrap();
    let ref_wave = t.get("wave").unwrap().to_dtype(DType::F32).unwrap();
    let audio_v = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Cross-check Rust mel against the fixture's own mel before synthesizing.
    let mel = ext.extract_offline(&audio_v).unwrap();
    if let Some(mel_ref) = t.get("mel") {
        let mr = mel_ref.to_dtype(DType::F32).unwrap();
        let (msnr, mmax) = mel_stats(&mr, &mel);
        println!("e2e_resynth_freeC: mel vs fixture  SNR {msnr:.2} dB, max_abs {mmax:.3e}");
    }

    let wave = voc.forward(&mel).unwrap();
    assert_eq!(wave.dims(), ref_wave.dims(), "wave shape mismatch");

    let y_ref = ref_wave.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let y_rs = wave.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (snr, lag, xc) = best_lag(&y_ref, &y_rs, 256);
    println!("e2e_resynth_freeC: best-lag SNR {snr:.2} dB @lag{lag}, xcorr {xc:.6}");
    assert!(snr >= 40.0, "offline E2E best-lag SNR {snr:.2} dB < 40 dB");
}

#[test]
#[allow(non_snake_case)]
fn streaming_e2e_recovery_freeC() {
    let device = Device::Cpu;
    if !fixtures_ready() {
        eprintln!("skip streaming_e2e_recovery_freeC: fixtures absent (gitignored).");
        return;
    }
    let ext = MelExtractor::from_safetensors(
        &fixture("mel_basis_44k_2048_128.safetensors"),
        &device,
    )
    .unwrap();
    let voc =
        FreeVocoder::from_safetensors_with_grid(&fixture("freeC.safetensors"), Grid::FREEC, &device)
            .unwrap();

    let t = candle_core::safetensors::load(fixture("e2e_freeC.safetensors"), &device).unwrap();
    let audio = t.get("audio").unwrap().to_dtype(DType::F32).unwrap();
    let ref_wave = t.get("wave").unwrap().to_dtype(DType::F32).unwrap();
    let audio_v = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let y_ref = ref_wave.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Attribution 1: the lookahead-centered streaming mel must be bit-identical
    // to the offline mel over every frame it emits (it just stops a few frames
    // early where end reflect-pad would be needed).
    let mel_off = ext.extract_offline(&audio_v).unwrap();
    let mel_str = ext.extract_stream_all(&audio_v).unwrap();
    let (t_off, t_str) = (mel_off.dim(2).unwrap(), mel_str.dim(2).unwrap());
    let mel_off_pref = mel_off.narrow(2, 0, t_str).unwrap();
    let (mel_snr, mel_max) = mel_stats(&mel_off_pref, &mel_str);
    println!(
        "streaming_e2e_recovery_freeC: streaming mel vs offline mel (first {t_str}/{t_off} frames) \
         SNR {mel_snr:.2} dB, max_abs {mel_max:.3e}"
    );
    assert!(
        mel_snr >= 120.0,
        "streaming mel not identical to offline mel: SNR {mel_snr:.2} dB (framing regressed)"
    );

    // THE GATE: realtime path = lookahead-centered streaming mel → causal
    // streaming vocoder (K=4, the realtime operating point) vs the PyTorch E2E
    // reference wave. best-lag absorbs the causal `nfft/2` synthesis offset.
    let y_str = voc
        .stream_all_chunked(&mel_str, 4)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (snr, lag, xc) = best_lag(&y_ref, &y_str, 512);
    println!(
        "streaming_e2e_recovery_freeC: streaming E2E vs PyTorch ref  best-lag SNR {snr:.2} dB \
         @lag{lag}, xcorr {xc:.6}  (was 1.26 dB with left-aligned mel)"
    );

    // Attribution 2: offline E2E (centered mel + center=True iSTFT) vs the same
    // reference — the streaming path should recover to within a hair of it.
    let y_off = voc.forward(&mel_off).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (snr_off, lag_off, _) = best_lag(&y_ref, &y_off, 512);
    println!(
        "streaming_e2e_recovery_freeC: offline E2E vs ref  best-lag SNR {snr_off:.2} dB @lag{lag_off}"
    );

    assert!(xc >= 0.999, "streaming E2E xcorr {xc:.6} < 0.999 — framing still off");
    assert!(
        snr >= 45.0,
        "streaming E2E best-lag SNR {snr:.2} dB < 45 dB — not recovered (was 1.26 dB)"
    );
}

/// Deployed realtime path: `FreeResynth::process_chunk` fed incrementally in
/// `chunk_samples()` blocks (persistent mel rolling-buffer + vocoder state).
/// Exercises the cross-call drain logic the `extract_stream_all` single-push
/// test cannot, and confirms the same E2E recovery vs the PyTorch reference.
#[test]
#[allow(non_snake_case)]
fn free_resynth_incremental_recovery_freeC() {
    let device = Device::Cpu;
    if !fixtures_ready() {
        eprintln!("skip free_resynth_incremental_recovery_freeC: fixtures absent (gitignored).");
        return;
    }
    let mut rs = FreeResynth::new(
        &fixture("freeC.safetensors"),
        &fixture("mel_basis_44k_2048_128.safetensors"),
        4,
        device.clone(),
    )
    .unwrap();
    println!(
        "free_resynth_incremental_recovery_freeC: algorithmic latency {:.2} ms (K=4, chunk {} samp)",
        rs.algorithmic_latency_ms(),
        rs.chunk_samples()
    );

    let t = candle_core::safetensors::load(fixture("e2e_freeC.safetensors"), &device).unwrap();
    let audio = t.get("audio").unwrap().to_dtype(DType::F32).unwrap();
    let ref_wave = t.get("wave").unwrap().to_dtype(DType::F32).unwrap();
    let audio_v = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let y_ref = ref_wave.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let cs = rs.chunk_samples();
    let mut y = Vec::with_capacity(audio_v.len());
    for blk in audio_v.chunks(cs) {
        y.extend_from_slice(&rs.process_chunk(blk).unwrap());
    }
    let (snr, lag, xc) = best_lag(&y_ref, &y, 512);
    println!(
        "free_resynth_incremental_recovery_freeC: incremental E2E vs ref  best-lag SNR {snr:.2} dB \
         @lag{lag}, xcorr {xc:.6}  (emitted {} / ref {} samples)",
        y.len(),
        y_ref.len()
    );
    assert!(xc >= 0.999, "incremental E2E xcorr {xc:.6} < 0.999");
    assert!(snr >= 45.0, "incremental E2E best-lag SNR {snr:.2} dB < 45 dB — not recovered");
}

/// (SNR dB, max abs err) of two mel tensors of identical shape.
fn mel_stats(a: &candle_core::Tensor, b: &candle_core::Tensor) -> (f64, f64) {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut sig = 0f64;
    let mut err = 0f64;
    let mut mx = 0f64;
    for i in 0..av.len().min(bv.len()) {
        let (r, x) = (av[i] as f64, bv[i] as f64);
        sig += r * r;
        err += (r - x) * (r - x);
        mx = mx.max((r - x).abs());
    }
    (10.0 * (sig / err.max(1e-30)).log10(), mx)
}

/// Best-lag SNR over lags in `-max_lag..=max_lag`. Returns (best SNR, lag, xcorr).
fn best_lag(reference: &[f32], test: &[f32], max_lag: i64) -> (f64, i64, f64) {
    best_lag_from(reference, test, max_lag, 0)
}

fn best_lag_from(reference: &[f32], test: &[f32], max_lag: i64, skip: usize) -> (f64, i64, f64) {
    let mut best = (f64::NEG_INFINITY, 0i64, 0f64);
    for lag in -max_lag..=max_lag {
        let (snr, xc) = snr_xcorr(reference, test, lag, skip);
        if snr > best.0 {
            best = (snr, lag, xc);
        }
    }
    best
}

fn snr_xcorr(reference: &[f32], test: &[f32], lag: i64, skip: usize) -> (f64, f64) {
    let mut sig = 0f64;
    let mut err = 0f64;
    let mut dot = 0f64;
    let mut sr = 0f64;
    let mut st = 0f64;
    for i in skip..reference.len() {
        let j = i as i64 + lag;
        if j < 0 || j as usize >= test.len() {
            continue;
        }
        let r = reference[i] as f64;
        let x = test[j as usize] as f64;
        sig += r * r;
        err += (r - x) * (r - x);
        dot += r * x;
        sr += r * r;
        st += x * x;
    }
    let snr = 10.0 * (sig / err.max(1e-30)).log10();
    let xc = dot / (sr.sqrt() * st.sqrt()).max(1e-30);
    (snr, xc)
}
