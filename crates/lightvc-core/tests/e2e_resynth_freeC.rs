//! M3 gates for the FreeVocoder resynthesis path (Rust mel → freeC vocoder).
//!
//! `e2e_resynth_freeC`  — offline: `audio → Rust mel → FreeVocoder.forward` vs
//!   the PyTorch end-to-end reference wave (`e2e_freeC.safetensors`). Reports
//!   best-lag SNR (expect ≈60 dB near lag 0; the only delta vs the 88 dB
//!   `parity_freeC` gate is the Rust-vs-Python mel).
//!
//! `streaming_causal_framing_freeC` — HONEST framing audit: the realtime path
//!   (causal left-aligned streaming mel + causal streaming vocoder) cannot
//!   reflect-pad or center like training's `center=False`+960-pad analysis, so
//!   its frames are right-aligned trailing windows. This quantifies how far the
//!   streaming output drifts from the offline E2E output — whether the offset
//!   is a pure shift (absorbable, high best-lag SNR) or real degradation.

use std::path::PathBuf;

use candle_core::{DType, Device};
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
fn streaming_causal_framing_freeC() {
    let device = Device::Cpu;
    if !fixtures_ready() {
        eprintln!("skip streaming_causal_framing_freeC: fixtures absent (gitignored).");
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
    let audio_v = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Offline E2E reference (centered mel + center=True iSTFT).
    let mel_off = ext.extract_offline(&audio_v).unwrap();
    let y_off = voc.forward(&mel_off).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Realtime path: causal left-aligned streaming mel + causal streaming
    // vocoder (K=4 chunk, the realtime operating point).
    let mel_str = ext.extract_stream_all(&audio_v).unwrap();
    let y_str = voc
        .stream_all_chunked(&mel_str, 4)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Attribution: run the SAME causal streaming mel through the OFFLINE
    // (center=True) vocoder. Isolates the mel-framing damage from the vocoder
    // streaming (which chunk_stream_freeC already shows is ~90 dB innocent).
    let y_str_off = voc
        .forward(&mel_str)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (snr_mel, lag_mel, xc_mel) = best_lag(&y_off, &y_str_off, 2048);
    println!(
        "streaming_causal_framing_freeC: mel-framing only (offline voc) SNR {snr_mel:.2} dB \
         @lag{lag_mel}, xcorr {xc_mel:.6}"
    );

    // Framing offset can be up to ~win/2 - hop/2 (mel) + nfft/2 (vocoder OLA),
    // so search a wide lag window and report the best alignment honestly.
    let (snr, lag, xc) = best_lag(&y_off, &y_str, 2048);
    // Also report the residual once the streaming warm-up (first ~16 mel
    // frames = 2048 samples, trailing window still filling) is skipped.
    let warm = 2048usize;
    let (snr_w, _, xc_w) = best_lag_from(&y_off, &y_str, 2048, warm);
    println!(
        "streaming_causal_framing_freeC: full best-lag SNR {snr:.2} dB @lag{lag}, xcorr {xc:.6}"
    );
    println!(
        "streaming_causal_framing_freeC: post-warmup(skip {warm}) SNR {snr_w:.2} dB, xcorr {xc_w:.6}"
    );
    println!(
        "streaming_causal_framing_freeC: interpretation — high xcorr + low SNR ⇒ near-shift with \
         fractional-frame framing error; low xcorr ⇒ genuine analysis mismatch (needs causal-\
         integrated mel or shorter analysis window)."
    );
    // Not a hard pass/fail gate: this is a design-characterization test. Only
    // guard against a total collapse (mel/vocoder wiring broken).
    assert!(xc > 0.3, "streaming output uncorrelated with offline (xcorr {xc:.6}) — path broken");
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
