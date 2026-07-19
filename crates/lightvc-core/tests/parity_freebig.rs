//! E4 parity gate: Rust/Candle FreeVocoder (`freebig`) vs PyTorch reference.
//!
//! Loads a fixed mel `[1,128,80]` and the expected wave `[1,40448]` produced by
//! `training/free_vocoder.py::FreeVocoder`, runs the Candle port, and asserts
//! SNR >= 60 dB and normalized cross-correlation >= 0.9999.

use std::path::PathBuf;

use candle_core::{DType, Device};
use lightvc_core::free_vocoder::FreeVocoder;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/parity")
        .join(name)
}

#[test]
fn parity_freebig() {
    let device = Device::Cpu;

    let weights = fixture("freebig.safetensors");
    if !weights.exists() || !fixture("parity_freebig.safetensors").exists() {
        eprintln!(
            "skip parity_freebig: fixtures absent (gitignored). Regenerate with \
             training/ export (see current/candle_vocoder_port.md)."
        );
        return;
    }
    let voc = FreeVocoder::from_safetensors(&weights, &device)
        .expect("load freebig weights");

    // Load parity fixture (mel + expected wave).
    let fix = fixture("parity_freebig.safetensors");
    let tensors = candle_core::safetensors::load(&fix, &device).expect("load fixture");
    let mel = tensors.get("mel").expect("mel tensor").to_dtype(DType::F32).unwrap();
    let expected = tensors.get("wave").expect("wave tensor").to_dtype(DType::F32).unwrap();

    let wave = voc.forward(&mel).expect("forward");

    assert_eq!(
        wave.dims(),
        expected.dims(),
        "output shape mismatch: got {:?}, expected {:?}",
        wave.dims(),
        expected.dims()
    );

    let y_rs = wave.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let y_py = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let (snr, xcorr) = metrics(&y_py, &y_rs);
    println!("parity_freebig: SNR = {snr:.2} dB, xcorr = {xcorr:.6}");

    assert!(snr >= 60.0, "SNR {snr:.2} dB < 60 dB");
    assert!(xcorr >= 0.9999, "xcorr {xcorr:.6} < 0.9999");
}

/// Returns (SNR dB, zero-lag normalized cross-correlation).
fn metrics(reference: &[f32], test: &[f32]) -> (f64, f64) {
    let n = reference.len();
    let mut sig = 0f64;
    let mut err = 0f64;
    let mut dot = 0f64;
    let mut ss_r = 0f64;
    let mut ss_t = 0f64;
    for i in 0..n {
        let r = reference[i] as f64;
        let t = test[i] as f64;
        sig += r * r;
        err += (r - t) * (r - t);
        dot += r * t;
        ss_r += r * r;
        ss_t += t * t;
    }
    let snr = 10.0 * (sig / err.max(1e-30)).log10();
    let xcorr = dot / (ss_r.sqrt() * ss_t.sqrt()).max(1e-30);
    (snr, xcorr)
}
