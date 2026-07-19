//! M1 gate: Rust offline mel extraction vs the bigvgan `mel_spectrogram`
//! reference. Loads `audio [1,44100]` + expected `mel [1,128,344]` from
//! `mel_freeC.safetensors`, builds the analyzer from the standalone
//! `mel_basis_44k_2048_128.safetensors` (the exact filterbank the runtime
//! backend loads), and asserts per-element max abs error < 1e-2 and SNR >=
//! 60 dB. Skips when the gitignored fixtures are absent.

use std::path::PathBuf;

use candle_core::{DType, Device};
use lightvc_core::mel::MelExtractor;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity").join(name)
}

#[test]
#[allow(non_snake_case)]
fn mel_parity_freeC() {
    let device = Device::Cpu;
    let basis = fixture("mel_basis_44k_2048_128.safetensors");
    let melfix = fixture("mel_freeC.safetensors");
    if !basis.exists() || !melfix.exists() {
        eprintln!("skip mel_parity_freeC: fixtures absent (gitignored).");
        return;
    }

    let ext = MelExtractor::from_safetensors(&basis, &device).expect("load mel_basis");

    let t = candle_core::safetensors::load(&melfix, &device).expect("load mel fixture");
    let audio = t.get("audio").expect("audio").to_dtype(DType::F32).unwrap();
    let expected = t.get("mel").expect("mel").to_dtype(DType::F32).unwrap();

    let audio_v = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mel = ext.extract_offline(&audio_v).expect("extract_offline");

    assert_eq!(
        mel.dims(),
        expected.dims(),
        "mel shape mismatch: got {:?}, expected {:?}",
        mel.dims(),
        expected.dims()
    );

    let a = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = mel.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let mut max_abs = 0f64;
    let mut sig = 0f64;
    let mut err = 0f64;
    for i in 0..a.len() {
        let (r, x) = (a[i] as f64, b[i] as f64);
        let d = (r - x).abs();
        if d > max_abs {
            max_abs = d;
        }
        sig += r * r;
        err += (r - x) * (r - x);
    }
    let snr = 10.0 * (sig / err.max(1e-30)).log10();
    println!("mel_parity_freeC: max_abs_err = {max_abs:.3e}, SNR = {snr:.2} dB");

    assert!(max_abs < 1e-2, "max abs err {max_abs:.3e} >= 1e-2");
    assert!(snr >= 60.0, "mel SNR {snr:.2} dB < 60 dB");
}
