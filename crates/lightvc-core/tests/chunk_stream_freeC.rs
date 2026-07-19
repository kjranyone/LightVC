//! Correctness gate for chunk-streaming: `step_chunk` (K frames/step) must be
//! numerically equivalent to per-frame `step` (K=1) — same causal conv
//! left-context, same causal-OLA rolling — and must track the offline
//! (center=True) reconstruction at the causal framing lag (nfft/2). Skips when
//! the gitignored freeC fixtures are absent.

use std::path::PathBuf;

use candle_core::{DType, Device, D};
use lightvc_core::free_vocoder::{FreeVocoder, Grid};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity").join(name)
}

#[test]
#[allow(non_snake_case)]
fn chunk_stream_matches_per_frame_freeC() {
    let device = Device::Cpu;
    let weights = fixture("freeC.safetensors");
    if !weights.exists() || !fixture("parity_freeC.safetensors").exists() {
        eprintln!("skip chunk_stream_freeC: fixtures absent (gitignored).");
        return;
    }
    let grid = Grid::FREEC;
    let voc = FreeVocoder::from_safetensors_with_grid(&weights, grid, &device).unwrap();

    let fix = candle_core::safetensors::load(fixture("parity_freeC.safetensors"), &device).unwrap();
    let mel = fix.get("mel").unwrap().to_dtype(DType::F32).unwrap(); // [1,128,T]
    // trim to 64 frames: enough to cross every K boundary (K<=16) yet fast in
    // an unoptimized `cargo test` (debug) run.
    let t = mel.dim(2).unwrap().min(64);
    let mel = mel.narrow(2, 0, t).unwrap();

    let base = voc.stream_all_chunked(&mel, 1).unwrap().flatten_all().unwrap()
        .to_vec1::<f32>().unwrap();

    // Every chunk size must reproduce the per-frame stream to high SNR
    // (only GEMM float-summation order differs across chunk shapes).
    for k in [2usize, 3, 4, 8, 16] {
        let out = voc.stream_all_chunked(&mel, k).unwrap().flatten_all().unwrap()
            .to_vec1::<f32>().unwrap();
        assert_eq!(out.len(), base.len(), "K={k} length mismatch");
        let snr = snr_db(&base, &out, 0);
        let xc = xcorr(&base, &out);
        println!("chunk K={k} vs per-frame: SNR {snr:.2} dB, xcorr {xc:.6}");
        // Same causal math; only f32 GEMM reduction order differs (gemv at c=1
        // vs gemm at c>=2), which floors agreement near ~90 dB — far above the
        // 60 dB parity gate and inaudible. Guard against real divergence.
        assert!(snr >= 80.0, "K={k}: chunk vs per-frame SNR {snr:.2} dB < 80 dB");
        assert!(xc >= 0.99999, "K={k}: chunk vs per-frame xcorr {xc:.6} < 0.99999");
    }

    // Streaming (causal OLA) must match offline (center=True) at lag nfft/2.
    let y_off = voc.forward(&mel).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let out = voc.stream_all_chunked(&mel, 2).unwrap().flatten_all().unwrap()
        .to_vec1::<f32>().unwrap();
    let lag = (grid.nfft / 2) as i64;
    let snr = snr_db(&y_off, &out, lag);
    println!("chunk K=2 vs offline @lag{lag}: SNR {snr:.2} dB");
    assert!(snr >= 60.0, "chunk vs offline SNR {snr:.2} dB < 60 dB");
    let _ = mel.dim(D::Minus1);
}

fn xcorr(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut ea = 0f64;
    let mut eb = 0f64;
    for i in 0..a.len().min(b.len()) {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        ea += x * x;
        eb += y * y;
    }
    dot / (ea.sqrt() * eb.sqrt()).max(1e-30)
}

fn snr_db(a: &[f32], b: &[f32], lag: i64) -> f64 {
    let mut sig = 0f64;
    let mut err = 0f64;
    for i in 0..a.len() as i64 {
        let j = i + lag;
        if j < 0 || j as usize >= b.len() {
            continue;
        }
        let r = a[i as usize] as f64;
        let t = b[j as usize] as f64;
        sig += r * r;
        err += (r - t) * (r - t);
    }
    10.0 * (sig / err.max(1e-30)).log10()
}
