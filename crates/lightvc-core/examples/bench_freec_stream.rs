//! freeC streaming: single-thread CPU RTF + streaming(causal-OLA) vs
//! offline(center=True) framing-difference report. Honest numbers only.
//!
//! Run: `cargo run --release -p lightvc-core --example bench_freec_stream`

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor, D};
use lightvc_core::free_vocoder::{FreeVocoder, Grid};

const SR: f64 = 44100.0;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity").join(name)
}

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let grid = Grid::FREEC;

    let weights = fixture("freeC.safetensors");
    if !weights.exists() {
        eprintln!("skip: freeC.safetensors absent");
        return Ok(());
    }
    let voc = FreeVocoder::from_safetensors_with_grid(&weights, grid, &device)?;

    // Tile the parity mel to ~3 s of audio for a stable RTF measurement.
    let fix = candle_core::safetensors::load(fixture("parity_freeC.safetensors"), &device)?;
    let mel0 = fix.get("mel").unwrap().to_dtype(DType::F32)?; // [1,128,320]
    let reps = 5usize;
    let mel = Tensor::cat(&vec![&mel0; reps], D::Minus1)?; // [1,128,1600]
    let t = mel.dim(D::Minus1)?;
    let audio_secs = (t * grid.hop) as f64 / SR;
    println!("grid: nfft={} win={} hop={} causal={}", grid.nfft, grid.win, grid.hop, grid.causal);
    println!("mel frames T={t}  -> audio {:.3} s @ {SR} Hz", audio_secs);
    println!("algorithmic latency (causal win/hop): {:.2} ms",
             1000.0 * grid.win as f64 / SR);

    // ---- Warm-up (fill caches / one full pass) ----
    let _ = voc.stream_all(&mel)?;

    // ---- RTF: frame-by-frame streaming, single thread ----
    let mut st = voc.new_stream()?;
    let mut stream_out: Vec<f32> = Vec::with_capacity(t * grid.hop);
    let mut worst_frame_us = 0f64;
    let t0 = Instant::now();
    for f in 0..t {
        let frame = mel.narrow(D::Minus1, f, 1)?;
        let f0 = Instant::now();
        let emit = voc.step(&mut st, &frame)?;
        let dt = f0.elapsed().as_secs_f64() * 1e6;
        if dt > worst_frame_us { worst_frame_us = dt; }
        stream_out.extend_from_slice(&emit);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let rtf = elapsed / audio_secs;
    let per_frame_ms = 1000.0 * elapsed / t as f64;
    let hop_budget_ms = 1000.0 * grid.hop as f64 / SR;
    println!("\n=== RTF (single-thread CPU, streaming) ===");
    println!("total {:.3} s for {:.3} s audio  -> RTF = {:.4}", elapsed, audio_secs, rtf);
    println!("mean per-frame {:.3} ms  (hop budget {:.3} ms)  -> {:.1}x headroom",
             per_frame_ms, hop_budget_ms, hop_budget_ms / per_frame_ms);
    println!("worst per-frame {:.3} ms", worst_frame_us / 1000.0);

    // ---- Batched offline RTF (same model, whole-utterance GEMMs) for contrast ----
    let _ = voc.forward(&mel)?; // warm
    let o0 = Instant::now();
    let offline = voc.forward(&mel)?;
    let off_elapsed = o0.elapsed().as_secs_f64();
    println!("\n=== RTF (single-thread CPU, batched offline forward) ===");
    println!("total {:.3} s for {:.3} s audio  -> RTF = {:.4}  ({}x faster than per-frame)",
             off_elapsed, audio_secs, off_elapsed / audio_secs,
             (elapsed / off_elapsed).round() as i64);

    // ---- Framing difference: streaming (causal OLA) vs offline (center=True) ----
    let y_off = offline.flatten_all()?.to_vec1::<f32>()?;
    let y_str = &stream_out;
    println!("\n=== framing: streaming(causal OLA) vs offline(center=True) ===");
    println!("offline len {}  streaming len {}", y_off.len(), y_str.len());

    // Search a small integer lag: streaming[j] should track offline[j - lag].
    let (mut best_lag, mut best_snr, mut best_xc) = (0i64, f64::NEG_INFINITY, 0f64);
    for lag in 0..=(grid.win as i64) {
        let (snr, xc) = compare_shift(&y_off, y_str, lag);
        if snr > best_snr { best_snr = snr; best_xc = xc; best_lag = lag; }
    }
    let (snr0, xc0) = compare_shift(&y_off, y_str, 0);
    println!("zero-lag        : SNR {:.2} dB, xcorr {:.6}", snr0, xc0);
    println!("best-lag {:>4}   : SNR {:.2} dB, xcorr {:.6}  (nfft/2={})",
             best_lag, best_snr, best_xc, grid.nfft / 2);

    Ok(())
}

/// Compare offline[i] with streaming[i+lag] over the overlapping interior.
/// Returns (SNR dB, normalized xcorr).
fn compare_shift(off: &[f32], stream: &[f32], lag: i64) -> (f64, f64) {
    let mut sig = 0f64; let mut err = 0f64;
    let mut dot = 0f64; let mut sr = 0f64; let mut ststat = 0f64;
    let n = off.len() as i64;
    let mut count = 0i64;
    for i in 0..n {
        let j = i + lag;
        if j < 0 || j as usize >= stream.len() { continue; }
        let r = off[i as usize] as f64;
        let t = stream[j as usize] as f64;
        sig += r * r; err += (r - t) * (r - t);
        dot += r * t; sr += r * r; ststat += t * t;
        count += 1;
    }
    if count == 0 { return (f64::NEG_INFINITY, 0.0); }
    let snr = 10.0 * (sig / err.max(1e-30)).log10();
    let xc = dot / (sr.sqrt() * ststat.sqrt()).max(1e-30);
    (snr, xc)
}
