//! freeC streaming: single-thread CPU RTF sweep over chunk size K, plus
//! correctness (chunk vs per-frame vs offline) and E2E latency breakdown.
//! Honest numbers only — no synthetic FLOP inflation.
//!
//! Run: `RAYON_NUM_THREADS=1 cargo run --release -p lightvc-core \
//!       --example bench_freec_stream`

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

    // Tile the parity mel to ~4.6 s of audio for a stable RTF measurement.
    let fix = candle_core::safetensors::load(fixture("parity_freeC.safetensors"), &device)?;
    let mel0 = fix.get("mel").unwrap().to_dtype(DType::F32)?; // [1,128,320]
    let reps = 5usize;
    let mel = Tensor::cat(&vec![&mel0; reps], D::Minus1)?; // [1,128,1600]
    let t = mel.dim(D::Minus1)?;
    let audio_secs = (t * grid.hop) as f64 / SR;
    let hop_ms = 1000.0 * grid.hop as f64 / SR;
    let algo_ms = 1000.0 * grid.win as f64 / SR;
    println!("grid: nfft={} win={} hop={} causal={}", grid.nfft, grid.win, grid.hop, grid.causal);
    println!("mel frames T={t}  -> audio {:.3} s @ {SR} Hz", audio_secs);
    println!("hop = {:.3} ms   algorithmic latency (causal win/sr) = {:.2} ms", hop_ms, algo_ms);

    // ---- Offline batched forward (whole-utterance GEMMs) as the RT floor ----
    let _ = voc.forward(&mel)?; // warm
    let o0 = Instant::now();
    let offline = voc.forward(&mel)?;
    let off_elapsed = o0.elapsed().as_secs_f64();
    let y_off = offline.flatten_all()?.to_vec1::<f32>()?;
    println!("\n=== offline batched forward (RT floor) ===");
    println!("RTF = {:.4}", off_elapsed / audio_secs);

    // ---- RTF sweep over chunk size K ----
    println!("\n=== chunk-streaming RTF sweep (single-thread CPU) ===");
    println!("{:>4} | {:>8} | {:>10} | {:>12} | {:>10} | {:>12}",
             "K", "RTF", "mean ms/K", "worst ms/K", "hop-hdrm", "add-lat ms");
    println!("{}", "-".repeat(72));

    let ks = [1usize, 2, 4, 8, 16];
    let mut baseline: Option<Vec<f32>> = None;
    for &k in &ks {
        // Warm-up one full pass at this chunk size.
        let _ = voc.stream_all_chunked(&mel, k)?;

        let mut st = voc.new_stream()?;
        let mut out: Vec<f32> = Vec::with_capacity(t * grid.hop);
        let mut worst_us = 0f64;
        let mut n_steps = 0u64;
        let t0 = Instant::now();
        let mut f = 0usize;
        while f < t {
            let c = k.min(t - f);
            let mc = mel.narrow(D::Minus1, f, c)?;
            let s0 = Instant::now();
            let emit = voc.step_chunk(&mut st, &mc)?;
            let dt = s0.elapsed().as_secs_f64() * 1e6;
            if dt > worst_us { worst_us = dt; }
            out.extend_from_slice(&emit);
            n_steps += 1;
            f += c;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let rtf = elapsed / audio_secs;
        let mean_step_ms = 1000.0 * elapsed / n_steps as f64;
        let chunk_budget_ms = hop_ms * k as f64;
        let headroom = chunk_budget_ms / mean_step_ms;
        let add_lat_ms = hop_ms * (k as f64 - 1.0); // buffering beyond per-frame
        println!("{:>4} | {:>8.4} | {:>10.3} | {:>12.3} | {:>9.2}x | {:>12.2}",
                 k, rtf, mean_step_ms, worst_us / 1000.0, headroom, add_lat_ms);

        if k == 1 { baseline = Some(out.clone()); }
        else if let Some(b) = &baseline {
            let (snr, xc) = compare_shift(b, &out, 0);
            println!("       chunk(K={k}) vs per-frame(K=1): zero-lag SNR {:.2} dB, xcorr {:.6}", snr, xc);
        }
    }

    // ---- Correctness vs offline (center=True) at the causal framing lag ----
    let base = baseline.unwrap();
    println!("\n=== correctness: streaming(causal OLA) vs offline(center=True) ===");
    println!("offline len {}  streaming len {}", y_off.len(), base.len());
    let (mut best_lag, mut best_snr, mut best_xc) = (0i64, f64::NEG_INFINITY, 0f64);
    for lag in 0..=(grid.win as i64) {
        let (snr, xc) = compare_shift(&y_off, &base, lag);
        if snr > best_snr { best_snr = snr; best_xc = xc; best_lag = lag; }
    }
    let (snr0, xc0) = compare_shift(&y_off, &base, 0);
    println!("zero-lag        : SNR {:.2} dB, xcorr {:.6}", snr0, xc0);
    println!("best-lag {:>4}   : SNR {:.2} dB, xcorr {:.6}  (nfft/2={})",
             best_lag, best_snr, best_xc, grid.nfft / 2);

    // ---- E2E latency breakdown ----
    println!("\n=== E2E latency breakdown (per K, 50 ms budget) ===");
    println!("algorithmic (window/lookahead, causal): {:.2} ms", algo_ms);
    for &k in &ks {
        let add = hop_ms * (k as f64 - 1.0);
        let total = algo_ms + add;
        println!("K={:>2}: algo {:.2} + chunk-buffer {:.2} = {:.2} ms  [{}]",
                 k, algo_ms, add, total, if total < 50.0 { "within 50 ms" } else { "OVER" });
    }

    Ok(())
}

/// Compare a[i] with b[i+lag] over the overlapping interior. (SNR dB, xcorr).
fn compare_shift(a: &[f32], b: &[f32], lag: i64) -> (f64, f64) {
    let mut sig = 0f64; let mut err = 0f64;
    let mut dot = 0f64; let mut ea = 0f64; let mut eb = 0f64;
    let n = a.len() as i64;
    let mut count = 0i64;
    for i in 0..n {
        let j = i + lag;
        if j < 0 || j as usize >= b.len() { continue; }
        let r = a[i as usize] as f64;
        let t = b[j as usize] as f64;
        sig += r * r; err += (r - t) * (r - t);
        dot += r * t; ea += r * r; eb += t * t;
        count += 1;
    }
    if count == 0 { return (f64::NEG_INFINITY, 0.0); }
    let snr = 10.0 * (sig / err.max(1e-30)).log10();
    let xc = dot / (ea.sqrt() * eb.sqrt()).max(1e-30);
    (snr, xc)
}
