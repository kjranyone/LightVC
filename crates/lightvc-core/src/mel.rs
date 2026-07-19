//! Mel-spectrogram extraction (bigvgan `mel_spectrogram` parity, offline +
//! lookahead-centered streaming). Feeds `free_vocoder::FreeVocoder` (freeC grid).
//!
//! Offline algorithm (numerically matches `bigvgan meldataset.mel_spectrogram`,
//! `center=False`):
//!   reflect-pad `(n_fft-hop)/2 = 960` both ends → framed STFT
//!   (`n_fft=2048, hop=128, win=2048, Hann periodic`) → magnitude
//!   `sqrt(re^2+im^2+1e-9)` → `mel_basis @ mag` → `log(clamp(x, min=1e-5))`.
//!   In original coordinates, frame `t` spans `[t*hop-960, t*hop+1088)` — i.e.
//!   effectively **centered**, with ~`win/2` of forward lookahead.
//!
//! candle-core 0.10 has no FFT/complex dtype, so the STFT is a **matrix DFT**
//! (the `free_vocoder` iSTFT tables' forward twin). Only the magnitude is used,
//! so the sign convention of the imaginary part is irrelevant. The Hann window
//! is folded into the DFT matrices. `mel_basis` (librosa slaney) is loaded from
//! safetensors — librosa's mel filterbank is not reimplemented here.
//!
//! Streaming (`MelStreamState`): a **lookahead-centered** analyzer that
//! reproduces the offline framing *exactly*. Input samples are buffered in a
//! rolling window; frame `t` is emitted only once the stream reaches original
//! index `t*hop + (win-1-960)`, so its analysis window is the identical centered
//! `[t*hop-960, t*hop+1088)` span the offline path uses — the front reflect-pad
//! is reconstructed from real past samples (`x[1..=960]`), so every emitted
//! frame is bit-for-bit the offline frame. The trailing frames that would need
//! *end* reflect-pad (unknown future) are simply not emitted. Cost: a fixed
//! analysis lookahead of `MEL_LOOKAHEAD = win-960 = 1088` samples (24.7 ms @
//! 44.1 kHz). This replaces the old left-aligned trailing window (which put the
//! streaming mel on a different, ~7.5-frame-shifted grid than the freeC vocoder
//! was trained on, collapsing streaming E2E SNR to ~1.3 dB). Verified in
//! `tests/e2e_resynth_freeC.rs::streaming_e2e_recovery_freeC`.

use candle_core::{Device, Result, Tensor};

pub const N_FFT: usize = 2048;
pub const HOP: usize = 128;
pub const WIN: usize = 2048;
pub const N_MELS: usize = 128;
pub const NB: usize = N_FFT / 2 + 1; // 1025
const PAD: usize = (N_FFT - HOP) / 2; // 960
const MAG_EPS: f64 = 1e-9;
const LOG_CLAMP_MIN: f64 = 1e-5;

/// Forward-lookahead (in 44.1 kHz samples) the lookahead-centered streaming mel
/// must buffer before it can emit a frame on the offline grid: `win - PAD`
/// (= 1088, ≈ 24.7 ms). This is the analysis latency of `stream_push`; the
/// causal `free_vocoder` synthesis path adds none.
pub const MEL_LOOKAHEAD: usize = WIN - PAD; // 1088

/// Immutable mel analyzer: DFT tables (window folded in) + mel filterbank.
pub struct MelExtractor {
    device: Device,
    cos_mat: Tensor,     // [WIN, NB]  window[n] * cos(2*pi*k*n/nfft)
    sin_mat: Tensor,     // [WIN, NB]  window[n] * sin(2*pi*k*n/nfft)
    mel_basis_t: Tensor, // [NB, N_MELS]  (mel_basis transposed, contiguous)
}

/// Lookahead-centered streaming state: a rolling real-sample buffer indexed in
/// original (pre-pad) coordinates, plus the index of the next offline frame to
/// emit. Reproduces the offline reflect-pad + `center=False` framing exactly.
pub struct MelStreamState {
    buf: Vec<f32>,     // real input samples x[base .. base+buf.len())
    base: usize,       // original (pre-pad) index of buf[0]
    next_frame: usize, // index t of the next offline frame to emit
}

impl MelExtractor {
    /// Build from an in-memory `mel_basis` tensor `[N_MELS, NB]` (slaney).
    pub fn from_mel_basis(mel_basis: &Tensor, device: &Device) -> Result<Self> {
        let mb = mel_basis.to_dtype(candle_core::DType::F32)?;
        let (m, nb) = mb.dims2()?;
        assert_eq!(m, N_MELS, "mel_basis rows != {N_MELS}");
        assert_eq!(nb, NB, "mel_basis cols != {NB}");
        let mel_basis_t = mb.transpose(0, 1)?.contiguous()?; // [NB, N_MELS]

        let two_pi = 2.0f64 * std::f64::consts::PI;
        let window: Vec<f64> = (0..WIN)
            .map(|n| 0.5 - 0.5 * (two_pi * n as f64 / WIN as f64).cos())
            .collect();
        let mut cos_v = vec![0f32; WIN * NB];
        let mut sin_v = vec![0f32; WIN * NB];
        for n in 0..WIN {
            let w = window[n];
            for k in 0..NB {
                let theta = two_pi * (k as f64) * (n as f64) / (N_FFT as f64);
                cos_v[n * NB + k] = (w * theta.cos()) as f32;
                sin_v[n * NB + k] = (w * theta.sin()) as f32;
            }
        }
        let cos_mat = Tensor::from_vec(cos_v, (WIN, NB), device)?;
        let sin_mat = Tensor::from_vec(sin_v, (WIN, NB), device)?;
        Ok(Self { device: device.clone(), cos_mat, sin_mat, mel_basis_t })
    }

    /// Load the `mel_basis` tensor from a safetensors file (key `mel_basis`).
    pub fn from_safetensors(path: &std::path::Path, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let mel_basis = tensors
            .get("mel_basis")
            .ok_or_else(|| candle_core::Error::Msg("mel_basis missing".into()))?;
        Self::from_mel_basis(mel_basis, device)
    }

    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Shared frame stack -> log-mel. `frames`: `[T, WIN]` (raw, unwindowed;
    /// the Hann window is baked into the DFT matrices). Returns `[1, N_MELS, T]`.
    fn frames_to_mel(&self, frames: &Tensor, t: usize) -> Result<Tensor> {
        let re = frames.matmul(&self.cos_mat)?; // [T, NB]
        let im = frames.matmul(&self.sin_mat)?; // [T, NB]
        let mag = (re.sqr()? + im.sqr()?)?.affine(1.0, MAG_EPS)?.sqrt()?; // [T, NB]
        let mel = mag.matmul(&self.mel_basis_t)?; // [T, N_MELS]
        let mel = mel.clamp(LOG_CLAMP_MIN, 1e30)?.log()?;
        let mel = mel.transpose(0, 1)?.contiguous()?; // [N_MELS, T]
        mel.reshape((1, N_MELS, t))
    }

    /// Offline log-mel: reflect-pad + `center=False` STFT (after the 960-pad).
    /// `audio`: `[N]` mono. Returns `[1, N_MELS, T]`,
    /// `T = 1 + (N + 2*960 - 2048)/128`.
    pub fn extract_offline(&self, audio: &[f32]) -> Result<Tensor> {
        assert!(audio.len() > PAD, "audio shorter than reflect pad");
        let padded = reflect_pad(audio, PAD);
        let n = padded.len();
        let t = 1 + (n - N_FFT) / HOP;
        let mut frames = vec![0f32; t * WIN];
        for f in 0..t {
            let off = f * HOP;
            frames[f * WIN..f * WIN + WIN].copy_from_slice(&padded[off..off + WIN]);
        }
        let frames = Tensor::from_vec(frames, (t, WIN), &self.device)?;
        self.frames_to_mel(&frames, t)
    }

    /// Fresh lookahead-centered streaming state (empty buffer, frame 0 next).
    pub fn new_stream(&self) -> MelStreamState {
        MelStreamState { buf: Vec::new(), base: 0, next_frame: 0 }
    }

    /// Reconstruct the offline reflect-padded sample at padded index `p`
    /// (`p < PAD` ⇒ front reflect `x[PAD-p]`, else real `x[p-PAD]`), reading the
    /// rolling buffer. `p-PAD >= base` and `PAD-p >= base` are guaranteed for
    /// every index any not-yet-emitted frame references, so no bounds slack.
    #[inline]
    fn padded_at(st: &MelStreamState, p: usize) -> f32 {
        let orig = if p < PAD { PAD - p } else { p - PAD };
        st.buf[orig - st.base]
    }

    /// Push mono samples; emit every frame that has become fully available on
    /// the **offline centered grid** as a `[1, N_MELS, nf]` tensor (or `None`
    /// while still within the startup lookahead). Frame `t` uses the identical
    /// window offline frame `t` uses (`padded[t*HOP .. t*HOP+WIN]`), so it is
    /// bit-for-bit the offline mel. Costs `MEL_LOOKAHEAD` samples of latency;
    /// the trailing frames needing end reflect-pad are never emitted.
    pub fn stream_push(
        &self,
        st: &mut MelStreamState,
        samples: &[f32],
    ) -> Result<Option<Tensor>> {
        st.buf.extend_from_slice(samples);
        let n_seen = st.base + st.buf.len();
        // Frame t needs original samples through index t*HOP + (WIN-1-PAD),
        // i.e. n_seen >= t*HOP + MEL_LOOKAHEAD. Highest emittable frame:
        if n_seen < MEL_LOOKAHEAD {
            return Ok(None);
        }
        let last_t = (n_seen - MEL_LOOKAHEAD) / HOP;
        if last_t < st.next_frame {
            return Ok(None);
        }
        let nf = last_t - st.next_frame + 1;
        let mut frames = vec![0f32; nf * WIN];
        for i in 0..nf {
            let p0 = (st.next_frame + i) * HOP; // window start in padded coords
            for n in 0..WIN {
                frames[i * WIN + n] = Self::padded_at(st, p0 + n);
            }
        }
        st.next_frame = last_t + 1;

        // Drop samples no longer referenced. While any remaining frame still
        // reaches into the front reflect region (next_frame*HOP < PAD) keep the
        // full head (x[1..=PAD]); otherwise retain from the lowest real index
        // the next frame reads, next_frame*HOP - PAD.
        let retain_lo = (st.next_frame * HOP).saturating_sub(PAD);
        if retain_lo > st.base {
            st.buf.drain(..retain_lo - st.base);
            st.base = retain_lo;
        }

        let frames = Tensor::from_vec(frames, (nf, WIN), &self.device)?;
        Ok(Some(self.frames_to_mel(&frames, nf)?))
    }

    /// Convenience: lookahead-centered streaming log-mel over a whole signal
    /// (fresh state). Emits every frame on the offline grid that does not need
    /// end reflect-pad, so `T_stream = 1 + (N - MEL_LOOKAHEAD)/HOP` — a few
    /// frames short of the offline `T` at the tail. Returns `[1, N_MELS, T_stream]`.
    pub fn extract_stream_all(&self, audio: &[f32]) -> Result<Tensor> {
        let mut st = self.new_stream();
        match self.stream_push(&mut st, audio)? {
            Some(mel) => Ok(mel),
            None => Tensor::zeros((1, N_MELS, 0), candle_core::DType::F32, &self.device),
        }
    }
}

/// torch/numpy `reflect` padding (border sample not repeated).
fn reflect_pad(x: &[f32], p: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * p);
    for j in 0..p {
        out.push(x[p - j]);
    }
    out.extend_from_slice(x);
    for j in 0..p {
        out.push(x[n - 2 - j]);
    }
    out
}
