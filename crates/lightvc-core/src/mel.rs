//! Mel-spectrogram extraction (bigvgan `mel_spectrogram` parity, offline +
//! causal-streaming). Feeds `free_vocoder::FreeVocoder` (freeC grid).
//!
//! Offline algorithm (numerically matches `bigvgan meldataset.mel_spectrogram`,
//! `center=False`):
//!   reflect-pad `(n_fft-hop)/2 = 960` both ends → framed STFT
//!   (`n_fft=2048, hop=128, win=2048, Hann periodic`) → magnitude
//!   `sqrt(re^2+im^2+1e-9)` → `mel_basis @ mag` → `log(clamp(x, min=1e-5))`.
//!
//! candle-core 0.10 has no FFT/complex dtype, so the STFT is a **matrix DFT**
//! (the `free_vocoder` iSTFT tables' forward twin). Only the magnitude is used,
//! so the sign convention of the imaginary part is irrelevant. The Hann window
//! is folded into the DFT matrices. `mel_basis` (librosa slaney) is loaded from
//! safetensors — librosa's mel filterbank is not reimplemented here.
//!
//! Streaming (`MelStreamState`): reflect-pad and centering are impossible
//! without lookahead, so each frame is a **left-aligned trailing window** (the
//! last `win` real samples, zero-padded before the stream start). This is the
//! causal analysis the freeC vocoder is deployed with; the resulting framing
//! offset vs the offline (centered) path is quantified in
//! `tests/e2e_resynth_freeC.rs`.

use candle_core::{Device, Result, Tensor};

pub const N_FFT: usize = 2048;
pub const HOP: usize = 128;
pub const WIN: usize = 2048;
pub const N_MELS: usize = 128;
pub const NB: usize = N_FFT / 2 + 1; // 1025
const PAD: usize = (N_FFT - HOP) / 2; // 960
const MAG_EPS: f64 = 1e-9;
const LOG_CLAMP_MIN: f64 = 1e-5;

/// Immutable mel analyzer: DFT tables (window folded in) + mel filterbank.
pub struct MelExtractor {
    device: Device,
    cos_mat: Tensor,     // [WIN, NB]  window[n] * cos(2*pi*k*n/nfft)
    sin_mat: Tensor,     // [WIN, NB]  window[n] * sin(2*pi*k*n/nfft)
    mel_basis_t: Tensor, // [NB, N_MELS]  (mel_basis transposed, contiguous)
}

/// Causal-streaming state: rolling trailing window + sub-hop sample carry.
pub struct MelStreamState {
    hist: Vec<f32>,  // [WIN] most-recent samples, left zero-padded at start
    carry: Vec<f32>, // buffered samples not yet forming a full hop
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

    /// Fresh causal-streaming state (zeroed trailing window, empty carry).
    pub fn new_stream(&self) -> MelStreamState {
        MelStreamState { hist: vec![0f32; WIN], carry: Vec::with_capacity(HOP) }
    }

    /// Push mono samples; emit every newly-completed left-aligned frame as a
    /// `[1, N_MELS, nf]` tensor (or `None` if fewer than `HOP` samples have
    /// accumulated since the last emission). Each emitted frame is the DFT of
    /// the trailing `WIN` real samples — no lookahead, no reflect pad.
    pub fn stream_push(
        &self,
        st: &mut MelStreamState,
        samples: &[f32],
    ) -> Result<Option<Tensor>> {
        st.carry.extend_from_slice(samples);
        let nf = st.carry.len() / HOP;
        if nf == 0 {
            return Ok(None);
        }
        let mut frames = vec![0f32; nf * WIN];
        for f in 0..nf {
            // Roll the trailing window left by HOP, append the next hop.
            st.hist.copy_within(HOP.., 0);
            st.hist[WIN - HOP..].copy_from_slice(&st.carry[f * HOP..f * HOP + HOP]);
            frames[f * WIN..f * WIN + WIN].copy_from_slice(&st.hist);
        }
        st.carry.drain(..nf * HOP);
        let frames = Tensor::from_vec(frames, (nf, WIN), &self.device)?;
        Ok(Some(self.frames_to_mel(&frames, nf)?))
    }

    /// Convenience: causal-streaming log-mel over a whole signal (fresh state,
    /// one frame per `HOP` samples). Returns `[1, N_MELS, N/HOP]`.
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
