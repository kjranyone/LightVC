//! `FreeResynth` — mic → mel (Rust) → FreeVocoder → out resynthesis backend.
//!
//! Wires the causal-streaming `MelExtractor` (freeC analysis, `hop=128`) into
//! the causal `FreeVocoder` (`Grid::FREEC`, `hop=128`) for real-time vocoder
//! re-synthesis. There is *no* voice-conversion front-end yet: the input mel is
//! the mel of the input audio itself, so `FreeResynth` reconstructs the input
//! through the neural vocoder (a passthrough that exercises the full mel→wave
//! path on the realtime thread).
//!
//! `process_chunk` consumes exactly `chunk_samples() = k*hop` samples at
//! 44.1 kHz, produces `k` mel frames, and streams them through the vocoder for
//! `k*hop` output samples (1:1 length). Streaming state (mel trailing window,
//! vocoder conv left-context + OLA ring) persists across calls via `&mut self`.

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::free_vocoder::{FreeVocoder, Grid, StreamState};
use crate::mel::{MelExtractor, MelStreamState, HOP};

/// Default mel frames per processing chunk. `k=4` gives `chunk_samples=512`
/// (11.6 ms) and was measured RTF≈0.94 single-thread on CPU (realtime).
pub const DEFAULT_CHUNK_FRAMES: usize = 4;

pub struct FreeResynth {
    voc: FreeVocoder,
    mel: MelExtractor,
    voc_stream: StreamState,
    mel_stream: MelStreamState,
    k: usize,
    device: Device,
}

impl FreeResynth {
    /// Load the freeC vocoder weights (`voc_path`) and mel filterbank
    /// (`mel_basis_path`, key `mel_basis`). `k` = mel frames per chunk.
    pub fn new(
        voc_path: &std::path::Path,
        mel_basis_path: &std::path::Path,
        k: usize,
        device: Device,
    ) -> Result<Self> {
        assert!(k >= 1, "chunk frames k must be >= 1");
        let voc = FreeVocoder::from_safetensors_with_grid(voc_path, Grid::FREEC, &device)?;
        assert_eq!(voc.grid().hop, HOP, "vocoder hop must match mel hop");
        let mel = MelExtractor::from_safetensors(mel_basis_path, &device)?;
        let voc_stream = voc.new_stream()?;
        let mel_stream = mel.new_stream();
        Ok(Self { voc, mel, voc_stream, mel_stream, k, device })
    }

    /// Samples per streaming chunk (44.1 kHz): `k * hop`.
    pub fn chunk_samples(&self) -> usize {
        self.k * self.voc.grid().hop
    }

    /// Causal resynthesis of one `chunk_samples()`-sized 44.1 kHz block.
    /// Any tail shorter than a full `hop` is carried in the mel state to the
    /// next call, so output length equals `hop *` (whole mel frames produced).
    pub fn process_chunk(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let mel_chunk = self.mel.stream_push(&mut self.mel_stream, pcm)?;
        match mel_chunk {
            Some(mel) => Ok(self.voc.step_chunk(&mut self.voc_stream, &mel)?),
            None => Ok(Vec::new()),
        }
    }

    /// Offline resynthesis (centered mel + `center=True` iSTFT). `[1, hop*(T-1)]`
    /// worth of samples returned as a flat `Vec<f32>`.
    pub fn process_full(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let mel = self.mel.extract_offline(pcm)?;
        let wave = self.voc.forward(&mel)?;
        Ok(wave.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Reset streaming state (mel trailing window + vocoder context/OLA).
    pub fn reset(&mut self) {
        self.mel_stream = self.mel.new_stream();
        if let Ok(s) = self.voc.new_stream() {
            self.voc_stream = s;
        }
    }

    /// Algorithmic latency (ms) at 44.1 kHz: the causal synthesis window plus
    /// the `(k-1)*hop` chunk-buffering term. The left-aligned mel analysis adds
    /// no lookahead latency. freeC (`win=256`, `k=4`) ⇒ 14.5 ms.
    pub fn algorithmic_latency_ms(&self) -> f32 {
        let g = self.voc.grid();
        (g.win as f32 + (self.k as f32 - 1.0) * g.hop as f32) / 44.1
    }

    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Expose the mel of `pcm` (offline path) — used by tests / diagnostics.
    pub fn mel_offline(&self, pcm: &[f32]) -> Result<Tensor> {
        Ok(self.mel.extract_offline(pcm)?)
    }
}
