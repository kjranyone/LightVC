//! Streaming codec wrappers with conv-state caching and overlap-add.
//!
//! DAC is non-causal (symmetric padding). For real-time streaming we use
//! chunked processing: buffer `hop_length × n_frames` samples per chunk,
//! cache the input tail for receptive-field overlap, and trim the output
//! to only newly produced frames.

use std::collections::VecDeque;

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::{
    codec::{DacCodec, DacConfig},
    DAC_HOP_LENGTH,
};

/// Number of samples to retain as overlap to absorb DAC's non-causal
/// receptive field. This is a conservative estimate based on the cascaded
/// dilated convolutions in the encoder (dilations up to 9 × 7 = 63 samples
/// per residual unit, accumulated across 4 encoder blocks).
const ENCODER_OVERLAP: usize = DAC_HOP_LENGTH * 4;

/// Latency modes → chunk sizes (in frames and samples).
#[derive(Clone, Copy, Debug)]
pub enum ChunkMode {
    /// 1 frame per chunk (minimum latency, more boundary artifacts).
    Strict,
    /// 4 frames per chunk (~46 ms at 86 Hz).
    Balanced,
    /// 8 frames per chunk (~93 ms at 86 Hz).
    Quality,
}

impl ChunkMode {
    pub fn frames_per_chunk(&self) -> usize {
        match self {
            Self::Strict => 1,
            Self::Balanced => 4,
            Self::Quality => 8,
        }
    }

    pub fn samples_per_chunk(&self) -> usize {
        self.frames_per_chunk() * DAC_HOP_LENGTH
    }
}

/// Streaming DAC encoder: processes audio in chunks, caches input tail.
pub struct StreamingCodec {
    codec: DacCodec,
    chunk_mode: ChunkMode,
    /// Input sample tail for overlap (absorbs non-causal receptive field).
    input_tail: VecDeque<f32>,
    /// Cached for decode-side overlap.
    prev_output: Option<Vec<f32>>,
    /// Number of frames produced so far (for trimming).
    total_frames: usize,
}

impl StreamingCodec {
    pub fn new(
        weights_path: &std::path::Path,
        config: &DacConfig,
        chunk_mode: ChunkMode,
        device: Device,
    ) -> Result<Self> {
        let codec = DacCodec::from_file(weights_path, config, device)?;
        Ok(Self {
            codec,
            chunk_mode,
            input_tail: VecDeque::with_capacity(ENCODER_OVERLAP + DAC_HOP_LENGTH),
            prev_output: None,
            total_frames: 0,
        })
    }

    pub fn from_codec(codec: DacCodec, chunk_mode: ChunkMode) -> Self {
        Self {
            codec,
            chunk_mode,
            input_tail: VecDeque::with_capacity(ENCODER_OVERLAP + DAC_HOP_LENGTH),
            prev_output: None,
            total_frames: 0,
        }
    }

    pub fn codec(&self) -> &DacCodec {
        &self.codec
    }

    pub fn codec_mut(&mut self) -> &mut DacCodec {
        &mut self.codec
    }

    pub fn chunk_mode(&self) -> ChunkMode {
        self.chunk_mode
    }

    pub fn set_chunk_mode(&mut self, mode: ChunkMode) {
        self.chunk_mode = mode;
    }

    pub fn reset_state(&mut self) {
        self.input_tail.clear();
        self.prev_output = None;
        self.total_frames = 0;
    }

    /// Process one chunk of PCM samples → continuous latent frames.
    ///
    /// `chunk_pcm`: exactly `chunk_mode.samples_per_chunk()` samples at 44.1 kHz.
    /// Returns: latent tensor `[1, latent_dim, frames_per_chunk]`.
    pub fn encode_step(&mut self, chunk_pcm: &[f32]) -> Result<Tensor> {
        let expected = self.chunk_mode.samples_per_chunk();
        if chunk_pcm.len() != expected {
            anyhow::bail!(
                "chunk size mismatch: expected {expected} samples, got {}",
                chunk_pcm.len()
            );
        }

        // Prepend cached tail for receptive-field overlap
        let mut buffer: Vec<f32> = self.input_tail.iter().copied().collect();
        buffer.extend_from_slice(chunk_pcm);

        // Encode
        let latent = self.codec.encode_pcm(&buffer)?;

        // Update tail: keep last ENCODER_OVERLAP samples
        let tail_start = buffer.len().saturating_sub(ENCODER_OVERLAP);
        self.input_tail.clear();
        self.input_tail.extend(buffer[tail_start..].iter().copied());

        // Trim to only newly produced frames
        let n_new = self.chunk_mode.frames_per_chunk();
        let total_frames = latent.dim(2)?;
        let start = total_frames.saturating_sub(n_new);
        let new_latent = latent.narrow(2, start, n_new)?;
        self.total_frames += n_new;

        Ok(new_latent)
    }

    /// Decode latent frames → PCM samples (handles output overlap-add).
    ///
    /// `latent`: `[1, latent_dim, frames]`
    /// Returns: PCM samples at 44.1 kHz.
    pub fn decode_step(&mut self, latent: &Tensor) -> Result<Vec<f32>> {
        let pcm = self.codec.decode_to_pcm(latent)?;

        // Simple overlap-add: cross-fade the boundary region
        let output = match self.prev_output.take() {
            None => pcm,
            Some(prev) => {
                let overlap_len = prev.len().min(pcm.len()).min(DAC_HOP_LENGTH);
                let mut merged = prev;
                // Cross-fade overlap region
                for i in 0..overlap_len {
                    let w = i as f32 / overlap_len as f32;
                    let idx = merged.len() - overlap_len + i;
                    let faded = merged[idx] * (1.0 - w) + pcm[i] * w;
                    merged[idx] = faded;
                }
                // Append non-overlapping portion
                if pcm.len() > overlap_len {
                    merged.extend_from_slice(&pcm[overlap_len..]);
                }
                merged
            }
        };

        // Keep tail for next overlap
        let tail_len = DAC_HOP_LENGTH.min(output.len());
        self.prev_output = Some(output[output.len() - tail_len..].to_vec());

        // Return only the new (non-cached) portion
        let new_len = self.chunk_mode.frames_per_chunk() * DAC_HOP_LENGTH;
        let start = output.len().saturating_sub(new_len);
        Ok(output[start..].to_vec())
    }

    /// Full encode of an entire audio buffer (non-streaming, for references).
    pub fn encode_full(&self, pcm: &[f32]) -> Result<Tensor> {
        self.codec.encode_pcm(pcm)
    }
}
