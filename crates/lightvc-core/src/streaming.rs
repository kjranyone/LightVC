//! Streaming codec wrappers with FRC lookahead and overlap-add.
//!
//! DAC is non-causal (symmetric padding). For real-time streaming we use
//! Future-Receptive Chunking (FRC, cf. MeanVC2): buffer `lookahead` samples
//! of *future* audio before encoding the current chunk, so the encoder's
//! symmetric receptive field is satisfied on both sides. This eliminates
//! chunk-boundary artifacts that arise from feeding a symmetric-padded conv
//! a left-only-context buffer ([02-1], [02-4], [08-1]).
//!
//! Per-layer conv-state caching ([02-2]) is intentionally NOT implemented:
//! once both the left overlap (`ENCODER_OVERLAP`) and the right lookahead
//! cover the encoder's effective receptive field, re-encoding
//! `[tail | current | lookahead]` each call reproduces the full-batch result
//! to within float precision. Caching intermediate activations is a pure
//! performance optimization and is left for a future pass.

use std::collections::VecDeque;

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::{
    codec::{DacCodec, DacConfig},
    DAC_HOP_LENGTH, DAC_LATENT_DIM,
};

/// Number of past samples retained as left overlap to absorb the encoder's
/// non-causal receptive field on the left side. Combined with the right-side
/// lookahead (`ChunkMode::lookahead_samples`), this bounds the streaming vs
/// full-batch discrepancy.
const ENCODER_OVERLAP: usize = DAC_HOP_LENGTH * 4;

/// Latency modes → chunk sizes and FRC lookahead (in samples at 44.1 kHz).
#[derive(Clone, Copy, Debug)]
pub enum ChunkMode {
    /// 1 frame per chunk, no lookahead. Boundary artifacts accepted.
    Strict,
    /// 4 frames per chunk (~46 ms), ~46 ms lookahead.
    Balanced,
    /// 8 frames per chunk (~93 ms), ~93 ms lookahead.
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

    /// Future-Receptive Chunking lookahead in samples at 44.1 kHz.
    ///
    /// This many *future* samples are buffered before encoding the current
    /// chunk, so the DAC encoder's symmetric-padded convs see real context
    /// on the right edge instead of zeros.
    ///
    /// Returns a multiple of `DAC_HOP_LENGTH` so that latent-frame indexing
    /// stays exact. The value also equals the algorithmic latency that the
    /// mode adds on top of the chunk size.
    pub fn lookahead_samples(&self) -> usize {
        match self {
            Self::Strict => 0,
            Self::Balanced => DAC_HOP_LENGTH * 4,
            Self::Quality => DAC_HOP_LENGTH * 8,
        }
    }

    /// Total algorithmic input latency contributed by this mode, in samples.
    /// = chunk size + lookahead. Useful for reporting end-to-end latency.
    pub fn algorithmic_latency_samples(&self) -> usize {
        self.samples_per_chunk() + self.lookahead_samples()
    }
}

/// Streaming DAC codec: processes audio in chunks with FRC lookahead and
/// decode-side overlap-add.
pub struct StreamingCodec {
    codec: DacCodec,
    chunk_mode: ChunkMode,
    /// Past samples retained as left overlap for the encoder's receptive field.
    input_tail: VecDeque<f32>,
    /// Buffered input awaiting encode. In steady state this holds
    /// `lookahead_samples()` worth of *future* audio that follows the chunk
    /// currently being encoded.
    pending: VecDeque<f32>,
    /// Cached for decode-side overlap-add.
    prev_output: Option<Vec<f32>>,
    /// Number of frames produced so far (for trimming and warmup gating).
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
            pending: VecDeque::with_capacity(
                chunk_mode.samples_per_chunk() + chunk_mode.lookahead_samples() + DAC_HOP_LENGTH,
            ),
            prev_output: None,
            total_frames: 0,
        })
    }

    pub fn from_codec(codec: DacCodec, chunk_mode: ChunkMode) -> Self {
        Self {
            codec,
            chunk_mode,
            input_tail: VecDeque::with_capacity(ENCODER_OVERLAP + DAC_HOP_LENGTH),
            pending: VecDeque::with_capacity(
                chunk_mode.samples_per_chunk() + chunk_mode.lookahead_samples() + DAC_HOP_LENGTH,
            ),
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
        // Different lookahead means the pending buffer semantics change;
        // reset all streaming state to avoid mismatched frame indexing.
        if mode.lookahead_samples() != self.chunk_mode.lookahead_samples() {
            self.reset_state();
        }
        self.chunk_mode = mode;
    }

    pub fn reset_state(&mut self) {
        self.input_tail.clear();
        self.pending.clear();
        self.prev_output = None;
        self.total_frames = 0;
    }

    /// Algorithmic input latency (chunk + FRC lookahead), in 44.1 kHz samples.
    pub fn algorithmic_latency_samples(&self) -> usize {
        self.chunk_mode.algorithmic_latency_samples()
    }

    fn empty_latent(&self) -> Result<Tensor> {
        Tensor::zeros(
            (1, DAC_LATENT_DIM, 0),
            candle_core::DType::F32,
            self.codec.device(),
        )
        .map_err(Into::into)
    }

    /// Process one chunk of PCM samples → continuous latent frames.
    ///
    /// `chunk_pcm`: exactly `chunk_mode.samples_per_chunk()` samples at 44.1 kHz.
    /// Returns: latent tensor `[1, latent_dim, frames_per_chunk]`.
    ///
    /// During FRC warmup (the first `ceil(lookahead / chunk_size)` calls) the
    /// encoder has not yet accumulated enough future context, so a 0-frame
    /// tensor is returned and the input is buffered. Strict mode (lookahead=0)
    /// has no warmup. Callers should tolerate empty output (silence) for the
    /// warmup period; downstream `decode_step` on an empty latent yields empty
    /// PCM.
    pub fn encode_step(&mut self, chunk_pcm: &[f32]) -> Result<Tensor> {
        let chunk_sz = self.chunk_mode.samples_per_chunk();
        if chunk_pcm.len() != chunk_sz {
            anyhow::bail!(
                "chunk size mismatch: expected {chunk_sz} samples, got {}",
                chunk_pcm.len()
            );
        }

        let lookahead = self.chunk_mode.lookahead_samples();
        self.pending.extend(chunk_pcm.iter().copied());

        // Warmup gate: before producing any output, require chunk_sz + lookahead
        // samples so the first emitted chunk has real future context. After the
        // first output, steady state maintains >= chunk_sz in `pending`.
        let ready = if self.total_frames == 0 {
            self.pending.len() >= chunk_sz + lookahead
        } else {
            self.pending.len() >= chunk_sz
        };
        if !ready {
            return self.empty_latent();
        }

        let tail_len = self.input_tail.len();

        // Build buffer = [input_tail | current_chunk | lookahead].
        // Current chunk is drained from `pending`; lookahead is read without
        // draining (those samples belong to a future chunk) and zero-padded
        // if insufficient (stream tail).
        let current: Vec<f32> = self.pending.drain(..chunk_sz).collect();
        let real_future = self.pending.len().min(lookahead);

        let mut buf: Vec<f32> = Vec::with_capacity(tail_len + chunk_sz + lookahead);
        buf.extend(self.input_tail.iter().copied());
        buf.extend_from_slice(&current);
        buf.extend(self.pending.iter().take(real_future).copied());
        if real_future < lookahead {
            buf.resize(buf.len() + (lookahead - real_future), 0.0);
        }

        // Encode the assembled buffer.
        let latent = self.codec.encode_pcm(&buf)?;

        // Update input_tail: last ENCODER_OVERLAP samples of [tail | current].
        // `buf` is [tail | current | lookahead], so [tail | current] occupies
        // indices 0..combined_len.
        let combined_len = tail_len + chunk_sz;
        let tail_start = combined_len.saturating_sub(ENCODER_OVERLAP);
        self.input_tail.clear();
        self.input_tail
            .extend(buf[tail_start..combined_len].iter().copied());

        // Extract only the latent frames corresponding to the current chunk.
        // Frame i in the latent maps to input sample i * DAC_HOP_LENGTH, so the
        // current chunk (which starts at sample `tail_len`) occupies frames
        // [tail_len / hop, (tail_len + chunk_sz) / hop).
        let n_new = self.chunk_mode.frames_per_chunk();
        let start_frame = tail_len / DAC_HOP_LENGTH;
        let new_latent = latent.narrow(2, start_frame, n_new)?;
        self.total_frames += n_new;

        Ok(new_latent)
    }

    /// Decode latent frames → PCM samples (handles output overlap-add).
    ///
    /// `latent`: `[1, latent_dim, frames]`. Returns PCM samples at 44.1 kHz.
    /// An empty latent (`frames == 0`) returns an empty Vec (FRC warmup).
    pub fn decode_step(&mut self, latent: &Tensor) -> Result<Vec<f32>> {
        let frames = latent.dim(2)?;
        if frames == 0 {
            return Ok(Vec::new());
        }

        let pcm = self.codec.decode_to_pcm(latent)?;

        // Overlap-add: cross-fade the boundary region with the previous chunk.
        let output = match self.prev_output.take() {
            None => pcm,
            Some(prev) => {
                let overlap_len = prev.len().min(pcm.len()).min(DAC_HOP_LENGTH);
                let mut merged = prev;
                for i in 0..overlap_len {
                    let w = i as f32 / overlap_len as f32;
                    let idx = merged.len() - overlap_len + i;
                    let faded = merged[idx] * (1.0 - w) + pcm[i] * w;
                    merged[idx] = faded;
                }
                if pcm.len() > overlap_len {
                    merged.extend_from_slice(&pcm[overlap_len..]);
                }
                merged
            }
        };

        // Keep tail for next overlap.
        let tail_len = DAC_HOP_LENGTH.min(output.len());
        self.prev_output = Some(output[output.len() - tail_len..].to_vec());

        // Return only the newly produced portion.
        let new_len = frames * DAC_HOP_LENGTH;
        let start = output.len().saturating_sub(new_len);
        Ok(output[start..].to_vec())
    }

    /// Full encode of an entire audio buffer (non-streaming, for references).
    pub fn encode_full(&self, pcm: &[f32]) -> Result<Tensor> {
        self.codec.encode_pcm(pcm)
    }
}
