//! Full real-time VC pipeline orchestrator.
//!
//! Wires together streaming codec encode → converter → streaming codec decode.

use anyhow::{anyhow, Result};
use candle_core::Tensor;

use crate::{
    codec::{DacCodec, DacConfig},
    converter::{AnyConverter, LatencyMode},
    streaming::{ChunkMode, StreamingCodec},
    DAC_SAMPLE_RATE,
};

/// Minimum reference audio length (1 second) for a reliable speaker embedding.
pub const MIN_REFERENCE_SAMPLES: usize = DAC_SAMPLE_RATE as usize;

/// RMS threshold below which an input chunk is treated as silence and
/// short-circuits the encode/convert/decode path (CPU saving).
pub const SILENCE_RMS_THRESHOLD: f32 = 1e-4;

/// Holds cached target voice information for zero-shot VC.
pub struct TargetVoice {
    /// Reference latent `[1, latent_dim, T_ref]`
    pub ref_latent: Tensor,
}

/// Number of latent frames to retain as converter left-context.
///
/// The converter stacks 4 `CausalResBlock`s with dilations [1, 3, 9]
/// (kernel 7). Because the converter is strictly causal when conditioned
/// on a fixed reference, feeding `[context | new]` and trimming to the
/// last `n_new` frames reproduces the non-chunked output near-exactly.
fn converter_context_frames(mode: LatencyMode) -> usize {
    match mode {
        LatencyMode::Strict => 16,
        LatencyMode::Balanced => 32,
        LatencyMode::Quality => 64,
    }
}

/// The full VC pipeline: encode → convert → decode.
pub struct VcPipeline {
    stream_codec: StreamingCodec,
    converter: AnyConverter,
    target: Option<TargetVoice>,
    mode: LatencyMode,
    /// Cached source-latent left-context for causal conv history.
    src_context: Option<Tensor>,
    /// Velocity scale for flow-matching inference (guidance scale).
    pub velocity_scale: f64,
}

impl VcPipeline {
    /// Create a new pipeline from model weights.
    pub fn new(
        dac_path: &std::path::Path,
        dac_config: &DacConfig,
        converter: AnyConverter,
        mode: LatencyMode,
        device: candle_core::Device,
    ) -> Result<Self> {
        let codec = DacCodec::from_file(dac_path, dac_config, device)?;
        let chunk_mode = match mode {
            LatencyMode::Strict => ChunkMode::Strict,
            LatencyMode::Balanced => ChunkMode::Balanced,
            LatencyMode::Quality => ChunkMode::Quality,
        };
        let stream_codec = StreamingCodec::from_codec(codec, chunk_mode);
        Ok(Self {
            stream_codec,
            converter,
            target: None,
            mode,
            src_context: None,
            velocity_scale: 2.5,
        })
    }

    /// Set the target voice from reference audio (44.1 kHz PCM).
    ///
    /// Returns an error if the reference is shorter than
    /// [`MIN_REFERENCE_SAMPLES`] (1 second), since shorter clips yield
    /// unstable speaker embeddings.
    pub fn set_target(&mut self, reference_pcm: &[f32]) -> Result<()> {
        if reference_pcm.len() < MIN_REFERENCE_SAMPLES {
            return Err(anyhow!(
                "reference too short: {} samples < {} (1s @ 44.1 kHz)",
                reference_pcm.len(),
                MIN_REFERENCE_SAMPLES
            ));
        }
        let ref_latent = self.stream_codec.encode_full(reference_pcm)?;
        self.target = Some(TargetVoice { ref_latent });
        Ok(())
    }

    /// Process one chunk of 44.1 kHz PCM → 44.1 kHz PCM.
    ///
    /// Edge cases handled (ARCHITECTURE §8):
    ///   - No target set → passthrough.
    ///   - Near-silence input (RMS < [`SILENCE_RMS_THRESHOLD`]) → passthrough,
    ///     saves CPU during mute gaps.
    ///   - FRC warmup (empty latent) → empty output (silence).
    ///   - NaN/Inf in decoder output → clamped to [-1, 1], NaN → 0.
    pub fn process_chunk(&mut self, chunk_pcm: &[f32]) -> Result<Vec<f32>> {
        if self.target.is_none() {
            return Ok(chunk_pcm.to_vec());
        }

        // Silence detection: skip the full path on near-zero input.
        if rms(chunk_pcm) < SILENCE_RMS_THRESHOLD {
            return Ok(chunk_pcm.to_vec());
        }

        let latent = self.stream_codec.encode_step(chunk_pcm)?;

        // FRC warmup: encoder returned a 0-frame latent while buffering
        // lookahead. Skip convert/decode and emit silence for this chunk.
        if latent.dim(2)? == 0 {
            return Ok(Vec::new());
        }

        let ref_latent = &self.target.as_ref().unwrap().ref_latent;

        // Prepend converter left-context so causal convs see real history.
        let full_src = match &self.src_context {
            Some(ctx) => Tensor::cat(&[ctx, &latent], 2)?,
            None => latent.clone(),
        };
        let converted = self.converter.convert(&full_src, ref_latent, self.velocity_scale)?;

        // Keep only newly produced frames for the decoder.
        let n_new = latent.dim(2)?;
        let total = converted.dim(2)?;
        let start = total.saturating_sub(n_new);
        let new_converted = converted.narrow(2, start, n_new)?.contiguous()?;

        // Update context: retain last ctx_len frames.
        let ctx_len = converter_context_frames(self.mode);
        let src_total = full_src.dim(2)?;
        let ctx_start = src_total.saturating_sub(ctx_len);
        if ctx_start < src_total {
            self.src_context = Some(
                full_src
                    .narrow(2, ctx_start, src_total - ctx_start)?
                    .contiguous()?,
            );
        }

        let pcm_out = self.stream_codec.decode_step(&new_converted)?;

        // Clamp NaN/Inf to [-1, 1]; NaN → 0. Guards against decoder blow-ups
        // from out-of-distribution latents.
        let safe = pcm_out
            .into_iter()
            .map(|s| if s.is_nan() { 0.0 } else { s.clamp(-1.0, 1.0) })
            .collect();

        Ok(safe)
    }

    /// Get the underlying streaming codec (for direct encode/decode).
    pub fn codec(&self) -> &StreamingCodec {
        &self.stream_codec
    }

    pub fn codec_mut(&mut self) -> &mut StreamingCodec {
        &mut self.stream_codec
    }

    /// Reset all streaming state (call on device change or silence gap).
    pub fn reset(&mut self) {
        self.stream_codec.reset_state();
        self.src_context = None;
    }

    /// Offline whole-audio conversion (no chunking artifacts).
    ///
    /// Encodes the entire source, converts in one pass, decodes.
    /// SOTA-quality path for file processing.
    pub fn process_full(&mut self, source_pcm: &[f32]) -> Result<Vec<f32>> {
        if self.target.is_none() {
            return Ok(source_pcm.to_vec());
        }
        let ref_latent = &self.target.as_ref().unwrap().ref_latent;
        let latent = self.stream_codec.encode_full(source_pcm)?;
        let converted = self.converter.convert(&latent, ref_latent, self.velocity_scale)?;
        let pcm_out = self.stream_codec.codec().decode_to_pcm(&converted)?;
        Ok(pcm_out)
    }

    pub fn has_target(&self) -> bool {
        self.target.is_some()
    }

    /// Chunk size in samples (at 44.1 kHz) for the current mode.
    pub fn chunk_samples(&self) -> usize {
        self.stream_codec.chunk_mode().samples_per_chunk()
    }

    /// Chunk size in milliseconds.
    pub fn chunk_ms(&self) -> f32 {
        let samples = self.chunk_samples() as f32;
        samples / 44_100.0 * 1000.0
    }

    /// Algorithmic input latency contributed by the current chunk mode, in
    /// 44.1 kHz samples. = chunk size + FRC lookahead.
    pub fn algorithmic_latency_samples(&self) -> usize {
        self.stream_codec.algorithmic_latency_samples()
    }

    /// Algorithmic input latency in milliseconds.
    pub fn algorithmic_latency_ms(&self) -> f32 {
        self.algorithmic_latency_samples() as f32 / 44_100.0 * 1000.0
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
