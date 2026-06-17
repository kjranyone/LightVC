//! Full real-time VC pipeline orchestrator.
//!
//! Wires together streaming codec encode → converter → streaming codec decode.

use anyhow::Result;
use candle_core::Tensor;

use crate::{
    codec::{DacCodec, DacConfig},
    converter::{AnyConverter, LatencyMode},
    streaming::{ChunkMode, StreamingCodec},
};

/// Holds cached target voice information for zero-shot VC.
pub struct TargetVoice {
    /// Reference latent `[1, latent_dim, T_ref]`
    pub ref_latent: Tensor,
}

/// The full VC pipeline: encode → convert → decode.
pub struct VcPipeline {
    stream_codec: StreamingCodec,
    converter: AnyConverter,
    target: Option<TargetVoice>,
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
        })
    }

    /// Set the target voice from reference audio (44.1 kHz PCM).
    pub fn set_target(&mut self, reference_pcm: &[f32]) -> Result<()> {
        let ref_latent = self.stream_codec.encode_full(reference_pcm)?;
        self.target = Some(TargetVoice { ref_latent });
        Ok(())
    }

    /// Process one chunk of 44.1 kHz PCM → 44.1 kHz PCM.
    ///
    /// If no target voice is set, returns the input unchanged (passthrough).
    /// Returns an empty `Vec` during FRC warmup (the first chunk(s) before
    /// enough lookahead has accumulated); callers should treat this as
    /// silence.
    pub fn process_chunk(&mut self, chunk_pcm: &[f32]) -> Result<Vec<f32>> {
        if self.target.is_none() {
            return Ok(chunk_pcm.to_vec());
        }

        let latent = self.stream_codec.encode_step(chunk_pcm)?;

        // FRC warmup: encoder returned a 0-frame latent while buffering
        // lookahead. Skip convert/decode and emit silence for this chunk.
        if latent.dim(2)? == 0 {
            return Ok(Vec::new());
        }

        let ref_latent = &self.target.as_ref().unwrap().ref_latent;
        let converted = self.converter.convert(&latent, ref_latent)?;

        let pcm_out = self.stream_codec.decode_step(&converted)?;

        Ok(pcm_out)
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
}
