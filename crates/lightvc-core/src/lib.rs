//! LightVC core inference library.
//!
//! Provides the DAC codec wrapper, converter model, streaming pipeline,
//! and weight loading for real-time voice conversion.

pub mod b1_pipeline;
pub mod codec;
pub mod converter;
pub mod dac_model;
pub mod eg;
pub mod flow_converter;
pub mod free_resynth;
pub mod free_vocoder;
pub mod mel;
pub mod pipeline;
pub mod ship_front;
pub mod ys1_codec;
pub mod simd;
pub mod soft_rvq;
pub mod streaming;
pub mod utte_adapter;
pub mod v1d;
pub mod v2f_infer;
pub mod vc_stream;
pub mod weights;

pub use b1_pipeline::{B1Offline, B1Streaming, StageTimings};
pub use codec::{DacCodec, DacConfig};
pub use free_resynth::FreeResynth;
pub use converter::{AnyConverter, Converter, ConverterConfig, FlowConverter, LatencyMode};
pub use pipeline::VcPipeline;
pub use soft_rvq::SoftRVQ;
pub use streaming::StreamingCodec;
pub use utte_adapter::{UTTEAdapter, UTTEAdapterConfig};

use anyhow::Result;

pub enum Backend {
    Legacy(VcPipeline),
    B1(B1Streaming),
    /// FreeVocoder resynthesis: mic → mel (Rust) → freeC vocoder → out.
    FreeVoc(FreeResynth),
    /// バ美声変換: mic → front → E → G(f0 シフト) → v2f。ブロック 256 固定。
    Vc(vc_stream::VcStream),
}

impl Backend {
    pub fn process_chunk(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        match self {
            Backend::Legacy(p) => p.process_chunk(pcm),
            Backend::B1(p) => p.process_chunk(pcm),
            Backend::FreeVoc(p) => p.process_chunk(pcm),
            Backend::Vc(p) => {
                // 学習系の慣習: front/E/G は x32768、V 出力は [-1,1]
                anyhow::ensure!(pcm.len() == vc_stream::VcStream::BLOCK,
                                "Vc は 256 サンプル固定");
                let x32: Vec<f32> = pcm.iter().map(|v| v * 32768.0).collect();
                Ok(p.process_block(&x32))
            }
        }
    }

    pub fn chunk_samples(&self) -> usize {
        match self {
            Backend::Legacy(p) => p.chunk_samples(),
            Backend::B1(p) => p.chunk_samples(),
            Backend::FreeVoc(p) => p.chunk_samples(),
            Backend::Vc(_) => vc_stream::VcStream::BLOCK,
        }
    }

    pub fn algorithmic_latency_ms(&self) -> f32 {
        match self {
            Backend::Legacy(p) => p.algorithmic_latency_ms(),
            Backend::B1(p) => {
                let mode = p.chunk_mode();
                mode.algorithmic_latency_samples() as f32 / 44.1
            }
            Backend::FreeVoc(p) => p.algorithmic_latency_ms(),
            // V の OLA 固有遅延 384 サンプル (E/G の CTX は左文脈=履歴で遅延ではない)
            Backend::Vc(_) => 384.0 / 44.1,
        }
    }

    pub fn set_velocity_scale(&mut self, scale: f64) {
        if let Backend::Legacy(p) = self {
            p.set_velocity_scale(scale);
        }
    }

    pub fn set_mode(&mut self, mode: converter::LatencyMode) {
        if let Backend::Legacy(p) = self {
            p.set_mode(mode);
        }
    }

    pub fn is_b1(&self) -> bool {
        matches!(self, Backend::B1(_))
    }

    pub fn reset(&mut self) {
        match self {
            Backend::Legacy(p) => p.reset(),
            Backend::B1(p) => p.reset(),
            Backend::FreeVoc(p) => p.reset(),
            Backend::Vc(p) => p.reset(),
        }
    }

    pub fn set_target(&mut self, pcm: &[f32]) -> Result<()> {
        match self {
            Backend::Legacy(p) => p.set_target(pcm),
            Backend::B1(_) => Ok(()),
            // Resynthesis has no reference target (input mel === output mel).
            Backend::FreeVoc(_) => Ok(()),
            // Vc の目標は cartridge (G 重み) に焼き込み済み
            Backend::Vc(_) => Ok(()),
        }
    }

    pub fn process_full(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        match self {
            Backend::Legacy(p) => p.process_full(pcm),
            Backend::B1(p) => p.process_full(pcm),
            Backend::FreeVoc(p) => p.process_full(pcm),
            Backend::Vc(p) => {
                let n = (pcm.len() / vc_stream::VcStream::BLOCK)
                    * vc_stream::VcStream::BLOCK;
                let mut y = Vec::with_capacity(n);
                for b in pcm[..n].chunks(vc_stream::VcStream::BLOCK) {
                    let x32: Vec<f32> = b.iter().map(|v| v * 32768.0).collect();
                    y.extend(p.process_block(&x32));
                }
                Ok(y)
            }
        }
    }

    pub fn mode(&self) -> converter::LatencyMode {
        match self {
            Backend::Legacy(p) => p.mode(),
            Backend::B1(_) => converter::LatencyMode::Balanced,
            Backend::FreeVoc(_) => converter::LatencyMode::Balanced,
            Backend::Vc(_) => converter::LatencyMode::Balanced,
        }
    }

    pub fn set_prosody(&mut self, mode: converter::ProsodyMode, blend: f64) {
        if let Backend::Legacy(p) = self {
            p.set_prosody(mode, blend);
        }
    }

    pub fn codec_device(&self) -> &candle_core::Device {
        match self {
            Backend::Legacy(p) => p.codec().codec().device(),
            Backend::B1(p) => p.codec().device(),
            Backend::FreeVoc(p) => p.device(),
            Backend::Vc(_) => {
                static CPU: std::sync::OnceLock<candle_core::Device> =
                    std::sync::OnceLock::new();
                CPU.get_or_init(|| candle_core::Device::Cpu)
            }
        }
    }
}

/// Sample rate expected by DAC (44.1 kHz).
pub const DAC_SAMPLE_RATE: u32 = 44_100;

/// DAC hop length (encoder stride product: 2 × 4 × 8 × 8).
pub const DAC_HOP_LENGTH: usize = 512;

/// DAC frame rate at 44.1 kHz: 44100 / 512 ≈ 86.13 Hz.
pub const DAC_FRAME_RATE: f32 = 44_100.0 / DAC_HOP_LENGTH as f32;

/// DAC latent dimensionality.
pub const DAC_LATENT_DIM: usize = 1024;
