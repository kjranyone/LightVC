//! LightVC core inference library.
//!
//! Provides the DAC codec wrapper, converter model, streaming pipeline,
//! and weight loading for real-time voice conversion.

pub mod codec;
pub mod converter;
pub mod dac_model;
pub mod pipeline;
pub mod streaming;
pub mod weights;

pub use codec::{DacCodec, DacConfig};
pub use converter::{AnyConverter, Converter, ConverterConfig, FlowConverter, LatencyMode};
pub use pipeline::VcPipeline;
pub use streaming::StreamingCodec;

/// Sample rate expected by DAC (44.1 kHz).
pub const DAC_SAMPLE_RATE: u32 = 44_100;

/// DAC hop length (encoder stride product: 2 × 4 × 8 × 8).
pub const DAC_HOP_LENGTH: usize = 512;

/// DAC frame rate at 44.1 kHz: 44100 / 512 ≈ 86.13 Hz.
pub const DAC_FRAME_RATE: f32 = 44_100.0 / DAC_HOP_LENGTH as f32;

/// DAC latent dimensionality.
pub const DAC_LATENT_DIM: usize = 1024;
