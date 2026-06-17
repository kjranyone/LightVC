//! LightVC audio I/O library.
//!
//! Cross-platform real-time audio capture/playback via cpal,
//! lock-free ring buffers, and real-time-safe resampling.

pub mod engine;
pub mod resample;
pub mod ringbuf;
pub mod stream;

pub use engine::{AudioBuffers, AudioEngine};
pub use resample::Resampler;
pub use ringbuf::AudioRingBuffer;
pub use stream::{DeviceInfo, DuplexStream};
