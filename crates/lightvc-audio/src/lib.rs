//! LightVC audio I/O library.
//!
//! Cross-platform real-time audio capture/playback via cpal,
//! lock-free ring buffers, and real-time-safe resampling.

pub mod resample;
pub mod ringbuf;
pub mod stream;

pub use resample::Resampler;
pub use ringbuf::AudioRingBuffer;
pub use stream::{DuplexStream, DeviceInfo};
