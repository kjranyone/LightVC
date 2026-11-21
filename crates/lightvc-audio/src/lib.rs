//! LightVC audio I/O library.
//!
//! Cross-platform real-time audio capture/playback via cpal,
//! lock-free ring buffers, and real-time-safe resampling.

pub mod engine;
pub mod host;
pub mod resample;
pub mod stream;

pub use engine::{AudioBuffers, AudioEngine};
pub use host::HostPref;
pub use resample::RateConv;
pub use stream::{DeviceInfo, DuplexStream};
