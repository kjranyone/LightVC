//! LightVC audio I/O library.
//!
//! Cross-platform real-time audio capture/playback via cpal and native Windows COM ASIO,
//! lock-free ring buffers, and real-time-safe resampling.

#[cfg(all(windows, target_arch = "x86_64"))]
pub mod asio_win;
pub mod engine;
pub mod host;
pub mod resample;
pub mod stream;

pub use engine::{AudioBuffers, AudioEndpoint, AudioEngine};
pub use host::HostPref;
pub use resample::RateConv;
pub use stream::{generic_channel_names, DeviceInfo, DuplexStream};
