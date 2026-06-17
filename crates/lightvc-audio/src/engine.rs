//! High-level audio engine: owns cpal streams + ring buffers + fault flags.
//!
//! Decouples the cpal stream lifecycle from the inference loop. The inference
//! thread receives an [`AudioBuffers`] (capture consumer + playback producer)
//! and polls [`AudioEngine`] for device-disconnection / overrun status.
//!
//! Implements [05-3] (capture/playback/inference separation) and the
//! overrun + disconnection items of [07-4].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream};
use rtrb::{Consumer, Producer, RingBuffer};

/// Ring-buffer capacity in mono samples (~1.5s at 44.1kHz, ~0.7s at 96kHz).
const RING_CAPACITY: usize = 1 << 16;

/// Endpoints the inference thread reads from / writes to.
pub struct AudioBuffers {
    /// Capture stream → inference (device sample rate, mono).
    pub capture: Consumer<f32>,
    /// Inference → playback stream (device sample rate, mono).
    pub playback: Producer<f32>,
}

/// Fault flags shared between cpal callbacks and the engine owner.
#[derive(Default)]
struct FaultFlags {
    /// Set by the cpal error callback when the device is lost.
    disconnected: AtomicBool,
    /// Incremented when the capture ring buffer is full and a sample is
    /// forcibly evicted to make room ([07-4] overrun handling).
    overrun: AtomicU64,
    /// Incremented when the playback ring buffer underflows (silent output).
    underrun: AtomicU64,
}

/// Owns the cpal duplex streams and exposes fault telemetry.
///
/// Dropping this stops both streams (cpal `Stream` stops on drop).
pub struct AudioEngine {
    capture_stream: Stream,
    playback_stream: Stream,
    flags: Arc<FaultFlags>,
    pub capture_sample_rate: u32,
    pub playback_sample_rate: u32,
}

impl AudioEngine {
    /// Start with default devices, auto-negotiated config.
    pub fn start_default() -> Result<(Self, AudioBuffers)> {
        let input = default_input()?;
        let output = default_output()?;
        Self::start(&input, &output)
    }

    /// Start with explicit devices, auto-negotiated config from defaults.
    pub fn start(input_device: &Device, output_device: &Device) -> Result<(Self, AudioBuffers)> {
        let in_cfg = input_device.default_input_config()?;
        let out_cfg = output_device.default_output_config()?;
        Self::start_with(
            input_device,
            output_device,
            in_cfg.sample_rate(),
            out_cfg.sample_rate(),
            in_cfg.channels(),
            out_cfg.channels(),
            cpal::BufferSize::Default,
        )
    }

    /// Start with full control over stream parameters ([05-2]).
    #[allow(clippy::too_many_arguments)]
    pub fn start_with(
        input_device: &Device,
        output_device: &Device,
        capture_sr: cpal::SampleRate,
        playback_sr: cpal::SampleRate,
        in_channels: u16,
        out_channels: u16,
        buffer_size: cpal::BufferSize,
    ) -> Result<(Self, AudioBuffers)> {
        let (capture_tx, capture) = RingBuffer::<f32>::new(RING_CAPACITY);
        let (playback, playback_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
        let flags = Arc::new(FaultFlags::default());

        let in_config = cpal::StreamConfig {
            channels: in_channels,
            sample_rate: capture_sr,
            buffer_size,
        };
        let out_config = cpal::StreamConfig {
            channels: out_channels,
            sample_rate: playback_sr,
            buffer_size,
        };

        let n_in = in_channels.max(1) as usize;
        let cap_flags = flags.clone();
        let mut cap_tx = capture_tx;
        let capture_stream = input_device.build_input_stream(
            in_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let frames = data.len() / n_in;
                for i in 0..frames {
                    let sample: f32 =
                        (0..n_in).map(|ch| data[i * n_in + ch]).sum::<f32>() / n_in as f32;
                    // [07-4] overrun: if the ring is full the consumer is
                    // behind. Drop the incoming sample (brief gap) rather than
                    // blocking the real-time callback.
                    if cap_tx.push(sample).is_err() {
                        cap_flags.overrun.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            {
                let f = flags.clone();
                move |err| {
                    eprintln!("Audio capture error: {err}");
                    f.disconnected.store(true, Ordering::Release);
                }
            },
            None,
        )?;

        let n_out = out_channels.max(1) as usize;
        let pb_flags = flags.clone();
        let mut pb_rx = playback_rx;
        let playback_stream = output_device.build_output_stream(
            out_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for frame in data.chunks_mut(n_out) {
                    match pb_rx.pop() {
                        Ok(s) => {
                            for ch in frame.iter_mut() {
                                *ch = s;
                            }
                        }
                        Err(_) => {
                            // [07-4] underrun: inference too slow → silence.
                            for ch in frame.iter_mut() {
                                *ch = 0.0;
                            }
                            pb_flags.underrun.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            },
            {
                let f = flags.clone();
                move |err| {
                    eprintln!("Audio playback error: {err}");
                    f.disconnected.store(true, Ordering::Release);
                }
            },
            None,
        )?;

        capture_stream.play()?;
        playback_stream.play()?;

        let engine = Self {
            capture_stream,
            playback_stream,
            flags,
            capture_sample_rate: capture_sr,
            playback_sample_rate: playback_sr,
        };
        Ok((engine, AudioBuffers { capture, playback }))
    }

    /// True if either cpal stream reported an error (e.g. device removed).
    /// The inference loop should treat this as a hard stop ([07-4]).
    pub fn is_disconnected(&self) -> bool {
        self.flags.disconnected.load(Ordering::Acquire)
    }

    /// Number of capture samples forcibly evicted since start ([07-4] overrun).
    pub fn overrun_count(&self) -> u64 {
        self.flags.overrun.load(Ordering::Relaxed)
    }

    /// Number of playback frames emitted as silence since start ([07-4] underrun).
    pub fn underrun_count(&self) -> u64 {
        self.flags.underrun.load(Ordering::Relaxed)
    }

    pub fn pause(&self) -> Result<()> {
        self.capture_stream.pause()?;
        self.playback_stream.pause()?;
        Ok(())
    }

    pub fn resume(&self) -> Result<()> {
        self.capture_stream.play()?;
        self.playback_stream.play()?;
        Ok(())
    }
}

pub fn default_input() -> Result<Device> {
    cpal::default_host()
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No default input device found"))
}

pub fn default_output() -> Result<Device> {
    cpal::default_host()
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("No default output device found"))
}
