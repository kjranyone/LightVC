//! Duplex audio stream management via cpal.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use rtrb::Producer;

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

pub struct DuplexStream {
    capture_stream: Stream,
    playback_stream: Stream,
    pub capture_sample_rate: u32,
    pub playback_sample_rate: u32,
}

impl DuplexStream {
    pub fn list_input_devices() -> Result<Vec<DeviceInfo>> {
        let host = cpal::default_host();
        let devices = host.input_devices()?;
        let mut info = Vec::new();
        for dev in devices {
            let name = dev
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_default();
            if let Ok(cfg) = dev.default_input_config() {
                info.push(DeviceInfo {
                    name,
                    sample_rate: cfg.sample_rate(),
                    channels: cfg.channels(),
                });
            }
        }
        Ok(info)
    }

    pub fn list_output_devices() -> Result<Vec<DeviceInfo>> {
        let host = cpal::default_host();
        let devices = host.output_devices()?;
        let mut info = Vec::new();
        for dev in devices {
            let name = dev
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_default();
            if let Ok(cfg) = dev.default_output_config() {
                info.push(DeviceInfo {
                    name,
                    sample_rate: cfg.sample_rate(),
                    channels: cfg.channels(),
                });
            }
        }
        Ok(info)
    }

    pub fn default_input() -> Result<Device> {
        let host = cpal::default_host();
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device found"))
    }

    pub fn default_output() -> Result<Device> {
        let host = cpal::default_host();
        host.default_output_device()
            .ok_or_else(|| anyhow!("No default output device found"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start(
        input_device: &Device,
        output_device: &Device,
        mut capture_tx: Producer<f32>,
        mut playback_rx: rtrb::Consumer<f32>,
    ) -> Result<Self> {
        let input_cfg = input_device.default_input_config()?;
        let output_cfg = output_device.default_output_config()?;

        let capture_sr = input_cfg.sample_rate();
        let playback_sr = output_cfg.sample_rate();
        let in_channels = input_cfg.channels();
        let out_channels = output_cfg.channels();

        let in_config = StreamConfig {
            channels: in_channels,
            sample_rate: capture_sr,
            buffer_size: cpal::BufferSize::Default,
        };
        let out_config = StreamConfig {
            channels: out_channels,
            sample_rate: playback_sr,
            buffer_size: cpal::BufferSize::Default,
        };

        let n_in = in_channels as usize;
        let capture_stream = input_device.build_input_stream(
            in_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let frame_count = data.len() / n_in.max(1);
                for i in 0..frame_count {
                    let sample: f32 =
                        (0..n_in).map(|ch| data[i * n_in + ch]).sum::<f32>() / n_in as f32;
                    let _ = capture_tx.push(sample);
                }
            },
            |err| eprintln!("Audio capture error: {err}"),
            None,
        )?;

        let n_out = out_channels as usize;
        let playback_stream = output_device.build_output_stream(
            out_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for frame in data.chunks_mut(n_out.max(1)) {
                    let sample = playback_rx.pop().unwrap_or(0.0);
                    for ch in frame.iter_mut() {
                        *ch = sample;
                    }
                }
            },
            |err| eprintln!("Audio playback error: {err}"),
            None,
        )?;

        capture_stream.play()?;
        playback_stream.play()?;

        Ok(Self {
            capture_stream,
            playback_stream,
            capture_sample_rate: capture_sr,
            playback_sample_rate: playback_sr,
        })
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
