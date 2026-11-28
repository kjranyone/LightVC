//! Duplex audio stream management via cpal and Windows COM ASIO.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use rtrb::Producer;

use crate::HostPref;

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    /// チャンネルの名前(ASIO はドライバの名前・ほかは 2 ch なら L / R、それ以外は番号)。
    pub channel_names: Vec<String>,
    pub host: HostPref,
    pub host_index: usize,
}

/// ドライバが名前を返さない装置のチャンネル名。
pub fn generic_channel_names(channels: u16) -> Vec<String> {
    if channels == 2 { vec!["L".into(), "R".into()] } else { (1..=channels).map(|c| c.to_string()).collect() }
}

pub struct DuplexStream {
    capture_stream: Stream,
    playback_stream: Stream,
    pub capture_sample_rate: u32,
    pub playback_sample_rate: u32,
}

impl DuplexStream {
    pub fn list_input_devices() -> Result<Vec<DeviceInfo>> {
        Self::list_inputs(HostPref::Default)
    }

    pub fn list_inputs(pref: HostPref) -> Result<Vec<DeviceInfo>> {
        match crate::host::resolve(pref)? {
            HostPref::Asio => {
                #[cfg(all(windows, target_arch = "x86_64"))]
                {
                    Ok(crate::asio_win::list_devices()
                        .into_iter()
                        .filter(|d| d.channels_in > 0)
                        .enumerate()
                        .map(|(i, d)| DeviceInfo {
                            name: d.name,
                            sample_rate: d.sample_rate,
                            channels: d.channels_in,
                            channel_names: d.in_channel_names,
                            host: HostPref::Asio,
                            host_index: i,
                        })
                        .collect())
                }
                #[cfg(all(windows, not(target_arch = "x86_64"), feature = "asio"))]
                {
                    let host = crate::host::pick(HostPref::Asio)?;
                    return Self::list_input_devices_with_host(&host, HostPref::Asio);
                }
                #[cfg(not(any(all(windows, target_arch = "x86_64"), all(windows, feature = "asio"))))]
                {
                    Ok(Vec::new())
                }
            }
            resolved => {
                let host = crate::host::pick(resolved)?;
                Self::list_input_devices_with_host(&host, resolved)
            }
        }
    }

    pub fn list_input_devices_on(host: &cpal::Host) -> Result<Vec<DeviceInfo>> {
        let h = if cfg!(windows) { HostPref::Wasapi } else { HostPref::Default };
        Self::list_input_devices_with_host(host, h)
    }

    fn list_input_devices_with_host(host: &cpal::Host, pref: HostPref) -> Result<Vec<DeviceInfo>> {
        let devices = host.input_devices()?;
        let mut info = Vec::new();
        for dev in devices {
            let name = dev
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_default();
            if let Ok(cfg) = dev.default_input_config() {
                let idx = info.len();
                info.push(DeviceInfo {
                    name,
                    sample_rate: cfg.sample_rate(),
                    channels: cfg.channels(),
                    channel_names: generic_channel_names(cfg.channels()),
                    host: pref,
                    host_index: idx,
                });
            }
        }
        Ok(info)
    }

    pub fn list_output_devices() -> Result<Vec<DeviceInfo>> {
        Self::list_outputs(HostPref::Default)
    }

    pub fn list_outputs(pref: HostPref) -> Result<Vec<DeviceInfo>> {
        match crate::host::resolve(pref)? {
            HostPref::Asio => {
                #[cfg(all(windows, target_arch = "x86_64"))]
                {
                    Ok(crate::asio_win::list_devices()
                        .into_iter()
                        .filter(|d| d.channels_out > 0)
                        .enumerate()
                        .map(|(i, d)| DeviceInfo {
                            name: d.name,
                            sample_rate: d.sample_rate,
                            channels: d.channels_out,
                            channel_names: d.out_channel_names,
                            host: HostPref::Asio,
                            host_index: i,
                        })
                        .collect())
                }
                #[cfg(all(windows, not(target_arch = "x86_64"), feature = "asio"))]
                {
                    let host = crate::host::pick(HostPref::Asio)?;
                    return Self::list_output_devices_with_host(&host, HostPref::Asio);
                }
                #[cfg(not(any(all(windows, target_arch = "x86_64"), all(windows, feature = "asio"))))]
                {
                    Ok(Vec::new())
                }
            }
            resolved => {
                let host = crate::host::pick(resolved)?;
                Self::list_output_devices_with_host(&host, resolved)
            }
        }
    }

    pub fn list_output_devices_on(host: &cpal::Host) -> Result<Vec<DeviceInfo>> {
        let h = if cfg!(windows) { HostPref::Wasapi } else { HostPref::Default };
        Self::list_output_devices_with_host(host, h)
    }

    fn list_output_devices_with_host(host: &cpal::Host, pref: HostPref) -> Result<Vec<DeviceInfo>> {
        let devices = host.output_devices()?;
        let mut info = Vec::new();
        for dev in devices {
            let name = dev
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_default();
            if let Ok(cfg) = dev.default_output_config() {
                let idx = info.len();
                info.push(DeviceInfo {
                    name,
                    sample_rate: cfg.sample_rate(),
                    channels: cfg.channels(),
                    channel_names: generic_channel_names(cfg.channels()),
                    host: pref,
                    host_index: idx,
                });
            }
        }
        Ok(info)
    }

    pub fn default_input() -> Result<Device> {
        let host = crate::host::pick(HostPref::Default)?;
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device found"))
    }

    pub fn default_output() -> Result<Device> {
        let host = crate::host::pick(HostPref::Default)?;
        host.default_output_device()
            .ok_or_else(|| anyhow!("No default output device found"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start(
        input_device: &Device,
        output_device: &Device,
        capture_tx: Producer<f32>,
        playback_rx: rtrb::Consumer<f32>,
    ) -> Result<Self> {
        let input_cfg = input_device.default_input_config()?;
        let output_cfg = output_device.default_output_config()?;

        let capture_sr = input_cfg.sample_rate();
        let playback_sr = output_cfg.sample_rate();
        let in_channels = input_cfg.channels();
        let out_channels = output_cfg.channels();

        Self::start_with(
            input_device,
            output_device,
            capture_sr,
            playback_sr,
            in_channels,
            out_channels,
            cpal::BufferSize::Default,
            capture_tx,
            playback_rx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_with(
        input_device: &Device,
        output_device: &Device,
        capture_sr: cpal::SampleRate,
        playback_sr: cpal::SampleRate,
        in_channels: u16,
        out_channels: u16,
        buffer_size: cpal::BufferSize,
        mut capture_tx: Producer<f32>,
        mut playback_rx: rtrb::Consumer<f32>,
    ) -> Result<Self> {
        let in_config = StreamConfig {
            channels: in_channels,
            sample_rate: capture_sr,
            buffer_size,
        };
        let out_config = StreamConfig {
            channels: out_channels,
            sample_rate: playback_sr,
            buffer_size,
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

    pub fn supported_input_configs(
        device: &Device,
    ) -> Result<Vec<cpal::SupportedStreamConfigRange>> {
        Ok(device.supported_input_configs()?.collect())
    }

    pub fn supported_output_configs(
        device: &Device,
    ) -> Result<Vec<cpal::SupportedStreamConfigRange>> {
        Ok(device.supported_output_configs()?.collect())
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
