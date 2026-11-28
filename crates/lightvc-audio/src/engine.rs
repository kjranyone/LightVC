//! High-level audio engine: owns cpal / ASIO streams + ring buffers + fault flags.
//!
//! Supports WASAPI, native Windows COM ASIO (full-duplex on one ASIO driver), and mixed hosts
//! (e.g., WASAPI input + ASIO output or ASIO input + WASAPI output).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::HostPref;

/// Ring-buffer capacity in mono samples (~1.5s at 44.1kHz, ~0.7s at 96kHz).
const RING_CAPACITY: usize = 1 << 16;

#[derive(Clone, Debug, Default)]
pub struct AudioEndpoint {
    pub host: HostPref,
    pub device_index: Option<usize>,
    pub device_name: Option<String>,
}

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
    disconnected: AtomicBool,
    overrun: AtomicU64,
    underrun: AtomicU64,
}

/// Owns the active audio streams (cpal and/or ASIO) and exposes fault telemetry.
pub struct AudioEngine {
    capture_stream: Option<Stream>,
    playback_stream: Option<Stream>,
    #[cfg(all(windows, target_arch = "x86_64"))]
    _asio_session: Option<crate::asio_win::AsioSession>,
    flags: Arc<FaultFlags>,
    #[cfg(all(windows, target_arch = "x86_64"))]
    asio_flags: Option<Arc<crate::asio_win::AsioFaultFlags>>,
    pub capture_sample_rate: u32,
    pub playback_sample_rate: u32,
    pub capture_buffer_frames: u32,
    pub playback_buffer_frames: u32,
    pub host_label: String,
}

fn pick_cpal_input(host: &cpal::Host, ep: &AudioEndpoint) -> Result<Device> {
    let devs: Vec<Device> = host
        .input_devices()?
        .filter(|d| d.default_input_config().is_ok())
        .collect();
    if let Some(ref name) = ep.device_name
        && let Some(d) = devs
            .iter()
            .find(|d| d.description().map(|x| x.name() == name).unwrap_or(false))
    {
        return Ok(d.clone());
    }
    if let Some(idx) = ep.device_index
        && let Some(d) = devs.get(idx)
    {
        return Ok(d.clone());
    }
    if ep.device_index.is_some() || ep.device_name.is_some() {
        bail!("指定された入力デバイスが見つかりません");
    }
    host.default_input_device()
        .ok_or_else(|| anyhow!("No default input device found"))
}

fn pick_cpal_output(host: &cpal::Host, ep: &AudioEndpoint) -> Result<Device> {
    let devs: Vec<Device> = host
        .output_devices()?
        .filter(|d| d.default_output_config().is_ok())
        .collect();
    if let Some(ref name) = ep.device_name
        && let Some(d) = devs
            .iter()
            .find(|d| d.description().map(|x| x.name() == name).unwrap_or(false))
    {
        return Ok(d.clone());
    }
    if let Some(idx) = ep.device_index
        && let Some(d) = devs.get(idx)
    {
        return Ok(d.clone());
    }
    if ep.device_index.is_some() || ep.device_name.is_some() {
        bail!("指定された出力デバイスが見つかりません");
    }
    host.default_output_device()
        .ok_or_else(|| anyhow!("No default output device found"))
}

#[cfg(all(windows, target_arch = "x86_64"))]
fn pick_asio_driver_name(ep: &AudioEndpoint, is_input: bool) -> Result<String> {
    let list = if is_input {
        crate::DuplexStream::list_inputs(HostPref::Asio)?
    } else {
        crate::DuplexStream::list_outputs(HostPref::Asio)?
    };
    if list.is_empty() {
        bail!(
            "使える ASIO {}デバイスがありません",
            if is_input { "入力" } else { "出力" }
        );
    }
    if let Some(ref name) = ep.device_name
        && let Some(d) = list.iter().find(|d| d.name.eq_ignore_ascii_case(name))
    {
        return Ok(d.name.clone());
    }
    if let Some(idx) = ep.device_index {
        return list
            .get(idx)
            .map(|d| d.name.clone())
            .ok_or_else(|| anyhow!("ASIO デバイス番号 {idx} が見つかりません"));
    }
    Ok(list[0].name.clone())
}

fn build_cpal_input_with_retry(
    dev: &Device,
    requested_sr: u32,
    buffer_frames: u32,
    flags: Arc<FaultFlags>,
) -> Result<(Stream, Consumer<f32>, u32, u32)> {
    let def = dev.default_input_config()?;
    let ch = def.channels();
    let def_sr = def.sample_rate();
    let pref_buf = if buffer_frames == 0 {
        cpal::BufferSize::Default
    } else {
        cpal::BufferSize::Fixed(buffer_frames)
    };
    let mut attempts = vec![(requested_sr, pref_buf)];
    if def_sr != requested_sr {
        attempts.push((def_sr, pref_buf));
    }
    if buffer_frames != 0 {
        attempts.push((def_sr, cpal::BufferSize::Default));
    }

    let mut last_err = None;
    for (sr, bsz) in attempts {
        let (mut cap_tx, cap_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
        let cfg = cpal::StreamConfig {
            channels: ch,
            sample_rate: sr,
            buffer_size: bsz,
        };
        let n_in = ch.max(1) as usize;
        let cap_flags = flags.clone();
        let err_flags = flags.clone();
        match dev.build_input_stream(
            cfg,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let frames = data.len() / n_in;
                for i in 0..frames {
                    let sample: f32 =
                        (0..n_in).map(|c| data[i * n_in + c]).sum::<f32>() / n_in as f32;
                    if cap_tx.push(sample).is_err() {
                        cap_flags.overrun.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            move |err| {
                eprintln!("Audio capture error: {err}");
                err_flags.disconnected.store(true, Ordering::Release);
            },
            None,
        ) {
            Ok(s) => {
                if let Err(e) = s.play() {
                    last_err = Some(anyhow::Error::from(e));
                    continue;
                }
                let actual_buf = if buffer_frames > 0 && matches!(bsz, cpal::BufferSize::Fixed(_)) {
                    buffer_frames
                } else {
                    480
                };
                return Ok((s, cap_rx, sr, actual_buf));
            }
            Err(e) => last_err = Some(anyhow::Error::from(e)),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("Failed to open capture stream")))
}

fn build_cpal_output_with_retry(
    dev: &Device,
    requested_sr: u32,
    buffer_frames: u32,
    flags: Arc<FaultFlags>,
) -> Result<(Stream, Producer<f32>, u32, u32)> {
    let def = dev.default_output_config()?;
    let ch = def.channels();
    let def_sr = def.sample_rate();
    let pref_buf = if buffer_frames == 0 {
        cpal::BufferSize::Default
    } else {
        cpal::BufferSize::Fixed(buffer_frames)
    };
    let mut attempts = vec![(requested_sr, pref_buf)];
    if def_sr != requested_sr {
        attempts.push((def_sr, pref_buf));
    }
    if buffer_frames != 0 {
        attempts.push((def_sr, cpal::BufferSize::Default));
    }

    let mut last_err = None;
    for (sr, bsz) in attempts {
        let (pb_tx, mut pb_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
        let cfg = cpal::StreamConfig {
            channels: ch,
            sample_rate: sr,
            buffer_size: bsz,
        };
        let n_out = ch.max(1) as usize;
        let pb_flags = flags.clone();
        let err_flags = flags.clone();
        match dev.build_output_stream(
            cfg,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for frame in data.chunks_mut(n_out) {
                    match pb_rx.pop() {
                        Ok(s) => {
                            for c in frame.iter_mut() {
                                *c = s;
                            }
                        }
                        Err(_) => {
                            for c in frame.iter_mut() {
                                *c = 0.0;
                            }
                            pb_flags.underrun.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            },
            move |err| {
                eprintln!("Audio playback error: {err}");
                err_flags.disconnected.store(true, Ordering::Release);
            },
            None,
        ) {
            Ok(s) => {
                if let Err(e) = s.play() {
                    last_err = Some(anyhow::Error::from(e));
                    continue;
                }
                let actual_buf = if buffer_frames > 0 && matches!(bsz, cpal::BufferSize::Fixed(_)) {
                    buffer_frames
                } else {
                    480
                };
                return Ok((s, pb_tx, sr, actual_buf));
            }
            Err(e) => last_err = Some(anyhow::Error::from(e)),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("Failed to open playback stream")))
}

impl AudioEngine {
    /// Start audio from explicit input and output endpoints (supports WASAPI, ASIO, and mixed hosts).
    pub fn start_endpoints(
        input: AudioEndpoint,
        output: AudioEndpoint,
        requested_sr: u32,
        buffer_frames: u32,
    ) -> Result<(Self, AudioBuffers)> {
        let in_host = crate::host::resolve(input.host)?;
        let out_host = crate::host::resolve(output.host)?;
        let flags = Arc::new(FaultFlags::default());

        #[cfg(all(windows, target_arch = "x86_64"))]
        {
            let buf_opt = (buffer_frames > 0).then_some(buffer_frames);
            match (in_host, out_host) {
                (HostPref::Asio, HostPref::Asio) => {
                    let in_name = pick_asio_driver_name(&input, true)?;
                    let out_name = pick_asio_driver_name(&output, false)?;
                    if !in_name.eq_ignore_ascii_case(&out_name) {
                        bail!(
                            "ASIO は同時に 1 つのドライバしか開けません(入力: {in_name} / 出力: {out_name})。同じ ASIO デバイスを選ぶか、片方を WASAPI にしてください"
                        );
                    }
                    let (cap_tx, cap_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
                    let (pb_tx, pb_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
                    let asio_flags = Arc::new(crate::asio_win::AsioFaultFlags::default());
                    let session = crate::asio_win::AsioSession::start(
                        &in_name,
                        requested_sr,
                        buf_opt,
                        Some(cap_tx),
                        Some(pb_rx),
                        asio_flags.clone(),
                    )?;
                    let sr = session.sample_rate;
                    let bf = session.buffer_frames;
                    return Ok((
                        Self {
                            capture_stream: None,
                            playback_stream: None,
                            _asio_session: Some(session),
                            flags,
                            asio_flags: Some(asio_flags),
                            capture_sample_rate: sr,
                            playback_sample_rate: sr,
                            capture_buffer_frames: bf,
                            playback_buffer_frames: bf,
                            host_label: "asio".to_string(),
                        },
                        AudioBuffers { capture: cap_rx, playback: pb_tx },
                    ));
                }
                (HostPref::Asio, out_h) => {
                    let in_name = pick_asio_driver_name(&input, true)?;
                    let cpal_out_host = crate::host::pick(out_h)?;
                    let out_dev = pick_cpal_output(&cpal_out_host, &output)?;
                    let (cap_tx, cap_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
                    let asio_flags = Arc::new(crate::asio_win::AsioFaultFlags::default());
                    let session = crate::asio_win::AsioSession::start(
                        &in_name,
                        requested_sr,
                        buf_opt,
                        Some(cap_tx),
                        None,
                        asio_flags.clone(),
                    )?;
                    let (pb_stream, pb_tx, pb_sr, pb_buf) =
                        build_cpal_output_with_retry(&out_dev, requested_sr, buffer_frames, flags.clone())?;
                    let cap_sr = session.sample_rate;
                    let cap_buf = session.buffer_frames;
                    return Ok((
                        Self {
                            capture_stream: None,
                            playback_stream: Some(pb_stream),
                            _asio_session: Some(session),
                            flags,
                            asio_flags: Some(asio_flags),
                            capture_sample_rate: cap_sr,
                            playback_sample_rate: pb_sr,
                            capture_buffer_frames: cap_buf,
                            playback_buffer_frames: pb_buf,
                            host_label: format!("asio+{}", out_h.label()),
                        },
                        AudioBuffers { capture: cap_rx, playback: pb_tx },
                    ));
                }
                (in_h, HostPref::Asio) => {
                    let out_name = pick_asio_driver_name(&output, false)?;
                    let cpal_in_host = crate::host::pick(in_h)?;
                    let in_dev = pick_cpal_input(&cpal_in_host, &input)?;
                    let (cap_stream, cap_rx, cap_sr, cap_buf) =
                        build_cpal_input_with_retry(&in_dev, requested_sr, buffer_frames, flags.clone())?;
                    let (pb_tx, pb_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
                    let asio_flags = Arc::new(crate::asio_win::AsioFaultFlags::default());
                    let session = crate::asio_win::AsioSession::start(
                        &out_name,
                        requested_sr,
                        buf_opt,
                        None,
                        Some(pb_rx),
                        asio_flags.clone(),
                    )?;
                    let pb_sr = session.sample_rate;
                    let pb_buf = session.buffer_frames;
                    return Ok((
                        Self {
                            capture_stream: Some(cap_stream),
                            playback_stream: None,
                            _asio_session: Some(session),
                            flags,
                            asio_flags: Some(asio_flags),
                            capture_sample_rate: cap_sr,
                            playback_sample_rate: pb_sr,
                            capture_buffer_frames: cap_buf,
                            playback_buffer_frames: pb_buf,
                            host_label: format!("{}+asio", in_h.label()),
                        },
                        AudioBuffers { capture: cap_rx, playback: pb_tx },
                    ));
                }
                _ => {}
            }
        }

        let cpal_in_host = crate::host::pick(in_host)?;
        let cpal_out_host = crate::host::pick(out_host)?;
        let in_dev = pick_cpal_input(&cpal_in_host, &input)?;
        let out_dev = pick_cpal_output(&cpal_out_host, &output)?;
        let (cap_stream, cap_rx, cap_sr, cap_buf) =
            build_cpal_input_with_retry(&in_dev, requested_sr, buffer_frames, flags.clone())?;
        let (pb_stream, pb_tx, pb_sr, pb_buf) =
            build_cpal_output_with_retry(&out_dev, requested_sr, buffer_frames, flags.clone())?;
        Ok((
            Self {
                capture_stream: Some(cap_stream),
                playback_stream: Some(pb_stream),
                #[cfg(all(windows, target_arch = "x86_64"))]
                _asio_session: None,
                flags,
                #[cfg(all(windows, target_arch = "x86_64"))]
                asio_flags: None,
                capture_sample_rate: cap_sr,
                playback_sample_rate: pb_sr,
                capture_buffer_frames: cap_buf,
                playback_buffer_frames: pb_buf,
                host_label: cpal_in_host.id().name().to_ascii_lowercase(),
            },
            AudioBuffers { capture: cap_rx, playback: pb_tx },
        ))
    }

    /// Start capture-only audio (used for operator microphone recording).
    pub fn start_capture_only(
        input: AudioEndpoint,
        requested_sr: u32,
        buffer_frames: u32,
    ) -> Result<(Self, Consumer<f32>)> {
        let in_host = crate::host::resolve(input.host)?;
        let flags = Arc::new(FaultFlags::default());

        #[cfg(all(windows, target_arch = "x86_64"))]
        if in_host == HostPref::Asio {
            let in_name = pick_asio_driver_name(&input, true)?;
            let (cap_tx, cap_rx) = RingBuffer::<f32>::new(RING_CAPACITY);
            let asio_flags = Arc::new(crate::asio_win::AsioFaultFlags::default());
            let buf_opt = (buffer_frames > 0).then_some(buffer_frames);
            let session = crate::asio_win::AsioSession::start(
                &in_name,
                requested_sr,
                buf_opt,
                Some(cap_tx),
                None,
                asio_flags.clone(),
            )?;
            let sr = session.sample_rate;
            let bf = session.buffer_frames;
            return Ok((
                Self {
                    capture_stream: None,
                    playback_stream: None,
                    _asio_session: Some(session),
                    flags,
                    asio_flags: Some(asio_flags),
                    capture_sample_rate: sr,
                    playback_sample_rate: sr,
                    capture_buffer_frames: bf,
                    playback_buffer_frames: 0,
                    host_label: "asio".to_string(),
                },
                cap_rx,
            ));
        }

        let cpal_in_host = crate::host::pick(in_host)?;
        let in_dev = pick_cpal_input(&cpal_in_host, &input)?;
        let (cap_stream, cap_rx, cap_sr, cap_buf) =
            build_cpal_input_with_retry(&in_dev, requested_sr, buffer_frames, flags.clone())?;
        Ok((
            Self {
                capture_stream: Some(cap_stream),
                playback_stream: None,
                #[cfg(all(windows, target_arch = "x86_64"))]
                _asio_session: None,
                flags,
                #[cfg(all(windows, target_arch = "x86_64"))]
                asio_flags: None,
                capture_sample_rate: cap_sr,
                playback_sample_rate: cap_sr,
                capture_buffer_frames: cap_buf,
                playback_buffer_frames: 0,
                host_label: cpal_in_host.id().name().to_ascii_lowercase(),
            },
            cap_rx,
        ))
    }

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

    /// Start with default devices and an explicit fixed buffer size.
    pub fn start_default_buffered(
        buffer_size: cpal::BufferSize,
    ) -> Result<(Self, AudioBuffers)> {
        let input = default_input()?;
        let output = default_output()?;
        Self::start_buffered(&input, &output, buffer_size)
    }

    /// Start with explicit devices and an explicit fixed buffer size.
    pub fn start_buffered(
        input_device: &Device,
        output_device: &Device,
        buffer_size: cpal::BufferSize,
    ) -> Result<(Self, AudioBuffers)> {
        let in_cfg = input_device.default_input_config()?;
        let out_cfg = output_device.default_output_config()?;
        Self::start_with(
            input_device,
            output_device,
            in_cfg.sample_rate(),
            out_cfg.sample_rate(),
            in_cfg.channels(),
            out_cfg.channels(),
            buffer_size,
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

        let bf = match buffer_size {
            cpal::BufferSize::Fixed(n) => n,
            cpal::BufferSize::Default => 480,
        };

        let engine = Self {
            capture_stream: Some(capture_stream),
            playback_stream: Some(playback_stream),
            #[cfg(all(windows, target_arch = "x86_64"))]
            _asio_session: None,
            flags,
            #[cfg(all(windows, target_arch = "x86_64"))]
            asio_flags: None,
            capture_sample_rate: capture_sr,
            playback_sample_rate: playback_sr,
            capture_buffer_frames: bf,
            playback_buffer_frames: bf,
            host_label: "wasapi".to_string(),
        };
        Ok((engine, AudioBuffers { capture, playback }))
    }

    pub fn is_disconnected(&self) -> bool {
        let cpal_disc = self.flags.disconnected.load(Ordering::Acquire);
        #[cfg(all(windows, target_arch = "x86_64"))]
        let asio_disc = self
            .asio_flags
            .as_ref()
            .is_some_and(|f| f.disconnected.load(Ordering::Acquire));
        #[cfg(not(all(windows, target_arch = "x86_64")))]
        let asio_disc = false;
        cpal_disc || asio_disc
    }

    pub fn overrun_count(&self) -> u64 {
        let cpal_ov = self.flags.overrun.load(Ordering::Relaxed);
        #[cfg(all(windows, target_arch = "x86_64"))]
        let asio_ov = self
            .asio_flags
            .as_ref()
            .map_or(0, |f| f.overrun.load(Ordering::Relaxed));
        #[cfg(not(all(windows, target_arch = "x86_64")))]
        let asio_ov = 0;
        cpal_ov + asio_ov
    }

    pub fn underrun_count(&self) -> u64 {
        let cpal_un = self.flags.underrun.load(Ordering::Relaxed);
        #[cfg(all(windows, target_arch = "x86_64"))]
        let asio_un = self
            .asio_flags
            .as_ref()
            .map_or(0, |f| f.underrun.load(Ordering::Relaxed));
        #[cfg(not(all(windows, target_arch = "x86_64")))]
        let asio_un = 0;
        cpal_un + asio_un
    }

    pub fn pause(&self) -> Result<()> {
        if let Some(ref s) = self.capture_stream {
            s.pause()?;
        }
        if let Some(ref s) = self.playback_stream {
            s.pause()?;
        }
        Ok(())
    }

    pub fn resume(&self) -> Result<()> {
        if let Some(ref s) = self.capture_stream {
            s.play()?;
        }
        if let Some(ref s) = self.playback_stream {
            s.play()?;
        }
        Ok(())
    }
}

/// 入力の指定(未指定なら既定)が実際に開く装置の表示名。
pub fn input_device_name(input: &AudioEndpoint) -> Result<String> {
    let in_host = crate::host::resolve(input.host)?;
    #[cfg(all(windows, target_arch = "x86_64"))]
    if in_host == HostPref::Asio {
        return pick_asio_driver_name(input, true);
    }
    let dev = pick_cpal_input(&crate::host::pick(in_host)?, input)?;
    Ok(dev.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "入力装置".to_string()))
}

pub fn default_input() -> Result<Device> {
    crate::host::pick(HostPref::Default)?
        .default_input_device()
        .ok_or_else(|| anyhow!("No default input device found"))
}

pub fn default_output() -> Result<Device> {
    crate::host::pick(HostPref::Default)?
        .default_output_device()
        .ok_or_else(|| anyhow!("No default output device found"))
}
