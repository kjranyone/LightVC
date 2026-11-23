//! Simple WAV playback via rodio-free approach.
//! Uses a background thread with cpal output (WASAPI on Windows so it never collides with ASIO).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Audio player: plays a WAV buffer once.
pub struct AudioPlayer {
    playing: Arc<AtomicBool>,
    _thread: Option<std::thread::JoinHandle<()>>,
}

impl AudioPlayer {
    /// Play a mono f32 buffer at 44100 Hz.
    pub fn play(samples: Vec<f32>) -> Result<Self> {
        let playing = Arc::new(AtomicBool::new(true));
        let playing_clone = playing.clone();

        let thread = std::thread::spawn(move || {
            let host = match lightvc_audio::host::pick(lightvc_audio::HostPref::Wasapi) {
                Ok(h) => h,
                Err(_) => return,
            };
            let Some(device) = host.default_output_device() else {
                return;
            };
            let def = device.default_output_config().ok();
            let target_sr = def.as_ref().map_or(44_100, |c| c.sample_rate());
            let ch = def.as_ref().map_or(1, |c| c.channels().max(1));
            let resampled = if target_sr == 44_100 {
                samples
            } else {
                resample_linear_sr(&samples, 44_100, target_sr)
            };
            let config = cpal::StreamConfig {
                channels: ch,
                sample_rate: target_sr,
                buffer_size: cpal::BufferSize::Default,
            };

            let idx = Arc::new(Mutex::new(0usize));
            let idx_clone = idx.clone();
            let samples_arc = Arc::new(resampled);
            let samples_for_closure = samples_arc.clone();
            let n_ch = ch as usize;

            let stream = device.build_output_stream(
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut i = idx_clone.lock().unwrap();
                    for frame in data.chunks_mut(n_ch) {
                        let s = if *i < samples_for_closure.len() {
                            let v = samples_for_closure[*i];
                            *i += 1;
                            v
                        } else {
                            0.0
                        };
                        for slot in frame.iter_mut() {
                            *slot = s;
                        }
                    }
                },
                |err| eprintln!("Playback error: {err}"),
                None,
            );

            if let Ok(s) = stream {
                let _ = s.play();
                while playing_clone.load(Ordering::Relaxed) {
                    let pos = idx.lock().unwrap();
                    if *pos >= samples_arc.len() {
                        break;
                    }
                    drop(pos);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
            playing_clone.store(false, Ordering::Relaxed);
        });

        Ok(Self {
            playing,
            _thread: Some(thread),
        })
    }

    pub fn stop(&self) {
        self.playing.store(false, Ordering::Relaxed);
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }
}

/// Load a WAV file as mono f32 at its native sample rate.
pub fn load_wav_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        _ => {
            let max_val = match spec.bits_per_sample {
                16 => 32768.0f32,
                24 => 8388608.0,
                32 => 2147483648.0,
                _ => 32768.0,
            };
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    Ok((mono, sr))
}

/// Save mono f32 as 32-bit float WAV.
pub fn save_wav_mono(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    Ok(())
}

/// Resample to 44100 Hz using linear interpolation (simple, sufficient for UI).
pub fn resample_linear(input: &[f32], from_sr: u32) -> Vec<f32> {
    resample_linear_sr(input, from_sr, 44_100)
}

fn resample_linear_sr(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_sr as f32 / from_sr as f32;
    let new_len = (input.len() as f32 * ratio).round() as usize;
    (0..new_len)
        .map(|i| {
            let src_idx = i as f32 / ratio;
            let i0 = src_idx.floor() as usize;
            let i1 = (i0 + 1).min(input.len() - 1);
            let frac = src_idx - i0 as f32;
            input[i0] * (1.0 - frac) + input[i1] * frac
        })
        .collect()
}
