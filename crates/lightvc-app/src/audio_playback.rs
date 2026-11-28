//! Simple WAV playback via rodio-free approach.
//! Uses a background thread with cpal output (WASAPI on Windows so it never collides with ASIO).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// 44.1 kHz のモノラルを 1 回鳴らす(WASAPI の既定の出力)。位置を読め、途中へ飛べる。
pub struct AudioPlayer {
    playing: Arc<AtomicBool>,
    /// 今の位置と全体の長さ(出力の標本化周波数のサンプル数)。
    pos: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    /// 飛び先の割合 ×1e6(NO_SEEK = なし)。コールバックが拾う。
    seek: Arc<AtomicUsize>,
    _thread: Option<std::thread::JoinHandle<()>>,
}

const NO_SEEK: usize = usize::MAX;
const SEEK_SCALE: f32 = 1e6;

impl AudioPlayer {
    /// Play a mono f32 buffer at 44100 Hz.
    pub fn play(samples: Vec<f32>) -> Result<Self> {
        Self::play_from(samples, 0.0)
    }

    /// `start` = 鳴らし始める位置(全体に対する割合 0〜1)。
    pub fn play_from(samples: Vec<f32>, start: f32) -> Result<Self> {
        let playing = Arc::new(AtomicBool::new(true));
        let pos = Arc::new(AtomicUsize::new(0));
        let total = Arc::new(AtomicUsize::new(0));
        let seek = Arc::new(AtomicUsize::new((start.clamp(0.0, 1.0) * SEEK_SCALE) as usize));
        let (playing_t, pos_t, total_t, seek_t) = (playing.clone(), pos.clone(), total.clone(), seek.clone());

        let thread = std::thread::spawn(move || {
            let host = match lightvc_audio::host::pick(lightvc_audio::HostPref::Wasapi) {
                Ok(h) => h,
                Err(_) => {
                    playing_t.store(false, Ordering::Relaxed);
                    return;
                }
            };
            let Some(device) = host.default_output_device() else {
                playing_t.store(false, Ordering::Relaxed);
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
            let len = resampled.len();
            total_t.store(len, Ordering::Relaxed);
            let config = cpal::StreamConfig {
                channels: ch,
                sample_rate: target_sr,
                buffer_size: cpal::BufferSize::Default,
            };
            let n_ch = ch as usize;
            let (pos_cb, seek_cb) = (pos_t.clone(), seek_t.clone());

            let stream = device.build_output_stream(
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let req = seek_cb.swap(NO_SEEK, Ordering::Relaxed);
                    let mut i = if req == NO_SEEK { pos_cb.load(Ordering::Relaxed) } else { ((req as f32 / SEEK_SCALE) * len as f32) as usize };
                    for frame in data.chunks_mut(n_ch) {
                        let s = if i < len {
                            i += 1;
                            resampled[i - 1]
                        } else {
                            0.0
                        };
                        for slot in frame.iter_mut() {
                            *slot = s;
                        }
                    }
                    pos_cb.store(i, Ordering::Relaxed);
                },
                |err| eprintln!("Playback error: {err}"),
                None,
            );

            if let Ok(s) = stream {
                let _ = s.play();
                while playing_t.load(Ordering::Relaxed) {
                    if pos_t.load(Ordering::Relaxed) >= len && seek_t.load(Ordering::Relaxed) == NO_SEEK {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
            playing_t.store(false, Ordering::Relaxed);
        });

        Ok(Self { playing, pos, total, seek, _thread: Some(thread) })
    }

    pub fn stop(&self) {
        self.playing.store(false, Ordering::Relaxed);
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    /// 今の位置(全体に対する割合 0〜1・飛んでいる途中なら飛び先)。
    pub fn progress(&self) -> f32 {
        let req = self.seek.load(Ordering::Relaxed);
        if req != NO_SEEK {
            return req as f32 / SEEK_SCALE;
        }
        let t = self.total.load(Ordering::Relaxed);
        if t == 0 { 0.0 } else { (self.pos.load(Ordering::Relaxed) as f32 / t as f32).min(1.0) }
    }

    /// 位置(全体に対する割合 0〜1)へ飛ぶ。
    pub fn seek(&self, frac: f32) {
        self.seek.store((frac.clamp(0.0, 1.0) * SEEK_SCALE) as usize, Ordering::Relaxed);
    }
}

/// Load a WAV or FLAC file as mono f32 at its native sample rate.
/// 読める音声の拡張子(ファイルの選択の絞り込みにも使う)。
pub const AUDIO_EXTENSIONS: [&str; 4] = ["wav", "flac", "mp3", "mp2"];

/// MP3 / MP2 を読む(nanomp3・タグと符号器の遅延・詰め物を除く)。インターリーブのまま返す。
fn load_mpeg(path: &std::path::Path) -> Result<(Vec<f32>, u32, usize)> {
    let bytes = std::fs::read(path)?;
    let mut r = nanomp3::SliceReader::<f32>::new(&bytes);
    let mut v = Vec::new();
    let mut buf = vec![0f32; 4608];
    loop {
        let n = r.read(&mut buf).map_err(|e| anyhow::anyhow!("{}: MP3 を復号できません({e:?})", path.display()))?;
        if n == 0 {
            break;
        }
        v.extend_from_slice(&buf[..n]);
    }
    let channels = r.channels().map(|c| c.num() as usize).unwrap_or(0);
    anyhow::ensure!(!v.is_empty() && channels > 0, "{}: MP3 の音声が見つかりません", path.display());
    Ok((v, r.sample_rate(), channels))
}

pub fn load_audio_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    anyhow::ensure!(
        AUDIO_EXTENSIONS.contains(&ext.as_str()),
        "{}: 読めない形式です(読めるのは WAV / FLAC / MP3)",
        path.display()
    );
    let is_flac = ext == "flac";
    let (samples, sr, channels, scale) = if ext == "mp3" || ext == "mp2" {
        let (v, sr, ch) = load_mpeg(path)?;
        (v, sr, ch, 1.0)
    } else if is_flac {
        let mut reader = claxon::FlacReader::open(path)?;
        let info = reader.streaminfo();
        let mut v = Vec::with_capacity(info.samples.unwrap_or(0).saturating_mul(info.channels as u64) as usize);
        for s in reader.samples() {
            v.push(s? as f32);
        }
        (v, info.sample_rate, info.channels as usize, 2f32.powi(info.bits_per_sample as i32 - 1))
    } else {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        // 整数の WAV は hound が f32 として読めない(i32 で読んで尺度で割る)
        let v: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
            reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
        } else {
            let max_val: f32 = match spec.bits_per_sample {
                24 => 8388608.0,
                32 => 2147483648.0,
                _ => 32768.0,
            };
            reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max_val).collect()
        };
        (v, spec.sample_rate, spec.channels as usize, 1.0)
    };
    anyhow::ensure!(channels >= 1 && sr > 0, "{}: 声道か標本化レートが不正です", path.display());

    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };
    let mono = if scale == 1.0 { mono } else { mono.iter().map(|s| s / scale).collect() };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_wav(name: &str, sr: u32) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("lightvc-audio-{name}-{}.wav", std::process::id()));
        let spec = hound::WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for s in [0i16, 16384, -16384, 8192] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        path
    }

    /// 整数の WAV は標本化レートを保ち、-1..1 に尺度を合わせて読む(操作者の登録・自分で追加した声)。
    #[test]
    fn wav_keeps_its_rate_and_scales_integers() {
        let path = tmp_wav("int16", 24_000);
        let (x, sr) = load_audio_mono(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(sr, 24_000);
        assert_eq!(x, vec![0.0, 0.5, -0.5, 0.25]);
    }

    /// 同梱のプリセット(FLAC・48 kHz)が読める(`voices/` が無い環境では読み飛ばす)。
    #[test]
    fn flac_decodes_to_the_stream_length() {
        let Some(dir) = crate::voices::voices_dir() else {
            eprintln!("skip: voices/ が無い");
            return;
        };
        let path = dir.join("imouto.flac");
        let info = claxon::FlacReader::open(&path).unwrap().streaminfo();
        assert_eq!((info.sample_rate, info.channels), (48_000, 1));
        let (x, sr) = load_audio_mono(&path).unwrap();
        assert_eq!((sr, x.len() as u64), (48_000, info.samples.unwrap()));
        assert!(x.iter().all(|s| s.abs() <= 1.0) && x.iter().any(|s| s.abs() > 0.1));
    }
}
