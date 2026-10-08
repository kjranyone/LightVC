//! Real-time-safe resampling between device sample rate and 44.1 kHz.
//!
//! Uses rubato v3 `Async` (sinc, fixed-input/output) with `process_into_buffer`
//! and pre-allocated work buffers. Steady-state operation performs no heap
//! allocation inside the resampler itself. Callers that need an owned
//! `Vec<f32>` must copy out of the returned slice — that copy is the only
//! remaining allocation and is deferred entirely to the caller side.

use anyhow::{anyhow, Result};
use rubato::{
    audioadapter_buffers::direct::SequentialSlice, Async, FixedAsync, Resampler as RubatoResampler,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Mono resampler: device sample rate ↔ 44.1 kHz.
pub struct Resampler {
    up: Async<f32>,
    down: Async<f32>,
    up_out: Vec<f32>,
    down_out: Vec<f32>,
}

impl Resampler {
    /// `device_sr`: host device sample rate.
    /// `chunk_size`: number of frames on the *fixed* side of each direction,
    /// i.e. input frames for `process_up` and output frames for `process_down`.
    pub fn new(device_sr: usize, chunk_size: usize) -> Result<Self> {
        if device_sr == 0 {
            return Err(anyhow!("device_sr must be non-zero"));
        }
        if chunk_size == 0 {
            return Err(anyhow!("chunk_size must be non-zero"));
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        // rubato's resample_ratio = output_sr / input_sr.
        let ratio_up = 44_100.0 / device_sr as f64;
        let ratio_down = device_sr as f64 / 44_100.0;

        // Up: device → 44.1k. Input size is fixed = chunk_size.
        let up = Async::<f32>::new_sinc(ratio_up, 2.0, &params, chunk_size, 1, FixedAsync::Input)?;

        // Down: 44.1k → device. Output size is fixed.
        let down_chunk = (chunk_size as f64 * ratio_down).round().max(1.0) as usize;
        let down =
            Async::<f32>::new_sinc(ratio_down, 2.0, &params, down_chunk, 1, FixedAsync::Output)?;

        // Pre-allocate output buffers with headroom. process_into_buffer writes
        // at most output_frames_max() frames per call; the slices we expose
        // are trimmed to the actually-written length.
        let up_out_cap = up.output_frames_max();
        let down_out_cap = down.output_frames_max();

        Ok(Self {
            up,
            down,
            up_out: vec![0.0; up_out_cap],
            down_out: vec![0.0; down_out_cap],
        })
    }

    /// Resample device_sr PCM → 44.1 kHz PCM.
    ///
    /// `input.len()` must be at least `input_frames_needed_up()`. Returns a
    /// slice into the internal output buffer; valid until the next `process_up`.
    pub fn process_up(&mut self, input: &[f32]) -> Result<&[f32]> {
        let needed = self.up.input_frames_next();
        if input.len() < needed {
            return Err(anyhow!(
                "process_up: need {needed} input frames, got {}",
                input.len()
            ));
        }
        let frames_out = self.up.output_frames_next();
        let in_adapter =
            SequentialSlice::new(input, 1, needed).map_err(|e| anyhow!("input adapter: {e}"))?;
        let mut out_adapter = SequentialSlice::new_mut(&mut self.up_out, 1, frames_out)
            .map_err(|e| anyhow!("output adapter: {e}"))?;
        let (_used, written) = self
            .up
            .process_into_buffer(&in_adapter, &mut out_adapter, None)?;
        Ok(&self.up_out[..written])
    }

    /// Resample 44.1 kHz PCM → device_sr PCM.
    ///
    /// `input.len()` must be at least `input_frames_needed_down()`. Returns a
    /// slice into the internal output buffer; valid until the next `process_down`.
    pub fn process_down(&mut self, input: &[f32]) -> Result<&[f32]> {
        let needed = self.down.input_frames_next();
        if input.len() < needed {
            return Err(anyhow!(
                "process_down: need {needed} input frames, got {}",
                input.len()
            ));
        }
        let frames_out = self.down.output_frames_next();
        let in_adapter =
            SequentialSlice::new(input, 1, needed).map_err(|e| anyhow!("input adapter: {e}"))?;
        let mut out_adapter = SequentialSlice::new_mut(&mut self.down_out, 1, frames_out)
            .map_err(|e| anyhow!("output adapter: {e}"))?;
        let (_used, written) =
            self.down
                .process_into_buffer(&in_adapter, &mut out_adapter, None)?;
        Ok(&self.down_out[..written])
    }

    /// Number of input frames the next `process_up` call will consume.
    /// For `FixedAsync::Input` this equals the `chunk_size` from construction.
    pub fn input_frames_needed_up(&self) -> usize {
        self.up.input_frames_next()
    }

    /// Number of input frames the next `process_down` call will consume.
    /// For `FixedAsync::Output` this varies from call to call.
    pub fn input_frames_needed_down(&self) -> usize {
        self.down.input_frames_next()
    }
}

/// 任意レート間の変換器（モデル側 48 kHz などに合わせる。入力側のチャンク長固定）。
pub struct RateConv {
    r: Async<f32>,
    out: Vec<f32>,
}

impl RateConv {
    pub fn new(from_sr: usize, to_sr: usize, chunk_size: usize) -> Result<Self> {
        if from_sr == 0 || to_sr == 0 || chunk_size == 0 {
            return Err(anyhow!("RateConv: rates and chunk_size must be non-zero"));
        }
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let r = Async::<f32>::new_sinc(to_sr as f64 / from_sr as f64, 2.0, &params, chunk_size, 1, FixedAsync::Input)?;
        let cap = r.output_frames_max();
        Ok(Self { r, out: vec![0.0; cap] })
    }

    pub fn input_frames_next(&self) -> usize {
        self.r.input_frames_next()
    }

    pub fn process(&mut self, input: &[f32]) -> Result<&[f32]> {
        let needed = self.r.input_frames_next();
        if input.len() < needed {
            return Err(anyhow!("RateConv: need {needed} input frames, got {}", input.len()));
        }
        let frames_out = self.r.output_frames_next();
        let in_adapter = SequentialSlice::new(input, 1, needed).map_err(|e| anyhow!("input adapter: {e}"))?;
        let mut out_adapter =
            SequentialSlice::new_mut(&mut self.out, 1, frames_out).map_err(|e| anyhow!("output adapter: {e}"))?;
        let (_used, written) = self.r.process_into_buffer(&in_adapter, &mut out_adapter, None)?;
        Ok(&self.out[..written])
    }
}

#[cfg(test)]
mod rate_conv_tests {
    use super::RateConv;

    #[test]
    fn roundtrip_44k_48k_keeps_length_and_level() {
        let sr = 44_100usize;
        let x: Vec<f32> = (0..sr).map(|i| 0.3 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin()).collect();
        let mut up = RateConv::new(sr, 48_000, 240).unwrap();
        let mut down = RateConv::new(48_000, sr, 240).unwrap();
        let (mut mid, mut out) = (Vec::new(), Vec::new());
        let mut i = 0;
        while i + up.input_frames_next() <= x.len() {
            let k = up.input_frames_next();
            mid.extend_from_slice(up.process(&x[i..i + k]).unwrap());
            i += k;
        }
        let mut j = 0;
        while j + down.input_frames_next() <= mid.len() {
            let k = down.input_frames_next();
            out.extend_from_slice(down.process(&mid[j..j + k]).unwrap());
            j += k;
        }
        let ratio = mid.len() as f64 / i as f64;
        let rms = |v: &[f32]| (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32).sqrt();
        let tail = |v: &[f32]| v[v.len() / 2..].to_vec();
        assert!((ratio - 48_000.0 / 44_100.0).abs() < 0.01, "length ratio {ratio}");
        assert!((rms(&tail(&out)) / rms(&tail(&x)) - 1.0).abs() < 0.02, "level");
    }
}
