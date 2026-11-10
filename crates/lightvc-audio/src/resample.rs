//! Real-time-safe resampling between the device sample rate and the model rate (48 kHz).
//!
//! Uses rubato v3 `Async` (sinc, fixed-input) with `process_into_buffer`
//! and pre-allocated work buffers. Steady-state operation performs no heap
//! allocation inside the resampler itself. Callers that need an owned
//! `Vec<f32>` must copy out of the returned slice — that copy is the only
//! remaining allocation and is deferred entirely to the caller side.

use anyhow::{anyhow, Result};
use rubato::{
    audioadapter_buffers::direct::SequentialSlice, Async, FixedAsync, Resampler as RubatoResampler,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

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

    /// 出力の遅延(出力のサンプル数)。オフラインで頭を詰めるときに使う。
    pub fn output_delay(&self) -> usize {
        self.r.output_delay()
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
