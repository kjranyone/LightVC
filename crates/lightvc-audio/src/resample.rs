//! Real-time-safe resampling between device sample rate and 44.1 kHz.

use anyhow::Result;
use rubato::{
    Resampler as RubatoResampler, SincFixedIn, SincFixedOut, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};

/// Resampler that converts between device sample rate and DAC's 44.1 kHz.
pub struct Resampler {
    up: SincFixedIn<f32>,
    down: SincFixedOut<f32>,
}

impl Resampler {
    pub fn new(device_sr: usize, chunk_size: usize) -> Result<Self> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let ratio_up = 44_100.0 / device_sr as f64;
        let ratio_down = device_sr as f64 / 44_100.0;

        let up = SincFixedIn::<f32>::new(ratio_up, 2.0, params, chunk_size, 1)?;

        let params2 = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let down = SincFixedOut::<f32>::new(
            ratio_down,
            2.0,
            params2,
            (chunk_size as f64 * ratio_down).round() as usize,
            1,
        )?;

        Ok(Self { up, down })
    }

    pub fn process_up(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let waves_in = vec![input.to_vec()];
        let waves_out = self.up.process(&waves_in, None)?;
        Ok(waves_out.into_iter().next().unwrap_or_default())
    }

    pub fn process_down(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let waves_in = vec![input.to_vec()];
        let waves_out = self.down.process(&waves_in, None)?;
        Ok(waves_out.into_iter().next().unwrap_or_default())
    }

    pub fn input_frames_needed_up(&self) -> usize {
        self.up.input_frames_next()
    }

    pub fn input_frames_needed_down(&self) -> usize {
        self.down.input_frames_next()
    }
}
