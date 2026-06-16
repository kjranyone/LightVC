//! DAC wrapper providing continuous latent encode/decode.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::dac_model::{DacModel, DacModelConfig};
use crate::{DAC_HOP_LENGTH, DAC_LATENT_DIM};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct DacConfig {
    pub latent_dim: usize,
    pub encoder_d_model: usize,
    pub encoder_strides: Vec<usize>,
    pub decoder_d_model: usize,
    pub decoder_rates: Vec<usize>,
}

impl Default for DacConfig {
    fn default() -> Self {
        let cfg = DacModelConfig::default();
        Self {
            latent_dim: cfg.latent_dim,
            encoder_d_model: cfg.encoder_d_model,
            encoder_strides: cfg.encoder_strides,
            decoder_d_model: cfg.decoder_d_model,
            decoder_rates: cfg.decoder_rates,
        }
    }
}

impl From<&DacConfig> for DacModelConfig {
    fn from(c: &DacConfig) -> Self {
        DacModelConfig {
            latent_dim: c.latent_dim,
            encoder_d_model: c.encoder_d_model,
            encoder_strides: c.encoder_strides.clone(),
            decoder_d_model: c.decoder_d_model,
            decoder_rates: c.decoder_rates.clone(),
        }
    }
}

/// DAC codec providing continuous latent encode/decode (no quantization).
pub struct DacCodec {
    model: DacModel,
    device: Device,
}

impl DacCodec {
    pub fn from_file(
        weights_path: &std::path::Path,
        config: &DacConfig,
        device: Device,
    ) -> Result<Self> {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
        Self::from_varbuilder(vb, config, device)
    }

    pub fn from_varbuilder(vb: VarBuilder, config: &DacConfig, device: Device) -> Result<Self> {
        let model = DacModel::new(&config.into(), vb)?;
        Ok(Self { model, device })
    }

    /// Encode PCM `[batch, 1, samples]` → continuous latent `[batch, latent_dim, frames]`.
    pub fn encode(&self, pcm: &Tensor) -> Result<Tensor> {
        Ok(self.model.encoder.forward(pcm)?)
    }

    /// Decode continuous latent `[batch, latent_dim, frames]` → PCM `[batch, 1, samples]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        Ok(self.model.decoder.forward(latent)?)
    }

    pub fn round_trip(&self, pcm: &Tensor) -> Result<Tensor> {
        let latent = self.encode(pcm)?;
        self.decode(&latent)
    }

    pub fn encode_pcm(&self, samples: &[f32]) -> Result<Tensor> {
        let n = samples.len();
        let pcm = Tensor::from_slice(samples, n, &self.device)?.reshape((1, 1, n))?;
        self.encode(&pcm)
    }

    pub fn decode_to_pcm(&self, latent: &Tensor) -> Result<Vec<f32>> {
        let pcm = self.decode(latent)?;
        let flat = pcm.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        Ok(flat)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

pub fn pad_to_hop(samples: Vec<f32>) -> Vec<f32> {
    let len = samples.len();
    let rem = len % DAC_HOP_LENGTH;
    if rem == 0 {
        samples
    } else {
        let pad = DAC_HOP_LENGTH - rem;
        let mut out = samples;
        out.resize(len + pad, 0.0);
        out
    }
}

#[allow(dead_code)]
const _: usize = DAC_LATENT_DIM;
