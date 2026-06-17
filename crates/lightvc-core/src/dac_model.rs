//! Native DAC model matching HuggingFace safetensors key naming.
//!
//! Uses `candle_core::Result` internally for `Module` trait compatibility.
//! The public `DacCodec` wrapper (in `codec.rs`) converts to `anyhow::Result`.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

// ---------------------------------------------------------------------------
// Snake1d activation
// ---------------------------------------------------------------------------

pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shape = xs.shape();
        let xs_flat = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs_flat)?.sin()?;
        let sin_sq = (&sin * &sin)?;
        // Matches Python: 1.0 / (alpha + 1e-9). affined = alpha * 1.0 + 1e-9.
        let alpha_safe = self.alpha.affine(1.0, 1e-9)?;
        let out = (&xs_flat + alpha_safe.recip()?.broadcast_mul(&sin_sq)?)?;
        out.reshape(shape)
    }
}

impl Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Snake1d::forward(self, xs)
    }
}

// ---------------------------------------------------------------------------
// Plain Conv1d loader (weight + bias, no weight_norm)
// ---------------------------------------------------------------------------

fn conv1d_plain(
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_ch, in_ch, kernel_size), "weight")?;
    let bias = vb.get((out_ch,), "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn conv_transpose1d_plain(
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    config: ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight = vb.get((in_ch, out_ch, kernel_size), "weight")?;
    let bias = vb.get((out_ch,), "bias")?;
    Ok(ConvTranspose1d::new(weight, Some(bias), config))
}

// ---------------------------------------------------------------------------
// Residual Unit (dilated conv stack)
// ---------------------------------------------------------------------------

pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let snake1 = Snake1d::new(dim, vb.pp("snake1"))?;
        let snake2 = Snake1d::new(dim, vb.pp("snake2"))?;

        let pad1 = ((7 - 1) * dilation) / 2;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad1,
            ..Default::default()
        };
        let conv1 = conv1d_plain(dim, dim, 7, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d_plain(dim, dim, 1, Default::default(), vb.pp("conv2"))?;

        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs
            .apply(&self.snake1)?
            .apply(&self.conv1)?
            .apply(&self.snake2)?
            .apply(&self.conv2)?;

        let pad = (xs.dim(D::Minus1)? - ys.dim(D::Minus1)?) / 2;
        if pad > 0 {
            let xs_cropped = xs.narrow(D::Minus1, pad, ys.dim(D::Minus1)?)?;
            &ys + &xs_cropped
        } else {
            &ys + xs
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder Block
// ---------------------------------------------------------------------------

pub struct EncoderBlock {
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    snake1: Snake1d,
    conv1: Conv1d,
}

impl EncoderBlock {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let res1 = ResidualUnit::new(in_dim, 1, vb.pp("res_unit1"))?;
        let res2 = ResidualUnit::new(in_dim, 3, vb.pp("res_unit2"))?;
        let res3 = ResidualUnit::new(in_dim, 9, vb.pp("res_unit3"))?;
        let snake1 = Snake1d::new(in_dim, vb.pp("snake1"))?;

        let cfg = Conv1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv1 = conv1d_plain(in_dim, out_dim, 2 * stride, cfg, vb.pp("conv1"))?;

        Ok(Self {
            res1,
            res2,
            res3,
            snake1,
            conv1,
        })
    }
}

impl Module for EncoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)?
            .apply(&self.snake1)?
            .apply(&self.conv1)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

pub struct Encoder {
    conv1: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Encoder {
    pub fn new(
        mut d_model: usize,
        strides: &[usize],
        latent_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = conv1d_plain(
            1,
            d_model,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;

        let mut blocks = Vec::with_capacity(strides.len());
        for (i, &stride) in strides.iter().enumerate() {
            let in_dim = d_model;
            d_model *= 2;
            let block = EncoderBlock::new(in_dim, d_model, stride, vb.pp(&format!("block.{i}")))?;
            blocks.push(block);
        }

        let snake1 = Snake1d::new(d_model, vb.pp("snake1"))?;
        let conv2 = conv1d_plain(
            d_model,
            latent_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in &self.blocks {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

// ---------------------------------------------------------------------------
// Decoder Block
// ---------------------------------------------------------------------------

pub struct DecoderBlock {
    snake1: Snake1d,
    conv_t1: ConvTranspose1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let snake1 = Snake1d::new(in_dim, vb.pp("snake1"))?;

        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv_t1 = conv_transpose1d_plain(in_dim, out_dim, 2 * stride, cfg, vb.pp("conv_t1"))?;

        let res1 = ResidualUnit::new(out_dim, 1, vb.pp("res_unit1"))?;
        let res2 = ResidualUnit::new(out_dim, 3, vb.pp("res_unit2"))?;
        let res3 = ResidualUnit::new(out_dim, 9, vb.pp("res_unit3"))?;

        Ok(Self {
            snake1,
            conv_t1,
            res1,
            res2,
            res3,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_t1)?
            .apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

pub struct Decoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Decoder {
    pub fn new(
        in_dim: usize,
        mut channels: usize,
        rates: &[usize],
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = conv1d_plain(
            in_dim,
            channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;

        let mut blocks = Vec::with_capacity(rates.len());
        for (i, &stride) in rates.iter().enumerate() {
            let out_dim = channels / 2;
            let block = DecoderBlock::new(channels, out_dim, stride, vb.pp(&format!("block.{i}")))?;
            channels = out_dim;
            blocks.push(block);
        }

        let snake1 = Snake1d::new(channels, vb.pp("snake1"))?;
        let conv2 = conv1d_plain(
            channels,
            d_out,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in &self.blocks {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

// ---------------------------------------------------------------------------
// Full DAC Model (encoder + decoder)
// ---------------------------------------------------------------------------

pub struct DacModel {
    pub encoder: Encoder,
    pub decoder: Decoder,
    /// RVQ quantizer. Only loaded for Phase 3 ([07-1]); `None` for normal
    /// continuous-latent inference.
    pub quantizer: Option<Quantizer>,
}

#[derive(Clone, Debug)]
pub struct DacModelConfig {
    pub latent_dim: usize,
    pub encoder_d_model: usize,
    pub encoder_strides: Vec<usize>,
    pub decoder_d_model: usize,
    pub decoder_rates: Vec<usize>,
}

impl Default for DacModelConfig {
    fn default() -> Self {
        Self {
            latent_dim: 1024,
            encoder_d_model: 64,
            encoder_strides: vec![2, 4, 8, 8],
            decoder_d_model: 1536,
            decoder_rates: vec![8, 8, 4, 2],
        }
    }
}

impl DacModel {
    pub fn new(config: &DacModelConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(
            config.encoder_d_model,
            &config.encoder_strides,
            config.latent_dim,
            vb.pp("encoder"),
        )?;
        let decoder = Decoder::new(
            config.latent_dim,
            config.decoder_d_model,
            &config.decoder_rates,
            1,
            vb.pp("decoder"),
        )?;
        Ok(Self {
            encoder,
            decoder,
            quantizer: None,
        })
    }

    /// Load the model including the RVQ quantizer ([07-1] Phase 3 prep).
    /// The quantizer is only present in the full `descript/dac_44khz`
    /// checkpoint; it is absent from LightVC's exported inference weights.
    pub fn with_quantizer(mut self, vb: VarBuilder) -> Result<Self> {
        let q = Quantizer::new(
            config_default_latent_dim(),
            DAC_CODEBOOK_DIM,
            DAC_N_CODES,
            DAC_N_CODEBOOKS,
            vb.pp("quantizer"),
        )?;
        self.quantizer = Some(q);
        Ok(self)
    }
}

fn config_default_latent_dim() -> usize {
    DacModelConfig::default().latent_dim
}

// ---------------------------------------------------------------------------
// Residual Vector Quantizer ([07-1] Phase 3 preparation, Rust side only)
// ---------------------------------------------------------------------------

/// DAC factorized codebook dimensions (descript/dac_44khz).
const DAC_CODEBOOK_DIM: usize = 8;
const DAC_N_CODES: usize = 1024;
const DAC_N_CODEBOOKS: usize = 9;

/// Single codebook: nearest-neighbor lookup in factorized space.
pub struct QuantizerLayer {
    codebook: Tensor, // [n_codes, codebook_dim]
}

impl QuantizerLayer {
    pub fn new(n_codes: usize, codebook_dim: usize, vb: VarBuilder) -> Result<Self> {
        let codebook = vb.get((n_codes, codebook_dim), "codebook")?;
        Ok(Self { codebook })
    }

    /// `z: [B, codebook_dim, T]` → `(codes [B, T], quantized [B, codebook_dim, T])`
    pub fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let z_t = z.permute((0, 2, 1))?; // [B, T, D]
                                         // L2 distance: ||z||^2 - 2*z·c + ||c||^2
        let z_sq = (&z_t * &z_t)?.sum(D::Minus1)?; // [B, T]
        let c_sq = (&self.codebook * &self.codebook)?.sum(D::Minus1)?; // [n_codes]
        let zc = z_t.matmul(&self.codebook.t()?)?; // [B, T, n_codes]
        let z_sq_e = z_sq.unsqueeze(D::Minus1)?; // [B, T, 1]
        let c_sq_e = c_sq.unsqueeze(0)?; // [1, n_codes]
        let dist = (z_sq_e.broadcast_add(&c_sq_e)? - (&zc * 2.0)?)?; // [B, T, n_codes]
        let codes = dist.argmin(D::Minus1)?; // [B, T]
        let quantized = Tensor::embedding(&self.codebook, &codes)?; // [B, T, D]
        let quantized = quantized.permute((0, 2, 1))?; // [B, D, T]
        Ok((codes, quantized))
    }
}

/// Residual vector quantizer: projects to codebook space, runs `n_codebooks`
/// residual layers, projects back to latent space.
pub struct Quantizer {
    in_proj: Conv1d,  // latent_dim → codebook_dim (1×1, no bias)
    out_proj: Conv1d, // codebook_dim → latent_dim (1×1, no bias)
    layers: Vec<QuantizerLayer>,
}

impl Quantizer {
    pub fn new(
        latent_dim: usize,
        codebook_dim: usize,
        n_codes: usize,
        n_codebooks: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_weight = vb.get((codebook_dim, latent_dim, 1), "in_proj.weight")?;
        let in_proj = Conv1d::new(in_weight, None, Conv1dConfig::default());
        let out_weight = vb.get((latent_dim, codebook_dim, 1), "out_proj.weight")?;
        let out_proj = Conv1d::new(out_weight, None, Conv1dConfig::default());
        let mut layers = Vec::with_capacity(n_codebooks);
        for i in 0..n_codebooks {
            layers.push(QuantizerLayer::new(
                n_codes,
                codebook_dim,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Self {
            in_proj,
            out_proj,
            layers,
        })
    }

    /// `z: [B, latent_dim, T]` → `(codes [B, n_codebooks, T], quantized [B, latent_dim, T])`
    pub fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let z_q = self.in_proj.forward(z)?; // [B, codebook_dim, T]
        let mut residual = z_q.clone();
        let mut all_codes = Vec::with_capacity(self.layers.len());
        let mut quantized_sum = Tensor::zeros_like(&z_q)?;
        for layer in &self.layers {
            let (codes, quantized) = layer.forward(&residual)?;
            all_codes.push(codes);
            quantized_sum = (&quantized_sum + &quantized)?;
            residual = (&residual - &quantized)?;
        }
        let codes = Tensor::stack(&all_codes, 1)?; // [B, n_codebooks, T]
        let quantized_out = self.out_proj.forward(&quantized_sum)?;
        Ok((codes, quantized_out))
    }
}
