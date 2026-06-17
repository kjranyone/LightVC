//! Voice conversion converter model.
//!
//! Phase 1: Causal Conv1d latent converter with FiLM speaker injection.
//! Phase 2: Universal Timbre Token Encoder (UTTE) with cross-attention.
//!
//! The converter transforms continuous DAC latents (1024-dim, ~86 Hz) in a
//! single forward pass — no flow matching, no ODE loop, no AR generation.
//!
//! Phase C flow-matching modules (`FlowConverter`, `AnyConverter`,
//! `BottleneckEncoder`, `TimeEmbed`, `CondMlp`) live in
//! [`crate::flow_converter`] and are re-exported from here for convenience.

use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder};

use crate::DAC_LATENT_DIM;

// ---------------------------------------------------------------------------
// Helper: create linear layer from VarBuilder
// ---------------------------------------------------------------------------

pub(crate) fn linear_layer(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> Result<candle_nn::Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get((out_dim,), "bias")?;
    Ok(candle_nn::Linear::new(weight, Some(bias)))
}

fn conv1d_layer(
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    dilation: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        dilation,
        padding: 0,
        stride: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    let weight = vb.get((out_ch, in_ch, kernel_size), "weight")?;
    let bias = vb.get((out_ch,), "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

fn layer_norm_layer(dim: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get((dim,), "weight")?;
    let bias = vb.get((dim,), "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

// ---------------------------------------------------------------------------
// Snake1d activation (matches DAC internals for latent-space compatibility)
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
        out.reshape(shape).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Causal Conv1d (left-pad only, no future context)
// ---------------------------------------------------------------------------

pub struct CausalConv1d {
    conv: Conv1d,
    kernel_size: usize,
    dilation: usize,
}

impl CausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            dilation,
            padding: 0,
            stride: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        // Python CausalConv1d stores conv as self.conv = nn.Conv1d(...),
        // so safetensors keys are "conv.weight" / "conv.bias".
        let weight = vb.get((out_channels, in_channels, kernel_size), "conv.weight")?;
        let bias = vb.get((out_channels,), "conv.bias")?;
        let conv = Conv1d::new(weight, Some(bias), cfg);
        Ok(Self {
            conv,
            kernel_size,
            dilation,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let pad = (self.kernel_size - 1) * self.dilation;
        let xs = if pad > 0 {
            xs.pad_with_zeros(D::Minus1, pad, 0)?
        } else {
            xs.clone()
        };
        Ok(self.conv.forward(&xs)?)
    }
}

// ---------------------------------------------------------------------------
// Causal Residual Conv Block (multi-dilation like DAC's ResidualUnit)
// ---------------------------------------------------------------------------

pub struct CausalResBlock {
    proj_in: candle_nn::Conv1d,
    snake1: Snake1d,
    c1: CausalConv1d,
    snake2: Snake1d,
    c2: CausalConv1d,
    snake3: Snake1d,
    c3: CausalConv1d,
    proj_out: candle_nn::Conv1d,
}

impl CausalResBlock {
    /// latent_dim → hidden_dim → conv blocks → hidden_dim → latent_dim
    pub fn new(latent_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj_in_w = vb.get((hidden_dim, latent_dim, 1), "proj_in.weight")?;
        let proj_in_b = vb.get((hidden_dim,), "proj_in.bias")?;
        let proj_in = candle_nn::Conv1d::new(proj_in_w, Some(proj_in_b), Default::default());

        let snake1 = Snake1d::new(hidden_dim, vb.pp("snake1"))?;
        let c1 = CausalConv1d::new(hidden_dim, hidden_dim, 7, 1, vb.pp("c1"))?;
        let snake2 = Snake1d::new(hidden_dim, vb.pp("snake2"))?;
        let c2 = CausalConv1d::new(hidden_dim, hidden_dim, 7, 3, vb.pp("c2"))?;
        let snake3 = Snake1d::new(hidden_dim, vb.pp("snake3"))?;
        let c3 = CausalConv1d::new(hidden_dim, hidden_dim, 7, 9, vb.pp("c3"))?;

        let proj_out_w = vb.get((latent_dim, hidden_dim, 1), "proj_out.weight")?;
        let proj_out_b = vb.get((latent_dim,), "proj_out.bias")?;
        let proj_out = candle_nn::Conv1d::new(proj_out_w, Some(proj_out_b), Default::default());

        Ok(Self {
            proj_in,
            snake1,
            c1,
            snake2,
            c2,
            snake3,
            c3,
            proj_out,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let h = self.proj_in.forward(xs)?;
        let h = self.c1.forward(&self.snake1.forward(&h)?)?;
        let h = self.c2.forward(&self.snake2.forward(&h)?)?;
        let h = self.c3.forward(&self.snake3.forward(&h)?)?;
        let h = self.proj_out.forward(&h)?;
        Ok((&h + residual)?)
    }
}

// ---------------------------------------------------------------------------
// FiLM conditioning: γ * z + β
// ---------------------------------------------------------------------------

pub struct FilmCond {
    proj: candle_nn::Linear,
    latent_dim: usize,
}

impl FilmCond {
    pub fn new(embed_dim: usize, latent_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear_layer(embed_dim, latent_dim * 2, vb.pp("film"))?;
        Ok(Self { proj, latent_dim })
    }

    pub fn forward(&self, z: &Tensor, embed: &Tensor) -> Result<Tensor> {
        let gb = self.proj.forward(embed)?;
        let gamma = gb.narrow(D::Minus1, 0, self.latent_dim)?;
        let beta = gb.narrow(D::Minus1, self.latent_dim, self.latent_dim)?;
        let gamma = gamma.unsqueeze(D::Minus1)?;
        let beta = beta.unsqueeze(D::Minus1)?;
        let z_scaled = gamma.broadcast_mul(z)?;
        Ok(beta.broadcast_add(&z_scaled)?)
    }
}

// ---------------------------------------------------------------------------
// Speaker Encoder: reference latent → global speaker embedding
// ---------------------------------------------------------------------------

pub struct SpeakerEncoder {
    proj1: candle_nn::Linear,
    proj2: candle_nn::Linear,
}

impl SpeakerEncoder {
    pub fn new(latent_dim: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj1 = linear_layer(latent_dim * 2, latent_dim / 2, vb.pp("p1"))?;
        let proj2 = linear_layer(latent_dim / 2, embed_dim, vb.pp("p2"))?;
        Ok(Self { proj1, proj2 })
    }

    pub fn forward(&self, ref_latent: &Tensor) -> Result<Tensor> {
        let t_len = ref_latent.dim(2)?;
        let mean = ref_latent.mean(D::Minus1)?;
        let std = {
            let mean_b = mean.unsqueeze(D::Minus1)?.broadcast_as(ref_latent.shape())?;
            let diff = (ref_latent - &mean_b)?;
            let sum_sq = diff.sqr()?.sum(D::Minus1)?;
            // PyTorch std(unbiased=True) divides by (N-1), not N.
            let n = (t_len as f64).max(1.0);
            let var = sum_sq.affine(1.0 / (n - 1.0), 0.0)?;
            var.sqrt()?
        };
        let pooled = Tensor::cat(&[&mean, &std], D::Minus1)?;
        let h = self.proj1.forward(&pooled)?;
        let h = h.gelu()?;
        Ok(self.proj2.forward(&h)?)
    }
}

// ---------------------------------------------------------------------------
// Universal Timbre Token Encoder (Phase 2, from MeanVC2)
// ---------------------------------------------------------------------------

pub struct TimbreTokenBank {
    n_tokens: usize,
    embed_dim: usize,
    key_prior: Tensor,
    val_prior: Tensor,
    key_proj: candle_nn::Linear,
    val_proj: candle_nn::Linear,
}

impl TimbreTokenBank {
    pub fn new(embed_dim: usize, n_tokens: usize, vb: VarBuilder) -> Result<Self> {
        let key_prior = vb.get((n_tokens, embed_dim), "key_prior")?;
        let val_prior = vb.get((n_tokens, embed_dim), "val_prior")?;
        let key_proj = linear_layer(embed_dim, embed_dim * n_tokens, vb.pp("key_proj"))?;
        let val_proj = linear_layer(embed_dim, embed_dim * n_tokens, vb.pp("val_proj"))?;
        Ok(Self {
            n_tokens,
            embed_dim,
            key_prior,
            val_prior,
            key_proj,
            val_proj,
        })
    }

    pub fn forward(&self, speaker_embed: &Tensor) -> Result<(Tensor, Tensor)> {
        let b = speaker_embed.dim(0)?;
        let keys_flat = self.key_proj.forward(speaker_embed)?;
        let keys = keys_flat.reshape((b, self.n_tokens, self.embed_dim))?;
        let keys = (&keys + self.key_prior.tanh()?)?;

        let vals_flat = self.val_proj.forward(speaker_embed)?;
        let vals = vals_flat.reshape((b, self.n_tokens, self.embed_dim))?;
        let vals = (&vals + self.val_prior.tanh()?)?;

        Ok((keys, vals))
    }
}

// ---------------------------------------------------------------------------
// Cross-attention block (z queries timbre tokens)
// ---------------------------------------------------------------------------

pub struct CrossAttnBlock {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    n_heads: usize,
    attn_dim: usize,
    norm: LayerNorm,
}

impl CrossAttnBlock {
    pub fn new(q_dim: usize, kv_dim: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let attn_dim = n_heads * (kv_dim / n_heads);
        let head_dim = attn_dim / n_heads;
        let q_proj = linear_layer(q_dim, attn_dim, vb.pp("q"))?;
        let k_proj = linear_layer(kv_dim, attn_dim, vb.pp("k"))?;
        let v_proj = linear_layer(kv_dim, attn_dim, vb.pp("v"))?;
        let out_proj = linear_layer(attn_dim, q_dim, vb.pp("o"))?;
        let norm = layer_norm_layer(q_dim, 1e-5, vb.pp("norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads,
            attn_dim,
            norm,
        })
    }

    pub fn forward(&self, z: &Tensor, keys: &Tensor, vals: &Tensor) -> Result<Tensor> {
        let (b, _d, t) = z.dims3()?;
        let z_t = z.transpose(1, 2)?;

        let q = self.q_proj.forward(&z_t)?;
        let k = self.k_proj.forward(keys)?;
        let v = self.v_proj.forward(vals)?;

        let head_dim = self.attn_dim / self.n_heads;
        let q = q.reshape((b * self.n_heads, t, head_dim))?;
        let n_tok = k.dim(1)?;
        let k = k.reshape((b * self.n_heads, n_tok, head_dim))?;
        let v = v.reshape((b * self.n_heads, n_tok, head_dim))?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(1, 2)?)?;
        let attn = (attn * scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;

        let out = out.reshape((b, t, self.attn_dim))?;
        let out = self.out_proj.forward(&out)?;

        let z_norm = self.norm.forward(&z_t)?;
        Ok((&z_norm + &out)?.transpose(1, 2)?)
    }
}

// ---------------------------------------------------------------------------
// Converter Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ConverterConfig {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub n_conv_blocks: usize,
    pub speaker_embed_dim: usize,
    #[serde(default = "default_n_timbre")]
    pub n_timbre_tokens: usize,
    #[serde(default = "default_n_attn")]
    pub n_attn_heads: usize,
    #[serde(default)]
    pub enable_timbre: bool,
    #[serde(default = "default_bottleneck")]
    pub bottleneck_dim: usize,
    #[serde(default = "default_time_embed")]
    pub time_embed_dim: usize,
    #[serde(default = "default_model_type")]
    pub model_type: String,
}

fn default_n_timbre() -> usize {
    32
}
fn default_n_attn() -> usize {
    8
}
fn default_bottleneck() -> usize {
    256
}
fn default_time_embed() -> usize {
    128
}
fn default_model_type() -> String {
    "converter".to_string()
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            latent_dim: DAC_LATENT_DIM,
            hidden_dim: DAC_LATENT_DIM,
            n_conv_blocks: 4,
            speaker_embed_dim: 256,
            n_timbre_tokens: 32,
            n_attn_heads: 8,
            enable_timbre: false,
            bottleneck_dim: 256,
            time_embed_dim: 128,
            model_type: "converter".to_string(),
        }
    }
}

/// Latency / quality mode (MeanVC2 future-receptive chunking concept).
#[derive(Clone, Copy, Debug, serde::Deserialize, PartialEq, Eq)]
pub enum LatencyMode {
    Strict,
    Balanced,
    Quality,
}

impl Default for LatencyMode {
    fn default() -> Self {
        Self::Balanced
    }
}

// ---------------------------------------------------------------------------
// Converter Model
// ---------------------------------------------------------------------------

pub struct Converter {
    config: ConverterConfig,
    film: FilmCond,
    speaker_encoder: SpeakerEncoder,
    blocks: Vec<CausalResBlock>,
    out_proj: CausalConv1d,
    timbre_bank: Option<TimbreTokenBank>,
    cross_attns: Vec<CrossAttnBlock>,
}

impl Converter {
    pub fn new(config: ConverterConfig, vb: VarBuilder) -> Result<Self> {
        let latent_dim = config.latent_dim;
        let embed_dim = config.speaker_embed_dim;

        let film = FilmCond::new(embed_dim, latent_dim, vb.pp("film"))?;
        let speaker_encoder = SpeakerEncoder::new(latent_dim, embed_dim, vb.pp("speaker_encoder"))?;

        let mut blocks = Vec::with_capacity(config.n_conv_blocks);
        for i in 0..config.n_conv_blocks {
            let blk =
                CausalResBlock::new(latent_dim, config.hidden_dim, vb.pp(&format!("blocks.{i}")))?;
            blocks.push(blk);
        }
        let out_proj = CausalConv1d::new(latent_dim, latent_dim, 1, 1, vb.pp("out_proj"))?;

        let (timbre_bank, cross_attns) = if config.enable_timbre {
            let bank = TimbreTokenBank::new(embed_dim, config.n_timbre_tokens, vb.pp("timbre"))?;
            let mut attns = Vec::new();
            for i in 0..config.n_conv_blocks {
                let attn = CrossAttnBlock::new(
                    latent_dim,
                    embed_dim,
                    config.n_attn_heads,
                    vb.pp(&format!("xattn.{i}")),
                )?;
                attns.push(attn);
            }
            (Some(bank), attns)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            config,
            film,
            speaker_encoder,
            blocks,
            out_proj,
            timbre_bank,
            cross_attns,
        })
    }

    pub fn speaker_embedding(&self, ref_latent: &Tensor) -> Result<Tensor> {
        self.speaker_encoder.forward(ref_latent)
    }

    pub fn timbre_tokens(&self, ref_latent: &Tensor) -> Result<(Tensor, Tensor)> {
        let embed = self.speaker_encoder.forward(ref_latent)?;
        match &self.timbre_bank {
            Some(bank) => bank.forward(&embed),
            None => anyhow::bail!("timbre tokens not available: enable_timbre=false"),
        }
    }

    /// One-step forward conversion.
    ///
    /// Accepts both batched `[B, D, T]` and unbatched `[D, T]` inputs,
    /// matching the Python `Converter.forward` ([08-6]).
    pub fn forward(&self, src_latent: &Tensor, ref_latent: &Tensor) -> Result<Tensor> {
        let was_unbatched = src_latent.rank() == 2;
        let (src_latent, ref_latent) = if was_unbatched {
            (src_latent.unsqueeze(0)?, ref_latent.unsqueeze(0)?)
        } else {
            (src_latent.clone(), ref_latent.clone())
        };

        let speaker_embed = self.speaker_encoder.forward(&ref_latent)?;
        let z = self.film.forward(&src_latent, &speaker_embed)?;

        let timbre = if self.config.enable_timbre {
            if let Some(bank) = &self.timbre_bank {
                Some(bank.forward(&speaker_embed)?)
            } else {
                None
            }
        } else {
            None
        };

        let mut z = z;
        for (i, block) in self.blocks.iter().enumerate() {
            z = block.forward(&z)?;
            if let Some((keys, vals)) = timbre.as_ref() {
                if i < self.cross_attns.len() {
                    z = self.cross_attns[i].forward(&z, keys, vals)?;
                }
            }
        }

        let delta = self.out_proj.forward(&z)?;
        let result = (&src_latent + &delta)?;
        if was_unbatched {
            result.squeeze(0).map_err(Into::into)
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// Phase C flow-matching modules live in `flow_converter.rs` (declared at the
// crate root in lib.rs). Re-exported here so existing
// `crate::converter::FlowConverter` paths keep working.
// ===========================================================================

pub use crate::flow_converter::{
    AnyConverter, BottleneckEncoder, CondMlp, FlowConverter, TimeEmbed,
};
