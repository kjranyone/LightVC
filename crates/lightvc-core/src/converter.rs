//! Voice conversion converter model.
//!
//! Phase 1: Causal Conv1d latent converter with FiLM speaker injection.
//! Phase 2: Universal Timbre Token Encoder (UTTE) with cross-attention.
//!
//! The converter transforms continuous DAC latents (1024-dim, ~86 Hz) in a
//! single forward pass — no flow matching, no ODE loop, no AR generation.

use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder};

use crate::DAC_LATENT_DIM;

// ---------------------------------------------------------------------------
// Helper: create linear layer from VarBuilder
// ---------------------------------------------------------------------------

fn linear_layer(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
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
        // Try standard conv keys first, then depthwise keys
        let weight = vb
            .get((out_channels, in_channels, kernel_size), "weight")
            .or_else(|_| vb.get((out_channels, in_channels, kernel_size), "conv.weight"))?;
        let bias = vb
            .get((out_channels,), "bias")
            .or_else(|_| vb.get((out_channels,), "conv.bias"))?;
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
        let proj1 = linear_layer(latent_dim, latent_dim / 2, vb.pp("p1"))?;
        let proj2 = linear_layer(latent_dim / 2, embed_dim, vb.pp("p2"))?;
        Ok(Self { proj1, proj2 })
    }

    pub fn forward(&self, ref_latent: &Tensor) -> Result<Tensor> {
        let pooled = ref_latent.mean(D::Minus1)?;
        let h = self.proj1.forward(&pooled)?;
        let h = h.relu()?;
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
    head_dim: usize,
    norm: LayerNorm,
}

impl CrossAttnBlock {
    pub fn new(dim: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / n_heads;
        let q_proj = linear_layer(dim, dim, vb.pp("q"))?;
        let k_proj = linear_layer(dim, dim, vb.pp("k"))?;
        let v_proj = linear_layer(dim, dim, vb.pp("v"))?;
        let out_proj = linear_layer(dim, dim, vb.pp("o"))?;
        let norm = layer_norm_layer(dim, 1e-5, vb.pp("norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads,
            head_dim,
            norm,
        })
    }

    pub fn forward(&self, z: &Tensor, keys: &Tensor, vals: &Tensor) -> Result<Tensor> {
        let (b, _d, t) = z.dims3()?;
        let z_t = z.transpose(1, 2)?;

        let q = self.q_proj.forward(&z_t)?;
        let k = self.k_proj.forward(keys)?;
        let v = self.v_proj.forward(vals)?;

        let q = q.reshape((b * self.n_heads, t, self.head_dim))?;
        let n_tok = k.dim(1)?;
        let k = k.reshape((b * self.n_heads, n_tok, self.head_dim))?;
        let v = v.reshape((b * self.n_heads, n_tok, self.head_dim))?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(1, 2)?)?;
        let attn = (attn * scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;

        let out = out.reshape((b, t, self.n_heads * self.head_dim))?;
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
    pub fn forward(&self, src_latent: &Tensor, ref_latent: &Tensor) -> Result<Tensor> {
        let speaker_embed = self.speaker_encoder.forward(ref_latent)?;
        let z = self.film.forward(src_latent, &speaker_embed)?;

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
        Ok((src_latent + &delta)?)
    }
}

// ===========================================================================
// Flow matching modules (Phase C)
// ===========================================================================

// ---------------------------------------------------------------------------
// Bottleneck Encoder (content code, drops speaker info)
// ---------------------------------------------------------------------------

pub struct BottleneckEncoder {
    down: CausalConv1d,
    act: Snake1d,
    up: CausalConv1d,
}

impl BottleneckEncoder {
    pub fn new(latent_dim: usize, bottleneck_dim: usize, vb: VarBuilder) -> Result<Self> {
        let down = CausalConv1d::new(latent_dim, bottleneck_dim, 1, 1, vb.pp("down"))?;
        let act = Snake1d::new(bottleneck_dim, vb.pp("act"))?;
        let up = CausalConv1d::new(bottleneck_dim, latent_dim, 1, 1, vb.pp("up"))?;
        Ok(Self { down, act, up })
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let c = self.down.forward(z)?;
        let c = self.act.forward(&c)?;
        self.up.forward(&c)
    }
}

// ---------------------------------------------------------------------------
// Time Embedding (sinusoidal for flow-matching timestep)
// ---------------------------------------------------------------------------

pub struct TimeEmbed {
    freqs: Tensor,
    mlp0: candle_nn::Linear,
    mlp2: candle_nn::Linear,
}

impl TimeEmbed {
    pub fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let half = embed_dim / 2;
        let device = vb.device();
        let freqs_data: Vec<f32> = (0..half)
            .map(|i| (1.0f32 / 10000.0f32.powf(i as f32 / half as f32)))
            .collect();
        let freqs = Tensor::from_vec(freqs_data, half, device)?;

        let mlp0 = linear_layer(embed_dim, embed_dim * 2, vb.pp("mlp.0"))?;
        let mlp2 = linear_layer(embed_dim * 2, embed_dim, vb.pp("mlp.2"))?;

        Ok(Self { freqs, mlp0, mlp2 })
    }

    /// `t`: [B] in [0,1] → embed [B, embed_dim]
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let half = self.freqs.dim(0)?;
        let t_f = t.reshape((t.dim(0)?, 1))?;
        let freqs_b = self.freqs.broadcast_as((1, half))?;
        let args = t_f.broadcast_mul(&freqs_b)?;
        let scaled = args.affine(2.0 * std::f32::consts::PI as f64, 0.0)?;
        let sin = scaled.sin()?;
        let cos = scaled.cos()?;
        let emb = Tensor::cat(&[sin, cos], D::Minus1)?;
        let h = self.mlp0.forward(&emb)?;
        let h = h.gelu()?;
        Ok(self.mlp2.forward(&h)?)
    }
}

// ---------------------------------------------------------------------------
// Conditioning MLP (speaker + time → FiLM params)
// ---------------------------------------------------------------------------

pub struct CondMlp {
    l0: candle_nn::Linear,
    l2: candle_nn::Linear,
    out_dim: usize,
}

impl CondMlp {
    pub fn new(in_dim: usize, latent_dim: usize, vb: VarBuilder) -> Result<Self> {
        let l0 = linear_layer(in_dim, latent_dim, vb.pp("0"))?;
        let l2 = linear_layer(latent_dim, latent_dim * 2, vb.pp("2"))?;
        Ok(Self {
            l0,
            l2,
            out_dim: latent_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.l0.forward(x)?;
        let h = h.gelu()?;
        let out = self.l2.forward(&h)?;
        let gamma = out.narrow(D::Minus1, 0, self.out_dim)?;
        let beta = out.narrow(D::Minus1, self.out_dim, self.out_dim)?;
        Ok((gamma, beta))
    }
}

// ---------------------------------------------------------------------------
// FlowConverter (Phase C, the core model)
// ---------------------------------------------------------------------------

pub struct FlowConverter {
    config: ConverterConfig,
    bottleneck: BottleneckEncoder,
    speaker_encoder: SpeakerEncoder,
    time_embed: TimeEmbed,
    cond_mlp: CondMlp,
    blocks: Vec<CausalResBlock>,
    vel_proj: CausalConv1d,
    timbre_bank: Option<TimbreTokenBank>,
    cross_attns: Vec<CrossAttnBlock>,
}

impl FlowConverter {
    pub fn new(config: ConverterConfig, vb: VarBuilder) -> Result<Self> {
        let latent_dim = config.latent_dim;
        let embed_dim = config.speaker_embed_dim;
        let time_dim = config.time_embed_dim;

        let bottleneck =
            BottleneckEncoder::new(latent_dim, config.bottleneck_dim, vb.pp("bottleneck"))?;
        let speaker_encoder = SpeakerEncoder::new(latent_dim, embed_dim, vb.pp("speaker_encoder"))?;
        let time_embed = TimeEmbed::new(time_dim, vb.pp("time_embed"))?;
        let cond_mlp = CondMlp::new(embed_dim + time_dim, latent_dim, vb.pp("cond_mlp"))?;

        let mut blocks = Vec::with_capacity(config.n_conv_blocks);
        for i in 0..config.n_conv_blocks {
            let blk =
                CausalResBlock::new(latent_dim, config.hidden_dim, vb.pp(&format!("blocks.{i}")))?;
            blocks.push(blk);
        }
        let vel_proj = CausalConv1d::new(latent_dim, latent_dim, 1, 1, vb.pp("vel_proj"))?;

        let (timbre_bank, cross_attns) = if config.enable_timbre {
            let bank = TimbreTokenBank::new(embed_dim, config.n_timbre_tokens, vb.pp("timbre"))?;
            let mut attns = Vec::new();
            for i in 0..config.n_conv_blocks {
                let attn = CrossAttnBlock::new(
                    latent_dim,
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
            bottleneck,
            speaker_encoder,
            time_embed,
            cond_mlp,
            blocks,
            vel_proj,
            timbre_bank,
            cross_attns,
        })
    }

    /// Compute conditioning (FiLM params) from reference + time.
    fn compute_conditioning(&self, ref_latent: &Tensor, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let speaker_embed = self.speaker_encoder.forward(ref_latent)?;
        let time_embed = self.time_embed.forward(t)?;
        let cond = Tensor::cat(&[&speaker_embed, &time_embed], D::Minus1)?;
        let (gamma, beta) = self.cond_mlp.forward(&cond)?;
        // [B, latent_dim] → [B, latent_dim, 1]
        let gamma = gamma.unsqueeze(D::Minus1)?;
        let beta = beta.unsqueeze(D::Minus1)?;
        Ok((gamma, beta))
    }

    /// Predict velocity field. Used during training (Python side).
    /// At inference, use `convert()` instead.
    pub fn forward_velocity(
        &self,
        z_t: &Tensor,
        t: &Tensor,
        ref_latent: &Tensor,
    ) -> Result<Tensor> {
        let content = self.bottleneck.forward(z_t)?;
        let (gamma, beta) = self.compute_conditioning(ref_latent, t)?;
        let z = gamma.broadcast_mul(&content)?;
        let z = beta.broadcast_add(&z)?;

        let timbre = if self.config.enable_timbre {
            if let Some(bank) = &self.timbre_bank {
                let speaker_embed = self.speaker_encoder.forward(ref_latent)?;
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

        self.vel_proj.forward(&z)
    }

    /// One-step inference (mean-flow, 1-NFE).
    ///
    /// `z_converted = z_src + v_pred(z_src, t=1, ref)`
    pub fn convert(&self, z_src: &Tensor, ref_latent: &Tensor) -> Result<Tensor> {
        let batch = z_src.dim(0)?;
        let device = z_src.device();
        let t = Tensor::ones((batch,), candle_core::DType::F32, device)?;
        let v = self.forward_velocity(z_src, &t, ref_latent)?;
        Ok((z_src + &v)?)
    }

    pub fn speaker_embedding(&self, ref_latent: &Tensor) -> Result<Tensor> {
        self.speaker_encoder.forward(ref_latent)
    }
}

// ---------------------------------------------------------------------------
// Model enum: load either Converter or FlowConverter based on config
// ---------------------------------------------------------------------------

/// Either a warm-start Converter or a flow-matching FlowConverter.
/// The variant is determined by `model_type` in the JSON config.
pub enum AnyConverter {
    Warm(Converter),
    Flow(FlowConverter),
}

impl AnyConverter {
    /// Load from a config + VarBuilder. Selects variant based on `model_type`.
    pub fn new(config: ConverterConfig, vb: VarBuilder) -> Result<Self> {
        match config.model_type.as_str() {
            "flow" => Ok(Self::Flow(FlowConverter::new(config, vb)?)),
            _ => Ok(Self::Warm(Converter::new(config, vb)?)),
        }
    }

    /// One-step forward conversion.
    pub fn convert(&self, src_latent: &Tensor, ref_latent: &Tensor) -> Result<Tensor> {
        match self {
            Self::Warm(c) => c.forward(src_latent, ref_latent),
            Self::Flow(c) => c.convert(src_latent, ref_latent),
        }
    }

    pub fn speaker_embedding(&self, ref_latent: &Tensor) -> Result<Tensor> {
        match self {
            Self::Warm(c) => c.speaker_embedding(ref_latent),
            Self::Flow(c) => c.speaker_embedding(ref_latent),
        }
    }
}
