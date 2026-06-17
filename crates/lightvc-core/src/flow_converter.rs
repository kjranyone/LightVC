//! Phase C flow-matching converter and the `AnyConverter` dispatch enum.
//!
//! Split from `converter.rs` so that the Phase 1 warm-start `Converter` and
//! the Phase C `FlowConverter` live in separate modules. `AnyConverter`
//! selects between them at load time based on `ConverterConfig::model_type`.

use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::VarBuilder;

use crate::converter::{
    linear_layer, CausalConv1d, CausalResBlock, Converter, ConverterConfig, Snake1d,
    SpeakerEncoder, TimbreTokenBank, CrossAttnBlock,
};

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
            .map(|i| (1.0f64 / 10000.0f64.powf(i as f64 / half as f64)) as f32)
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
        let h = h.gelu_erf()?;
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
        let h = h.gelu_erf()?;
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
                CausalResBlock::new(latent_dim, config.hidden_dim, vb.pp(format!("blocks.{i}")))?;
            blocks.push(blk);
        }
        let vel_proj = CausalConv1d::new(latent_dim, latent_dim, 1, 1, vb.pp("vel_proj"))?;

        let (timbre_bank, cross_attns) = if config.enable_timbre {
            let bank = TimbreTokenBank::new(embed_dim, config.n_timbre_tokens, vb.pp("timbre"))?;
            let mut attns = Vec::new();
            for i in 0..config.n_conv_blocks {
                let attn = CrossAttnBlock::new(
                    latent_dim,
                    embed_dim,
                    config.n_attn_heads,
                    vb.pp(format!("xattn.{i}")),
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

    fn compute_conditioning(&self, ref_latent: &Tensor, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let speaker_embed = self.speaker_encoder.forward(ref_latent)?;
        let time_embed = self.time_embed.forward(t)?;
        let cond = Tensor::cat(&[&speaker_embed, &time_embed], D::Minus1)?;
        let (gamma, beta) = self.cond_mlp.forward(&cond)?;
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
            if let Some((keys, vals)) = timbre.as_ref()
                && i < self.cross_attns.len()
            {
                z = self.cross_attns[i].forward(&z, keys, vals)?;
            }
        }

        self.vel_proj.forward(&z)
    }

    /// One-step inference (mean-flow, 1-NFE).
    ///
    /// `z_converted = z_src + v_pred(z_src, t=1, ref)`
    ///
    /// Accepts both batched `[B, D, T]` and unbatched `[D, T]` inputs,
    /// matching the Python `FlowConverter.convert` ([08-6]).
    pub fn convert(
        &self,
        z_src: &Tensor,
        ref_latent: &Tensor,
        velocity_scale: f64,
    ) -> Result<Tensor> {
        let was_unbatched = z_src.rank() == 2;
        let (z_src, ref_latent) = if was_unbatched {
            (z_src.unsqueeze(0)?, ref_latent.unsqueeze(0)?)
        } else {
            (z_src.clone(), ref_latent.clone())
        };
        let batch = z_src.dim(0)?;
        let device = z_src.device();
        let t = Tensor::ones((batch,), candle_core::DType::F32, device)?;
        let v = self.forward_velocity(&z_src, &t, &ref_latent)?;
        let v_scaled = v.affine(velocity_scale, 0.0)?;
        let result = (&z_src + &v_scaled)?;
        if was_unbatched {
            result.squeeze(0).map_err(Into::into)
        } else {
            Ok(result)
        }
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
    pub fn convert(
        &self,
        src_latent: &Tensor,
        ref_latent: &Tensor,
        velocity_scale: f64,
    ) -> Result<Tensor> {
        match self {
            Self::Warm(c) => c.forward(src_latent, ref_latent),
            Self::Flow(c) => c.convert(src_latent, ref_latent, velocity_scale),
        }
    }

    pub fn speaker_embedding(&self, ref_latent: &Tensor) -> Result<Tensor> {
        match self {
            Self::Warm(c) => c.speaker_embedding(ref_latent),
            Self::Flow(c) => c.speaker_embedding(ref_latent),
        }
    }
}
