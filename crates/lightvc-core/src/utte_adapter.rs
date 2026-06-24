use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

pub struct UTTEAdapterConfig {
    pub latent_dim: usize,
    pub bottleneck: usize,
    pub timbre_dim: usize,
    pub n_tokens: usize,
    pub n_heads: usize,
    pub kernel: usize,
}

impl Default for UTTEAdapterConfig {
    fn default() -> Self {
        Self {
            latent_dim: 1024,
            bottleneck: 256,
            timbre_dim: 192,
            n_tokens: 32,
            n_heads: 4,
            kernel: 3,
        }
    }
}

pub struct UTTEAdapter {
    conv_in: Conv1d,
    conv_out: Conv1d,
    token_mlp: Linear,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    config: UTTEAdapterConfig,
}

fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get((out_dim,), "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

fn conv1d(in_ch: usize, out_ch: usize, kernel: usize, vb: VarBuilder) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        padding: kernel / 2,
        stride: 1,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    let weight = vb.get((out_ch, in_ch, kernel), "weight")?;
    let bias = vb.get((out_ch,), "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

impl UTTEAdapter {
    pub fn from_varbuilder(vb: VarBuilder, config: UTTEAdapterConfig) -> Result<Self> {
        let c = &config;

        let conv_in = conv1d(c.latent_dim, c.bottleneck, c.kernel, vb.pp("conv_in"))?;
        let conv_out = conv1d(c.bottleneck, c.latent_dim, c.kernel, vb.pp("conv_out"))?;

        let token_mlp = linear(c.timbre_dim, c.n_tokens * c.bottleneck, vb.pp("token_mlp"))?;

        let attn_vb = vb.pp("attn");
        let q_proj = linear(c.bottleneck, c.bottleneck, attn_vb.pp("q"))?;
        let k_proj = linear(c.bottleneck, c.bottleneck, attn_vb.pp("k"))?;
        let v_proj = linear(c.bottleneck, c.bottleneck, attn_vb.pp("v"))?;
        let o_proj = linear(c.bottleneck, c.bottleneck, attn_vb.pp("o"))?;

        Ok(Self {
            conv_in,
            conv_out,
            token_mlp,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config,
        })
    }

    pub fn forward(&self, z_q: &Tensor, timbre: &Tensor) -> Result<Tensor> {
        let (b, _, t) = z_q.dims3()?;
        let c = &self.config;
        let head_dim = c.bottleneck / c.n_heads;

        let tokens = self.token_mlp.forward(timbre)?;
        let tokens = tokens.reshape((b, c.n_tokens, c.bottleneck))?;

        let x = self.conv_in.forward(z_q)?;
        let h = x.transpose(1, 2)?;

        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&tokens)?;
        let v = self.v_proj.forward(&tokens)?;

        let q = q
            .reshape((b, t, c.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * c.n_heads, t, head_dim))?;
        let k = k
            .reshape((b, c.n_tokens, c.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * c.n_heads, c.n_tokens, head_dim))?;
        let v = v
            .reshape((b, c.n_tokens, c.n_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * c.n_heads, c.n_tokens, head_dim))?;

        let scale = 1.0f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?;
        let scores = (scores * scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let out = attn.matmul(&v)?;
        let out = out
            .reshape((b, c.n_heads, t, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c.bottleneck))?;

        let out = self.o_proj.forward(&out)?;
        let h = (h + out)?;

        let h = h.gelu_erf()?;
        let h = h.transpose(1, 2)?;

        let delta = self.conv_out.forward(&h)?;
        Ok((z_q + delta)?)
    }
}

pub fn load_adapter(
    weights_path: &std::path::Path,
    device: &Device,
) -> Result<UTTEAdapter> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
    };
    let config = UTTEAdapterConfig::default();
    UTTEAdapter::from_varbuilder(vb, config)
}
