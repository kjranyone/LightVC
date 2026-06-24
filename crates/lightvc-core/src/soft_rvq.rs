use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use candle_nn::ops::softmax;

pub struct SoftRVQ {
    in_projs: Vec<Tensor>,
    in_biases: Vec<Tensor>,
    out_projs: Vec<Tensor>,
    out_biases: Vec<Tensor>,
    codebooks: Vec<Tensor>,
}

impl SoftRVQ {
    pub fn from_varbuilder(vb: VarBuilder) -> Result<Self> {
        let qvb = vb.pp("quantizer").pp("quantizers");
        let mut in_projs = Vec::with_capacity(9);
        let mut in_biases = Vec::with_capacity(9);
        let mut out_projs = Vec::with_capacity(9);
        let mut out_biases = Vec::with_capacity(9);
        let mut codebooks = Vec::with_capacity(9);

        for d in 0..9 {
            let ip = qvb.get((8, 1024, 1), &format!("{d}.in_proj.weight"))?;
            let ib = qvb.get((8,), &format!("{d}.in_proj.bias"))?;
            let op = qvb.get((1024, 8, 1), &format!("{d}.out_proj.weight"))?;
            let ob = qvb.get((1024,), &format!("{d}.out_proj.bias"))?;
            let cb = qvb.get((1024, 8), &format!("{d}.codebook.weight"))?;
            in_projs.push(ip);
            in_biases.push(ib);
            out_projs.push(op);
            out_biases.push(ob);
            codebooks.push(cb);
        }

        Ok(Self { in_projs, in_biases, out_projs, out_biases, codebooks })
    }

    pub fn soft_requantize(
        &self,
        q0_source: &Tensor,
        z_input: &Tensor,
        tau: f64,
    ) -> Result<Tensor> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };

        let mut z_q = q0_source.clone();
        let mut residual = (z_input - q0_source)?;

        for d in 1..9 {
            let conv_in = Conv1d::new(self.in_projs[d].clone(), Some(self.in_biases[d].clone()), cfg);
            let z_e = conv_in.forward(&residual)?;

            let z_t = z_e.permute((0, 2, 1))?;
            let cb = &self.codebooks[d];

            let (b, t, _d) = z_t.dims3()?;
            let z_flat = z_t.reshape((b * t, 8))?;
            let cb_t = cb.t()?;
            let zc = z_flat.matmul(&cb_t)?;
            let zc = zc.reshape((b, t, 1024))?;

            let z_sq = (&z_t * &z_t)?.sum(D::Minus1)?;
            let c_sq = (cb * cb)?.sum(D::Minus1)?;
            let z_sq_e = z_sq.unsqueeze(D::Minus1)?;
            let c_sq_e = c_sq.unsqueeze(0)?;
            let dist = (z_sq_e.broadcast_add(&c_sq_e)? - (&zc * 2.0)?)?;

            let neg_scaled = (dist * (-1.0 / tau))?;
            let weights = softmax(&neg_scaled, D::Minus1)?;

            let w_flat = weights.reshape((b * t, 1024))?;
            let z_soft = w_flat.matmul(cb)?;
            let z_soft = z_soft.reshape((b, t, 8))?.permute((0, 2, 1))?;

            let conv_out = Conv1d::new(self.out_projs[d].clone(), Some(self.out_biases[d].clone()), cfg);
            let q_depth = conv_out.forward(&z_soft)?;

            z_q = (&z_q + &q_depth)?;
            residual = (&residual - &q_depth)?;
        }

        Ok(z_q)
    }

    pub fn quantize_q0(&self, z_s: &Tensor) -> Result<Tensor> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };

        let conv_in = Conv1d::new(self.in_projs[0].clone(), Some(self.in_biases[0].clone()), cfg);
        let z_e = conv_in.forward(z_s)?;
        let z_t = z_e.permute((0, 2, 1))?;
        let cb = &self.codebooks[0];

        let (b, t, _d) = z_t.dims3()?;
        let z_flat = z_t.reshape((b * t, 8))?;
        let cb_t = cb.t()?;
        let zc = z_flat.matmul(&cb_t)?.reshape((b, t, 1024))?;

        let z_sq = (&z_t * &z_t)?.sum(D::Minus1)?;
        let c_sq = (cb * cb)?.sum(D::Minus1)?;
        let z_sq_e = z_sq.unsqueeze(D::Minus1)?;
        let c_sq_e = c_sq.unsqueeze(0)?;
        let dist = (z_sq_e.broadcast_add(&c_sq_e)? - (&zc * 2.0)?)?;
        let codes = dist.argmin(D::Minus1)?;

        let codes_flat = codes.reshape((b * t,))?;
        let quantized = Tensor::embedding(cb, &codes_flat)?;
        let quantized = quantized.reshape((b, t, 8))?.permute((0, 2, 1))?;

        let conv_out = Conv1d::new(self.out_projs[0].clone(), Some(self.out_biases[0].clone()), cfg);
        Ok(conv_out.forward(&quantized)?)
    }
}

pub fn load_soft_rvq(
    dac_weights: &std::path::Path,
    device: &Device,
) -> Result<SoftRVQ> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dac_weights], DType::F32, device)?
    };
    SoftRVQ::from_varbuilder(vb)
}
