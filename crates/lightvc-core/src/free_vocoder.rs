//! FreeVocoder (`freebig`) — F0-free ISTFT-head neural vocoder, Candle port.
//!
//! Numerically matches `training/free_vocoder.py::FreeVocoder` (non-causal,
//! `nfft=2048, hop=512, win=2048, NB=1025, dim=512, n_layers=8, n_mels=128`).
//! state_dict keys match PyTorch exactly (`embed`, `blocks.{i}.{dw,norm,pw1,pw2}`,
//! `norm`, `head`). The `window` buffer is not loaded (Hann is regenerated here).
//!
//! candle-core 0.10 has no FFT / complex dtype, so the `torch.istft(center=True)`
//! head is reimplemented as a matrix inverse-DFT + overlap-add with the NOLA
//! window-squared normalization (cf. `ltv_render._build_ola_mm`).

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

const N_MELS: usize = 128;
const DIM: usize = 512;
const N_LAYERS: usize = 8;
const NFFT: usize = 2048;
const WIN: usize = 2048;
const HOP: usize = 512;
const NB: usize = NFFT / 2 + 1; // 1025

// ---------------------------------------------------------------------------
// Loader helpers (plain Conv1d / Linear / LayerNorm, PyTorch layout)
// ---------------------------------------------------------------------------

fn conv1d_plain(
    in_ch: usize,
    out_ch: usize,
    k: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let w = vb.get((out_ch, in_ch, k), "weight")?;
    let b = vb.get((out_ch,), "bias")?;
    Ok(Conv1d::new(w, Some(b), cfg))
}

fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    let b = vb.get((out_dim,), "bias")?;
    Ok(Linear::new(w, Some(b)))
}

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let w = vb.get((dim,), "weight")?;
    let b = vb.get((dim,), "bias")?;
    Ok(LayerNorm::new(w, b, 1e-5))
}

// ---------------------------------------------------------------------------
// ConvNeXtBlock1d (non-causal), matches kansei_vocoder.ConvNeXtBlock1d
// ---------------------------------------------------------------------------

struct ConvNeXtBlock1d {
    dw: Conv1d,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
}

impl ConvNeXtBlock1d {
    fn new(vb: VarBuilder) -> Result<Self> {
        // dw: Conv1d(dim, dim, 7, groups=1, padding=0) — manual pad in forward.
        let dw = conv1d_plain(DIM, DIM, 7, Conv1dConfig::default(), vb.pp("dw"))?;
        let norm = layer_norm(DIM, vb.pp("norm"))?;
        let pw1 = linear(DIM, DIM * 3, vb.pp("pw1"))?;
        let pw2 = linear(DIM * 3, DIM, vb.pp("pw2"))?;
        Ok(Self {
            dw,
            norm,
            pw1,
            pw2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // r = x; non-causal pad (k//2, k-1-k//2) = (3, 3)
        let r = x;
        let h = x.pad_with_zeros(D::Minus1, 3, 3)?;
        let h = self.dw.forward(&h)?; // [B, dim, T]
        let h = h.transpose(1, 2)?.contiguous()?; // [B, T, dim]
        let h = self.norm.forward(&h)?;
        let h = self.pw1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.pw2.forward(&h)?;
        let h = h.transpose(1, 2)?; // [B, dim, T]
        r + &h
    }
}

// ---------------------------------------------------------------------------
// FreeVocoder
// ---------------------------------------------------------------------------

pub struct FreeVocoder {
    embed: Conv1d,
    blocks: Vec<ConvNeXtBlock1d>,
    norm: LayerNorm,
    head: Linear,
    // iSTFT constant tables (all f32, computed in f64).
    cos_mat: Tensor,  // [NB, WIN]  cos(2*pi*k*n/NFFT)
    sin_mat: Tensor,  // [NB, WIN]  sin(2*pi*k*n/NFFT)
    wk: Tensor,       // [1, NB]    irfft one-sided weights (folds 1/NFFT scale)
    window: Vec<f32>, // [WIN] Hann (periodic)
}

impl FreeVocoder {
    pub fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let embed = conv1d_plain(N_MELS, DIM, 7, Conv1dConfig::default(), vb.pp("embed"))?;
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            blocks.push(ConvNeXtBlock1d::new(vb.pp(format!("blocks.{i}")))?);
        }
        let norm = layer_norm(DIM, vb.pp("norm"))?;
        let head = linear(DIM, 2 * NB, vb.pp("head"))?;

        // Inverse-DFT tables: y_frame[n] = sum_k Wk*(re[k]*cos - im[k]*sin),
        // n in 0..WIN, k in 0..NB. Wk = 2/NFFT (interior), 1/NFFT (DC & Nyquist).
        // (DC/Nyquist imaginary parts vanish since sin(0)=sin(pi*n)=0, matching
        // torch.fft.irfft's Hermitian assumption.)
        let mut cos_v = vec![0f32; NB * WIN];
        let mut sin_v = vec![0f32; NB * WIN];
        let two_pi = 2.0f64 * std::f64::consts::PI;
        for k in 0..NB {
            for n in 0..WIN {
                let theta = two_pi * (k as f64) * (n as f64) / (NFFT as f64);
                cos_v[k * WIN + n] = theta.cos() as f32;
                sin_v[k * WIN + n] = theta.sin() as f32;
            }
        }
        let mut wk_v = vec![(2.0f64 / NFFT as f64) as f32; NB];
        wk_v[0] = (1.0f64 / NFFT as f64) as f32;
        wk_v[NB - 1] = (1.0f64 / NFFT as f64) as f32;

        let window: Vec<f32> = (0..WIN)
            .map(|n| (0.5 - 0.5 * (two_pi * n as f64 / WIN as f64).cos()) as f32)
            .collect();

        let cos_mat = Tensor::from_vec(cos_v, (NB, WIN), device)?;
        let sin_mat = Tensor::from_vec(sin_v, (NB, WIN), device)?;
        let wk = Tensor::from_vec(wk_v, (1, NB), device)?;

        Ok(Self {
            embed,
            blocks,
            norm,
            head,
            cos_mat,
            sin_mat,
            wk,
            window,
        })
    }

    /// mel: [1, 128, T] -> wave: [1, hop*(T-1)]
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Backbone (non-causal pad (3,3)).
        let x = mel.pad_with_zeros(D::Minus1, 3, 3)?;
        let mut x = self.embed.forward(&x)?; // [1, dim, T]
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        let x = x.transpose(1, 2)?.contiguous()?; // [1, T, dim]
        let x = self.norm.forward(&x)?;
        let h = self.head.forward(&x)?; // [1, T, 2*NB]

        // Split magnitude / phase on the last (2*NB) axis. h is [1, T, 2*NB].
        let mag = h.narrow(D::Minus1, 0, NB)?.exp()?.clamp(0.0, 1e2)?; // [1,T,NB]
        let p = h.narrow(D::Minus1, NB, NB)?; // [1,T,NB]
        let re = (&mag * p.cos()?)?; // [1,T,NB]
        let im = (&mag * p.sin()?)?; // [1,T,NB]

        let t = re.dim(1)?;
        let re = re.reshape((t, NB))?; // [T, NB]
        let im = im.reshape((t, NB))?;

        // Apply one-sided irfft weights, then matrix inverse-DFT to [T, WIN].
        let re_w = re.broadcast_mul(&self.wk)?; // [T, NB]
        let im_w = im.broadcast_mul(&self.wk)?;
        let y_cos = re_w.matmul(&self.cos_mat)?; // [T, WIN]
        let y_sin = im_w.matmul(&self.sin_mat)?; // [T, WIN]
        let y_frame = (y_cos - y_sin)?; // [T, WIN]

        // Overlap-add + NOLA (window^2) normalization + center trim, in plain f32.
        let frames = y_frame.to_vec2::<f32>()?; // [T][WIN]
        self.istft_ola(&frames, t)
    }

    fn istft_ola(&self, frames: &[Vec<f32>], t: usize) -> Result<Tensor> {
        let total = WIN + HOP * (t - 1);
        let mut ola = vec![0f64; total];
        let mut env = vec![0f64; total];
        let win2: Vec<f64> = self
            .window
            .iter()
            .map(|&w| (w as f64) * (w as f64))
            .collect();
        for (f, frame) in frames.iter().enumerate() {
            let off = f * HOP;
            for n in 0..WIN {
                let syn = (frame[n] as f64) * (self.window[n] as f64);
                ola[off + n] += syn;
                env[off + n] += win2[n];
            }
        }
        // center=True: drop NFFT/2 from the front, keep hop*(T-1) samples.
        let start = NFFT / 2;
        let out_len = HOP * (t - 1);
        let mut out = vec![0f32; out_len];
        for i in 0..out_len {
            let e = env[start + i];
            out[i] = if e > 1e-11 {
                (ola[start + i] / e) as f32
            } else {
                0.0
            };
        }
        Tensor::from_vec(out, (1, out_len), &Device::Cpu)
    }
}

// ---------------------------------------------------------------------------
// Convenience loader
// ---------------------------------------------------------------------------

impl FreeVocoder {
    pub fn from_safetensors(path: &std::path::Path, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(vb, device)
    }
}
