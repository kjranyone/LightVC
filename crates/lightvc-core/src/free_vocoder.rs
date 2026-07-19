//! FreeVocoder — F0-free ISTFT-head neural vocoder, Candle port.
//!
//! Numerically matches `training/free_vocoder.py::FreeVocoder`. The synthesis
//! grid (`nfft`/`win`/`hop`) and the `causal` flag are configurable so the same
//! code serves both shipped lattices:
//!   * `freebig` — non-causal, `nfft=2048, hop=512, win=2048, NB=1025`.
//!   * `freeC`   — causal (low latency), `nfft=256, hop=128, win=256, NB=129`.
//! Fixed dims: `dim=512, n_layers=8, n_mels=128, kernel=7`.
//! state_dict keys match PyTorch exactly (`embed`, `blocks.{i}.{dw,norm,pw1,pw2}`,
//! `norm`, `head`). The `window` buffer is not loaded (Hann is regenerated here).
//!
//! candle-core 0.10 has no FFT / complex dtype, so the `torch.istft(center=True)`
//! head is reimplemented as a matrix inverse-DFT + overlap-add with the NOLA
//! window-squared normalization. A causal frame-by-frame streaming path
//! (`StreamState` / `step`) reconstructs with a causal OLA (no center trim, no
//! future lookahead) for real-time deployment.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

const N_MELS: usize = 128;
const DIM: usize = 512;
const N_LAYERS: usize = 8;
const K: usize = 7; // conv kernel (embed + depthwise)

/// Synthesis lattice + causality. `nb = nfft/2 + 1`.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub nfft: usize,
    pub win: usize,
    pub hop: usize,
    pub causal: bool,
}

impl Grid {
    pub const FREEBIG: Grid = Grid { nfft: 2048, win: 2048, hop: 512, causal: false };
    pub const FREEC: Grid = Grid { nfft: 256, win: 256, hop: 128, causal: true };

    #[inline]
    pub fn nb(&self) -> usize {
        self.nfft / 2 + 1
    }
}

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
// ConvNeXtBlock1d, matches kansei_vocoder.ConvNeXtBlock1d (causal-capable)
// ---------------------------------------------------------------------------

struct ConvNeXtBlock1d {
    dw: Conv1d,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
    causal: bool,
}

impl ConvNeXtBlock1d {
    fn new(vb: VarBuilder, causal: bool) -> Result<Self> {
        // dw: Conv1d(dim, dim, 7, groups=1, padding=0) — manual pad in forward.
        let dw = conv1d_plain(DIM, DIM, K, Conv1dConfig::default(), vb.pp("dw"))?;
        let norm = layer_norm(DIM, vb.pp("norm"))?;
        let pw1 = linear(DIM, DIM * 3, vb.pp("pw1"))?;
        let pw2 = linear(DIM * 3, DIM, vb.pp("pw2"))?;
        Ok(Self { dw, norm, pw1, pw2, causal })
    }

    /// Pointwise tail (LN + MLP), shared by offline and streaming. `h`:[B,dim,T].
    fn pointwise(&self, h: &Tensor) -> Result<Tensor> {
        let h = h.transpose(1, 2)?.contiguous()?; // [B, T, dim]
        let h = self.norm.forward(&h)?;
        let h = self.pw1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.pw2.forward(&h)?;
        h.transpose(1, 2) // [B, dim, T]
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // r = x; causal pad (k-1, 0), non-causal pad (k/2, k-1-k/2) = (3, 3).
        let r = x;
        let h = if self.causal {
            x.pad_with_zeros(D::Minus1, K - 1, 0)?
        } else {
            x.pad_with_zeros(D::Minus1, K / 2, K - 1 - K / 2)?
        };
        let h = self.dw.forward(&h)?; // [B, dim, T]
        let h = self.pointwise(&h)?;
        r + &h
    }
}

// ---------------------------------------------------------------------------
// FreeVocoder
// ---------------------------------------------------------------------------

pub struct FreeVocoder {
    grid: Grid,
    device: Device,
    embed: Conv1d,
    blocks: Vec<ConvNeXtBlock1d>,
    norm: LayerNorm,
    head: Linear,
    // iSTFT constant tables (all f32, computed in f64).
    cos_mat: Tensor,  // [NB, WIN]  cos(2*pi*k*n/NFFT)
    sin_mat: Tensor,  // [NB, WIN]  sin(2*pi*k*n/NFFT)
    wk: Tensor,       // [1, NB]    irfft one-sided weights (folds 1/NFFT scale)
    window: Vec<f32>, // [WIN] Hann (periodic)
    win2: Vec<f64>,   // [WIN] window^2 (NOLA denominator term)
}

impl FreeVocoder {
    pub fn new(vb: VarBuilder, grid: Grid, device: &Device) -> Result<Self> {
        let embed = conv1d_plain(N_MELS, DIM, K, Conv1dConfig::default(), vb.pp("embed"))?;
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            blocks.push(ConvNeXtBlock1d::new(vb.pp(format!("blocks.{i}")), grid.causal)?);
        }
        let norm = layer_norm(DIM, vb.pp("norm"))?;
        let head = linear(DIM, 2 * grid.nb(), vb.pp("head"))?;

        let (nfft, win, nb) = (grid.nfft, grid.win, grid.nb());
        // Inverse-DFT tables: y_frame[n] = sum_k Wk*(re[k]*cos - im[k]*sin),
        // n in 0..WIN, k in 0..NB. Wk = 2/NFFT (interior), 1/NFFT (DC & Nyquist).
        let mut cos_v = vec![0f32; nb * win];
        let mut sin_v = vec![0f32; nb * win];
        let two_pi = 2.0f64 * std::f64::consts::PI;
        for k in 0..nb {
            for n in 0..win {
                let theta = two_pi * (k as f64) * (n as f64) / (nfft as f64);
                cos_v[k * win + n] = theta.cos() as f32;
                sin_v[k * win + n] = theta.sin() as f32;
            }
        }
        let mut wk_v = vec![(2.0f64 / nfft as f64) as f32; nb];
        wk_v[0] = (1.0f64 / nfft as f64) as f32;
        wk_v[nb - 1] = (1.0f64 / nfft as f64) as f32;

        let window: Vec<f32> = (0..win)
            .map(|n| (0.5 - 0.5 * (two_pi * n as f64 / win as f64).cos()) as f32)
            .collect();
        let win2: Vec<f64> = window.iter().map(|&w| (w as f64) * (w as f64)).collect();

        let cos_mat = Tensor::from_vec(cos_v, (nb, win), device)?;
        let sin_mat = Tensor::from_vec(sin_v, (nb, win), device)?;
        let wk = Tensor::from_vec(wk_v, (1, nb), device)?;

        Ok(Self {
            grid,
            device: device.clone(),
            embed,
            blocks,
            norm,
            head,
            cos_mat,
            sin_mat,
            wk,
            window,
            win2,
        })
    }

    #[inline]
    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// Backbone: mel[1,128,T] -> spectral frames re,im each [T, NB].
    fn backbone(&self, mel: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = if self.grid.causal {
            mel.pad_with_zeros(D::Minus1, K - 1, 0)?
        } else {
            mel.pad_with_zeros(D::Minus1, K / 2, K - 1 - K / 2)?
        };
        let mut x = self.embed.forward(&x)?; // [1, dim, T]
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        let x = x.transpose(1, 2)?.contiguous()?; // [1, T, dim]
        let x = self.norm.forward(&x)?;
        let h = self.head.forward(&x)?; // [1, T, 2*NB]

        let nb = self.grid.nb();
        let mag = h.narrow(D::Minus1, 0, nb)?.exp()?.clamp(0.0, 1e2)?; // [1,T,NB]
        let p = h.narrow(D::Minus1, nb, nb)?; // [1,T,NB]
        let re = (&mag * p.cos()?)?;
        let im = (&mag * p.sin()?)?;
        let t = re.dim(1)?;
        Ok((re.reshape((t, nb))?, im.reshape((t, nb))?))
    }

    /// re,im: [T, NB] -> y_frame: [T][WIN] (windowed inverse-DFT, pre-OLA).
    fn synth_frames(&self, re: &Tensor, im: &Tensor) -> Result<Vec<Vec<f32>>> {
        let re_w = re.broadcast_mul(&self.wk)?; // [T, NB]
        let im_w = im.broadcast_mul(&self.wk)?;
        let y_cos = re_w.matmul(&self.cos_mat)?; // [T, WIN]
        let y_sin = im_w.matmul(&self.sin_mat)?;
        let y_frame = (y_cos - y_sin)?; // [T, WIN]
        y_frame.to_vec2::<f32>()
    }

    /// Offline forward: mel:[1,128,T] -> wave:[1, hop*(T-1)] (center=True iSTFT).
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (re, im) = self.backbone(mel)?;
        let t = re.dim(0)?;
        let frames = self.synth_frames(&re, &im)?;
        self.istft_ola_center(&frames, t)
    }

    /// Offline center=True overlap-add + NOLA normalization + center trim.
    fn istft_ola_center(&self, frames: &[Vec<f32>], t: usize) -> Result<Tensor> {
        let (win, hop, nfft) = (self.grid.win, self.grid.hop, self.grid.nfft);
        let total = win + hop * (t - 1);
        let mut ola = vec![0f64; total];
        let mut env = vec![0f64; total];
        for (f, frame) in frames.iter().enumerate() {
            let off = f * hop;
            for n in 0..win {
                ola[off + n] += (frame[n] as f64) * (self.window[n] as f64);
                env[off + n] += self.win2[n];
            }
        }
        // center=True: drop NFFT/2 from the front, keep hop*(T-1) samples.
        let start = nfft / 2;
        let out_len = hop * (t - 1);
        let mut out = vec![0f32; out_len];
        for i in 0..out_len {
            let e = env[start + i];
            out[i] = if e > 1e-11 { (ola[start + i] / e) as f32 } else { 0.0 };
        }
        Tensor::from_vec(out, (1, out_len), &self.device)
    }

    // -----------------------------------------------------------------------
    // Streaming (causal OLA, frame-by-frame). Requires grid.causal == true.
    // -----------------------------------------------------------------------

    /// Fresh streaming state (zeroed conv left-context + OLA ring buffers).
    pub fn new_stream(&self) -> Result<StreamState> {
        let embed_ctx = Tensor::zeros((1, N_MELS, K - 1), DType::F32, &self.device)?;
        let mut block_ctx = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            block_ctx.push(Tensor::zeros((1, DIM, K - 1), DType::F32, &self.device)?);
        }
        Ok(StreamState {
            embed_ctx,
            block_ctx,
            ola: vec![0f64; self.grid.win],
            env: vec![0f64; self.grid.win],
        })
    }

    /// Push one mel frame (`[1,128,1]` or `[128]`), emit `hop` finalized samples.
    ///
    /// Causal OLA: no center trim, no future lookahead. Output sample `j` is the
    /// causal-framing reconstruction at absolute position `frame_idx*hop + j`.
    pub fn step(&self, st: &mut StreamState, mel_frame: &Tensor) -> Result<Vec<f32>> {
        assert!(self.grid.causal, "streaming requires a causal grid");
        let frame = mel_frame.reshape((1, N_MELS, 1))?;

        // embed: conv over [ctx(K-1) | frame(1)] -> 1 output frame.
        let inp = Tensor::cat(&[&st.embed_ctx, &frame], D::Minus1)?; // [1,128,K]
        let mut x = self.embed.forward(&inp)?; // [1,dim,1]
        st.embed_ctx = inp.narrow(D::Minus1, 1, K - 1)?.contiguous()?;

        // ConvNeXt blocks.
        for (i, b) in self.blocks.iter().enumerate() {
            let r = x.clone();
            let ci = Tensor::cat(&[&st.block_ctx[i], &x], D::Minus1)?; // [1,dim,K]
            let h = b.dw.forward(&ci)?; // [1,dim,1]
            let h = b.pointwise(&h)?;
            st.block_ctx[i] = ci.narrow(D::Minus1, 1, K - 1)?.contiguous()?;
            x = (r + h)?;
        }

        // norm + head on the single frame.
        let x = x.transpose(1, 2)?.contiguous()?; // [1,1,dim]
        let x = self.norm.forward(&x)?;
        let h = self.head.forward(&x)?; // [1,1,2*NB]
        let nb = self.grid.nb();
        let mag = h.narrow(D::Minus1, 0, nb)?.exp()?.clamp(0.0, 1e2)?;
        let p = h.narrow(D::Minus1, nb, nb)?;
        let re = (&mag * p.cos()?)?.reshape((1, nb))?;
        let im = (&mag * p.sin()?)?.reshape((1, nb))?;
        let y = self.synth_frames(&re, &im)?; // [1][WIN]
        let y = &y[0];

        // Causal OLA into the rolling [win] buffer, emit leading hop samples.
        let (win, hop) = (self.grid.win, self.grid.hop);
        for n in 0..win {
            st.ola[n] += (y[n] as f64) * (self.window[n] as f64);
            st.env[n] += self.win2[n];
        }
        let mut out = vec![0f32; hop];
        for j in 0..hop {
            let e = st.env[j];
            out[j] = if e > 1e-11 { (st.ola[j] / e) as f32 } else { 0.0 };
        }
        // Roll left by hop (advance to next frame origin).
        for n in 0..(win - hop) {
            st.ola[n] = st.ola[n + hop];
            st.env[n] = st.env[n + hop];
        }
        for n in (win - hop)..win {
            st.ola[n] = 0.0;
            st.env[n] = 0.0;
        }
        Ok(out)
    }

    /// Convenience: run all mel frames through the streaming path, concatenating
    /// the `hop`-sized emissions. Returns `[1, hop*T]`.
    pub fn stream_all(&self, mel: &Tensor) -> Result<Tensor> {
        let t = mel.dim(D::Minus1)?;
        let mut st = self.new_stream()?;
        let mut out: Vec<f32> = Vec::with_capacity(t * self.grid.hop);
        for f in 0..t {
            let frame = mel.narrow(D::Minus1, f, 1)?;
            out.extend_from_slice(&self.step(&mut st, &frame)?);
        }
        let n = out.len();
        Tensor::from_vec(out, (1, n), &self.device)
    }
}

/// Streaming state: conv left-context tensors + causal-OLA ring buffers.
pub struct StreamState {
    embed_ctx: Tensor,        // [1, 128, K-1]
    block_ctx: Vec<Tensor>,   // N_LAYERS x [1, dim, K-1]
    ola: Vec<f64>,            // [win]
    env: Vec<f64>,            // [win]
}

// ---------------------------------------------------------------------------
// Convenience loaders
// ---------------------------------------------------------------------------

impl FreeVocoder {
    pub fn from_safetensors_with_grid(
        path: &std::path::Path,
        grid: Grid,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(vb, grid, device)
    }

    /// Backwards-compatible: loads the `freebig` (non-causal 2048/512) lattice.
    pub fn from_safetensors(path: &std::path::Path, device: &Device) -> Result<Self> {
        Self::from_safetensors_with_grid(path, Grid::FREEBIG, device)
    }
}
