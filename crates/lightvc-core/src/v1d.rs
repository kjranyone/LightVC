//! V1D — additive complex residual on top of the harmonic prior.
//!
//! Mirrors `training/v1d.py`. Keys match PyTorch exactly (`CLAUDE.md`):
//! `inp.{weight,bias}` / `blocks.{i}.dw.{weight,bias}` / `blocks.{i}.norm.*` /
//! `blocks.{i}.pw1.*` / `blocks.{i}.pw2.*` / `norm.*` / `out.*`.
//!
//! Left padding only — zero lookahead. Phase comes from the prior; the trunk
//! never predicts it freely.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

pub const N_MEL: usize = 80;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V1dCfg {
    pub dim: usize,
    pub layers: usize,
    pub k_in: usize,
    pub k: usize,
    pub nbin: usize,
}

impl V1dCfg {
    pub const DEFAULT: V1dCfg = V1dCfg { dim: 256, layers: 6, k_in: 7, k: 3, nbin: 257 };

    #[inline]
    pub fn cin(&self) -> usize {
        N_MEL + 3 * self.nbin
    }

    /// Left context of the trunk, in synthesis frames.
    #[inline]
    pub fn ctx(&self) -> usize {
        (self.k_in - 1) + self.layers * (self.k - 1)
    }
}

fn conv1d_plain(i: usize, o: usize, k: usize, vb: VarBuilder) -> Result<Conv1d> {
    let w = vb.get((o, i, k), "weight")?;
    let b = vb.get((o,), "bias")?;
    Ok(Conv1d::new(w, Some(b), Conv1dConfig::default()))
}

fn linear(i: usize, o: usize, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get((o, i), "weight")?;
    let b = vb.get((o,), "bias")?;
    Ok(Linear::new(w, Some(b)))
}

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let w = vb.get((dim,), "weight")?;
    let b = vb.get((dim,), "bias")?;
    Ok(LayerNorm::new(w, b, 1e-5))
}

struct Block {
    dw: Conv1d,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
    k: usize,
}

impl Block {
    fn new(cfg: &V1dCfg, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dw: conv1d_plain(cfg.dim, cfg.dim, cfg.k, vb.pp("dw"))?,
            norm: layer_norm(cfg.dim, vb.pp("norm"))?,
            pw1: linear(cfg.dim, cfg.dim * 3, vb.pp("pw1"))?,
            pw2: linear(cfg.dim * 3, cfg.dim, vb.pp("pw2"))?,
            k: cfg.k,
        })
    }

    /// `h` already carries the k-1 frames of left context; returns [B,dim,T].
    fn tail(&self, h: &Tensor, r: &Tensor) -> Result<Tensor> {
        let y = self.dw.forward(h)?;
        let y = y.transpose(1, 2)?.contiguous()?;
        let y = self.norm.forward(&y)?;
        let y = self.pw2.forward(&self.pw1.forward(&y)?.gelu_erf()?)?;
        r + &y.transpose(1, 2)?
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.pad_with_zeros(D::Minus1, self.k - 1, 0)?;
        self.tail(&h, x)
    }
}

pub struct V1d {
    cfg: V1dCfg,
    inp: Conv1d,
    blocks: Vec<Block>,
    norm: LayerNorm,
    out: Linear,
}

impl V1d {
    pub fn load(cfg: V1dCfg, vb: VarBuilder) -> Result<Self> {
        let inp = conv1d_plain(cfg.cin(), cfg.dim, cfg.k_in, vb.pp("inp"))?;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            blocks.push(Block::new(&cfg, vb.pp(format!("blocks.{i}")))?);
        }
        let norm = layer_norm(cfg.dim, vb.pp("norm"))?;
        let out = linear(cfg.dim, 2 * cfg.nbin, vb.pp("out"))?;
        Ok(Self { cfg, inp, blocks, norm, out })
    }

    pub fn cfg(&self) -> V1dCfg {
        self.cfg
    }

    /// Trunk. `x`: [B, cin, T] -> [B, 2*nbin, T]. Canonical signature.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.pad_with_zeros(D::Minus1, self.cfg.k_in - 1, 0)?;
        let mut h = self.inp.forward(&h)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        let h = h.transpose(1, 2)?.contiguous()?;
        let h = self.out.forward(&self.norm.forward(&h)?)?;
        h.transpose(1, 2)
    }

    /// Concatenate the trunk input the same way `V1D.features` does.
    pub fn features(mel: &Tensor, pre: &Tensor, pim: &Tensor, plog: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[mel, pre, pim, plog], 1)
    }

    /// Additive complex residual. Returns (real, imag), each [B, nbin, T].
    pub fn apply(&self, x: &Tensor, pre: &Tensor, pim: &Tensor) -> Result<(Tensor, Tensor)> {
        let o = self.forward(x)?;
        let nb = self.cfg.nbin;
        Ok((
            (pre + o.narrow(1, 0, nb)?)?,
            (pim + o.narrow(1, nb, nb)?)?,
        ))
    }
}

/// Per-layer left-context caches (block execution). Mirrors `V1DStream`.
pub struct V1dStream {
    c_in: Tensor,
    c_blk: Vec<Tensor>,
}

impl V1dStream {
    pub fn new(net: &V1d, dev: &candle_core::Device) -> Result<Self> {
        let c = net.cfg;
        Ok(Self {
            c_in: Tensor::zeros((1, c.cin(), c.k_in - 1), candle_core::DType::F32, dev)?,
            c_blk: (0..c.layers)
                .map(|_| Tensor::zeros((1, c.dim, c.k - 1), candle_core::DType::F32, dev))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    /// Emit `T` frames; advances state. `x`: [1, cin, T].
    pub fn step(&mut self, net: &V1d, x: &Tensor, pre: &Tensor, pim: &Tensor)
        -> Result<(Tensor, Tensor)>
    {
        let c = net.cfg;
        let h = Tensor::cat(&[&self.c_in, x], D::Minus1)?;
        let n = h.dim(D::Minus1)?;
        self.c_in = h.narrow(D::Minus1, n - (c.k_in - 1), c.k_in - 1)?.contiguous()?;
        let mut y = net.inp.forward(&h)?;
        for (i, b) in net.blocks.iter().enumerate() {
            let hh = Tensor::cat(&[&self.c_blk[i], &y], D::Minus1)?;
            let m = hh.dim(D::Minus1)?;
            self.c_blk[i] = hh.narrow(D::Minus1, m - (c.k - 1), c.k - 1)?.contiguous()?;
            let r = hh.narrow(D::Minus1, c.k - 1, m - (c.k - 1))?.contiguous()?;
            y = b.tail(&hh, &r)?;
        }
        let y = y.transpose(1, 2)?.contiguous()?;
        let o = net.out.forward(&net.norm.forward(&y)?)?.transpose(1, 2)?;
        let nb = c.nbin;
        Ok(((pre + o.narrow(1, 0, nb)?)?, (pim + o.narrow(1, nb, nb)?)?))
    }
}

// ---------------------------------------------------------------------------
// C.3 parity tests (torch ≡ candle ≤ 1e-4, streaming ≡ offline ≥ 80 dB)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata")
    }

    fn read_f32(name: &str) -> Vec<f32> {
        let b = std::fs::read(dir().join(name)).expect("testdata");
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn load(dev: &Device) -> (V1d, V1dCfg) {
        let cfg = V1dCfg::DEFAULT;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir().join("v1d_test.safetensors")], DType::F32, dev).unwrap()
        };
        (V1d::load(cfg, vb).unwrap(), cfg)
    }

    #[test]
    fn torch_parity() {
        let dev = Device::Cpu;
        let (net, cfg) = load(&dev);
        let t = 32usize;
        let x = Tensor::from_vec(read_f32("tv_x.bin"), (1, cfg.cin(), t), &dev).unwrap();
        let pre = Tensor::from_vec(read_f32("tv_pre.bin"), (1, cfg.nbin, t), &dev).unwrap();
        let pim = Tensor::from_vec(read_f32("tv_pim.bin"), (1, cfg.nbin, t), &dev).unwrap();
        let (re, im) = net.apply(&x, &pre, &pim).unwrap();
        for (got, want) in [(re, "tv_re.bin"), (im, "tv_im.bin")] {
            let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let w = read_f32(want);
            let e = g.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(e <= 1e-4, "torch != candle: max abs err {e} ({want})");
        }
    }

    #[test]
    fn streaming_equals_offline() {
        let dev = Device::Cpu;
        let (net, cfg) = load(&dev);
        let t = 32usize;
        let x = Tensor::from_vec(read_f32("tv_x.bin"), (1, cfg.cin(), t), &dev).unwrap();
        let pre = Tensor::from_vec(read_f32("tv_pre.bin"), (1, cfg.nbin, t), &dev).unwrap();
        let pim = Tensor::from_vec(read_f32("tv_pim.bin"), (1, cfg.nbin, t), &dev).unwrap();
        let (fre, _) = net.apply(&x, &pre, &pim).unwrap();

        let mut st = V1dStream::new(&net, &dev).unwrap();
        let mut parts = Vec::new();
        let blk = 2usize;
        for i in (0..t).step_by(blk) {
            let xs = x.narrow(D::Minus1, i, blk).unwrap();
            let ps = pre.narrow(D::Minus1, i, blk).unwrap();
            let qs = pim.narrow(D::Minus1, i, blk).unwrap();
            parts.push(st.step(&net, &xs, &ps, &qs).unwrap().0);
        }
        let sre = Tensor::cat(&parts, D::Minus1).unwrap();

        // ⚠ 頭 (NFFT_S - HOP_S)/HOP_S = 3 合成フレームを除外して測る（C.3）。
        let skip = 3usize;
        let a = sre.narrow(D::Minus1, skip, t - skip).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = fre.narrow(D::Minus1, skip, t - skip).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut e, mut s) = (0f64, 0f64);
        for (p, q) in a.iter().zip(&b) {
            e += ((p - q) as f64).powi(2);
            s += (*q as f64).powi(2);
        }
        let snr = 10.0 * (s / e.max(1e-30)).log10();
        assert!(snr >= 80.0, "streaming != offline: SNR {snr:.2} dB");
    }

    #[test]
    fn zero_lookahead() {
        // 時刻 c 以降の入力を書き換えても、c 以前の出力は 1 bit も動かない。
        let dev = Device::Cpu;
        let (net, cfg) = load(&dev);
        let t = 32usize;
        let x = Tensor::from_vec(read_f32("tv_x.bin"), (1, cfg.cin(), t), &dev).unwrap();
        let pre = Tensor::from_vec(read_f32("tv_pre.bin"), (1, cfg.nbin, t), &dev).unwrap();
        let pim = Tensor::from_vec(read_f32("tv_pim.bin"), (1, cfg.nbin, t), &dev).unwrap();
        let base = net.apply(&x, &pre, &pim).unwrap().0;

        let c = 20usize;
        let head = x.narrow(D::Minus1, 0, c).unwrap();
        let tail = Tensor::ones((1, cfg.cin(), t - c), DType::F32, &dev).unwrap();
        let x2 = Tensor::cat(&[&head, &tail], D::Minus1).unwrap();
        let alt = net.apply(&x2, &pre, &pim).unwrap().0;

        let a = base.narrow(D::Minus1, 0, c).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = alt.narrow(D::Minus1, 0, c).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let e = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(e == 0.0, "先読みがある: 頭 {c} フレームが {e} 動いた");
    }
}
