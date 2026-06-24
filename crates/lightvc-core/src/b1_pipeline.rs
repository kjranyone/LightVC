use anyhow::Result;
use candle_core::{Device, Tensor};
use crate::codec::{DacCodec, DacConfig};
use crate::soft_rvq::SoftRVQ;
use crate::streaming::{ChunkMode, StreamingCodec};
use crate::utte_adapter::UTTEAdapter;
use std::path::Path;
use std::time::Instant;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[derive(Debug, Clone)]
pub struct StageTimings {
    pub encode: Vec<f64>,
    pub q0_rvq: Vec<f64>,
    pub adapter: Vec<f64>,
    pub decode: Vec<f64>,
    pub total: Vec<f64>,
}

impl StageTimings {
    fn new() -> Self {
        Self {
            encode: Vec::new(),
            q0_rvq: Vec::new(),
            adapter: Vec::new(),
            decode: Vec::new(),
            total: Vec::new(),
        }
    }

    pub fn summary(&self) {
        if self.total.is_empty() {
            println!("  (no samples)");
            return;
        }
        let mut total = self.total.clone();
        total.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  total: p50={:.2}ms p95={:.2}ms p99={:.2}ms (n={})",
            percentile(&total, 0.5),
            percentile(&total, 0.95),
            percentile(&total, 0.99),
            total.len(),
        );
    }
}

pub struct B1Streaming {
    streaming: StreamingCodec,
    soft_rvq: SoftRVQ,
    adapter: UTTEAdapter,
    timbre: Option<Tensor>,
    tau: f64,
    pub timings: StageTimings,
}

impl B1Streaming {
    pub fn new(
        dac_weights: &Path,
        quantizer_weights: &Path,
        adapter_weights: &Path,
        mode: ChunkMode,
        device: Device,
    ) -> Result<Self> {
        let streaming = StreamingCodec::new(dac_weights, &DacConfig::default(), mode, device.clone())?;
        let soft_rvq = SoftRVQ::from_varbuilder(
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[quantizer_weights], candle_core::DType::F32, &device)? },
        )?;
        let adapter = UTTEAdapter::from_varbuilder(
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[adapter_weights], candle_core::DType::F32, &device)? },
            crate::utte_adapter::UTTEAdapterConfig::default(),
        )?;
        Ok(Self {
            streaming,
            soft_rvq,
            adapter,
            timbre: None,
            tau: 5.0,
            timings: StageTimings::new(),
        })
    }

    pub fn set_timbre(&mut self, timbre: Tensor) {
        self.timbre = Some(timbre);
    }

    pub fn set_tau(&mut self, tau: f64) {
        self.tau = tau;
    }

    pub fn chunk_mode(&self) -> ChunkMode {
        self.streaming.chunk_mode()
    }

    pub fn reset(&mut self) {
        self.streaming.reset_state();
        self.timings = StageTimings::new();
    }

    pub fn process_chunk(&mut self, chunk_pcm: &[f32]) -> Result<Vec<f32>> {
        let timbre = self
            .timbre
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("timbre not set"))?;

        let t_total = Instant::now();

        let t0 = Instant::now();
        let latent = self.streaming.encode_step(chunk_pcm)?;
        let us_enc = t0.elapsed().as_micros() as f64;

        let frames = latent.dim(2)?;
        if frames == 0 {
            self.timings.encode.push(us_enc);
            self.timings.total.push(us_enc);
            return Ok(Vec::new());
        }

        let t0 = Instant::now();
        let q0 = self.soft_rvq.quantize_q0(&latent)?;
        let z_q = self.soft_rvq.soft_requantize(&q0, &latent, self.tau)?;
        let us_rvq = t0.elapsed().as_micros() as f64;

        let t0 = Instant::now();
        let z_qa = self.adapter.forward(&z_q, timbre)?;
        let us_ad = t0.elapsed().as_micros() as f64;

        let t0 = Instant::now();
        let output = self.streaming.decode_step(&z_qa)?;
        let us_dec = t0.elapsed().as_micros() as f64;

        let us_tot = t_total.elapsed().as_micros() as f64;
        self.timings.encode.push(us_enc);
        self.timings.q0_rvq.push(us_rvq);
        self.timings.adapter.push(us_ad);
        self.timings.decode.push(us_dec);
        self.timings.total.push(us_tot);

        Ok(output)
    }

    pub fn process_full(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let chunk_sz = self.streaming.chunk_mode().samples_per_chunk();
        let mut output = Vec::new();

        let mut pos = 0;
        while pos < pcm.len() {
            let end = (pos + chunk_sz).min(pcm.len());
            let chunk = &pcm[pos..end];
            if chunk.len() < chunk_sz {
                let mut padded = chunk.to_vec();
                padded.resize(chunk_sz, 0.0);
                let out = self.process_chunk(&padded)?;
                let real_frames = chunk.len() / crate::DAC_HOP_LENGTH;
                let real_samples = real_frames * crate::DAC_HOP_LENGTH;
                output.extend_from_slice(&out[..real_samples.min(out.len())]);
            } else {
                let out = self.process_chunk(chunk)?;
                output.extend_from_slice(&out);
            }
            pos += chunk_sz;
        }
        Ok(output)
    }
}

pub struct B1Offline {
    codec: DacCodec,
    soft_rvq: SoftRVQ,
    adapter: UTTEAdapter,
    timbre: Option<Tensor>,
    tau: f64,
}

impl B1Offline {
    pub fn new(
        dac_weights: &Path,
        quantizer_weights: &Path,
        adapter_weights: &Path,
        device: Device,
    ) -> Result<Self> {
        let codec = DacCodec::from_file(dac_weights, &DacConfig::default(), device.clone())?;
        let soft_rvq = SoftRVQ::from_varbuilder(
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[quantizer_weights], candle_core::DType::F32, &device)? },
        )?;
        let adapter = UTTEAdapter::from_varbuilder(
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[adapter_weights], candle_core::DType::F32, &device)? },
            crate::utte_adapter::UTTEAdapterConfig::default(),
        )?;
        Ok(Self { codec, soft_rvq, adapter, timbre: None, tau: 5.0 })
    }

    pub fn set_timbre(&mut self, timbre: Tensor) {
        self.timbre = Some(timbre);
    }

    pub fn process(&self, pcm: &Tensor) -> Result<Tensor> {
        let timbre = self
            .timbre
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("timbre not set"))?;
        let z_s = self.codec.encode(pcm)?;
        let q0 = self.soft_rvq.quantize_q0(&z_s)?;
        let z_q = self.soft_rvq.soft_requantize(&q0, &z_s, self.tau)?;
        let z_qa = self.adapter.forward(&z_q, timbre)?;
        self.codec.decode(&z_qa)
    }
}
