//! CLI argument parsing and subcommand dispatch.

use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use lightvc_audio::Resampler;
use lightvc_core::{
    converter::{AnyConverter, ConverterConfig, LatencyMode},
    pipeline::VcPipeline,
    DacConfig, FreeResynth,
};

#[derive(Parser)]
#[command(
    name = "lightvc",
    version,
    about = "LightVC real-time voice conversion"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Validate DAC encode/decode round-trip on a WAV file.
    Roundtrip(RoundtripCmd),
    /// Apply converter to a WAV file (offline processing).
    Convert(ConvertCmd),
    /// Apply B1 UTTE adapter pipeline to a WAV file (offline).
    ConvertB1(ConvertB1Cmd),
    /// FreeVocoder resynthesis (WAV → Rust mel → freeC vocoder → WAV),
    /// streamed through the deployed `chunk_samples()` realtime path.
    Resynth(ResynthCmd),
    /// v2f (harmonic-prior, causal, 30 ms class) resynthesis: WAV → WAV.
    V2f(V2fCmd),
    /// バ美声変換 (front→E→G→V ストリーム): WAV → WAV。
    Vc(VcCmd),
    /// Launch desktop GUI (3 tabs: offline/realtime/catalog).
    Gui(GuiCmd),
}

#[derive(Parser)]
pub struct GuiCmd {
    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: Option<PathBuf>,
    #[arg(long)]
    pub cuda: bool,
    #[arg(long)]
    pub metal: bool,
    /// Launch the GUI with mock data for screenshot capture. No model,
    /// no audio devices required. The user switches tabs inside the app.
    #[arg(long)]
    pub demo: bool,
}

#[derive(Parser)]
pub struct RoundtripCmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(short, long, default_value = "roundtrip_output.wav")]
    pub output: PathBuf,

    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: PathBuf,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub metal: bool,
}

#[derive(Parser)]
pub struct ConvertCmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(short, long)]
    pub reference: PathBuf,

    #[arg(short, long, default_value = "converted_output.wav")]
    pub output: PathBuf,

    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: PathBuf,

    #[arg(long, env = "LIGHTVC_CONVERTER_WEIGHTS")]
    pub converter_weights: PathBuf,

    #[arg(long)]
    pub converter_config: Option<PathBuf>,

    #[arg(
        long,
        default_value = "balanced",
        help = "strict | balanced | quality | full"
    )]
    pub mode: String,

    #[arg(
        long,
        default_value = "1.0",
        help = "Velocity scale (guidance). 1.0 = training-matched, >1 amplifies conversion"
    )]
    pub velocity_scale: f64,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub metal: bool,
}

#[derive(Parser)]
pub struct ConvertB1Cmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(short, long, help = "Pre-computed ECAPA timbre safetensors")]
    pub timbre: PathBuf,

    #[arg(short, long, default_value = "converted_b1.wav")]
    pub output: PathBuf,

    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: PathBuf,

    #[arg(long, default_value = "models/dac_quantizer.safetensors")]
    pub quantizer_weights: PathBuf,

    #[arg(long, default_value = "models/utte_adapter_b1.safetensors")]
    pub adapter_weights: PathBuf,

    #[arg(long, default_value = "balanced", help = "strict | balanced")]
    pub mode: String,

    #[arg(long, default_value = "5.0")]
    pub tau: f64,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub metal: bool,
}

#[derive(Parser)]
pub struct ResynthCmd {
    #[arg(short, long)]
    pub input: PathBuf,

    #[arg(long, env = "LIGHTVC_FREEC_WEIGHTS", help = "freeC vocoder safetensors")]
    pub weights: PathBuf,

    #[arg(
        long,
        env = "LIGHTVC_MEL_BASIS",
        help = "librosa slaney mel filterbank safetensors (key `mel_basis`)"
    )]
    pub mel_basis: PathBuf,

    #[arg(short, long, default_value = "resynth_output.wav")]
    pub output: PathBuf,

    #[arg(long, default_value = "4", help = "Mel frames per streaming chunk (k*hop samples)")]
    pub k: usize,
}

pub fn select_device(cuda: bool, metal: bool) -> Result<Device> {
    if cuda {
        Ok(Device::new_cuda(0)?)
    } else if metal {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn run_roundtrip(cmd: RoundtripCmd) -> Result<()> {
    println!("LightVC Phase 0: DAC Round-Trip Test");
    println!("  Input:  {}", cmd.input.display());
    println!("  Output: {}", cmd.output.display());

    let device = select_device(cmd.cuda, cmd.metal)?;
    println!("  Device: {:?}", device);

    let (input_pcm, input_sr) = load_wav_mono(&cmd.input)?;
    println!(
        "  Loaded: {} samples at {} Hz ({:.1}s)",
        input_pcm.len(),
        input_sr,
        input_pcm.len() as f32 / input_sr as f32
    );

    let pcm_44k = if input_sr != 44_100 {
        println!("  Resampling {} → 44100 Hz...", input_sr);
        resample_to_44100(&input_pcm, input_sr)?
    } else {
        input_pcm
    };

    let padded = lightvc_core::codec::pad_to_hop(pcm_44k);
    println!("  Padded: {} samples", padded.len());

    println!("  Loading DAC weights...");
    let dac_config = DacConfig::default();
    let codec = lightvc_core::DacCodec::from_file(&cmd.dac_weights, &dac_config, device)?;

    println!("  Encoding...");
    let latent = codec.encode_pcm(&padded)?;
    println!("  Latent shape: {:?}", latent.shape());

    println!("  Decoding...");
    let output_pcm = codec.decode_to_pcm(&latent)?;
    println!("  Output: {} samples", output_pcm.len());

    let trim_len = padded.len().min(output_pcm.len());
    let output_trimmed = &output_pcm[..trim_len];

    save_wav_mono(&cmd.output, output_trimmed, 44_100)?;
    println!("  Saved: {}", cmd.output.display());

    let mse: f64 = output_trimmed
        .iter()
        .zip(padded.iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum::<f64>()
        / trim_len as f64;
    println!("  MSE: {:.6}", mse);
    println!("  RMSE: {:.6}", mse.sqrt());

    println!("Done.");
    Ok(())
}

pub fn run_convert(cmd: ConvertCmd) -> Result<()> {
    println!("LightVC Offline Conversion");

    let device = select_device(cmd.cuda, cmd.metal)?;

    let converter_config = if let Some(cfg_path) = &cmd.converter_config {
        let cfg_str = std::fs::read_to_string(cfg_path)?;
        serde_json::from_str(&cfg_str)?
    } else {
        ConverterConfig::default()
    };

    println!("Loading DAC...");
    let dac_config = DacConfig::default();

    let mode = match cmd.mode.as_str() {
        "strict" => LatencyMode::Strict,
        "quality" => LatencyMode::Quality,
        _ => LatencyMode::Balanced,
    };
    let use_full = cmd.mode == "full";

    println!("Loading converter...");
    let vb = lightvc_core::weights::load_varbuilder(&cmd.converter_weights, DType::F32, &device)?;
    let converter = AnyConverter::new(converter_config, vb)?;

    let mut pipeline = VcPipeline::new(&cmd.dac_weights, &dac_config, converter, mode, device)?;
    pipeline.velocity_scale = cmd.velocity_scale;

    println!("Loading reference: {}", cmd.reference.display());
    let (ref_pcm, ref_sr) = load_wav_mono(&cmd.reference)?;
    let ref_44k = if ref_sr != 44_100 {
        resample_to_44100(&ref_pcm, ref_sr)?
    } else {
        ref_pcm
    };
    pipeline.set_target(&ref_44k)?;

    println!("Loading source: {}", cmd.input.display());
    let (src_pcm, src_sr) = load_wav_mono(&cmd.input)?;
    let src_44k = if src_sr != 44_100 {
        resample_to_44100(&src_pcm, src_sr)?
    } else {
        src_pcm
    };
    let padded = lightvc_core::codec::pad_to_hop(src_44k);

    if use_full {
        println!("Mode: full (offline, no chunking) — SOTA quality");
        println!("Processing...");
        let output = pipeline.process_full(&padded)?;
        save_wav_mono(&cmd.output, &output, 44_100)?;
        println!("Saved: {}", cmd.output.display());
        return Ok(());
    }

    let chunk_size = pipeline.chunk_samples();
    println!(
        "Chunk size: {} samples ({:.1} ms)",
        chunk_size,
        pipeline.chunk_ms()
    );
    println!("Processing...");

    let mut output = Vec::with_capacity(padded.len());
    let mut i = 0;
    while i < padded.len() {
        let end = (i + chunk_size).min(padded.len());
        let chunk = &padded[i..end];

        let chunk_padded = if chunk.len() < chunk_size {
            let mut c = chunk.to_vec();
            c.resize(chunk_size, 0.0);
            c
        } else {
            chunk.to_vec()
        };

        let out_chunk = pipeline.process_chunk(&chunk_padded)?;
        output.extend_from_slice(&out_chunk[..end - i]);
        i = end;
    }

    save_wav_mono(&cmd.output, &output, 44_100)?;
    println!("Saved: {}", cmd.output.display());

    Ok(())
}

pub fn run_convert_b1(cmd: ConvertB1Cmd) -> Result<()> {
    println!("LightVC B1 Offline Conversion");
    println!("  Input:  {}", cmd.input.display());
    println!("  Timbre: {}", cmd.timbre.display());
    println!("  Output: {}", cmd.output.display());

    let device = select_device(cmd.cuda, cmd.metal)?;
    println!("  Device: {:?}", device);

    let (input_pcm, input_sr) = load_wav_mono(&cmd.input)?;
    println!(
        "  Loaded: {} samples at {} Hz ({:.1}s)",
        input_pcm.len(),
        input_sr,
        input_pcm.len() as f32 / input_sr as f32
    );

    let pcm_44k = resample_to_44100(&input_pcm, input_sr)?;

    let timbre = {
        let vb = lightvc_core::weights::load_varbuilder(&cmd.timbre, DType::F32, &device)?;
        vb.get((1, 192), "timbre")?
    };

    let chunk_mode = match cmd.mode.as_str() {
        "strict" => {
            eprintln!(
                "  WARNING: Strict mode (1-frame decode) produces C/D quality with B1.\n\
                 \x20          Balanced is recommended. See issue #7."
            );
            lightvc_core::streaming::ChunkMode::Strict
        }
        _ => lightvc_core::streaming::ChunkMode::Balanced,
    };

    let mut b1 = lightvc_core::b1_pipeline::B1Streaming::new(
        &cmd.dac_weights,
        &cmd.quantizer_weights,
        &cmd.adapter_weights,
        chunk_mode,
        device,
    )?;
    b1.set_timbre(timbre);
    b1.set_tau(cmd.tau);

    println!(
        "  Pipeline: B1 UTTE adapter ({}, tau={})",
        cmd.mode, cmd.tau
    );
    println!("  Processing...");
    let output = b1.process_full(&pcm_44k)?;
    println!(
        "  Output: {} samples ({:.1}s)",
        output.len(),
        output.len() as f32 / 44_100.0
    );

    save_wav_mono(&cmd.output, &output, 44_100)?;
    println!("  Saved: {}", cmd.output.display());

    Ok(())
}

#[derive(clap::Args)]
pub struct V2fCmd {
    /// Input WAV (44.1 kHz mono)
    pub input: std::path::PathBuf,
    /// Flat f32 weights (training/export_v2f.py)
    #[arg(short, long, default_value = "models/v2f.bin")]
    pub weights: std::path::PathBuf,
    /// Mel filterbank [80][513] flat f32
    #[arg(long, default_value = "models/mel_fb_1024_80.bin")]
    pub mel_fb: std::path::PathBuf,
    /// Mel->linear map [257][80] flat f32
    #[arg(long, default_value = "models/mel2lin_W.bin")]
    pub mel2lin: std::path::PathBuf,
    #[arg(short, long, default_value = "v2f_output.wav")]
    pub output: std::path::PathBuf,
}

#[derive(clap::Args)]
pub struct VcCmd {
    /// Input WAV (44.1 kHz mono, 男声)
    pub input: std::path::PathBuf,
    /// E 重み (training/export_eg.py --kind e)
    #[arg(long, default_value = "models/e1.bin")]
    pub e: std::path::PathBuf,
    /// G 重み = voice cartridge (training/export_eg.py --kind g)
    #[arg(long, default_value = "models/g1.bin")]
    pub g: std::path::PathBuf,
    /// V 重み
    #[arg(long, default_value = "models/v2f.bin")]
    pub v: std::path::PathBuf,
    #[arg(long, default_value = "models/mel_fb_1024_80.bin")]
    pub mel_fb: std::path::PathBuf,
    #[arg(long, default_value = "models/mel2lin_W.bin")]
    pub mel2lin: std::path::PathBuf,
    /// f0 シフト (半音・cartridge 定数)
    #[arg(long, default_value_t = 15.0)]
    pub shift: f32,
    /// E/G を f16 重みで実行 (帯域半減で ~2 倍速。要 F16C)。proxy 判定は f32 と
    /// 同値を確認済み (SECS 0.722 / CER 一致)。--f16=false で f32 に戻せる
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub f16: bool,
    #[arg(short, long, default_value = "vc_output.wav")]
    pub output: std::path::PathBuf,
}

pub fn run_vc(cmd: VcCmd) -> Result<()> {
    use lightvc_core::eg::Eg1d;
    use lightvc_core::v2f_infer::{V2fEngine, V2fStream};
    use lightvc_core::vc_stream::VcStream;
    let rdf = |p: &std::path::Path| -> Result<Vec<f32>> {
        Ok(std::fs::read(p)
            .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    println!("LightVC voice conversion (causal, streaming)");
    let e = std::sync::Arc::new(Eg1d::from_flat(&rdf(&cmd.e)?, 80, 768, 256, 8));
    let g = std::sync::Arc::new(Eg1d::from_flat(&rdf(&cmd.g)?, 770, 80, 256, 6));
    let eng = std::sync::Arc::new(
        V2fEngine::load(&cmd.v, &cmd.mel_fb, &cmd.mel2lin, 24, 8)
            .map_err(|e| anyhow::anyhow!("weights: {e}"))?);
    let fb = rdf(&cmd.mel_fb)?;
    let (x, sr) = load_wav_mono(&cmd.input)?;
    anyhow::ensure!(sr == 44_100, "44.1 kHz expected, got {sr}");
    // 学習系の慣習: front/E/G は ×32768 スケール、V 出力は [-1,1]
    let x32: Vec<f32> = x.iter().map(|v| v * 32768.0).collect();
    let v = V2fStream::with_threads(eng, 2);
    let mut vc = if cmd.f16 {
        use lightvc_core::eg::Eg1dH;
        use lightvc_core::vc_stream::EgAny;
        VcStream::new_any(
            EgAny::F16(std::sync::Arc::new(Eg1dH::from_net(&e))),
            EgAny::F16(std::sync::Arc::new(Eg1dH::from_net(&g))),
            v, fb, cmd.shift)
    } else {
        VcStream::new(e, g, v, fb, cmd.shift)
    };
    let n = (x32.len() / VcStream::BLOCK) * VcStream::BLOCK;
    let mut y = Vec::with_capacity(n);
    let t0 = std::time::Instant::now();
    for b in x32[..n].chunks(VcStream::BLOCK) {
        y.extend(vc.process_block(b));
    }
    let el = t0.elapsed().as_secs_f64();
    println!("  {} samples  shift {:+.1} st  RTF {:.3}", n, cmd.shift,
             el / (n as f64 / 44_100.0));
    save_wav_mono_f32(&cmd.output, &y, 44_100)?;
    println!("  Saved: {}", cmd.output.display());
    Ok(())
}

pub fn run_v2f(cmd: V2fCmd) -> Result<()> {
    use lightvc_core::v2f_infer::V2fEngine;
    println!("LightVC v2f resynthesis (causal, 30 ms class)");
    let eng = V2fEngine::load(&cmd.weights, &cmd.mel_fb, &cmd.mel2lin, 24, 8)
        .map_err(|e| anyhow::anyhow!("weights: {e}"))?;
    let (x, sr) = load_wav_mono(&cmd.input)?;
    anyhow::ensure!(sr == 44_100, "44.1 kHz expected, got {sr}");
    // prior の雑音は決定的に（xorshift。乱数品質は要らない、再現性が要る）
    let mut st = 0x9e3779b97f4a7c15u64;
    let noise: Vec<f32> = (0..x.len()).map(|_| {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 40) as f32 / 8388608.0) - 1.0
    }).collect();
    let t0 = std::time::Instant::now();
    let y = eng.process(&x, &noise);
    let el = t0.elapsed().as_secs_f64();
    println!("  {} samples  RTF {:.3}", y.len(), el / (x.len() as f64 / 44_100.0));
    save_wav_mono(&cmd.output, &y, 44_100)?;
    println!("  Saved: {}", cmd.output.display());
    Ok(())
}

pub fn run_resynth(cmd: ResynthCmd) -> Result<()> {
    println!("LightVC FreeVocoder Resynthesis");
    println!("  Input:     {}", cmd.input.display());
    println!("  Weights:   {}", cmd.weights.display());
    println!("  Mel basis: {}", cmd.mel_basis.display());
    println!("  Output:    {}", cmd.output.display());

    let mut rs = FreeResynth::new(&cmd.weights, &cmd.mel_basis, cmd.k, Device::Cpu)?;
    let chunk = rs.chunk_samples();
    println!(
        "  Streaming: K={} → {} samples/chunk ({:.1} ms), algorithmic latency {:.1} ms",
        cmd.k,
        chunk,
        chunk as f32 / 44.1,
        rs.algorithmic_latency_ms()
    );

    let (src_pcm, src_sr) = load_wav_mono(&cmd.input)?;
    let pcm_44k = if src_sr != 44_100 {
        println!("  Resampling {} → 44100 Hz...", src_sr);
        resample_to_44100(&src_pcm, src_sr)?
    } else {
        src_pcm
    };
    println!("  Loaded: {} samples @ 44.1 kHz", pcm_44k.len());

    // Feed the deployed realtime path in fixed `chunk_samples()` blocks. The
    // trailing remainder (< chunk, hence < one whole mel frame after the mel
    // state's carry) is fed as-is; any sub-frame tail stays buffered in the mel
    // state and is simply dropped when the stream ends (the honest streaming
    // semantics — no artificial zero-padding of the analysis window).
    let mut output = Vec::with_capacity(pcm_44k.len());
    for blk in pcm_44k.chunks(chunk) {
        output.extend_from_slice(&rs.process_chunk(blk)?);
    }

    save_wav_mono_f32(&cmd.output, &output, 44_100)?;
    println!("  Saved: {} ({} samples)", cmd.output.display(), output.len());
    Ok(())
}

pub fn run_gui(cmd: GuiCmd) -> Result<()> {
    println!("LightVC GUI starting...");

    let icon = crate::assets::load_icon();

    let dac_weights = cmd
        .dac_weights
        .unwrap_or_else(|| std::path::PathBuf::from("models/dac_44khz.safetensors"));
    let mut app = crate::app::LightVcApp::new(dac_weights);
    if cmd.demo {
        app.enable_demo();
    }
    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size([800.0, 600.0])
        .with_title("LightVC");
    if let Some(icon_data) = icon {
        viewport = viewport.with_icon(std::sync::Arc::new(icon_data));
    }
    let opts = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native("LightVC", opts, Box::new(move |_cc| Ok(Box::new(app))))?;

    Ok(())
}

// --- WAV I/O helpers ---

pub fn load_wav_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = match spec.bits_per_sample {
                16 => 32768.0f32,
                24 => 8388608.0,
                32 => 2147483648.0,
                _ => 32768.0,
            };
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

pub fn save_wav_mono(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    Ok(())
}

/// Float WAV writer that preserves the sample's native range (no `[-1, 1]`
/// clamp). Used for FreeVocoder resynthesis, whose output legitimately peaks
/// above unity (like the training reference); clamping would clip those peaks
/// and corrupt the resynthesis fidelity.
pub fn save_wav_mono_f32(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn resample_to_44100(input: &[f32], input_sr: u32) -> Result<Vec<f32>> {
    if input_sr == 44_100 {
        return Ok(input.to_vec());
    }
    let mut resampler = Resampler::new(input_sr as usize, 4096)?;
    let mut output = Vec::new();
    let chunk_size = resampler.input_frames_needed_up();

    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let chunk = &input[pos..end];
        if chunk.len() < chunk_size {
            let mut padded = chunk.to_vec();
            padded.resize(chunk_size, 0.0);
            let resampled = resampler.process_up(&padded)?;
            output.extend_from_slice(resampled);
        } else {
            let resampled = resampler.process_up(chunk)?;
            output.extend_from_slice(resampled);
        }
        pos = end;
    }

    Ok(output)
}
