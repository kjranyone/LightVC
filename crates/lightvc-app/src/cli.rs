//! CLI argument parsing and subcommand dispatch.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use lightvc_audio::Resampler;
use lightvc_core::{
    converter::{AnyConverter, ConverterConfig, LatencyMode},
    pipeline::VcPipeline,
    DacConfig,
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
    /// Launch desktop GUI (3 tabs: offline/realtime/catalog).
    Gui(GuiCmd),
}

#[derive(Parser)]
pub struct GuiCmd {
    #[arg(long, env = "LIGHTVC_DAC_WEIGHTS")]
    pub dac_weights: PathBuf,
    #[arg(long)]
    pub cuda: bool,
    #[arg(long)]
    pub metal: bool,
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

    #[arg(long, default_value = "balanced")]
    pub mode: String,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub metal: bool,
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

    println!("Loading converter...");
    let vb = lightvc_core::weights::load_varbuilder(&cmd.converter_weights, DType::F32, &device)?;
    let converter = AnyConverter::new(converter_config, vb)?;

    let mut pipeline = VcPipeline::new(&cmd.dac_weights, &dac_config, converter, mode, device)?;

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

pub fn run_gui(cmd: GuiCmd) -> Result<()> {
    println!("LightVC GUI starting...");

    let icon = crate::assets::load_icon();

    let mut app = crate::app::LightVcApp::new(cmd.dac_weights);
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
    eframe::run_simple_native("LightVC", opts, move |ctx, _frame| {
        app.render(ctx);
    })?;

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
            output.extend(resampler.process_up(&padded)?);
        } else {
            output.extend(resampler.process_up(chunk)?);
        }
        pos = end;
    }

    Ok(output)
}
