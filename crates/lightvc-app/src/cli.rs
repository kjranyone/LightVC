//! CLI argument parsing and subcommand dispatch.
//!
//! 製品の経路は paper_snv の段 P(前段 v3 → E / C → rvoc3hi・`lightvc_core::snv_stream::SnvStream`)だけ。

use anyhow::Result;
use clap::{Parser, Subcommand};

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
    /// paper_snv の製品経路(前段 v3 → E / C → rvoc3hi)のオフライン変換: WAV(48 kHz)→ WAV。240 サンプル単位の流しをそのまま通す。
    Snv(SnvCmd),
    /// paper_snv の製品経路のリアルタイム変換(マイク → 48 kHz・240 サンプル単位 → スピーカー)。
    RealtimeSnv(RealtimeSnvCmd),
    /// Launch desktop GUI (3 tabs: offline/realtime/voices).
    Gui(GuiCmd),
}

#[derive(Parser)]
pub struct GuiCmd {
    /// Launch the GUI with mock data for screenshot capture. No model,
    /// no audio devices required. The user switches tabs inside the app.
    #[arg(long)]
    pub demo: bool,
}

pub fn run_gui(cmd: GuiCmd) -> Result<()> {
    println!("LightVC GUI starting...");
    let icon = crate::assets::load_icon();
    let mut app = crate::app::LightVcApp::new();
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

/// paper_snv の製品経路の部品(training/export_{pfront,snv_model,rvoc}.py の出力)と登録の波形(48 kHz)。
#[derive(Parser, Clone)]
pub struct SnvModel {
    #[arg(long, default_value = "results/snv_p45_gate_t80_export")]
    pub front: std::path::PathBuf,
    #[arg(long, default_value = "results/snv_k3g16kv_export")]
    pub model: std::path::PathBuf,
    #[arg(long, default_value = "results/rvoc3hi/export_330k")]
    pub voc: std::path::PathBuf,
    /// 操作者の登録(自分の普通の発話・計 30 秒程度・複数可)
    #[arg(long, required = true, num_args = 1..)]
    pub operator: Vec<std::path::PathBuf>,
    /// 目標の参照(目標の声の普通の発話・計 20 秒以上・複数可)
    #[arg(long, required = true, num_args = 1..)]
    pub target: Vec<std::path::PathBuf>,
    /// 固定の半音(運用 = 0 / 5 / 7 / 12)
    #[arg(long, default_value_t = 0.0)]
    pub semitones: f32,
    /// E / C のスレッド数(1 = 全部逐次・2 / 4 = 出力チャンネルの分割 + 描画器の並列)
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
}

impl SnvModel {
    fn load(&self) -> Result<lightvc_core::snv_stream::SnvStream> {
        let mut s = lightvc_core::snv_stream::SnvStream::load(&self.front, &self.model, &self.voc)?;
        if self.threads > 1 {
            s.enable_parallel(self.threads);
        }
        s.set_semitones(self.semitones);
        let rd = |ps: &[std::path::PathBuf], who: &str| -> Result<Vec<Vec<f32>>> {
            let mut v = Vec::new();
            let mut sec = 0.0;
            for p in ps {
                let (x, sr) = load_wav_mono(p)?;
                anyhow::ensure!(sr == 48_000, "{who} {} は 48 kHz が必要 (got {sr})", p.display());
                sec += x.len() as f64 / 48_000.0;
                v.push(x);
            }
            println!("{who}: {} 本・{sec:.1} 秒", v.len());
            Ok(v)
        };
        let t0 = std::time::Instant::now();
        let op = rd(&self.operator, "操作者の登録")?;
        s.enroll_operator(&op.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
        let tg = rd(&self.target, "目標の参照")?;
        s.enroll_target(&tg.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
        println!("登録 {:.1} s", t0.elapsed().as_secs_f64());
        Ok(s)
    }
}

#[derive(Parser)]
pub struct SnvCmd {
    #[arg(short, long)]
    pub input: std::path::PathBuf,
    #[arg(short, long, default_value = "snv_output.wav")]
    pub output: std::path::PathBuf,
    #[command(flatten)]
    pub model: SnvModel,
    /// 出荷検査: t 秒以降を時間反転した入力でも t より前の出力がビット一致するか(未来不変性・実音声)と 1 ブロックの処理時間の分布を JSON で出す
    #[arg(long)]
    pub probe_future: Option<f32>,
}

#[derive(Parser)]
pub struct RealtimeSnvCmd {
    #[command(flatten)]
    pub model: SnvModel,
    /// 実行秒数 (0 = 停止まで)
    #[arg(long, default_value_t = 0)]
    pub seconds: u64,
    /// cpal のバッファ (フレーム数・0 = 既定)
    #[arg(long, default_value_t = 240)]
    pub buffer: u32,
}

pub fn run_snv(cmd: SnvCmd) -> Result<()> {
    use lightvc_core::snv_stream::SnvStream;
    const HOP: usize = SnvStream::BLOCK;
    let (x, sr) = load_wav_mono(&cmd.input)?;
    anyhow::ensure!(sr == 48_000, "入力は 48 kHz が必要 (got {sr})");
    let mut vc = cmd.model.load()?;
    let n = x.len() / HOP;
    if let Some(ts) = cmd.probe_future {
        let cut = ((ts * 48_000.0) as usize / HOP).min(n - 1) * HOP;
        let mut x2 = x[..n * HOP].to_vec();
        x2[cut..].reverse();
        let mut vc2 = cmd.model.load()?;
        let (mut y1, mut y2) = (vec![0f32; n * HOP], vec![0f32; n * HOP]);
        let mut ms = Vec::with_capacity(n);
        for i in 0..n {
            let t = std::time::Instant::now();
            vc.process_block(&x[i * HOP..(i + 1) * HOP], &mut y1[i * HOP..(i + 1) * HOP]);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        for i in 0..n {
            vc2.process_block(&x2[i * HOP..(i + 1) * HOP], &mut y2[i * HOP..(i + 1) * HOP]);
        }
        let pre = y1[..cut].iter().zip(&y2[..cut]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let first = y1[..cut].iter().zip(&y2[..cut]).position(|(a, b)| a.to_bits() != b.to_bits());
        let post = (y1[cut..].iter().zip(&y2[cut..]).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / (n * HOP - cut) as f64).sqrt();
        let rms = (y1[cut..].iter().map(|a| (*a as f64).powi(2)).sum::<f64>() / (n * HOP - cut) as f64).sqrt();
        let mut q = ms.clone();
        q.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pc = |p: f64| q[((q.len() - 1) as f64 * p).round() as usize];
        let verdict = if pre > 0 { "FAIL" } else if post < 1e-6 * rms.max(1e-9) { "INCONCLUSIVE" } else { "PASS" };
        println!(
            "{{\"cut_s\": {:.3}, \"pre_mismatch\": {pre}, \"first_mismatch_ms\": {}, \"post_rms_diff\": {post:.3e}, \"post_rms\": {rms:.3e}, \"verdict\": \"{verdict}\", \"block_ms\": {{\"mean\": {:.4}, \"p50\": {:.4}, \"p95\": {:.4}, \"p99\": {:.4}, \"max\": {:.4}}}, \"rtf\": {:.4}, \"n_blocks\": {n}, \"threads\": {}, \"latency_ms\": {:.1}}}",
            cut as f64 / 48_000.0,
            first.map_or("null".to_string(), |f| format!("{:.2}", f as f64 / 48.0)),
            ms.iter().sum::<f64>() / n as f64, pc(0.5), pc(0.95), pc(0.99), q[q.len() - 1],
            ms.iter().sum::<f64>() / n as f64 / (HOP as f64 / 48.0),
            cmd.model.threads, SnvStream::LATENCY_SAMPLES as f64 / 48.0
        );
        return Ok(());
    }
    let mut y = vec![0f32; n * HOP];
    let t0 = std::time::Instant::now();
    for i in 0..n {
        vc.process_block(&x[i * HOP..(i + 1) * HOP], &mut y[i * HOP..(i + 1) * HOP]);
    }
    let el = t0.elapsed().as_secs_f64();
    let pk = y.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-9);
    if pk > 0.99 {
        y.iter_mut().for_each(|v| *v *= 0.99 / pk);
    }
    save_wav_mono(&cmd.output, &y, 48_000)?;
    println!("snv: {} s → {} | RTF {:.3}(壁時計)", n * HOP / 48_000, cmd.output.display(), el / (n as f64 * HOP as f64 / 48_000.0));
    Ok(())
}

pub fn run_realtime_snv(cmd: RealtimeSnvCmd) -> Result<()> {
    use cpal::traits::DeviceTrait;
    use lightvc_core::snv_stream::SnvStream;
    const HOP: usize = SnvStream::BLOCK;
    const SR: usize = 48_000;
    let mut vc = cmd.model.load()?;
    let input = lightvc_audio::engine::default_input()?;
    let output = lightvc_audio::engine::default_output()?;
    let in_cfg = input.default_input_config()?;
    let out_cfg = output.default_output_config()?;
    let buf = if cmd.buffer == 0 { cpal::BufferSize::Default } else { cpal::BufferSize::Fixed(cmd.buffer) };
    let (engine, mut io, dev_sr) = match lightvc_audio::AudioEngine::start_with(
        &input, &output, SR as u32, SR as u32, in_cfg.channels(), out_cfg.channels(), buf) {
        Ok((e, io)) => (e, io, SR),
        Err(e48) => {
            let (isr, osr) = (in_cfg.sample_rate(), out_cfg.sample_rate());
            anyhow::ensure!(isr == osr, "48 kHz で開けず({e48})、入出力の既定レートも不一致 ({isr} / {osr})");
            let (e, io) = lightvc_audio::AudioEngine::start_with(&input, &output, isr, osr, in_cfg.channels(), out_cfg.channels(), buf)?;
            (e, io, isr as usize)
        }
    };
    let conv = dev_sr != SR;
    let mut up = if conv { Some(lightvc_audio::RateConv::new(dev_sr, SR, HOP)?) } else { None };
    let mut down = if conv { Some(lightvc_audio::RateConv::new(SR, dev_sr, HOP)?) } else { None };
    println!("snv realtime | device {} Hz{} | algorithmic latency {:.1} ms(ブロック + 出力部)+ resampler + device buffers | 半音 {:+} | スレッド {}",
             dev_sr, if conv { " (48 kHz へ変換)" } else { "" }, SnvStream::LATENCY_SAMPLES as f64 / 48.0, cmd.model.semitones, cmd.model.threads);
    for _ in 0..HOP {
        let _ = io.playback.push(0.0);
    }
    let mut cap: Vec<f32> = Vec::new();
    let mut in48: Vec<f32> = Vec::new();
    let mut out48: Vec<f32> = Vec::new();
    let mut out = vec![0f32; HOP];
    let mut times: Vec<f64> = Vec::new();
    let t_start = std::time::Instant::now();
    let mut last_report = std::time::Instant::now();
    loop {
        if engine.is_disconnected() {
            anyhow::bail!("オーディオデバイスが切断された");
        }
        if cmd.seconds > 0 && t_start.elapsed().as_secs() >= cmd.seconds {
            break;
        }
        let n = io.capture.slots();
        if n == 0 {
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }
        for _ in 0..n {
            cap.push(io.capture.pop().unwrap_or(0.0));
        }
        match up.as_mut() {
            Some(u) => {
                while cap.len() >= u.input_frames_next() {
                    let k = u.input_frames_next();
                    in48.extend_from_slice(u.process(&cap[..k])?);
                    cap.drain(..k);
                }
            }
            None => in48.append(&mut cap),
        }
        while in48.len() >= HOP {
            let t0 = std::time::Instant::now();
            vc.process_block(&in48[..HOP], &mut out);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
            in48.drain(..HOP);
            out48.extend(out.iter().map(|v| v.clamp(-1.0, 1.0)));
        }
        match down.as_mut() {
            Some(d) => {
                while out48.len() >= d.input_frames_next() {
                    let k = d.input_frames_next();
                    for v in d.process(&out48[..k])? {
                        let _ = io.playback.push(*v);
                    }
                    out48.drain(..k);
                }
            }
            None => {
                for v in out48.drain(..) {
                    let _ = io.playback.push(v);
                }
            }
        }
        if last_report.elapsed().as_secs() >= 5 && !times.is_empty() {
            let mut t = times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| t[((t.len() as f64 - 1.0) * p) as usize];
            println!("  blocks {} | compute ms p50 {:.2} p95 {:.2} p99 {:.2} (予算 5.0) | overrun {} underrun {}",
                     t.len(), q(0.5), q(0.95), q(0.99), engine.overrun_count(), engine.underrun_count());
            times.clear();
            last_report = std::time::Instant::now();
        }
    }
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

/// 任意のレートの波形を 48 kHz へ(登録と GUI のオフライン変換の入力)。変換器の遅延の分は頭を詰める。
pub fn resample_to_48k(input: &[f32], input_sr: u32) -> Result<Vec<f32>> {
    const SR: usize = 48_000;
    if input_sr as usize == SR {
        return Ok(input.to_vec());
    }
    let mut conv = lightvc_audio::RateConv::new(input_sr as usize, SR, 1024)?;
    let want = (input.len() as f64 * SR as f64 / input_sr as f64).round() as usize;
    let delay = conv.output_delay();
    let mut out = Vec::with_capacity(want + delay + 2048);
    let mut pos = 0;
    while out.len() < want + delay {
        let k = conv.input_frames_next();
        let mut chunk = vec![0.0f32; k];
        let n = input.len().saturating_sub(pos).min(k);
        chunk[..n].copy_from_slice(&input[pos..pos + n]);
        out.extend_from_slice(conv.process(&chunk)?);
        pos += k;
    }
    out.drain(..delay);
    out.truncate(want);
    Ok(out)
}
