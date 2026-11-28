//! CLI argument parsing and subcommand dispatch.
//!
//! 製品の経路は paper_snv の段 P126(門と重みつきの前段 v4_dual → K3G-v4 → 出力部 snv_p123_s・`lightvc_core::snv_stream::SnvStream`)だけ。

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
    /// paper_snv の製品経路(段 P126: 前段 v4_dual → K3G-v4 → snv_p123_s)のオフライン変換: WAV(48 kHz)→ WAV。240 サンプル単位の流しをそのまま通す。
    Snv(SnvCmd),
    /// paper_snv の製品経路のリアルタイム変換(マイク → 48 kHz・240 サンプル単位 → スピーカー)。
    RealtimeSnv(RealtimeSnvCmd),
    /// 使えるホストと入出力装置の一覧(既定の選択の確認)。
    Devices,
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
    let state = std::sync::Arc::new(std::sync::Mutex::new(crate::state::AppState::new()));
    if cmd.demo {
        crate::ui::enable_demo(&state);
    }
    crate::ui::run(state, cmd.demo)
}

fn default_export(i: usize) -> std::path::PathBuf {
    let (sub, dev) = crate::state::EXPORTS[i];
    crate::state::resolve_export(sub, dev).into()
}

/// paper_snv の製品経路の部品(training/export_{snv_gate,snv_k3g,rvoc}.py の出力)と登録の波形(48 kHz)。
#[derive(Parser, Clone)]
pub struct SnvModel {
    /// 前段と門(既定 = 実行形式の隣の models/front_gate → results/snv_p126_gate_export)
    #[arg(long, default_value_os_t = default_export(0))]
    pub front: std::path::PathBuf,
    /// 変換網 K3G-v4(既定 = models/k3g → results/snv_p126_k3g16kv_export)
    #[arg(long, default_value_os_t = default_export(1))]
    pub model: std::path::PathBuf,
    /// 出力部 snv_p123_s(既定 = models/rvoc3hi → results/snv_p123_s/export_20k)
    #[arg(long, default_value_os_t = default_export(2))]
    pub voc: std::path::PathBuf,
    /// 操作者の登録(自分の普通の発話・計 30 秒程度・複数可)
    #[arg(long, required = true, num_args = 1..)]
    pub operator: Vec<std::path::PathBuf>,
    /// 目標の参照(目標の声の普通の発話・計 20 秒以上・複数可。同梱のプリセットは `voices/<id>.flac`)
    #[arg(long, required = true, num_args = 1..)]
    pub target: Vec<std::path::PathBuf>,
    /// 固定の半音(運用 = 0 / 5 / 7 / 12)
    #[arg(long, default_value_t = 0.0)]
    pub semitones: f32,
    /// 変換網のスレッド数(1 = 全部逐次・2 / 4 = K3G の組の分割 + 門と描画器の並列・実時間には 4)
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
                let (x, sr) = crate::audio_playback::load_audio_mono(p)?;
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
    /// オーディオホスト (default / wasapi / asio。未指定なら LVC_HOST → 既定の方針)
    #[arg(long, default_value = "default")]
    pub host: String,
    /// 入力オーディオホスト (省略時は --host)
    #[arg(long)]
    pub input_host: Option<String>,
    /// 出力オーディオホスト (省略時は --host)
    #[arg(long)]
    pub output_host: Option<String>,
    /// 入力装置(`devices` の一覧の番号・未指定 = 既定)
    #[arg(long)]
    pub input: Option<usize>,
    /// 出力装置(`devices` の一覧の番号・未指定 = 既定)
    #[arg(long)]
    pub output: Option<usize>,
}

pub fn run_devices() -> Result<()> {
    let mut shown = Vec::new();
    for pref in [lightvc_audio::HostPref::Asio, lightvc_audio::HostPref::Wasapi] {
        let Ok(inputs) = lightvc_audio::DuplexStream::list_inputs(pref) else { continue };
        let Ok(outputs) = lightvc_audio::DuplexStream::list_outputs(pref) else { continue };
        if inputs.is_empty() && outputs.is_empty() {
            continue;
        }
        let name = pref.label().to_string();
        shown.push(name.clone());
        println!("host {name}:");
        for (i, d) in inputs.iter().enumerate() {
            println!("  in  {i}: {} ({}Hz, {}ch)", d.name, d.sample_rate, d.channels);
        }
        for (i, d) in outputs.iter().enumerate() {
            println!("  out {i}: {} ({}Hz, {}ch)", d.name, d.sample_rate, d.channels);
        }
    }
    println!("hosts: {:?}", shown);
    Ok(())
}

pub fn run_snv(cmd: SnvCmd) -> Result<()> {
    use lightvc_core::snv_stream::SnvStream;
    const HOP: usize = SnvStream::BLOCK;
    let (x, sr) = crate::audio_playback::load_audio_mono(&cmd.input)?;
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
            cmd.model.threads, vc.latency_samples() as f64 / 48.0
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
    use lightvc_core::snv_stream::SnvStream;
    const HOP: usize = SnvStream::BLOCK;
    const SR: usize = 48_000;
    lightvc_core::affinity::pin_audio_driver();
    lightvc_core::affinity::raise_timer_resolution();
    let mut vc = cmd.model.load()?;
    {
        // 重みのページインと常駐スレッドの起動を済ませる(開始直後の無音の窓を防ぐ)。
        let warm = vec![0f32; HOP];
        let mut sink = vec![0f32; HOP];
        vc.process_block(&warm, &mut sink);
    }
    let in_host = lightvc_audio::HostPref::parse(cmd.input_host.as_deref().unwrap_or(&cmd.host))?;
    let out_host = lightvc_audio::HostPref::parse(cmd.output_host.as_deref().unwrap_or(&cmd.host))?;
    let (engine, mut io) = lightvc_audio::AudioEngine::start_endpoints(
        lightvc_audio::AudioEndpoint { host: in_host, device_index: cmd.input, device_name: None },
        lightvc_audio::AudioEndpoint { host: out_host, device_index: cmd.output, device_name: None },
        SR as u32,
        cmd.buffer,
    )?;
    let host_name = engine.host_label.clone();
    let (csr, psr) = (engine.capture_sample_rate as usize, engine.playback_sample_rate as usize);
    let conv = csr != SR || psr != SR;
    let mk = |a: usize, b: usize| if a == b { Ok(None) } else { lightvc_audio::RateConv::new(a, b, HOP).map(Some) };
    let mut up = mk(csr, SR)?;
    let mut down = mk(SR, psr)?;
    println!("snv realtime | host {host_name} | in {csr} Hz・out {psr} Hz{} | algorithmic latency {:.1} ms(ブロック + 出力部)+ resampler + device buffers | 半音 {:+} | スレッド {}",
             if conv { " (48 kHz へ/からの変換)" } else { "" }, vc.latency_samples() as f64 / 48.0, cmd.model.semitones, cmd.model.threads);
    for _ in 0..1024 {
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
        // 遅れが溜まったら古い入力を捨てて実時間に追いつく(GUI と同じ・drain の O(n) の蓄積崩壊を防ぐ)。
        if in48.len() > 8 * HOP {
            let excess = in48.len() - 2 * HOP;
            in48.drain(..excess);
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
