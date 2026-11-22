//! リアルタイム変換の実行部(音声スレッド・UI からは state.rs の制御の列で駆動)。

use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use lightvc_core::snv_stream::SnvStream;

use crate::state::{RtControl, RtMetrics};

const SR: usize = 48_000;
const HOP: usize = SnvStream::BLOCK;

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

// =========================================================================
// Inference thread
// =========================================================================

fn start_engine(
    host_pref: lightvc_audio::HostPref,
    input_idx: Option<usize>,
    output_idx: Option<usize>,
    buffer_frames: u32,
) -> anyhow::Result<(lightvc_audio::AudioEngine, lightvc_audio::AudioBuffers, String)> {
    use cpal::traits::HostTrait as _;
    let buf = if buffer_frames == 0 { cpal::BufferSize::Default } else { cpal::BufferSize::Fixed(buffer_frames) };
    let host = lightvc_audio::host::pick(host_pref)?;
    let host_name = host.id().name().to_ascii_lowercase();
    let in_dev = match input_idx {
        Some(i) => host.input_devices().ok().and_then(|mut d| d.nth(i)),
        None => host.default_input_device(),
    };
    let out_dev = match output_idx {
        Some(i) => host.output_devices().ok().and_then(|mut d| d.nth(i)),
        None => host.default_output_device(),
    };
    match (in_dev, out_dev) {
        (Some(i), Some(o)) => lightvc_audio::AudioEngine::start_buffered(&i, &o, buf).map(|(e, b)| (e, b, host_name)),
        _ => anyhow::bail!("選んだホストに入出力の装置が無い({host_name})"),
    }
}

/// Device-rate capture → 48 kHz → SnvStream (240-sample blocks) → device-rate playback.
/// この糸が変換器を持ち(ロックなし)・P コアに置かれる。登録と読み込みは別の糸で済ませたものを RtControl::Stream で受け取る。
pub fn inference_loop(control_rx: Receiver<RtControl>, metrics_tx: Sender<RtMetrics>) {
    lightvc_core::affinity::pin_audio_driver();
    lightvc_core::affinity::raise_timer_resolution();
    let mut running = false;
    let mut bypass = false;
    let mut muted = false;
    let mut buffer_frames: u32 = 256;
    let mut stream: Option<SnvStream> = None;
    let mut engine: Option<lightvc_audio::AudioEngine> = None;
    let mut cap_rx: Option<rtrb::Consumer<f32>> = None;
    let mut pb_tx: Option<rtrb::Producer<f32>> = None;
    let mut up: Option<lightvc_audio::RateConv> = None;
    let mut down: Option<lightvc_audio::RateConv> = None;
    let (mut cap, mut in48, mut out48): (Vec<f32>, Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
    let mut out = vec![0f32; HOP];
    let (mut in_rms, mut out_rms, mut rtf) = (0.0f32, 0.0f32, 0.0f32);
    let algo_ms = SnvStream::LATENCY_SAMPLES as f32 / 48.0;
    let mut times: Vec<f64> = Vec::new();
    let mut window = (0.0f32, 0.0f32, 0.0f32, 0u32);
    let mut gate_open = false;
    let mut host_name = String::new();
    let mut last_report = Instant::now();

    loop {
        while let Ok(msg) = control_rx.try_recv() {
            match msg {
                RtControl::StartWithDevices { host, input_idx, output_idx } => {
                    if running {
                        continue;
                    }
                    match start_engine(host, input_idx, output_idx, buffer_frames) {
                        Ok((eng, bufs, hname)) => {
                            let (csr, psr) = (eng.capture_sample_rate as usize, eng.playback_sample_rate as usize);
                            let mk = |a: usize, b: usize| if a == b { Ok(None) } else { lightvc_audio::RateConv::new(a, b, HOP).map(Some) };
                            match (mk(csr, SR), mk(SR, psr)) {
                                (Ok(u), Ok(d)) => {
                                    up = u;
                                    down = d;
                                    cap.clear();
                                    in48.clear();
                                    out48.clear();
                                    cap_rx = Some(bufs.capture);
                                    pb_tx = Some(bufs.playback);
                                    engine = Some(eng);
                                    host_name = hname;
                                    running = true;
                                    last_report = Instant::now();
                                }
                                (Err(e), _) | (_, Err(e)) => eprintln!("Resampler: {e}"),
                            }
                        }
                        Err(e) => eprintln!("Audio: {e}"),
                    }
                }
                RtControl::Stream(s) => stream = Some(*s),
                RtControl::Stop => {
                    running = false;
                    engine = None;
                    cap_rx = None;
                    pb_tx = None;
                    up = None;
                    down = None;
                }
                RtControl::Bypass(b) => bypass = b,
                RtControl::Mute(m) => muted = m,
                RtControl::SetBufferSize(n) => buffer_frames = n,
                RtControl::SetSemitones(k) => {
                    if let Some(s) = stream.as_mut() {
                        s.set_semitones(k);
                    }
                }
            }
        }

        if !running {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        if engine.as_ref().map(|e| e.is_disconnected()).unwrap_or(false) {
            eprintln!("Audio device disconnected — stopping");
            running = false;
            engine = None;
            cap_rx = None;
            pb_tx = None;
            let _ = metrics_tx.send(RtMetrics { disconnected: true, ..Default::default() });
            continue;
        }

        let (Some(crx), Some(ptx)) = (cap_rx.as_mut(), pb_tx.as_mut()) else {
            continue;
        };

        let mut did_work = false;
        while let Ok(s) = crx.pop() {
            cap.push(s);
            did_work = true;
            if cap.len() >= 8 * HOP {
                break;
            }
        }
        match up.as_mut() {
            Some(u) => {
                while cap.len() >= u.input_frames_next() {
                    let k = u.input_frames_next();
                    match u.process(&cap[..k]) {
                        Ok(y) => in48.extend_from_slice(y),
                        Err(e) => eprintln!("resample up: {e}"),
                    }
                    cap.drain(..k);
                }
            }
            None => in48.append(&mut cap),
        }
        // Drop the oldest input if processing falls behind (stay near real time).
        if in48.len() > 8 * HOP {
            let excess = in48.len() - 2 * HOP;
            in48.drain(..excess);
        }

        while in48.len() >= HOP {
            let t0 = Instant::now();
            let blk = &in48[..HOP];
            let mut converted = false;
            if !bypass && let Some(s) = stream.as_mut() {
                s.process_block(blk, &mut out);
                gate_open = s.gate_open();
                converted = true;
            }
            if !converted {
                out.copy_from_slice(blk);
            }
            if muted {
                out.iter_mut().for_each(|v| *v = 0.0);
            }
            in_rms = rms(blk);
            out_rms = rms(&out);
            rtf = t0.elapsed().as_secs_f32() / (HOP as f32 / SR as f32);
            out48.extend(out.iter().map(|v| v.clamp(-1.0, 1.0)));
            in48.drain(..HOP);
            times.push(t0.elapsed().as_secs_f64() * 1e3);
            did_work = true;
        }

        match down.as_mut() {
            Some(d) => {
                while out48.len() >= d.input_frames_next() {
                    let k = d.input_frames_next();
                    match d.process(&out48[..k]) {
                        Ok(y) => {
                            for v in y {
                                let _ = ptx.push(*v);
                            }
                        }
                        Err(e) => eprintln!("resample down: {e}"),
                    }
                    out48.drain(..k);
                }
            }
            None => {
                for v in out48.drain(..) {
                    let _ = ptx.push(v);
                }
            }
        }

        // 5 秒ごとの 1 ブロックの計算時間(realtime-snv と同じ定義・p99 < 5.0 が線)。
        if last_report.elapsed().as_secs() >= 5 && !times.is_empty() {
            let mut t = times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| t[((t.len() as f64 - 1.0) * p) as usize] as f32;
            window = (q(0.5), q(0.95), q(0.99), t.len() as u32);
            times.clear();
            last_report = Instant::now();
        }

        let (overrun, underrun) = engine.as_ref().map(|e| (e.overrun_count(), e.underrun_count())).unwrap_or((0, 0));
        let dev_sr = engine.as_ref().map(|e| e.capture_sample_rate).unwrap_or(SR as u32) as f32;
        let buf_ms = buffer_frames as f32 * 1000.0 / dev_sr;
        let _ = metrics_tx.send(RtMetrics {
            input_rms: in_rms,
            output_rms: out_rms,
            latency_ms: buf_ms + 3.0 + algo_ms + 3.0 + buf_ms,
            rtf,
            disconnected: false,
            overrun,
            underrun,
            compute_p50: window.0,
            compute_p95: window.1,
            compute_p99: window.2,
            blocks: window.3,
            gate_open,
            host: host_name.clone(),
        });

        if !did_work {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
