//! Tab 2: Real-time voice conversion.
//! Mic input → 48 kHz・240 サンプル → SnvStream(前段 v3 → E / C → rvoc3hi)→ speaker output.
//!
//! Contains both UI rendering and the inference thread loop.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use lightvc_core::snv_stream::SnvStream;

use crate::app::{AppState, RtControl, RtMetrics};
use crate::file_pick::FilePick;
use crate::widgets;

const SR: usize = 48_000;
const HOP: usize = SnvStream::BLOCK;
const SEMITONES: [(f32, &str); 4] = [(0.0, "+0"), (5.0, "+5"), (7.0, "+7"), (12.0, "+12")];

/// Render the realtime tab.
#[allow(clippy::too_many_arguments)]
pub fn render(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    operator_pick: &FilePick,
    state: &Arc<Mutex<AppState>>,
    running: &mut bool,
    bypass: &mut bool,
    muted: &mut bool,
    buffer_frames: &mut u32,
    metrics: &RtMetrics,
    mut on_load: impl FnMut(),
    mut on_ensure_thread: impl FnMut(),
    on_record: impl Fn(u32),
    on_control: impl Fn(RtControl),
    assets: &mut crate::assets::AssetCache,
) {
    let stop_icon = assets.icon_stop(ctx).clone();
    on_ensure_thread();

    let (has_engine, loading, mut cfg, ref_name) = {
        let s = state.lock().unwrap();
        let name = s.selected_voice.and_then(|i| s.voices.get(i).map(|v| v.name.clone()));
        (s.model_loaded, s.loading, s.settings.clone(), name)
    };
    let force_bypass = !has_engine;

    let (status_color, status_label) = if *running {
        if *bypass || force_bypass {
            (crate::theme::colors::STATUS_BYPASS, "BYPASS")
        } else {
            (crate::theme::colors::STATUS_CONVERTING, "CONVERTING")
        }
    } else {
        (crate::theme::colors::STATUS_STOPPED, "STOPPED")
    };

    let col_gap = crate::theme::space::MEDIUM;
    let col_w = (ui.available_width() - col_gap) / 2.0;
    let card_min_h = 150.0;

    ui.horizontal_top(|ui| {
        // LEFT: Model + enrollment card
        ui.allocate_ui_with_layout(
            egui::vec2(col_w, card_min_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let frame = if has_engine {
                    crate::theme::info_card_frame()
                } else {
                    crate::theme::glow_card_frame()
                };
                frame.show(ui, |ui| {
                    ui.set_min_height(card_min_h - 40.0);
                    crate::theme::subheading(ui, "Model");
                    egui::CollapsingHeader::new(
                        egui::RichText::new("[*] Exports")
                            .size(11.0)
                            .color(crate::theme::colors::TEXT_DIM),
                    )
                    .default_open(!has_engine)
                    .show(ui, |ui| {
                        for (label, v) in [("front", &mut cfg.front), ("E / C", &mut cfg.model), ("voc", &mut cfg.voc)] {
                            ui.horizontal(|ui| {
                                crate::theme::form_label(ui, label);
                                crate::theme::path_text_edit(ui, v, "export dir");
                            });
                        }
                        ui.horizontal(|ui| {
                            crate::theme::form_label(ui, "threads");
                            for n in [1usize, 2, 4] {
                                if crate::theme::pill_button(ui, &n.to_string(), cfg.threads == n) {
                                    cfg.threads = n;
                                }
                            }
                        });
                    });
                    ui.add_space(crate::theme::space::TIGHT);
                    ui.label(
                        egui::RichText::new(format!("Operator: {} file(s)", cfg.operator.len()))
                            .size(11.0)
                            .color(crate::theme::colors::TEXT_DIM),
                    );
                    for p in &cfg.operator {
                        ui.label(
                            egui::RichText::new(format!("  {}", p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()))
                                .size(10.0)
                                .color(crate::theme::colors::TEXT_MUTED),
                        );
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Add operator WAV").clicked() {
                            operator_pick.open(ctx);
                        }
                        let (recording, remaining) = {
                            let s = state.lock().unwrap();
                            let rem = s.operator_rec.as_ref().map(|r| r.seconds.saturating_sub(r.started.elapsed().as_secs() as u32));
                            (s.operator_rec.is_some(), rem)
                        };
                        if let Some(rem) = remaining {
                            ui.label(
                                egui::RichText::new(format!("●REC 残り {rem} s"))
                                    .size(11.0)
                                    .color(crate::theme::colors::ERROR),
                            );
                        } else if ui.button("Record operator (30 s)").clicked() && !*running {
                            on_record(30);
                        }
                        if !cfg.operator.is_empty() && ui.button("Clear").clicked() {
                            cfg.operator.clear();
                        }
                        let _ = recording;
                    });
                    ui.add_space(crate::theme::space::TIGHT);
                    let can_load = !cfg.operator.is_empty() && ref_name.is_some() && !loading;
                    ui.horizontal(|ui| {
                        if crate::theme::primary_button(ui, if has_engine { "Reload" } else { "Load" }, can_load) && can_load {
                            on_load();
                        }
                        if loading {
                            ui.spinner();
                        } else if ref_name.is_none() {
                            ui.label(
                                egui::RichText::new("select a target in Voices")
                                    .size(10.0)
                                    .color(crate::theme::colors::ERROR),
                            );
                        } else if !has_engine {
                            ui.label(
                                egui::RichText::new("no model -> BYPASS")
                                    .size(10.0)
                                    .color(crate::theme::colors::ERROR),
                            );
                        }
                    });
                });
            },
        );

        ui.add_space(col_gap);

        // RIGHT: Status card
        ui.allocate_ui_with_layout(
            egui::vec2(col_w, card_min_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let frame = if *running && !*bypass && has_engine {
                    crate::theme::glow_card_frame()
                } else {
                    crate::theme::info_card_frame()
                };
                frame.show(ui, |ui| {
                    ui.set_min_height(card_min_h - 40.0);
                    crate::theme::subheading(ui, "Status");
                    ui.horizontal(|ui| {
                        crate::theme::status_badge(ui, status_label, status_color);
                        ui.add_space(crate::theme::space::SMALL);
                        let (txt, col) = match &ref_name {
                            Some(name) => (format!("> {name}"), crate::theme::colors::TEXT),
                            None => ("no voice".to_string(), crate::theme::colors::TEXT_MUTED),
                        };
                        ui.label(egui::RichText::new(txt).size(12.0).color(col));
                        if *running && !*bypass && has_engine {
                            let (g, c) = if metrics.gate_open {
                                ("GATE", crate::theme::colors::MINT)
                            } else {
                                ("gate", crate::theme::colors::TEXT_MUTED)
                            };
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(egui::RichText::new(g).size(10.0).color(c).strong());
                            });
                        }
                    });
                    ui.add_space(crate::theme::space::TIGHT);
                    ui.horizontal(|ui| {
                        crate::theme::stat_card(ui, &format!("{:.0}", metrics.latency_ms), "ms", crate::theme::colors::TEXT);
                        ui.add_space(crate::theme::space::TIGHT);
                        crate::theme::stat_card(ui, &format!("{:.2}", metrics.rtf), "RTF", crate::theme::colors::TEXT);
                        if metrics.overrun > 0 || metrics.underrun > 0 {
                            ui.add_space(crate::theme::space::TIGHT);
                            ui.label(
                                egui::RichText::new(format!("in:{}  out:{}", metrics.overrun, metrics.underrun))
                                    .size(10.0)
                                    .color(crate::theme::colors::ERROR),
                            );
                        }
                    });
                    ui.add_space(crate::theme::space::TIGHT);
                    // 5 秒ごとの 1 ブロックの計算時間(realtime-snv と同じ定義・p99 < 5.0 が線)。
                    ui.horizontal(|ui| {
                        let col = |v: f32| if v >= 5.0 { crate::theme::colors::ERROR } else { crate::theme::colors::MINT };
                        ui.label(egui::RichText::new("compute ms").size(10.0).color(crate::theme::colors::TEXT_DIM));
                        for (name, v) in [("p50", metrics.compute_p50), ("p95", metrics.compute_p95), ("p99", metrics.compute_p99)] {
                            ui.label(egui::RichText::new(format!("{name} {v:.2}")).size(10.0).color(col(v)));
                            ui.add_space(crate::theme::space::TIGHT);
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let n = if metrics.blocks == 0 && metrics.compute_p99 == 0.0 { String::new() } else { format!("{} blk", metrics.blocks) };
                            ui.label(egui::RichText::new(n).size(10.0).color(crate::theme::colors::TEXT_MUTED));
                            ui.label(egui::RichText::new(format!("host {}", if metrics.host.is_empty() { "-" } else { &metrics.host })).size(10.0).color(crate::theme::colors::TEXT_DIM));
                        });
                    });
                });
            },
        );
    });

    if let Some(path) = operator_pick.take() {
        cfg.operator.push(path);
    }
    let semitones_before = cfg.semitones;

    ui.add_space(crate::theme::space::MEDIUM);

    // Input signal meter
    crate::theme::info_card_frame().show(ui, |ui| {
        ui.horizontal(|ui| {
            crate::theme::subheading(ui, "Input");
            ui.add_space(crate::theme::space::SMALL);
            crate::theme::level_meter_kind_compact(ui, metrics.input_rms, crate::theme::MeterKind::Input);
        });
    });

    ui.add_space(crate::theme::space::MEDIUM);

    // Latency card: E2E estimate (device buffers + resample + algorithmic) + buffer selector.
    {
        let buf_ms = *buffer_frames as f32 / 48.0;
        let resamp_ms = 3.0;
        let algo_ms = SnvStream::LATENCY_SAMPLES as f32 / 48.0;
        let e2e = buf_ms * 2.0 + resamp_ms * 2.0 + algo_ms;
        let segs: [(&str, f32, egui::Color32); 5] = [
            ("buf", buf_ms, crate::theme::colors::CYAN),
            ("resamp", resamp_ms, crate::theme::colors::MINT),
            ("algo", algo_ms, crate::theme::colors::PINK),
            ("resamp", resamp_ms, crate::theme::colors::MINT),
            ("buf", buf_ms, crate::theme::colors::CYAN),
        ];
        crate::theme::info_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                crate::theme::subheading(ui, "Latency");
                ui.add_space(crate::theme::space::SMALL);
                ui.label(
                    egui::RichText::new(format!("{e2e:.0} ms E2E"))
                        .size(16.0)
                        .strong()
                        .color(crate::theme::colors::TEXT),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let old_buf = *buffer_frames;
                    egui::ComboBox::from_id_salt("rt_buffer_size")
                        .selected_text(format!("{} frames", *buffer_frames))
                        .show_ui(ui, |ui| {
                            for n in [128u32, 240, 256, 512, 1024] {
                                ui.selectable_value(buffer_frames, n, n.to_string());
                            }
                        });
                    if *buffer_frames != old_buf {
                        on_control(RtControl::SetBufferSize(*buffer_frames));
                        if *running {
                            on_control(RtControl::Stop);
                            on_control(start_msg(state));
                        }
                    }
                    ui.label(egui::RichText::new("Buffer").size(11.0).color(crate::theme::colors::TEXT_DIM));
                });
            });
            ui.add_space(crate::theme::space::TIGHT);
            let bar_h = 14.0;
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), bar_h), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            let total = e2e.max(0.001);
            let mut x = rect.min.x;
            for (_, ms, color) in segs.iter() {
                let w = rect.width() * (ms / total);
                painter.rect_filled(
                    egui::Rect::from_min_size(egui::pos2(x, rect.min.y), egui::vec2(w, bar_h)),
                    2.0,
                    crate::theme::with_alpha(*color, 200),
                );
                x += w;
            }
            ui.add_space(crate::theme::space::TIGHT);
            ui.horizontal(|ui| {
                for (name, ms, color) in segs.iter() {
                    ui.label(egui::RichText::new(format!("{name} {ms:.0}")).size(10.0).color(*color));
                    ui.add_space(crate::theme::space::TIGHT);
                }
            });
        });
        ui.add_space(crate::theme::space::MEDIUM);
    }

    // Operation bar: Bypass | Mute | semitones | Start/Stop
    crate::theme::info_card_frame().show(ui, |ui| {
        ui.horizontal(|ui| {
            if !force_bypass {
                let old_bp = *bypass;
                if crate::theme::operation_button(ui, if *bypass { "BYPASS" } else { "Bypass" }, crate::theme::OpKind::Bypass, *bypass, None) {
                    *bypass = !*bypass;
                }
                if *bypass != old_bp {
                    on_control(RtControl::Bypass(*bypass));
                }
                ui.add_space(crate::theme::space::SMALL);
                let old_mute = *muted;
                if crate::theme::pill_button(ui, if *muted { "MUTED" } else { "Mute" }, *muted) {
                    *muted = !*muted;
                }
                if *muted != old_mute {
                    on_control(RtControl::Mute(*muted));
                }
            }

            ui.add_space(crate::theme::space::SMALL);
            ui.label(egui::RichText::new("Pitch").size(11.0).color(crate::theme::colors::TEXT_DIM));
            for (k, name) in SEMITONES {
                if crate::theme::pill_button(ui, name, cfg.semitones == k) {
                    cfg.semitones = k;
                }
            }
            if cfg.semitones != semitones_before {
                on_control(RtControl::SetSemitones(cfg.semitones));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                if !*running {
                    if crate::theme::operation_button(ui, "Start", crate::theme::OpKind::Start, true, None) {
                        on_control(RtControl::Bypass(*bypass || force_bypass));
                        on_control(start_msg(state));
                        *running = true;
                    }
                } else if crate::theme::operation_button(ui, "Stop", crate::theme::OpKind::Stop, true, Some(&stop_icon)) {
                    on_control(RtControl::Stop);
                    *running = false;
                }
            });
        });
    });

    state.lock().unwrap().settings = cfg;
    ui.add_space(crate::theme::space::SMALL);
}

fn start_msg(state: &Arc<Mutex<AppState>>) -> RtControl {
    let s = state.lock().unwrap();
    RtControl::StartWithDevices { host: s.selected_host, input_idx: s.selected_input, output_idx: s.selected_output }
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
            in_rms = widgets::rms(blk);
            out_rms = widgets::rms(&out);
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
