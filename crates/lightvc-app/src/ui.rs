//! LightVC の GUI(GPUI・Material 3 Expressive)。navigation rail + 4 画面。
//!
//! デザインの正本は DESIGN.md。トークンと部品は m3.rs。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gpui::{
    div, prelude::*, px, relative, size, Animation, AnimationExt as _, AnyElement, App, Application,
    Bounds, Context, Div, FontWeight, Render, Rgba, SharedString, TitlebarOptions, Window,
    WindowAppearance, WindowBounds, WindowOptions,
};

use crate::audio_playback::{self, AudioPlayer};
use crate::file_pick::FilePick;
use crate::m3::{self, c, icon, shape, Btn, Typo};
use crate::state::{self, AppState, RtControl, RtMetrics};
use crate::voices::{VoiceEntry, VoiceGroup};

const SR: u32 = 48_000;
const SEMITONES: [(f32, &str); 4] = [(0.0, "+0"), (5.0, "+5"), (7.0, "+7"), (12.0, "+12")];
const BUFFERS: [u32; 5] = [128, 240, 256, 512, 1024];
const THREADS: [usize; 3] = [1, 2, 4];

#[derive(Copy, Clone, PartialEq)]
enum Tab { Realtime, Offline, Voices, Devices, Credits }

#[derive(Default)]
struct OfflineUi {
    source: Option<PathBuf>,
    reference: Option<PathBuf>,
    converting: bool,
    converted: Option<Vec<f32>>,
    player: Option<AudioPlayer>,
    source_preview: Option<AudioPlayer>,
    reference_preview: Option<AudioPlayer>,
}

#[derive(Default)]
struct CatalogUi {
    player: Option<AudioPlayer>,
    playing: Option<usize>,
}

#[derive(Clone, Default)]
struct Picks {
    source: FilePick,
    reference: FilePick,
    operator: FilePick,
    front: FilePick,
    model: FilePick,
    voc: FilePick,
    add: FilePick,
}

impl Picks {
    fn any_pending(&self) -> bool {
        self.source.pending() || self.reference.pending() || self.operator.pending() || self.front.pending() || self.model.pending() || self.voc.pending() || self.add.pending()
    }
}

#[derive(Default)]
struct DeviceCache {
    inputs: Vec<lightvc_audio::DeviceInfo>,
    outputs: Vec<lightvc_audio::DeviceInfo>,
}

pub struct LightVcUi {
    state: Arc<Mutex<AppState>>,
    tab: Tab,
    demo: bool,
    dark_override: Option<bool>,
    rt_running: bool,
    rt_bypass: bool,
    rt_muted: bool,
    rt_buffer: u32,
    metrics: RtMetrics,
    picks: Picks,
    offline: OfflineUi,
    catalog: CatalogUi,
    devices: DeviceCache,
    /// ボイスの画面で同梱のプリセットに掛ける表示フィルタ(None = すべて)。
    voice_filter: Option<VoiceGroup>,
    /// リアルタイム・ファイル変換の「目標の声」に出す chip の区分。
    voice_group: VoiceGroup,
    show_full_notice: bool,
}

pub fn run(state: Arc<Mutex<AppState>>, demo: bool) -> anyhow::Result<()> {
    Application::new().with_assets(m3::Assets).run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1120.), px(780.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(900.), px(600.))),
                titlebar: Some(TitlebarOptions { title: Some("LightVC".into()), ..Default::default() }),
                app_id: Some("lightvc".into()),
                ..Default::default()
            },
            |window, cx| {
                cx.new(|cx| {
                    cx.observe_window_appearance(window, |_, _, cx| cx.notify()).detach();
                    let mut ui = LightVcUi {
                        state: state.clone(),
                        tab: match std::env::var("LVC_TAB").as_deref() {
                            Ok("voices") => Tab::Voices,
                            Ok("offline") => Tab::Offline,
                            Ok("devices") => Tab::Devices,
                            Ok("credits") => Tab::Credits,
                            _ => Tab::Realtime,
                        },
                        demo,
                        dark_override: match std::env::var("LVC_THEME").as_deref() {
                            Ok("dark") => Some(true),
                            Ok("light") => Some(false),
                            _ => None,
                        },
                        rt_running: demo,
                        rt_bypass: false,
                        rt_muted: false,
                        rt_buffer: 256,
                        metrics: if demo {
                            RtMetrics {
                                input_rms: 0.11, output_rms: 0.08, latency_ms: 31.0, rtf: 0.34,
                                compute_p50: 4.1, compute_p95: 4.6, compute_p99: 4.8,
                                blocks: 960, gate_open: true, host: "wasapi".into(),
                                ..Default::default()
                            }
                        } else { RtMetrics::default() },
                        picks: Picks::default(),
                        offline: OfflineUi::default(),
                        catalog: CatalogUi::default(),
                        devices: DeviceCache::default(),
                        voice_filter: None,
                        voice_group: VoiceGroup::Female,
                        show_full_notice: false,
                    };
                    if ui.tab == Tab::Devices { ui.refresh_devices(); }
                    ui
                })
            },
        ).unwrap();
        cx.activate(true);
    });
    Ok(())
}

pub fn enable_demo(state: &Arc<Mutex<AppState>>) {
    let mut s = state.lock().unwrap();
    s.model_loaded = true;
    s.settings.operator = vec!["samples/operator.wav".into()];
    // 声は本物のプリセット(参照の音声も試聴できる)。操作者の登録だけは模擬の値。
    s.voices = crate::voices::bundled_voices();
    s.selected_voice = s.voices.first().map(|_| 0);
    s.status = "デモ表示(模擬の値)".into();
    if std::env::var("LVC_DEMO_STATE").as_deref() == Ok("loading") { s.loading = true; }
}

impl LightVcUi {
    fn pump(&mut self) {
        let mut disconnected = false;
        let mut rt_error: Option<String> = None;
        {
            let mut s = self.state.lock().unwrap();
            if let Some(ref rx) = s.rt_metrics_rx {
                while let Ok(m) = rx.try_recv() {
                    disconnected |= m.disconnected;
                    if let Some(ref err) = m.error {
                        rt_error = Some(err.clone());
                    }
                    self.metrics = m;
                }
            }
            if disconnected {
                s.error = Some(rt_error.unwrap_or_else(|| "オーディオ装置が切断されました".to_string()));
            }
            if self.offline.converting && (s.status.starts_with("Converted:") || s.error.is_some()) {
                self.offline.converting = false;
            }
            if let Some(r) = s.offline_result.take() { self.offline.converted = Some(r); }
        }
        if disconnected { self.rt_running = false; }
        if let Some(p) = self.picks.source.take() { self.offline.source = Some(p); }
        if let Some(p) = self.picks.reference.take() { self.offline.reference = Some(p); }
        if let Some(p) = self.picks.operator.take() { self.state.lock().unwrap().settings.operator.push(p); }
        if let Some(p) = self.picks.front.take() { self.state.lock().unwrap().settings.front = p.display().to_string(); }
        if let Some(p) = self.picks.model.take() { self.state.lock().unwrap().settings.model = p.display().to_string(); }
        if let Some(p) = self.picks.voc.take() { self.state.lock().unwrap().settings.voc = p.display().to_string(); }
        if let Some(p) = self.picks.add.take() {
            let name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            self.state.lock().unwrap().voices.push(VoiceEntry { name, path: p, group: VoiceGroup::Own });
        }
    }

    fn needs_tick(&self) -> bool {
        self.rt_running || self.picks.any_pending()
            || self.state.lock().map(|s| s.loading || s.operator_rec.is_some()).unwrap_or(false)
            || self.offline.converting
            || self.offline.player.as_ref().is_some_and(|p| p.is_playing())
            || self.offline.source_preview.as_ref().is_some_and(|p| p.is_playing())
            || self.offline.reference_preview.as_ref().is_some_and(|p| p.is_playing())
            || self.catalog.player.as_ref().is_some_and(|p| p.is_playing())
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        if !self.demo {
            state::ensure_rt_thread(&self.state);
            let force = !self.state.lock().unwrap().model_loaded;
            state::send_control(&self.state, RtControl::Bypass(self.rt_bypass || force));
            state::send_control(&self.state, state::start_msg(&self.state));
        }
        self.rt_running = true;
        cx.notify();
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        state::send_control(&self.state, RtControl::Stop);
        self.rt_running = false;
        cx.notify();
    }

    fn set_tab(&mut self, tab: Tab, cx: &mut Context<Self>) {
        if tab == Tab::Devices && self.tab != Tab::Devices { self.refresh_devices(); }
        self.tab = tab;
        cx.notify();
    }

    fn refresh_devices(&mut self) {
        let filter = self.state.lock().unwrap().selected_host;
        let hosts: Vec<lightvc_audio::HostPref> = match filter {
            lightvc_audio::HostPref::Default => vec![lightvc_audio::HostPref::Asio, lightvc_audio::HostPref::Wasapi],
            other => vec![other],
        };
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        for h in hosts {
            if let Ok(v) = lightvc_audio::DuplexStream::list_inputs(h) { ins.extend(v); }
            if let Ok(v) = lightvc_audio::DuplexStream::list_outputs(h) { outs.extend(v); }
        }
        self.devices.inputs = ins;
        self.devices.outputs = outs;
    }

    fn set_semitones(&mut self, k: f32, cx: &mut Context<Self>) {
        let mut s = self.state.lock().unwrap();
        s.settings.semitones = k;
        if let Some(ref tx) = s.rt_control_tx { let _ = tx.send(RtControl::SetSemitones(k)); }
        cx.notify();
    }

    fn select_voice(&mut self, i: usize, cx: &mut Context<Self>) {
        let reload = {
            let mut s = self.state.lock().unwrap();
            let group = s.voices.get(i).map(|v| v.group);
            if let Some(group) = group {
                s.selected_voice = Some(i);
                // 選んだ声が「目標の声」の chip に出る区分に切り替える
                self.voice_group = group;
            }
            s.model_loaded && !s.settings.operator.is_empty()
        };
        if reload && !self.demo { state::load_engine(&self.state); }
        cx.notify();
    }

    /// 「目標の声」の区分(女性 / 男性 / 自分)の connected button group。
    fn voice_group_segments(&self, prefix: &'static str, cx: &mut Context<Self>) -> Div {
        let groups = [VoiceGroup::Female, VoiceGroup::Male, VoiceGroup::Own];
        let mut group = m3::segments();
        for (i, g) in groups.into_iter().enumerate() {
            group = group.child(
                m3::segment(SharedString::from(format!("{prefix}_group_{i}")), g.label(), self.voice_group == g, i == 0, i == groups.len() - 1)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.voice_group = g;
                        cx.notify();
                    })),
            );
        }
        group
    }

    /// 今の区分の声の chip。`selected` は状態の voices の番号について、その声を使っているか。
    fn voice_chips(&self, prefix: &'static str, voices: &[VoiceEntry], selected: impl Fn(usize) -> bool, cx: &mut Context<Self>, on_pick: fn(&mut Self, usize, &mut Context<Self>)) -> Div {
        let mut chips = div().flex().flex_wrap().gap_2();
        let mut shown = false;
        for (i, v) in voices.iter().enumerate() {
            if v.group != self.voice_group {
                continue;
            }
            shown = true;
            chips = chips.child(
                m3::chip(SharedString::from(format!("{prefix}_{i}")), v.name.clone(), selected(i))
                    .on_click(cx.listener(move |this, _, _, cx| on_pick(this, i, cx))),
            );
        }
        if !shown {
            let text = if self.voice_group == VoiceGroup::Own { "自分で追加したボイスはありません" } else { "この区分の声はありません" };
            chips = chips.child(supporting(text));
        }
        chips
    }

    /// ファイル変換の「目標の声」に登録済みの声を入れる。
    fn pick_offline_voice(this: &mut Self, i: usize, cx: &mut Context<Self>) {
        let p = this.state.lock().unwrap().voices.get(i).map(|v| v.path.clone());
        if let Some(p) = p {
            this.offline.reference = Some(p);
        }
        cx.notify();
    }

    fn pitch_group(&self, prefix: &'static str, semitones: f32, cx: &mut Context<Self>) -> Div {
        let mut group = m3::segments();
        for (i, (k, name)) in SEMITONES.iter().enumerate() {
            let k = *k;
            group = group.child(
                m3::segment(SharedString::from(format!("{prefix}_{i}")), *name, semitones == k, i == 0, i == SEMITONES.len() - 1)
                    .on_click(cx.listener(move |this, _, _, cx| this.set_semitones(k, cx))),
            );
        }
        group
    }
}

fn file_name(p: &std::path::Path) -> String {
    p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

fn stop_player(slot: &mut Option<AudioPlayer>) {
    if let Some(p) = slot.take() { p.stop(); }
}

fn play_path(slot: &mut Option<AudioPlayer>, path: &std::path::Path) -> anyhow::Result<()> {
    let (wav, sr) = audio_playback::load_audio_mono(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let wav44 = audio_playback::resample_linear(&wav, sr);
    *slot = Some(AudioPlayer::play(wav44)?);
    Ok(())
}

fn section_title(text: &str) -> Div {
    div().title_m().text_color(c::on_surface()).child(text.to_string())
}

fn supporting(text: impl Into<SharedString>) -> Div {
    div().body_s().text_color(c::on_surface_variant()).child(text.into())
}

fn field_label(text: &str) -> Div {
    div().label_l().text_color(c::on_surface_variant()).child(text.to_string())
}

fn page_header(title: &str, subtitle: &str) -> Div {
    div().flex().items_center().gap_4().min_h(px(56.))
        .child(
            div().flex().flex_col().flex_none()
                .child(div().headline_s().text_color(c::on_surface()).child(title.to_string()))
                .child(div().body_m().text_color(c::on_surface_variant()).child(subtitle.to_string())),
        )
        .child(div().flex_1())
}

/// 補助の数値(ラベル + headline)。
fn stat(label: &str, value: String, unit: &str, fg: Rgba) -> Div {
    div().flex().flex_col().flex_none()
        .child(div().label_m().text_color(m3::alpha(fg, 0.72)).child(label.to_string()))
        .child(
            div().flex().items_baseline().gap_1()
                .child(div().headline_s().text_color(fg).child(value))
                .child(div().label_m().text_color(m3::alpha(fg, 0.72)).child(unit.to_string())),
        )
}

impl Render for LightVcUi {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        m3::set_dark(self.dark_override.unwrap_or(matches!(window.appearance(), WindowAppearance::Dark | WindowAppearance::VibrantDark)));
        self.pump();
        if self.needs_tick() { window.request_animation_frame(); }

        let tab = self.tab;
        let rail = div()
            .w(px(96.)).h_full().flex_none().flex().flex_col().items_center().py_4().gap_2()
            .child(m3::nav_item("nav_rt", "graphic_eq", "リアルタイム", tab == Tab::Realtime).on_click(cx.listener(|this, _, _, cx| this.set_tab(Tab::Realtime, cx))))
            .child(m3::nav_item("nav_off", "audio_file", "ファイル変換", tab == Tab::Offline).on_click(cx.listener(|this, _, _, cx| this.set_tab(Tab::Offline, cx))))
            .child(m3::nav_item("nav_vo", "record_voice_over", "ボイス", tab == Tab::Voices).on_click(cx.listener(|this, _, _, cx| this.set_tab(Tab::Voices, cx))))
            .child(div().flex_1())
            .child(m3::nav_item("nav_dev", "settings", "デバイス", tab == Tab::Devices).on_click(cx.listener(|this, _, _, cx| this.set_tab(Tab::Devices, cx))))
            .child(m3::nav_item("nav_cred", "info", "クレジット", tab == Tab::Credits).on_click(cx.listener(|this, _, _, cx| this.set_tab(Tab::Credits, cx))));

        let (status, error) = {
            let s = self.state.lock().unwrap();
            (s.status.clone(), s.error.clone())
        };

        let page: AnyElement = match tab {
            Tab::Realtime => self.render_realtime(&status, cx),
            Tab::Offline => self.render_offline(&status, cx),
            Tab::Voices => self.render_voices(&status, cx),
            Tab::Devices => self.render_devices(&status, cx),
            Tab::Credits => self.render_credits(cx),
        };
        let page = div().w_full().p_6().child(page).with_animation(("page", tab as usize), Animation::new(Duration::from_millis(260)), |el, d| {
            let e = 1.0 - (1.0 - d).powi(3);
            el.opacity(e).pt(px(24.0 + 14.0 * (1.0 - e)))
        });

        let mut pane = div().relative().size_full().flex().flex_col().overflow_hidden().rounded(px(shape::XL)).bg(c::surface())
            .child(div().id("scroll").flex_1().min_h_0().overflow_y_scroll().child(page));
        if let Some(err) = error {
            pane = pane.child(
                div().absolute().bottom_4().left_0().right_0().flex().justify_center().px_4().child(
                    div().max_w(px(640.)).min_h(px(48.)).pl_4().pr_1().py_1().flex().items_center().gap_3()
                        .rounded(px(shape::XS)).bg(c::inverse_surface()).shadow_lg()
                        .child(icon("error", 20., c::inverse_primary()))
                        .child(div().flex_1().body_m().text_color(c::inverse_on_surface()).child(err))
                        .child(m3::icon_button("snack_close", "close", false, c::inverse_on_surface()).on_click(cx.listener(|this, _, _, cx| {
                            this.state.lock().unwrap().error = None;
                            cx.notify();
                        }))),
                ),
            );
        }

        div().id("root").size_full().flex().bg(c::surface_container()).text_color(c::on_surface())
            .child(rail)
            .child(div().flex_1().min_w_0().h_full().py_2().pr_2().child(pane))
    }
}

impl LightVcUi {
    // ===== リアルタイム =====

    fn render_realtime(&mut self, status: &str, cx: &mut Context<Self>) -> AnyElement {
        let m = self.metrics.clone();
        let (has_engine, is_loading, cfg, voices, selected_voice, rec) = {
            let s = self.state.lock().unwrap();
            (s.model_loaded, s.loading, s.settings.clone(), s.voices.clone(), s.selected_voice,
             s.operator_rec.as_ref().map(|r| r.seconds.saturating_sub(r.started.elapsed().as_secs() as u32)))
        };
        let force_bypass = !has_engine;
        let voice_name = selected_voice.and_then(|i| voices.get(i)).map(|v| v.name.clone());
        let converting = self.rt_running && !self.rt_bypass && !force_bypass;

        // ---- hero: 状態で色が替わる主役の容器 ----
        let (state_label, state_icon, bg, fg) = if !self.rt_running {
            ("停止中", "stop_fill1", c::surface_container_high(), c::on_surface())
        } else if converting {
            ("変換中", "graphic_eq", c::primary_container(), c::on_primary_container())
        } else {
            ("バイパス中", "alt_route", c::tertiary_container(), c::on_tertiary_container())
        };
        let dim = m3::alpha(fg, 0.72);

        let mut top = div().flex().items_center().gap_3()
            .child(
                div().h(px(32.)).pl_3().pr_4().flex().flex_none().items_center().gap_2().rounded_full().bg(fg)
                    .child(icon(state_icon, 18., bg))
                    .child(div().label_l().font_weight(FontWeight::BOLD).text_color(bg).child(state_label)),
            )
            .child(div().flex_1());
        if m.overrun > 0 || m.underrun > 0 {
            top = top.child(
                div().h(px(32.)).px_3().flex().flex_none().items_center().gap_2().rounded(px(shape::SM)).bg(c::error_container())
                    .child(icon("error", 18., c::on_error_container()))
                    .child(div().label_l().text_color(c::on_error_container()).child(format!("音切れ 入力 {}・出力 {}", m.overrun, m.underrun))),
            );
        }
        if converting {
            top = top.child(
                div().h(px(32.)).px_3().flex().flex_none().items_center().gap_2().rounded(px(shape::SM)).border_1().border_color(m3::alpha(fg, 0.3))
                    .child(div().size(px(8.)).rounded_full().bg(if m.gate_open { fg } else { m3::alpha(fg, 0.3) }))
                    .child(div().label_l().text_color(fg).child(if m.gate_open { "ゲート 開" } else { "ゲート 閉" })),
            );
        }

        let route = div().flex().items_center().gap_2()
            .child(icon("mic", 22., dim))
            .child(div().title_l().text_color(dim).child("あなた"))
            .child(icon("arrow_forward", 22., dim))
            .child(div().title_l().font_weight(FontWeight::BOLD).text_color(fg).truncate().child(voice_name.clone().unwrap_or_else(|| "目標の声が未選択".into())));

        let measuring = m.blocks > 0 || m.compute_p99 > 0.0;
        let verdict = if !self.rt_running || m.latency_ms <= 0.0 { None }
            else if m.latency_ms < 30.0 { Some(("目標 30 ms 以内", false)) }
            else if m.latency_ms < 50.0 { Some(("目標 30 ms を超過", false)) }
            else { Some(("上限 50 ms を超過", true)) };
        let latency = div().flex().flex_col().flex_none()
            .child(div().label_l().text_color(dim).child("E2E 遅延"))
            .child(
                div().flex().items_baseline().gap_2()
                    .child(div().display_l().text_color(fg).child(if m.latency_ms > 0.0 { format!("{:.0}", m.latency_ms) } else { "—".into() }))
                    .child(div().title_l().text_color(dim).child("ms")),
            )
            .when_some(verdict, |s, (text, bad)| s.child(div().label_m().text_color(if bad { c::error() } else { dim }).child(text)));
        let stats = div().flex().flex_wrap().items_end().gap_x_8().gap_y_3()
            .child(latency)
            .child(div().flex().flex_wrap().gap_x_8().gap_y_3().pb(px(18.))
                .child(stat("計算 p99", if measuring { format!("{:.2}", m.compute_p99) } else { "—".into() }, "ms", fg))
                .child(stat("RTF", if measuring { format!("{:.2}", m.rtf) } else { "—".into() }, "", fg))
                .child(stat("ホスト", if m.host.is_empty() { "—".into() } else { m.host.clone() }, "", fg)));

        let buf_ms = self.rt_buffer as f32 * 1000.0 / 48_000.0;
        let (in_buf_ms, out_buf_ms, resamp_ms) = if self.rt_running && m.capture_buffer_frames > 0 && m.playback_buffer_frames > 0 {
            let csr = m.capture_sample_rate.max(1) as f32;
            let psr = m.playback_sample_rate.max(1) as f32;
            let ri = if m.capture_sample_rate == SR { 0.0 } else { 3.0 };
            let ro = if m.playback_sample_rate == SR { 0.0 } else { 3.0 };
            (m.capture_buffer_frames as f32 * 1000.0 / csr, m.playback_buffer_frames as f32 * 1000.0 / psr, ri + ro)
        } else {
            (buf_ms, buf_ms, 6.0)
        };
        let algo_ms = lightvc_core::snv_stream::SnvStream::LATENCY_SAMPLES as f32 / 48.0;
        let total = (in_buf_ms + out_buf_ms + resamp_ms + algo_ms).max(0.001);
        let parts: [(&str, f32, f32); 3] = [("バッファ(入出力)", in_buf_ms + out_buf_ms, 0.45), ("リサンプル", resamp_ms, 0.25), ("モデル", algo_ms, 1.0)];
        let mut bar = div().w_full().h(px(8.)).flex().gap(px(2.));
        let mut legend = div().flex().flex_wrap().gap_x_4().gap_y_1();
        for (name, ms, a) in parts {
            bar = bar.child(div().h_full().rounded_full().bg(m3::alpha(fg, a)).w(relative(ms / total * 0.99)));
            legend = legend.child(
                div().flex().items_center().gap(px(6.))
                    .child(div().size(px(8.)).rounded_full().bg(m3::alpha(fg, a)))
                    .child(div().body_s().text_color(dim).child(format!("{name} {ms:.1} ms"))),
            );
        }
        let budget = div().flex().flex_col().gap_2()
            .child(div().label_m().text_color(dim).child(format!("遅延の内訳(設計値 {total:.0} ms)")))
            .child(bar)
            .child(legend);

        let track = m3::alpha(fg, 0.16);
        let meters = div().flex().gap_6()
            .child(div().flex_1().flex().items_center().gap_3().child(div().label_m().flex_none().text_color(dim).child("入力")).child(m3::meter(m.input_rms * 6.0, fg, track)))
            .child(div().flex_1().flex().items_center().gap_3().child(div().label_m().flex_none().text_color(dim).child("出力")).child(m3::meter(m.output_rms * 6.0, fg, track)));

        let hero = div().bg(bg).rounded(px(shape::XL)).p_6().flex().flex_col().gap_5()
            .child(top).child(route).child(stats).child(budget).child(meters);

        // ---- 操作: toolbar + 主役のボタン ----
        let mut toolbar = div().h(px(64.)).px_3().flex().items_center().gap_2().rounded_full().bg(c::surface_container_high());
        if force_bypass {
            toolbar = toolbar.flex_1().min_w_0().child(div().px_2().flex().items_center().gap_2()
                .child(icon("info", 20., c::on_surface_variant()))
                .child(div().body_m().text_color(c::on_surface_variant()).child("モデル未読み込み — 開始すると素通し(バイパス)で動きます")));
        } else {
            toolbar = toolbar
                .child(m3::toggle_button("bypass", "alt_route", "バイパス", self.rt_bypass).on_click(cx.listener(|this, _, _, cx| {
                    this.rt_bypass = !this.rt_bypass;
                    state::send_control(&this.state, RtControl::Bypass(this.rt_bypass));
                    cx.notify();
                })))
                .child(m3::toggle_button("mute", if self.rt_muted { "volume_off" } else { "volume_up" }, "ミュート", self.rt_muted).on_click(cx.listener(|this, _, _, cx| {
                    this.rt_muted = !this.rt_muted;
                    state::send_control(&this.state, RtControl::Mute(this.rt_muted));
                    cx.notify();
                })));
        }
        let main_btn = if self.rt_running {
            m3::button_m("stop_btn", "stop_fill1", "停止", c::tertiary(), c::on_tertiary(), true, true).on_click(cx.listener(|this, _, _, cx| this.stop(cx)))
        } else {
            m3::button_m("start_btn", "play_arrow_fill1", "開始", c::primary(), c::on_primary(), false, true).on_click(cx.listener(|this, _, _, cx| this.start(cx)))
        };
        let controls = div().flex().items_center().gap_3().child(toolbar).when(!force_bypass, |s| s.child(div().flex_1())).child(main_btn);

        // ---- 調整 ----
        let mut buf_group = m3::segments();
        for (i, n) in BUFFERS.into_iter().enumerate() {
            buf_group = buf_group.child(
                m3::segment(SharedString::from(format!("buf_{n}")), n.to_string(), self.rt_buffer == n, i == 0, i == BUFFERS.len() - 1).on_click(cx.listener(move |this, _, _, cx| {
                    state::send_control(&this.state, RtControl::SetBufferSize(n));
                    this.rt_buffer = n;
                    if this.rt_running && !this.demo {
                        state::send_control(&this.state, RtControl::Stop);
                        state::send_control(&this.state, state::start_msg(&this.state));
                    }
                    cx.notify();
                })),
            );
        }
        let tuning = m3::card_outlined()
            .child(div().flex().flex_wrap().items_center().gap_3()
                .child(div().w(px(88.)).flex_none().child(field_label("ピッチ")))
                .child(self.pitch_group("rt_pitch", cfg.semitones, cx))
                .child(supporting("半音・固定")))
            .child(div().flex().flex_wrap().items_center().gap_3()
                .child(div().w(px(88.)).flex_none().child(field_label("バッファ")))
                .child(buf_group)
                .child(supporting(format!("{buf_ms:.1} ms / 片道"))))
            .when(measuring, |s| s.child(supporting(format!("計算時間 p50 {:.2} ms・p95 {:.2} ms・p99 {:.2} ms(直近 5 秒)", m.compute_p50, m.compute_p95, m.compute_p99))));

        let main = div().flex_1().min_w_0().flex().flex_col().gap_4().child(hero).child(controls).child(tuning);

        // ---- 補助の列: 目標の声・操作者・モデル ----
        let mut target = m3::card().child(
            div().flex().items_center()
                .child(section_title("目標の声"))
                .child(div().flex_1())
                .child(m3::button("tgt_manage", None, "管理", Btn::Text, true).on_click(cx.listener(|this, _, _, cx| this.set_tab(Tab::Voices, cx)))),
        );
        if voices.is_empty() {
            target = target
                .child(supporting("まだ登録がありません。ボイスの画面で参照の WAV を追加します。"))
                .child(div().flex().child(m3::button("tgt_open", Some("add"), "ボイスを追加", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| this.set_tab(Tab::Voices, cx)))));
        } else {
            // 声が 30 以上あるので、区分(女性 / 男性 / 自分)ごとに chip を出す
            target = target
                .child(self.voice_group_segments("tgt", cx))
                .child(self.voice_chips("tgt", &voices, |i| selected_voice == Some(i), cx, Self::select_voice));
        }

        let mut operator = m3::card().child(
            div().flex().items_center().gap_3()
                .child(section_title("操作者"))
                .child(div().flex_1())
                .child(supporting(if cfg.operator.is_empty() { "未登録".to_string() } else { format!("{} 本を登録済み", cfg.operator.len()) })),
        );
        operator = operator.child(supporting("自分の普通の発話を 30 秒ほど登録します。"));
        let mut op_row = div().flex().flex_wrap().items_center().gap_2();
        if let Some(rem) = rec {
            op_row = op_row
                .child(div().h(px(40.)).px_4().flex().items_center().gap_2().rounded_full().bg(c::error_container())
                    .child(m3::rec_dot())
                    .child(div().label_l().text_color(c::on_error_container()).child(format!("録音中 残り {rem} 秒"))))
                .child(m3::button("op_cancel", None, "中止", Btn::Text, true).on_click(cx.listener(|this, _, _, cx| {
                    if let Some(r) = this.state.lock().unwrap().operator_rec.as_ref() {
                        r.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    cx.notify();
                })));
        } else {
            let can_rec = !self.rt_running;
            op_row = op_row
                .child(m3::button("op_rec", Some("mic"), "録音", Btn::Tonal, can_rec).on_click(cx.listener(|this, _, _, cx| {
                    if !this.rt_running && !this.demo { state::record_operator(&this.state, 30); }
                    cx.notify();
                })))
                .child(m3::button("op_add", Some("folder_open"), "WAV", Btn::Outlined, true).on_click(cx.listener(|this, _, _, cx| {
                    this.picks.operator.open();
                    cx.notify();
                })));
            if !cfg.operator.is_empty() {
                op_row = op_row.child(m3::button("op_clear", None, "消去", Btn::Text, true).on_click(cx.listener(|this, _, _, cx| {
                    this.state.lock().unwrap().settings.operator.clear();
                    cx.notify();
                })));
            }
        }
        operator = operator.child(op_row);

        let mut model = m3::card().child(section_title("モデル"));
        let mut paths = div().flex().flex_col();
        for (id, label, cur) in [("front", "前段", cfg.front.clone()), ("model", "変換", cfg.model.clone()), ("voc", "出力部", cfg.voc.clone())] {
            paths = paths.child(
                div().h(px(40.)).flex().items_center().gap_2()
                    .child(div().w(px(48.)).flex_none().label_m().text_color(c::on_surface_variant()).child(label))
                    .child(div().flex_1().min_w_0().body_m().text_color(c::on_surface()).truncate().child(file_name(std::path::Path::new(&cur))))
                    .child(m3::icon_button(SharedString::from(format!("pick_{id}")), "folder_open", false, c::on_surface_variant()).on_click(cx.listener(move |this, _, _, cx| {
                        match id {
                            "front" => this.picks.front.open(),
                            "model" => this.picks.model.open(),
                            _ => this.picks.voc.open(),
                        }
                        cx.notify();
                    }))),
            );
        }
        model = model.child(paths);
        let mut th_group = m3::segments();
        for (i, n) in THREADS.into_iter().enumerate() {
            th_group = th_group.child(
                m3::segment(SharedString::from(format!("th_{n}")), n.to_string(), cfg.threads == n, i == 0, i == THREADS.len() - 1).on_click(cx.listener(move |this, _, _, cx| {
                    this.state.lock().unwrap().settings.threads = n;
                    cx.notify();
                })),
            );
        }
        model = model.child(div().flex().items_center().gap_3().child(field_label("スレッド")).child(div().flex_1()).child(th_group));
        let can_load = !cfg.operator.is_empty() && voice_name.is_some() && !is_loading;
        let hint = if is_loading { status.to_string() }
            else if cfg.operator.is_empty() { "操作者の登録が必要です".into() }
            else if voice_name.is_none() { "目標の声を選んでください".into() }
            else if has_engine { "読み込み済み".into() }
            else { "準備ができました".into() };
        model = model.child(
            div().flex().items_center().gap_3()
                .child(m3::button("load", Some("refresh"), if has_engine { "再読み込み" } else { "読み込む" }, if has_engine { Btn::Tonal } else { Btn::Filled }, can_load).on_click(cx.listener(|this, _, _, cx| {
                    let ok = { let s = this.state.lock().unwrap(); !s.settings.operator.is_empty() && s.selected_voice.is_some() && !s.loading };
                    if ok && !this.demo { state::load_engine(&this.state); }
                    cx.notify();
                })))
                .when(is_loading, |s| s.child(m3::loading_indicator(32.)))
                .child(div().flex_1().min_w_0().child(supporting(hint))),
        );

        let side = div().w(px(320.)).flex_none().flex().flex_col().gap_3().child(target).child(operator).child(model);

        div().flex().flex_col().gap_4()
            .child(page_header("リアルタイム変換", "マイクの声を目標の声へ、その場で変換します"))
            .child(div().flex().items_start().gap_4().child(main).child(side))
            .into_any_element()
    }

    // ===== ファイル変換 =====

    fn render_offline(&mut self, status: &str, cx: &mut Context<Self>) -> AnyElement {
        let (semitones, has_operator, voices) = {
            let s = self.state.lock().unwrap();
            (s.settings.semitones, !s.settings.operator.is_empty(), s.voices.clone())
        };

        let slot = |title: &str, sub: &str, glyph: &str, path: &Option<PathBuf>| {
            m3::card().flex_1().min_w_0().gap_4()
                .child(div().flex().items_center().gap_3()
                    .child(m3::avatar(glyph, c::primary_container(), c::on_primary_container()))
                    .child(div().flex().flex_col().child(section_title(title)).child(supporting(sub.to_string()))))
                .child(match path {
                    Some(p) => div().body_l().text_color(c::on_surface()).truncate().child(file_name(p)),
                    None => div().body_l().text_color(c::on_surface_variant()).child("未選択"),
                })
        };

        let src_playing = self.offline.source_preview.as_ref().is_some_and(|p| p.is_playing());
        let source = slot("変換元", "変換したい音声(WAV)", "audio_file", &self.offline.source).child(
            div().flex().gap_2()
                .child(m3::button("src_browse", Some("folder_open"), "選ぶ", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| {
                    this.picks.source.open();
                    cx.notify();
                })))
                .child(m3::button("src_play", Some(if src_playing { "stop_fill1" } else { "play_arrow_fill1" }), if src_playing { "停止" } else { "試聴" }, Btn::Outlined, self.offline.source.is_some()).on_click(cx.listener(|this, _, _, cx| {
                    if this.offline.source_preview.is_some() {
                        stop_player(&mut this.offline.source_preview);
                    } else if let Some(p) = this.offline.source.clone()
                        && let Err(e) = play_path(&mut this.offline.source_preview, &p)
                    {
                        this.state.lock().unwrap().error = Some(format!("再生できません: {e}"));
                    }
                    cx.notify();
                }))),
        );

        let ref_playing = self.offline.reference_preview.as_ref().is_some_and(|p| p.is_playing());
        let mut reference = slot("目標の声", "普通の発話を 20 秒以上", "record_voice_over", &self.offline.reference).child(
            div().flex().gap_2()
                .child(m3::button("ref_browse", Some("folder_open"), "選ぶ", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| {
                    this.picks.reference.open();
                    cx.notify();
                })))
                .child(m3::button("ref_play", Some(if ref_playing { "stop_fill1" } else { "play_arrow_fill1" }), if ref_playing { "停止" } else { "試聴" }, Btn::Outlined, self.offline.reference.is_some()).on_click(cx.listener(|this, _, _, cx| {
                    if this.offline.reference_preview.is_some() {
                        stop_player(&mut this.offline.reference_preview);
                    } else if let Some(p) = this.offline.reference.clone()
                        && let Err(e) = play_path(&mut this.offline.reference_preview, &p)
                    {
                        this.state.lock().unwrap().error = Some(format!("再生できません: {e}"));
                    }
                    cx.notify();
                }))),
        );
        if !voices.is_empty() {
            let current = self.offline.reference.clone();
            reference = reference
                .child(self.voice_group_segments("cat", cx))
                .child(self.voice_chips("cat", &voices, |i| current.as_ref().is_some_and(|p| voices.get(i).is_some_and(|v| &v.path == p)), cx, Self::pick_offline_voice));
        }

        let can = self.offline.source.is_some() && self.offline.reference.is_some() && has_operator && !self.offline.converting;
        let hint = if self.offline.converting { "変換しています…".to_string() }
            else if !has_operator { "操作者の登録がありません(リアルタイムの画面で登録します)".into() }
            else if self.offline.source.is_none() { "変換元を選んでください".into() }
            else if self.offline.reference.is_none() { "目標の声を選んでください".into() }
            else { status.to_string() };
        let action = div().flex().flex_wrap().items_center().gap_4()
            .child(m3::button_m("convert", "auto_awesome_fill1", "変換する", c::primary(), c::on_primary(), false, can).on_click(cx.listener(|this, _, _, cx| {
                let has_operator = !this.state.lock().unwrap().settings.operator.is_empty();
                if this.offline.source.is_none() || this.offline.reference.is_none() || this.offline.converting || !has_operator || this.demo { return; }
                let (src, refp) = (this.offline.source.clone().unwrap(), this.offline.reference.clone().unwrap());
                { let mut s = this.state.lock().unwrap(); s.offline_result = None; s.error = None; s.status = "Converting...".to_string(); }
                this.offline.converted = None;
                this.offline.player = None;
                this.offline.converting = true;
                let st = this.state.clone();
                std::thread::spawn(move || crate::offline_tab::run_offline_conversion(st, src, refp));
                cx.notify();
            })))
            .when(self.offline.converting, |s| s.child(m3::loading_indicator(40.)))
            .child(div().flex().items_center().gap_3().child(field_label("ピッチ")).child(self.pitch_group("off_pitch", semitones, cx)))
            .child(div().flex_1().min_w_0().child(supporting(hint)));

        let mut el = div().flex().flex_col().gap_4()
            .child(page_header("ファイル変換", "録音済みの音声を目標の声に変換します"))
            .child(div().flex().gap_4().child(source).child(reference))
            .child(action);

        if let Some(samples) = &self.offline.converted {
            let out_playing = self.offline.player.as_ref().is_some_and(|p| p.is_playing());
            let seconds = samples.len() as f32 / SR as f32;
            let fg = c::on_tertiary_container();
            el = el.child(
                div().bg(c::tertiary_container()).rounded(px(shape::XL)).p_6().flex().flex_wrap().items_center().gap_6()
                    .child(m3::avatar("check", fg, c::tertiary_container()))
                    .child(stat("変換結果", format!("{seconds:.1}"), "秒・48 kHz", fg))
                    .child(div().flex_1())
                    .child(m3::button("out_play", Some(if out_playing { "stop_fill1" } else { "play_arrow_fill1" }), if out_playing { "停止" } else { "再生" }, Btn::Filled, true).on_click(cx.listener(|this, _, _, cx| {
                        if this.offline.player.is_some() {
                            stop_player(&mut this.offline.player);
                        } else if let Some(samples) = this.offline.converted.clone() {
                            let wav44 = audio_playback::resample_linear(&samples, SR);
                            this.offline.player = AudioPlayer::play(wav44).ok();
                        }
                        cx.notify();
                    })))
                    .child(m3::button("out_save", Some("download"), "保存", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| {
                        if let Some(samples) = this.offline.converted.clone()
                            && let Some(path) = rfd::FileDialog::new().save_file()
                        {
                            let mut s = this.state.lock().unwrap();
                            match audio_playback::save_wav_mono(&path, &samples, SR) {
                                Ok(()) => s.status = format!("保存しました: {}", path.display()),
                                Err(e) => s.error = Some(format!("保存できません: {e}")),
                            }
                        }
                        cx.notify();
                    }))),
            );
        }
        el.into_any_element()
    }

    // ===== ボイス =====

    fn render_voices(&mut self, _status: &str, cx: &mut Context<Self>) -> AnyElement {
        let (voices, selected_voice) = {
            let s = self.state.lock().unwrap();
            (s.voices.clone(), s.selected_voice)
        };
        let header = page_header("ボイス", "同梱のプリセットから選ぶか、目標の声の参照(普通の発話 20 秒以上)を追加します")
            .child(m3::button("voice_add", Some("add"), "ボイスを追加", Btn::Filled, true).on_click(cx.listener(|this, _, _, cx| {
                this.picks.add.open();
                cx.notify();
            })));
        let mut el = div().flex().flex_col().gap_4().child(header);
        if voices.is_empty() {
            return el.child(
                div().py_16().flex().flex_col().items_center().gap_3().rounded(px(shape::XL)).bg(c::surface_container())
                    .child(div().size(px(72.)).flex().items_center().justify_center().rounded(px(shape::XL)).bg(c::primary_container())
                        .child(icon("record_voice_over", 36., c::on_primary_container())))
                    .child(div().title_l().text_color(c::on_surface()).child("まだボイスがありません"))
                    .child(div().body_m().text_color(c::on_surface_variant()).child("WAV / FLAC を選ぶと、ファイル名で登録されます")),
            ).into_any_element();
        }

        // 行の並び(番号 = 状態の voices の番号)。同梱は表示フィルタの区分だけ、自分で追加した声は全部。
        let presets: Vec<usize> = (0..voices.len()).filter(|&i| voices[i].group != VoiceGroup::Own && self.voice_filter.is_none_or(|g| g == voices[i].group)).collect();
        let own: Vec<usize> = (0..voices.len()).filter(|&i| voices[i].group == VoiceGroup::Own).collect();

        if voices.iter().any(|v| v.group != VoiceGroup::Own) {
            let filters = [(None, "すべて"), (Some(VoiceGroup::Female), "女性"), (Some(VoiceGroup::Male), "男性")];
            let mut group = m3::segments();
            for (i, (g, label)) in filters.into_iter().enumerate() {
                group = group.child(
                    m3::segment(SharedString::from(format!("voice_filter_{i}")), label, self.voice_filter == g, i == 0, i == filters.len() - 1)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.voice_filter = g;
                            cx.notify();
                        })),
                );
            }
            el = el
                .child(div().flex().flex_wrap().items_center().gap_3().child(section_title("同梱のプリセット")).child(div().flex_1()).child(group))
                .child(supporting("Irodori-TTS で作った架空の声です。実在の人物の録音ではありません。"));
            el = if presets.is_empty() {
                el.child(supporting("この区分のプリセットはありません"))
            } else {
                el.child(self.voice_rows(&voices, &presets, selected_voice, false, cx))
            };
        }
        if !own.is_empty() {
            el = el.child(section_title("自分で追加したボイス")).child(self.voice_rows(&voices, &own, selected_voice, true, cx));
        }
        el.into_any_element()
    }

    /// 声の一覧の行。`ids` は状態の voices の番号(この並びで行の角を整える)。自分で追加した声(`own`)だけ削除できる。
    fn voice_rows(&self, voices: &[VoiceEntry], ids: &[usize], selected_voice: Option<usize>, own: bool, cx: &mut Context<Self>) -> Div {
        let mut list = m3::list();
        let n = ids.len();
        for (row, &i) in ids.iter().enumerate() {
            let v = &voices[i];
            let selected = selected_voice == Some(i);
            let playing = self.catalog.playing == Some(i) && self.catalog.player.as_ref().is_some_and(|p| p.is_playing());
            let path = v.path.clone();
            let sub = if selected { c::on_secondary_container() } else { c::on_surface_variant() };
            let detail = if own { path.display().to_string() } else { format!("{}・同梱", v.group.label()) };
            let mut item = m3::list_item(SharedString::from(format!("voice_{i}")), selected, row == 0, row == n - 1).min_h(px(72.))
                .child(
                    div().id(SharedString::from(format!("voice_sel_{i}"))).flex_1().min_w_0().flex().items_center().gap_4().cursor_pointer()
                        .on_click(cx.listener(move |this, _, _, cx| this.select_voice(i, cx)))
                        .child(if selected { m3::avatar("check", c::secondary(), c::on_secondary()) } else { m3::avatar("person", c::primary_container(), c::on_primary_container()) })
                        .child(div().flex_1().min_w_0().flex().flex_col()
                            .child(div().body_l().font_weight(FontWeight::MEDIUM).truncate().child(v.name.clone()))
                            .child(div().body_m().text_color(sub).truncate().child(detail)))
                        .when(selected, |s| s.child(div().label_m().flex_none().text_color(sub).child("使用中"))),
                )
                .child(m3::icon_button(SharedString::from(format!("voice_play_{i}")), if playing { "stop_fill1" } else { "play_arrow_fill1" }, true, sub).on_click(cx.listener(move |this, _, _, cx| {
                    let was_playing = this.catalog.playing == Some(i) && this.catalog.player.is_some();
                    stop_player(&mut this.catalog.player);
                    this.catalog.playing = None;
                    if !was_playing {
                        match play_path(&mut this.catalog.player, &path) {
                            Ok(()) => {
                                this.catalog.playing = Some(i);
                                this.state.lock().unwrap().error = None;
                            }
                            Err(e) => this.state.lock().unwrap().error = Some(format!("再生できません: {e}")),
                        }
                    }
                    cx.notify();
                })));
            if own {
                item = item.child(m3::icon_button(SharedString::from(format!("voice_del_{i}")), "delete", false, sub).on_click(cx.listener(move |this, _, _, cx| {
                    stop_player(&mut this.catalog.player);
                    this.catalog.playing = None;
                    let mut s = this.state.lock().unwrap();
                    if i < s.voices.len() {
                        s.voices.remove(i);
                        s.selected_voice = match s.selected_voice {
                            Some(sel) if sel == i => None,
                            Some(sel) if sel > i => Some(sel - 1),
                            other => other,
                        };
                    }
                    cx.notify();
                })));
            }
            list = list.child(item);
        }
        list
    }

    // ===== デバイス =====

    fn render_devices(&mut self, _status: &str, cx: &mut Context<Self>) -> AnyElement {
        let (sel_in_host, sel_in, sel_in_name, sel_out_host, sel_out, sel_out_name, sel_host) = {
            let s = self.state.lock().unwrap();
            (
                s.selected_input_host,
                s.selected_input,
                s.selected_input_name.clone(),
                s.selected_output_host,
                s.selected_output,
                s.selected_output_name.clone(),
                s.selected_host,
            )
        };
        let mut hosts: Vec<(String, lightvc_audio::HostPref)> = vec![("すべて".to_string(), lightvc_audio::HostPref::Default)];
        for h in lightvc_audio::host::available() {
            if let Ok(p) = lightvc_audio::HostPref::parse(h) { hosts.push((h.to_uppercase(), p)); }
        }
        let mut host_group = m3::segments();
        let n = hosts.len();
        for (i, (label, pref)) in hosts.into_iter().enumerate() {
            host_group = host_group.child(
                m3::segment(SharedString::from(format!("host_{i}")), label, pref == sel_host, i == 0, i == n - 1).on_click(cx.listener(move |this, _, _, cx| {
                    this.state.lock().unwrap().selected_host = pref;
                    this.refresh_devices();
                    cx.notify();
                })),
            );
        }
        let host_card = m3::card_outlined()
            .child(
                div().flex().flex_wrap().items_center().gap_4()
                    .child(field_label("表示フィルタ"))
                    .child(host_group)
                    .child(div().flex_1())
                    .child(m3::button("dev_refresh", Some("refresh"), "再読込", Btn::Tonal, true).on_click(cx.listener(|this, _, _, cx| {
                        this.refresh_devices();
                        cx.notify();
                    }))),
            )
            .child(supporting("入力と出力で異なるホスト(例: WASAPI 入力 + ASIO 出力)を混在できます。ASIO 同士は同じデバイスを選んでください(30 ms 級の遅延には ASIO 推奨)。"));

        let column = |title: &str, glyph: &str, input: bool, devices: &[lightvc_audio::DeviceInfo], sel_h: lightvc_audio::HostPref, sel_idx: Option<usize>, sel_name: Option<&str>, cx: &mut Context<Self>| {
            let mut list = m3::list();
            let n = devices.len();
            for (i, d) in devices.iter().enumerate() {
                let selected = if let Some(name) = sel_name {
                    d.host == sel_h && d.name == name
                } else {
                    sel_idx == Some(d.host_index) && (sel_h == lightvc_audio::HostPref::Default || d.host == sel_h)
                };
                let sub = if selected { c::on_secondary_container() } else { c::on_surface_variant() };
                let dev_host = d.host;
                let dev_idx = d.host_index;
                let dev_name = d.name.clone();
                let badge = dev_host.label().to_uppercase();
                list = list.child(
                    m3::list_item(SharedString::from(format!("{}_{i}", if input { "in" } else { "out" })), selected, i == 0, i == n - 1).cursor_pointer()
                        .on_click(cx.listener(move |this, _, _, cx| {
                            let mut s = this.state.lock().unwrap();
                            if input {
                                if selected {
                                    s.selected_input_host = lightvc_audio::HostPref::Default;
                                    s.selected_input = None;
                                    s.selected_input_name = None;
                                } else {
                                    s.selected_input_host = dev_host;
                                    s.selected_input = Some(dev_idx);
                                    s.selected_input_name = Some(dev_name.clone());
                                }
                            } else if selected {
                                s.selected_output_host = lightvc_audio::HostPref::Default;
                                s.selected_output = None;
                                s.selected_output_name = None;
                            } else {
                                s.selected_output_host = dev_host;
                                s.selected_output = Some(dev_idx);
                                s.selected_output_name = Some(dev_name.clone());
                            }
                            cx.notify();
                        }))
                        .child(div().flex_1().min_w_0().flex().flex_col()
                            .child(div().body_l().truncate().child(d.name.clone()))
                            .child(div().body_s().text_color(sub).child(format!("{badge}・{} Hz・{} ch", d.sample_rate, d.channels))))
                        .when(selected, |s| s.child(icon("check", 24., sub))),
                );
            }
            let status_txt = if sel_idx.is_none() && sel_name.is_none() {
                "システムの既定を使用".to_string()
            } else {
                format!("{} を選択中", sel_h.label().to_uppercase())
            };
            div().flex_1().min_w_0().flex().flex_col().gap_3()
                .child(div().flex().items_center().gap_2().child(icon(glyph, 22., c::on_surface_variant())).child(section_title(title))
                    .child(div().flex_1()).child(supporting(status_txt)))
                .when(devices.is_empty(), |s| s.child(m3::card().child(supporting("装置が見つかりません"))))
                .child(list)
        };
        let inputs = self.devices.inputs.clone();
        let outputs = self.devices.outputs.clone();
        let lists = div().flex().items_start().gap_4()
            .child(column("入力", "mic", true, &inputs, sel_in_host, sel_in, sel_in_name.as_deref(), cx))
            .child(column("出力", "volume_up", false, &outputs, sel_out_host, sel_out, sel_out_name.as_deref(), cx));

        div().flex().flex_col().gap_4()
            .child(page_header("デバイス", "音声の入出力とホストを選びます"))
            .child(host_card)
            .child(lists)
            .into_any_element()
    }

    // ===== クレジット =====

    fn render_credits(&mut self, cx: &mut Context<Self>) -> AnyElement {
        const NOTICE_TEXT: &str = include_str!("../../../NOTICE");
        let row = |title: &str, detail: &str| {
            div().flex().flex_col().gap(px(2.))
                .child(div().label_l().text_color(c::on_surface()).child(title.to_string()))
                .child(supporting(detail.to_string()))
        };

        let link_btn = |id: &'static str, label: &'static str, url: &'static str, kind: Btn, cx: &mut Context<Self>| {
            m3::button(id, Some("arrow_forward"), label, kind, true)
                .on_click(cx.listener(move |_, _, _, cx| cx.open_url(url)))
        };

        let app_card = m3::card()
            .child(
                div().flex().flex_wrap().items_baseline().gap_3()
                    .child(div().title_l().text_color(c::on_surface()).child(format!("LightVC v{}", env!("CARGO_PKG_VERSION"))))
                    .child(div().label_l().text_color(c::primary()).child("MIT License")),
            )
            .child(supporting("リアルタイム声質変換アプリケーション (推論経路: SnvStream・48 kHz・240 サンプル / 5 ms)"))
            .child(
                div().flex().flex_wrap().items_center().gap_2().pt_1()
                    .child(link_btn("cred_home", "作者ホームページ (oto-nari.com)", "https://oto-nari.com", Btn::Tonal, cx))
                    .child(link_btn("cred_x", "X (@mu_sette)", "https://x.com/mu_sette", Btn::Outlined, cx))
                    .child(link_btn("cred_gh", "GitHub (kjranyone/LightVC)", "https://github.com/kjranyone/LightVC", Btn::Outlined, cx)),
            );

        let models_card = m3::card()
            .child(section_title("学習済みモデルとデータの帰属"))
            .child(row(
                "前段・ゲート (models/front_gate)",
                "Irodori-TTS-v4.1-Small (Aratako / MIT)、WORLD (M. Morise / modified BSD)、非音声音源 (MUSAN・FMA・FSD50K の CC0 / Public Domain / CC BY のみ・音声除去済み。個別帰属は同梱の ATTRIBUTION_SOURCES.md)",
            ))
            .child(row(
                "変換モデル (models/k3g)",
                "Irodori-TTS-v4.1-Small (Aratako / MIT・VoiceDesign による架空の 564 話者、実在話者の参照なし。台詞出典の非公開データセットは非同梱)",
            ))
            .child(row(
                "出力部ボコーダ (models/rvoc3hi)",
                "非公開の音声データセット (非同梱) および同音声を参照とした Irodori-TTS-600M-v3-VoiceDesign (Aratako / MIT)",
            ))
            .child(row(
                "同梱のボイスプリセット (voices/)",
                "Irodori-TTS-v4.1 Anime (phasefield-audio / MIT) と Irodori-TTS-v4.1-Small (Aratako / MIT) で作った架空の声 30 人(女性 15・男性 15)。実在の人物の録音は使っていません",
            ));

        let third_party_items: [(&str, &str); 5] = [
            ("GPUI (Zed Industries / Apache-2.0)", "デスクトップ GUI フレームワーク"),
            ("Material Symbols Rounded (Google / Apache-2.0)", "UI アイコン (assets/icons/*.svg)"),
            ("WORLD (M. Morise / modified BSD)", "スペクトル包絡処理 (crates/lightvc-core/src/cheaptrick.rs)"),
            ("音声 I/O (cpal・rubato・rtrb・hound・claxon)", "WASAPI / リサンプリング / リングバッファ / FLAC の読み込み。ASIO は Windows ネイティブ COM 連携を使用し、Steinberg ASIO SDK は同梱していません"),
            ("クレート依存の許諾方針", "MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause / ISC / Zlib / Unicode-3.0 / CC0-1.0 (self_cell は Apache-2.0 を選択、GPLv3 依存なし)"),
        ];
        let mut third_party_list = m3::list();
        let n = third_party_items.len();
        for (i, (title, detail)) in third_party_items.into_iter().enumerate() {
            third_party_list = third_party_list.child(
                m3::list_item(SharedString::from(format!("cred_tp_{i}")), false, i == 0, i == n - 1)
                    .child(row(title, detail)),
            );
        }

        let show_notice = self.show_full_notice;
        let notice_toggle = div().flex().items_center().gap_3()
            .child(m3::button(
                "toggle_notice",
                Some("info"),
                if show_notice { "NOTICE 全文を閉じる" } else { "NOTICE 全文を表示" },
                if show_notice { Btn::Tonal } else { Btn::Outlined },
                true,
            ).on_click(cx.listener(|this, _, _, cx| {
                this.show_full_notice = !this.show_full_notice;
                cx.notify();
            })));

        let mut el = div().flex().flex_col().gap_4()
            .child(page_header("クレジット", "ソフトウェア・学習済みモデル・第三者ライセンスの帰属表示"))
            .child(app_card)
            .child(models_card)
            .child(div().flex().flex_col().gap_2().child(section_title("第三者ソフトウェア・素材")).child(third_party_list))
            .child(notice_toggle);

        if show_notice {
            el = el.child(
                m3::card_outlined().child(
                    div().body_s().text_color(c::on_surface_variant()).child(NOTICE_TEXT),
                ),
            );
        }

        el.into_any_element()
    }
}
