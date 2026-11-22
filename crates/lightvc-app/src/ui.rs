//! LightVC の GUI(GPUI)。3 タブ(Offline / Realtime / Voices)+ 設定 + 状態バー。
//!
//! 描画は要求駆動: 動作中(リアルタイム・読み込み・録音・変換・試聴・ダイアログ)だけ
//! request_animation_frame で次フレームを要求し、止まっているときは再描画しない。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use gpui::{
    div, prelude::*, px, rgb, rgba, size, AnyElement, App, Application, Bounds, Context,
    ElementId, FontWeight, InteractiveElement, Render, SharedString, Stateful, Styled,
    TitlebarOptions, Window, WindowBounds, WindowOptions,
};

use crate::audio_playback::{self, AudioPlayer};
use crate::file_pick::FilePick;
use crate::state::{self, AppState, RtControl, RtMetrics};

const SR: u32 = 48_000;
const SEMITONES: [(f32, &str); 4] = [(0.0, "+0"), (5.0, "+5"), (7.0, "+7"), (12.0, "+12")];
const BUFFERS: [u32; 5] = [128, 240, 256, 512, 1024];
const METER_W: f32 = 170.0;

// きゃわわ Future Bass の色(旧 egui テーマと同じ系統)。rgb/hsla は const fn ではないので関数で持つ。
mod col {
    use gpui::{hsla, rgb, Hsla, Rgba};
    pub fn bg() -> Rgba {
        rgb(0xEE_E4_F2)
    }
    pub fn card() -> Rgba {
        rgb(0xFD_FC_FE)
    }
    pub fn card_tint() -> Rgba {
        rgb(0xF8_F6_FC)
    }
    pub fn border() -> Hsla {
        hsla(280.0 / 360.0, 0.25, 0.72, 0.55)
    }
    pub fn text() -> Rgba {
        rgb(0x3A_2A_46)
    }
    pub fn text_dim() -> Rgba {
        rgb(0x7A_6A_88)
    }
    pub fn text_muted() -> Rgba {
        rgb(0xA8_98_B8)
    }
    pub fn pink() -> Rgba {
        rgb(0xE0_5A_8A)
    }
    pub fn pink_bright() -> Rgba {
        rgb(0xEC_80_A8)
    }
    pub fn lavender() -> Rgba {
        rgb(0x88_78_C0)
    }
    pub fn cyan() -> Rgba {
        rgb(0x38_A8_C8)
    }
    pub fn mint() -> Rgba {
        rgb(0x2E_B8_8C)
    }
    pub fn error() -> Rgba {
        rgb(0xD0_40_70)
    }
    pub fn white() -> Rgba {
        rgb(0xFF_FF_FF)
    }
    pub fn panel_dark() -> Rgba {
        rgb(0x1E_16_30)
    }
    pub fn track() -> Rgba {
        rgb(0xE8_E0_F0)
    }
}

#[derive(Copy, Clone, PartialEq)]
enum Tab {
    Offline,
    Realtime,
    Catalog,
}

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
struct SettingsCache {
    inputs: Vec<lightvc_audio::DeviceInfo>,
    outputs: Vec<lightvc_audio::DeviceInfo>,
    hosts: Vec<&'static str>,
}

pub struct LightVcUi {
    state: Arc<Mutex<AppState>>,
    tab: Tab,
    demo: bool,
    rt_running: bool,
    rt_bypass: bool,
    rt_muted: bool,
    rt_buffer: u32,
    metrics: RtMetrics,
    picks: Picks,
    offline: OfflineUi,
    catalog: CatalogUi,
    settings_open: bool,
    settings: SettingsCache,
}

pub fn run(state: Arc<Mutex<AppState>>, demo: bool) -> anyhow::Result<()> {
    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1000.), px(720.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions { title: Some("LightVC".into()), ..Default::default() }),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| LightVcUi {
                    state: state.clone(),
                    tab: match std::env::var("LVC_TAB").as_deref() {
                        Ok("voices") => Tab::Catalog,
                        Ok("realtime") => Tab::Realtime,
                        Ok("offline") => Tab::Offline,
                        _ if demo => Tab::Realtime,
                        _ => Tab::Offline,
                    },
                    demo,
                    rt_running: demo,
                    rt_bypass: false,
                    rt_muted: false,
                    rt_buffer: 256,
                    metrics: if demo {
                        RtMetrics {
                            input_rms: 0.18,
                            output_rms: 0.12,
                            latency_ms: 31.0,
                            rtf: 0.34,
                            compute_p50: 4.1,
                            compute_p95: 4.6,
                            compute_p99: 4.8,
                            blocks: 960,
                            gate_open: true,
                            host: "wasapi".into(),
                            ..Default::default()
                        }
                    } else {
                        RtMetrics::default()
                    },
                    picks: Picks::default(),
                    offline: OfflineUi::default(),
                    catalog: CatalogUi::default(),
                    settings_open: false,
                    settings: SettingsCache::default(),
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });
    Ok(())
}

/// デモ値を入れる(スクリーンショット撮影用・モデルも装置も要らない)。
pub fn enable_demo(state: &Arc<Mutex<AppState>>) {
    let mut s = state.lock().unwrap();
    s.model_loaded = true;
    s.settings.operator = vec!["samples/operator.wav".into()];
    s.voices = vec![
        state::VoiceEntry { name: "Venus (warm)".into(), path: "samples/venus.wav".into() },
        state::VoiceEntry { name: "Mars (bright)".into(), path: "samples/mars.wav".into() },
        state::VoiceEntry { name: "Lyra (soft)".into(), path: "samples/lyra.wav".into() },
    ];
    s.selected_voice = Some(0);
    s.status = "Demo mode".into();
}

impl LightVcUi {
    // ---- 台帳の汲み上げ(描画の頭でまとめて) ----
    fn pump(&mut self) {
        let mut disconnected = false;
        {
            let mut s = self.state.lock().unwrap();
            if let Some(ref rx) = s.rt_metrics_rx {
                while let Ok(m) = rx.try_recv() {
                    disconnected |= m.disconnected;
                    self.metrics = m;
                }
            }
            if disconnected {
                s.status = "Audio device disconnected".to_string();
            }
            if self.offline.converting && (s.status.starts_with("Converted:") || s.error.is_some()) {
                self.offline.converting = false;
            }
            if let Some(r) = s.offline_result.take() {
                self.offline.converted = Some(r);
            }
        }
        if disconnected {
            self.rt_running = false;
        }
        if let Some(p) = self.picks.source.take() {
            self.offline.source = Some(p);
        }
        if let Some(p) = self.picks.reference.take() {
            self.offline.reference = Some(p);
        }
        if let Some(p) = self.picks.operator.take() {
            self.state.lock().unwrap().settings.operator.push(p);
        }
        if let Some(p) = self.picks.front.take() {
            self.state.lock().unwrap().settings.front = p.display().to_string();
        }
        if let Some(p) = self.picks.model.take() {
            self.state.lock().unwrap().settings.model = p.display().to_string();
        }
        if let Some(p) = self.picks.voc.take() {
            self.state.lock().unwrap().settings.voc = p.display().to_string();
        }
        if let Some(p) = self.picks.add.take() {
            let name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            self.state.lock().unwrap().voices.push(state::VoiceEntry { name, path: p });
        }
    }

    fn needs_tick(&self) -> bool {
        self.demo
            || self.rt_running
            || self.picks.any_pending()
            || self.state.lock().map(|s| s.loading || s.operator_rec.is_some()).unwrap_or(false)
            || self.offline.converting
            || self.offline.player.as_ref().is_some_and(|p| p.is_playing())
            || self.offline.source_preview.as_ref().is_some_and(|p| p.is_playing())
            || self.offline.reference_preview.as_ref().is_some_and(|p| p.is_playing())
            || self.catalog.player.as_ref().is_some_and(|p| p.is_playing())
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        state::ensure_rt_thread(&self.state);
        let force = !self.state.lock().unwrap().model_loaded;
        state::send_control(&self.state, RtControl::Bypass(self.rt_bypass || force));
        state::send_control(&self.state, state::start_msg(&self.state));
        self.rt_running = true;
        cx.notify();
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        state::send_control(&self.state, RtControl::Stop);
        self.rt_running = false;
        cx.notify();
    }

    fn pick_settings(&mut self) {
        let host = lightvc_audio::host::pick(self.state.lock().unwrap().selected_host).unwrap_or_else(|_| cpal::default_host());
        self.settings.hosts = lightvc_audio::host::available();
        self.settings.inputs = lightvc_audio::DuplexStream::list_input_devices_on(&host).unwrap_or_default();
        self.settings.outputs = lightvc_audio::DuplexStream::list_output_devices_on(&host).unwrap_or_default();
    }
}

// ---- 部品 ----

enum Btn {
    Primary,
    Dark,
    Ghost,
    Danger,
}

fn btn(id: impl Into<ElementId>, label: impl Into<SharedString>, kind: Btn) -> Stateful<gpui::Div> {
    div()
        .id(id)
        .px_4()
        .py_1()
        .rounded_md()
        .text_size(px(13.))
        .map(|s| match kind {
            Btn::Primary => s.bg(col::pink()).text_color(col::white()),
            Btn::Dark => s.bg(col::panel_dark()).text_color(col::white()),
            Btn::Ghost => s.bg(col::card()).text_color(col::text()).border_1().border_color(col::border()),
            Btn::Danger => s.bg(col::card()).text_color(col::error()).border_1().border_color(col::border()),
        })
        .hover(|s| match kind {
            Btn::Primary => s.bg(col::pink_bright()),
            _ => s.bg(col::card_tint()),
        })
        .child(label.into())
}

fn pill(id: impl Into<ElementId>, label: impl Into<SharedString>, selected: bool) -> Stateful<gpui::Div> {
    div()
        .id(id)
        .px_3()
        .py_1()
        .rounded_full()
        .text_size(px(12.))
        .map(|s| {
            if selected {
                s.bg(col::pink()).text_color(col::white())
            } else {
                s.bg(col::card()).text_color(col::text_dim()).border_1().border_color(col::border())
            }
        })
        .hover(|s| s.bg(col::pink_bright()))
        .child(label.into())
}

fn card() -> gpui::Div {
    div().bg(col::card()).rounded_2xl().border_1().border_color(col::border()).p_4().flex().flex_col().gap_2()
}

fn heading(text: &str) -> gpui::Div {
    div().text_size(px(14.)).text_color(col::text()).font_weight(FontWeight::SEMIBOLD).child(text.to_string())
}

fn subtext(text: impl Into<SharedString>) -> gpui::Div {
    div().text_size(px(11.)).text_color(col::text_dim()).child(text.into())
}

fn meter(level: f32, color: gpui::Rgba) -> gpui::Div {
    let w = (METER_W * level.clamp(0.0, 1.0)).max(2.0);
    div().w(px(METER_W)).h_2().rounded_full().bg(col::track()).child(div().w(px(w)).h_full().rounded_full().bg(color))
}

fn stat(value: String, unit: &str) -> gpui::Div {
    div()
        .px_3()
        .py_1()
        .rounded_lg()
        .bg(col::card_tint())
        .flex()
        .items_center()
        .gap_1()
        .child(div().text_size(px(16.)).font_weight(FontWeight::BOLD).text_color(col::text()).child(value))
        .child(div().text_size(px(10.)).text_color(col::text_muted()).child(unit.to_string()))
}

fn toggle_stop(player_slot: &mut Option<AudioPlayer>) {
    if let Some(p) = player_slot.take() {
        p.stop();
    }
}

fn play_path(player_slot: &mut Option<AudioPlayer>, path: &PathBuf) {
    if let Ok((wav, sr)) = audio_playback::load_wav_mono(path) {
        let wav44 = audio_playback::resample_linear(&wav, sr);
        *player_slot = AudioPlayer::play(wav44).ok();
    }
}

impl Render for LightVcUi {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.pump();
        if self.needs_tick() {
            window.request_animation_frame();
        }

        let root = div().id("root").size_full().flex().flex_col().bg(col::bg());

        let tab = self.tab;
        let bar = div()
            .flex()
            .items_center()
            .gap_3()
            .px_5()
            .py_2()
            .bg(rgba(0xFD_FC_FE_F0))
            .border_b_1()
            .border_color(col::border())
            .child(div().text_size(px(18.)).font_weight(FontWeight::BOLD).text_color(col::pink()).child("✦ LightVC"))
            .child(pill("tab_off", "Offline", tab == Tab::Offline).on_click(cx.listener(|this, _, _, cx| {
                this.tab = Tab::Offline;
                cx.notify();
            })))
            .child(pill("tab_rt", "Realtime", tab == Tab::Realtime).on_click(cx.listener(|this, _, _, cx| {
                this.tab = Tab::Realtime;
                cx.notify();
            })))
            .child(pill("tab_vo", "Voices", tab == Tab::Catalog).on_click(cx.listener(|this, _, _, cx| {
                this.tab = Tab::Catalog;
                cx.notify();
            })))
            .child(div().flex_1())
            .child(
                div()
                    .id("settings")
                    .px_3()
                    .py_1()
                    .rounded_md()
                    .text_size(px(13.))
                    .bg(col::card())
                    .border_1()
                    .border_color(col::border())
                    .text_color(col::text())
                    .hover(|s| s.bg(col::card_tint()))
                    .on_click(cx.listener(|this, _, _, cx| {
                        if !this.settings_open {
                            this.pick_settings();
                        }
                        this.settings_open = !this.settings_open;
                        cx.notify();
                    }))
                    .child("⚙ Devices"),
            );

        let (status_line, status_color, err) = {
            let s = self.state.lock().unwrap();
            (s.status.clone(), if s.error.is_some() { col::error() } else { col::mint() }, s.error.clone())
        };

        let bottom = div()
            .flex()
            .items_center()
            .gap_3()
            .px_5()
            .py_2()
            .bg(rgba(0xFD_FC_FE_F0))
            .border_t_1()
            .border_color(col::border())
            .child(div().text_size(px(11.)).text_color(status_color).child(format!("● {}", err.unwrap_or(status_line))))
            .child(div().flex_1())
            .child(subtext("Out"))
            .child(meter(self.metrics.output_rms * 6.0, col::cyan()));

        let content: AnyElement = match self.tab {
            Tab::Offline => self.render_offline(cx),
            Tab::Realtime => self.render_realtime(cx),
            Tab::Catalog => self.render_catalog(cx),
        };

        let content = div().id("content").flex_1().flex().flex_col().px_5().py_3().gap_3().child(if self.settings_open {
            div().flex().flex_col().gap_3().child(self.render_settings(cx)).child(content).into_any_element()
        } else {
            content
        });

        root.child(bar).child(content).child(bottom)
    }
}

impl LightVcUi {
    fn render_settings(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let (sel_in, sel_out, sel_host) = {
            let s = self.state.lock().unwrap();
            (s.selected_input, s.selected_output, s.selected_host)
        };
        let mut el = card().child(heading("Audio Devices"));

        let mut host_row = div().flex().flex_wrap().items_center().gap_2().child(subtext("Host"));
        let mut host_pills: Vec<(String, lightvc_audio::HostPref)> = vec![("default".to_string(), lightvc_audio::HostPref::Default)];
        for h in lightvc_audio::host::available() {
            if let Ok(p) = lightvc_audio::HostPref::parse(h) {
                host_pills.push((h.to_string(), p));
            }
        }
        for (i, (label, pref)) in host_pills.iter().enumerate() {
            let selected = *pref == sel_host;
            let pref = *pref;
            let label2 = label.clone();
            host_row = host_row.child(
                pill(SharedString::from(format!("host_{i}")), label2.clone(), selected).on_click(cx.listener(move |this, _, _, cx| {
                    this.state.lock().unwrap().selected_host = pref;
                    this.pick_settings();
                    cx.notify();
                })),
            );
        }
        el = el.child(host_row);

        let mut device_rows = div().flex().flex_col().gap_1();
        device_rows = device_rows.child(subtext("Input"));
        let inputs = self.settings.inputs.clone();
        for (i, d) in inputs.iter().enumerate() {
            let sel = sel_in == Some(i);
            device_rows = device_rows.child(
                div()
                    .id(SharedString::from(format!("in_{i}")))
                    .px_2()
                    .py_1()
                    .rounded_md()
                    .text_size(px(11.))
                    .map(move |s| if sel { s.bg(col::card_tint()).text_color(col::text()) } else { s.text_color(col::text_dim()) })
                    .hover(|s| s.bg(col::card_tint()))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.state.lock().unwrap().selected_input = Some(i);
                        cx.notify();
                    }))
                    .child(format!("in  {i}: {} ({}Hz, {}ch)", d.name, d.sample_rate, d.channels)),
            );
        }
        device_rows = device_rows.child(subtext("Output"));
        let outputs = self.settings.outputs.clone();
        for (i, d) in outputs.iter().enumerate() {
            let sel = sel_out == Some(i);
            device_rows = device_rows.child(
                div()
                    .id(SharedString::from(format!("out_{i}")))
                    .px_2()
                    .py_1()
                    .rounded_md()
                    .text_size(px(11.))
                    .map(move |s| if sel { s.bg(col::card_tint()).text_color(col::text()) } else { s.text_color(col::text_dim()) })
                    .hover(|s| s.bg(col::card_tint()))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.state.lock().unwrap().selected_output = Some(i);
                        cx.notify();
                    }))
                    .child(format!("out {i}: {} ({}Hz, {}ch)", d.name, d.sample_rate, d.channels)),
            );
        }
        el.child(device_rows).into_any_element()
    }

    fn render_offline(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let (semitones, has_operator, voices, selected_voice) = {
            let s = self.state.lock().unwrap();
            (s.settings.semitones, !s.settings.operator.is_empty(), s.voices.iter().map(|v| v.name.clone()).collect::<Vec<_>>(), s.selected_voice)
        };

        let mut el = div().flex().flex_col().gap_3().w_full();

        let src_label = self.offline.source.as_ref().map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_else(|| "(未選択)".into());
        let src_playing = self.offline.source_preview.as_ref().is_some_and(|p| p.is_playing());
        el = el.child(
            card().child(heading("Source")).child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().flex_1().text_size(px(12.)).text_color(col::text()).child(src_label))
                    .child(btn("src_play", if src_playing { "Stop" } else { "Play" }, Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                        if this.offline.source_preview.is_some() {
                            toggle_stop(&mut this.offline.source_preview);
                        } else if let Some(p) = this.offline.source.clone() {
                            play_path(&mut this.offline.source_preview, &p);
                        }
                        cx.notify();
                    })))
                    .child(btn("src_browse", "Browse", Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                        this.picks.source.open();
                        cx.notify();
                    }))),
            ),
        );

        let ref_label = self.offline.reference.as_ref().map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_else(|| "(未選択)".into());
        let ref_playing = self.offline.reference_preview.as_ref().is_some_and(|p| p.is_playing());
        el = el.child(
            card().child(heading("Reference (目標の声・20 秒以上)")).child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().flex_1().text_size(px(12.)).text_color(col::text()).child(ref_label))
                    .child(btn("ref_play", if ref_playing { "Stop" } else { "Play" }, Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                        if this.offline.reference_preview.is_some() {
                            toggle_stop(&mut this.offline.reference_preview);
                        } else if let Some(p) = this.offline.reference.clone() {
                            play_path(&mut this.offline.reference_preview, &p);
                        }
                        cx.notify();
                    })))
                    .child(btn("ref_browse", "Browse", Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                        this.picks.reference.open();
                        cx.notify();
                    }))),
            ),
        );

        if !voices.is_empty() {
            let mut row = div().flex().flex_wrap().items_center().gap_2().child(subtext("Or pick from Voice Catalog"));
            for (i, name) in voices.iter().enumerate() {
                let selected = selected_voice == Some(i) && self.offline.reference.is_none();
                row = row.child(
                    pill(SharedString::from(format!("cat_{i}")), name.clone(), selected).on_click(cx.listener(move |this, _, _, cx| {
                        let p = this.state.lock().unwrap().voices.get(i).map(|v| v.path.clone());
                        if let Some(p) = p {
                            this.offline.reference = Some(p);
                        }
                        cx.notify();
                    })),
                );
            }
            el = el.child(card().child(row));
        }

        let mut pitch_row = div().flex().flex_wrap().items_center().gap_2().child(subtext("Pitch (固定の半音)"));
        for (i, (k, name)) in SEMITONES.iter().enumerate() {
            let selected = semitones == *k;
            let k = *k;
            pitch_row = pitch_row.child(
                pill(SharedString::from(format!("pitch_{i}")), *name, selected).on_click(cx.listener(move |this, _, _, cx| {
                    let mut s = this.state.lock().unwrap();
                    s.settings.semitones = k;
                    if let Some(ref tx) = s.rt_control_tx {
                        let _ = tx.send(RtControl::SetSemitones(k));
                    }
                    cx.notify();
                })),
            );
        }
        el = el.child(card().child(pitch_row));

        let can = self.offline.source.is_some() && self.offline.reference.is_some() && has_operator && !self.offline.converting;
        el = el.child(
            div().flex().items_center().gap_3().child(
                btn("convert", if self.offline.converting { "Converting…" } else { "Convert" }, if can { Btn::Primary } else { Btn::Ghost }).on_click(cx.listener(move |this, _, _, cx| {
                    if this.offline.source.is_none() || this.offline.reference.is_none() || this.offline.converting {
                        return;
                    }
                    let (src, refp) = (this.offline.source.clone().unwrap(), this.offline.reference.clone().unwrap());
                    {
                        let mut s = this.state.lock().unwrap();
                        s.offline_result = None;
                        s.error = None;
                        s.status = "Converting...".to_string();
                    }
                    this.offline.converted = None;
                    this.offline.player = None;
                    this.offline.converting = true;
                    let st = this.state.clone();
                    std::thread::spawn(move || crate::offline_tab::run_offline_conversion(st, src, refp));
                    cx.notify();
                })),
            ).when(self.offline.converting, |r| r.child(subtext("変換中…(登録と変換に数十秒)"))),
        );

        if let Some(samples) = &self.offline.converted {
            let out_playing = self.offline.player.as_ref().is_some_and(|p| p.is_playing());
            let seconds = samples.len() as f32 / SR as f32;
            el = el.child(
                card()
                    .child(heading("Output"))
                    .child(subtext(format!("{seconds:.1} s・48 kHz")))
                    .child(
                        div()
                            .flex()
                            .gap_3()
                            .child(btn("out_play", if out_playing { "Stop" } else { "Play Output" }, Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                                if this.offline.player.is_some() {
                                    toggle_stop(&mut this.offline.player);
                                } else if let Some(samples) = this.offline.converted.clone() {
                                    let wav44 = audio_playback::resample_linear(&samples, SR);
                                    this.offline.player = AudioPlayer::play(wav44).ok();
                                }
                                cx.notify();
                            })))
                            .child(btn("out_save", "Save As…", Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                                if let Some(samples) = this.offline.converted.clone() {
                                    if let Some(path) = rfd::FileDialog::new().save_file() {
                                        let _ = audio_playback::save_wav_mono(&path, &samples, SR);
                                        this.state.lock().unwrap().status = format!("保存した: {}", path.display());
                                    }
                                }
                                cx.notify();
                            }))),
                    ),
            );
        }
        if !has_operator {
            el = el.child(div().text_size(px(12.)).text_color(col::error()).child("操作者の登録がありません(Realtime タブで追加)"));
        }
        el.into_any_element()
    }

    fn render_realtime(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let m = self.metrics.clone();
        let (has_engine, loading, cfg, voices, selected_voice, rec) = {
            let s = self.state.lock().unwrap();
            (
                s.model_loaded,
                s.loading,
                s.settings.clone(),
                s.voices.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
                s.selected_voice,
                s.operator_rec.as_ref().map(|r| r.seconds.saturating_sub(r.started.elapsed().as_secs() as u32)),
            )
        };
        let force_bypass = !has_engine;
        let ref_name = selected_voice.and_then(|i| voices.get(i).cloned());

        let mut el = div().flex().flex_col().gap_3().w_full();

        // 1 行目: Model + Status
        let mut model_card = card().child(heading("Model"));
        let mut ex_row = div().flex().flex_wrap().items_center().gap_2();
        for (id, label, cur) in [("front", "front", cfg.front.clone()), ("model", "E / C", cfg.model.clone()), ("voc", "voc", cfg.voc.clone())] {
            let short = std::path::Path::new(&cur).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(cur.clone());
            ex_row = ex_row.child(
                div().flex().items_center().gap_1().child(subtext(format!("{label}: {short}"))).child(
                    btn(SharedString::from(format!("pick_{id}")), "…", Btn::Ghost).on_click({
                        let id = id.to_string();
                        cx.listener(move |this, _, _, cx| {
                            match id.as_str() {
                                "front" => this.picks.front.open(),
                                "model" => this.picks.model.open(),
                                _ => this.picks.voc.open(),
                            }
                            cx.notify();
                        })
                    }),
                ),
            );
        }
        model_card = model_card.child(ex_row);
        let mut th_row = div().flex().flex_wrap().items_center().gap_2().child(subtext("threads"));
        for n in [1usize, 2, 4] {
            let selected = cfg.threads == n;
            th_row = th_row.child(
                pill(SharedString::from(format!("th_{n}")), n.to_string(), selected).on_click(cx.listener(move |this, _, _, cx| {
                    this.state.lock().unwrap().settings.threads = n;
                    cx.notify();
                })),
            );
        }
        model_card = model_card.child(th_row);
        model_card = model_card.child(subtext(format!("Operator: {} 本", cfg.operator.len())));
        let mut op_row = div().flex().flex_wrap().items_center().gap_2();
        op_row = op_row.child(btn("op_add", "Add operator WAV", Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
            this.picks.operator.open();
            cx.notify();
        })));
        if let Some(rem) = rec {
            op_row = op_row
                .child(div().text_size(px(12.)).font_weight(FontWeight::BOLD).text_color(col::error()).child(format!("●REC 残り {rem} s")))
                .child(btn("op_cancel", "Cancel", Btn::Danger).on_click(cx.listener(|this, _, _, cx| {
                    if let Some(r) = this.state.lock().unwrap().operator_rec.as_ref() {
                        r.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    cx.notify();
                })));
        } else {
            op_row = op_row.child(btn("op_rec", "Record operator (30 s)", Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                if !this.rt_running {
                    state::record_operator(&this.state, 30);
                }
                cx.notify();
            })));
        }
        if !cfg.operator.is_empty() {
            op_row = op_row.child(btn("op_clear", "Clear", Btn::Danger).on_click(cx.listener(|this, _, _, cx| {
                this.state.lock().unwrap().settings.operator.clear();
                cx.notify();
            })));
        }
        model_card = model_card.child(op_row);
        let can_load = !cfg.operator.is_empty() && ref_name.is_some() && !loading;
        model_card = model_card.child(
            div()
                .flex()
                .items_center()
                .gap_3()
                .child(btn("load", if has_engine { "Reload" } else { "Load" }, if can_load { Btn::Primary } else { Btn::Ghost }).on_click(cx.listener(|this, _, _, cx| {
                    let ok = {
                        let s = this.state.lock().unwrap();
                        !s.settings.operator.is_empty() && s.selected_voice.is_some()
                    };
                    if ok {
                        state::load_engine(&this.state);
                    }
                    cx.notify();
                })))
                .when(loading, |r| r.child(subtext("Loading model and enrolling…")))
                .when(ref_name.is_none(), |r| r.child(div().text_size(px(11.)).text_color(col::error()).child("下の Target で目標を選ぶ"))),
        );

        let status_label = if self.rt_running {
            if self.rt_bypass || force_bypass { "BYPASS" } else { "CONVERTING" }
        } else {
            "STOPPED"
        };
        let status_color = if self.rt_running {
            if self.rt_bypass || force_bypass { col::lavender() } else { col::mint() }
        } else {
            col::text_muted()
        };
        let mut status_card = card().child(
            div().flex().items_center().gap_2().child(
                div().px_3().py_1().rounded_full().text_size(px(11.)).font_weight(FontWeight::BOLD).bg(status_color).text_color(col::white()).child(status_label),
            ).child(div().text_size(px(12.)).text_color(col::text()).child(format!("> {}", ref_name.clone().unwrap_or_else(|| "no voice".into())))),
        );
        if self.rt_running && !self.rt_bypass && has_engine {
            status_card = status_card.child(
                div().flex().justify_end().child(
                    div().text_size(px(10.)).font_weight(FontWeight::SEMIBOLD).text_color(if m.gate_open { col::mint() } else { col::text_muted() }).child(if m.gate_open { "GATE" } else { "gate" }),
                ),
            );
        }
        status_card = status_card.child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_2()
                .child(stat(format!("{:.0}", m.latency_ms), "ms"))
                .child(stat(format!("{:.2}", m.rtf), "RTF"))
                .when(m.overrun > 0 || m.underrun > 0, |r| {
                    r.child(div().text_size(px(10.)).text_color(col::error()).child(format!("in:{} out:{}", m.overrun, m.underrun)))
                }),
        );
        let measuring = m.blocks > 0 || m.compute_p99 > 0.0;
        let cc = |v: f32| if !measuring { col::text_muted() } else if v >= 5.0 { col::error() } else { col::mint() };
        status_card = status_card.child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_2()
                .child(subtext("compute ms"))
                .child(div().text_size(px(11.)).text_color(cc(m.compute_p50)).child(if measuring { format!("p50 {:.2}", m.compute_p50) } else { "p50 —".to_string() }))
                .child(div().text_size(px(11.)).text_color(cc(m.compute_p95)).child(if measuring { format!("p95 {:.2}", m.compute_p95) } else { "p95 —".to_string() }))
                .child(div().text_size(px(11.)).text_color(cc(m.compute_p99)).child(if measuring { format!("p99 {:.2}", m.compute_p99) } else { "p99 —".to_string() }))
                .child(subtext(format!("host {}", if m.host.is_empty() { "-" } else { &m.host }))),
        );

        el = el.child(div().flex().gap_3().child(div().flex_1().child(model_card)).child(div().flex_1().child(status_card)));

        // Target 簡易選択
        if !voices.is_empty() {
            let mut row = div().flex().flex_wrap().items_center().gap_2().child(subtext("Target"));
            for (i, name) in voices.iter().enumerate() {
                let selected = selected_voice == Some(i);
                row = row.child(
                    pill(SharedString::from(format!("tgt_{i}")), name.clone(), selected).on_click(cx.listener(move |this, _, _, cx| {
                        let reload = {
                            let mut s = this.state.lock().unwrap();
                            if s.voices.get(i).is_some() {
                                s.selected_voice = Some(i);
                            }
                            s.model_loaded && !s.settings.operator.is_empty()
                        };
                        if reload {
                            state::load_engine(&this.state);
                        }
                        cx.notify();
                    })),
                );
            }
            row = row.child(
                div()
                    .id("tgt_manage")
                    .px_2()
                    .text_size(px(11.))
                    .text_color(col::lavender())
                    .hover(|s| s.text_color(col::pink()))
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.tab = Tab::Catalog;
                        cx.notify();
                    }))
                    .child("管理…"),
            );
            el = el.child(card().child(row));
        } else {
            el = el.child(card().child(
                div().flex().items_center().gap_3().child(subtext("目標の声が未登録")).child(btn("tgt_open", "Open Voices →", Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                    this.tab = Tab::Catalog;
                    cx.notify();
                }))),
            ));
        }

        // 入力メーター
        el = el.child(card().child(div().flex().items_center().gap_3().child(subtext("Input")).child(meter(self.metrics.input_rms * 6.0, col::pink()))));

        // Latency
        let buf_ms = self.rt_buffer as f32 * 1000.0 / 48_000.0;
        let resamp_ms = 3.0;
        let algo_ms = lightvc_core::snv_stream::SnvStream::LATENCY_SAMPLES as f32 / 48.0;
        let e2e = buf_ms * 2.0 + resamp_ms * 2.0 + algo_ms;
        let total = e2e.max(0.001);
        let segs: [(f32, gpui::Rgba); 5] = [(buf_ms, col::cyan()), (resamp_ms, col::mint()), (algo_ms, col::pink()), (resamp_ms, col::mint()), (buf_ms, col::cyan())];
        let mut seg_row = div().flex().w_full().h_3().rounded_full();
        for (ms, c) in segs {
            seg_row = seg_row.child(div().w(px(600.0 * ms / total)).h_full().bg(c));
        }
        let mut buf_row = div().flex().flex_wrap().items_center().gap_2().child(subtext("Buffer"));
        for n in BUFFERS {
            let selected = self.rt_buffer == n;
            buf_row = buf_row.child(
                pill(SharedString::from(format!("buf_{n}")), n.to_string(), selected).on_click(cx.listener(move |this, _, _, cx| {
                    state::send_control(&this.state, RtControl::SetBufferSize(n));
                    this.rt_buffer = n;
                    if this.rt_running {
                        state::send_control(&this.state, RtControl::Stop);
                        state::send_control(&this.state, state::start_msg(&this.state));
                    }
                    cx.notify();
                })),
            );
        }
        el = el.child(
            card().child(div().flex().flex_wrap().items_center().gap_3().child(div().text_size(px(16.)).font_weight(FontWeight::BOLD).text_color(col::text()).child(format!("{e2e:.0} ms E2E"))).child(div().flex_1()).child(buf_row)).child(seg_row),
        );

        // 運用バー
        let mut ops = div().flex().flex_wrap().items_center().gap_2();
        if !force_bypass {
            ops = ops
                .child(pill("bypass", if self.rt_bypass { "BYPASS" } else { "Bypass" }, self.rt_bypass).on_click(cx.listener(|this, _, _, cx| {
                    this.rt_bypass = !this.rt_bypass;
                    state::send_control(&this.state, RtControl::Bypass(this.rt_bypass));
                    cx.notify();
                })))
                .child(pill("mute", if self.rt_muted { "MUTED" } else { "Mute" }, self.rt_muted).on_click(cx.listener(|this, _, _, cx| {
                    this.rt_muted = !this.rt_muted;
                    state::send_control(&this.state, RtControl::Mute(this.rt_muted));
                    cx.notify();
                })));
        }
        ops = ops.child(subtext("Pitch"));
        for (i, (k, name)) in SEMITONES.iter().enumerate() {
            let selected = cfg.semitones == *k;
            let k = *k;
            ops = ops.child(
                pill(SharedString::from(format!("rt_pitch_{i}")), *name, selected).on_click(cx.listener(move |this, _, _, cx| {
                    let mut s = this.state.lock().unwrap();
                    s.settings.semitones = k;
                    if let Some(ref tx) = s.rt_control_tx {
                        let _ = tx.send(RtControl::SetSemitones(k));
                    }
                    cx.notify();
                })),
            );
        }
        ops = ops.child(div().flex_1());
        if self.rt_running {
            ops = ops.child(btn("stop_btn", "■ Stop", Btn::Dark).on_click(cx.listener(|this, _, _, cx| {
                this.stop(cx);
            })));
        } else {
            ops = ops.child(btn("start_btn", "▶ Start", Btn::Primary).on_click(cx.listener(|this, _, _, cx| {
                this.start(cx);
            })));
        }
        el = el.child(card().child(ops));

        el.into_any_element()
    }

    fn render_catalog(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let (voices, selected_voice) = {
            let s = self.state.lock().unwrap();
            (s.voices.iter().map(|v| (v.name.clone(), v.path.clone())).collect::<Vec<_>>(), s.selected_voice)
        };
        let mut el = div().flex().flex_col().gap_3().w_full();
        el = el.child(subtext("目標の声の参照を登録する(zero-shot・普通の発話 20 秒以上)"));

        el = el.child(
            card().child(heading("Add Voice")).child(
                div().flex().items_center().gap_3().child(subtext("WAV を選ぶとファイル名で登録します")).child(btn("voice_add", "Browse…", Btn::Ghost).on_click(cx.listener(|this, _, _, cx| {
                    this.picks.add.open();
                    cx.notify();
                }))),
            ),
        );

        if voices.is_empty() {
            el = el.child(card().child(div().py_6().flex().justify_center().child(subtext("まだ登録がありません"))));
        } else {
            let mut list = card().child(heading("Voices"));
            for (i, (name, path)) in voices.iter().enumerate() {
                let selected = selected_voice == Some(i);
                let playing = self.catalog.playing == Some(i) && self.catalog.player.as_ref().is_some_and(|p| p.is_playing());
                let path2 = path.clone();
                list = list.child(
                    div()
                        .id(SharedString::from(format!("voice_{i}")))
                        .flex()
                        .items_center()
                        .gap_3()
                        .px_2()
                        .py_1()
                        .rounded_lg()
                        .map(move |s| if selected { s.bg(col::card_tint()) } else { s })
                        .hover(|s| s.bg(col::card_tint()))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            let reload = {
                                let mut s = this.state.lock().unwrap();
                                s.selected_voice = Some(i);
                                s.model_loaded && !s.settings.operator.is_empty()
                            };
                            if reload {
                                state::load_engine(&this.state);
                            }
                            cx.notify();
                        }))
                        .child(div().text_size(px(13.)).font_weight(FontWeight::SEMIBOLD).text_color(if selected { col::pink() } else { col::text() }).child(name.clone()))
                        .child(div().flex_1().text_size(px(10.)).text_color(col::text_muted()).child(path.display().to_string()))
                        .child(btn(SharedString::from(format!("voice_play_{i}")), if playing { "Stop" } else { "Play" }, Btn::Ghost).on_click({
                            let path = path2.clone();
                            cx.listener(move |this, _, _, cx| {
                                if this.catalog.playing == Some(i) && this.catalog.player.is_some() {
                                    toggle_stop(&mut this.catalog.player);
                                    this.catalog.playing = None;
                                } else {
                                    toggle_stop(&mut this.catalog.player);
                                    play_path(&mut this.catalog.player, &path);
                                    this.catalog.playing = Some(i);
                                }
                                cx.notify();
                            })
                        }))
                        .child(btn(SharedString::from(format!("voice_del_{i}")), "✕", Btn::Danger).on_click(cx.listener(move |this, _, _, cx| {
                            toggle_stop(&mut this.catalog.player);
                            let mut s = this.state.lock().unwrap();
                            if s.voices.len() == 1 {
                                s.voices.clear();
                                s.selected_voice = None;
                            } else if let Some(sel) = s.selected_voice {
                                s.voices.remove(i);
                                s.selected_voice = if sel == i { None } else if sel > i { Some(sel - 1) } else { Some(sel) };
                            }
                            cx.notify();
                        }))),
                );
            }
            el = el.child(list);
        }
        el.into_any_element()
    }
}
