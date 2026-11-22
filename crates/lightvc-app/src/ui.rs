//! LightVC の GUI(GPUI・Material 3 Expressive 準拠)。3 タブ + 設定 + 状態バー。
//! デザインの正本は DESIGN.md(色のロール・タイプスケール・シェイプ・コンポーネントの写像)。
//!
//! 描画は要求駆動: 動作中(リアルタイム・読み込み・録音・変換・試聴・ダイアログ)だけ
//! request_animation_frame で次フレームを要求し、止まっているときは再描画しない。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gpui::{
    bounce, div, ease_in_out, prelude::*, px, rgb, size, Animation, AnimationExt as _, AnyElement,
    App, Application, Bounds, Context, ElementId, FontWeight, InteractiveElement, Render,
    SharedString, Stateful, Styled, TitlebarOptions, Window, WindowBounds, WindowOptions,
};

use crate::audio_playback::{self, AudioPlayer};
use crate::file_pick::FilePick;
use crate::state::{self, AppState, RtControl, RtMetrics};

const SR: u32 = 48_000;
const SEMITONES: [(f32, &str); 4] = [(0.0, "+0"), (5.0, "+5"), (7.0, "+7"), (12.0, "+12")];
const BUFFERS: [u32; 5] = [128, 240, 256, 512, 1024];
const METER_W: f32 = 180.0;

/// M3E の色ロール・シェイプ(DESIGN.md §1〜§3)。seed = きゃわわピンク/ラベンダン。rgb/hsla は const fn ではないので関数で持つ。
mod md {
    use gpui::{hsla, rgb, Hsla, Rgba};

    pub fn primary() -> Rgba {
        rgb(0xE0_5A_8A)
    }
    pub fn on_primary() -> Rgba {
        rgb(0xFF_FF_FF)
    }
    pub fn primary_container() -> Rgba {
        rgb(0xFB_D9_E5)
    }
    pub fn on_primary_container() -> Rgba {
        rgb(0x4A_1B_31)
    }
    pub fn secondary() -> Rgba {
        rgb(0x7C_68_BE)
    }
    pub fn secondary_container() -> Rgba {
        rgb(0xE7_DF_F8)
    }
    pub fn on_secondary_container() -> Rgba {
        rgb(0x2A_20_50)
    }
    pub fn tertiary() -> Rgba {
        rgb(0x2E_B8_8C)
    }
    pub fn tertiary_container() -> Rgba {
        rgb(0xCC_F1_E2)
    }
    pub fn on_tertiary_container() -> Rgba {
        rgb(0x06_38_28)
    }
    pub fn error() -> Rgba {
        rgb(0xC0_3B_5E)
    }
    pub fn on_error() -> Rgba {
        rgb(0xFF_FF_FF)
    }
    pub fn error_container() -> Rgba {
        rgb(0xF9_D6_DF)
    }
    pub fn on_error_container() -> Rgba {
        rgb(0x45_09_0F)
    }
    pub fn surface() -> Rgba {
        rgb(0xF7_F1_F9)
    }
    pub fn surface_low() -> Rgba {
        rgb(0xF2_EB_F5)
    }
    pub fn surface_high() -> Rgba {
        rgb(0xE6_DC_EB)
    }
    pub fn surface_lowest() -> Rgba {
        rgb(0xFF_FF_FF)
    }
    pub fn on_surface() -> Rgba {
        rgb(0x2B_22_33)
    }
    pub fn on_surface_variant() -> Rgba {
        rgb(0x5D_52_66)
    }
    pub fn outline() -> Hsla {
        hsla(280.0 / 360.0, 0.14, 0.55, 1.0)
    }
    pub fn outline_variant() -> Rgba {
        rgb(0xD9_CD_DF)
    }

    // シェイプスケール(§3)。tension = 診断だけ四角。
    pub const SHAPE_XS: f32 = 4.0;
    pub const SHAPE_MD: f32 = 10.0;
    pub const SHAPE_LG: f32 = 14.0;
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
        let bounds = Bounds::centered(None, size(px(1040.), px(760.)), cx);
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

// ---- M3E 部品キット(DESIGN.md §4) ------------------------------------------------

/// 強調の階層。filled = 高(画面に 1 個)・tonal = 中・outlined = 低付き枠・text = 最低。
enum Emph {
    Filled,
    Tonal,
    Outlined,
    Text,
}

fn btn(id: impl Into<ElementId>, label: impl Into<SharedString>, e: Emph) -> Stateful<gpui::Div> {
    div()
        .id(id)
        .map(|s| match e {
            Emph::Filled => s.px_5().py_2().rounded_full().bg(md::primary()).text_color(md::on_primary()),
            Emph::Tonal => s.px_4().py_2().rounded_full().bg(md::secondary_container()).text_color(md::on_secondary_container()),
            Emph::Outlined => s.px_4().py_2().rounded_full().bg(gpui::transparent_black()).text_color(md::on_surface()).border_1().border_color(md::outline()),
            Emph::Text => s.px_2().py_1().rounded_md().bg(gpui::transparent_black()).text_color(md::on_surface_variant()),
        })
        .text_size(px(13.))
        .font_weight(FontWeight::MEDIUM)
        .hover(|s| match e {
            Emph::Filled => s.bg(rgb(0xEC_80_A8)),
            Emph::Tonal => s.bg(rgb(0xD9_CE_F2)),
            Emph::Outlined => s.bg(md::surface_low()),
            Emph::Text => s.bg(md::surface_high()),
        })
        .child(label.into())
}

/// connected button group の一肢。選択肢だけ膨らみ・primary を纏う(M3E の shape/width 変化)。
fn group_seg(id: impl Into<ElementId>, label: impl Into<SharedString>, selected: bool) -> Stateful<gpui::Div> {
    div()
        .id(id)
        .rounded_full()
        .text_size(px(12.))
        .font_weight(FontWeight::MEDIUM)
        .map(|s| {
            if selected {
                s.px_4().py_1().bg(md::primary()).text_color(md::on_primary())
            } else {
                s.px_3().py_1().text_color(md::on_surface_variant())
            }
        })
        .hover(|s| s.bg(md::primary_container()))
        .child(label.into())
}

fn group_container() -> gpui::Div {
    div().flex().items_center().gap_1().px_1().py_1().rounded_full().bg(md::surface_high())
}

/// filter chip(選択で primary-container + ✓)。
fn chip(id: impl Into<ElementId>, label: impl Into<SharedString>, selected: bool) -> Stateful<gpui::Div> {
    let label: SharedString = label.into();
    let text: SharedString = if selected { format!("✓ {label}").into() } else { label };
    div()
        .id(id)
        .px_3()
        .py_1()
        .rounded_full()
        .text_size(px(12.))
        .font_weight(FontWeight::MEDIUM)
        .map(|s| {
            if selected {
                s.bg(md::primary_container()).text_color(md::on_primary_container()).border_1().border_color(md::primary())
            } else {
                s.bg(md::surface_lowest()).text_color(md::on_surface_variant()).border_1().border_color(md::outline_variant())
            }
        })
        .hover(|s| s.bg(md::primary_container()))
        .child(text)
}

fn badge(text: &str, kind: BadgeKind) -> gpui::Div {
    let (bg, fg) = match kind {
        BadgeKind::Good => (md::tertiary(), md::on_primary()),
        BadgeKind::Mid => (md::secondary(), md::on_primary()),
        BadgeKind::Off => (md::outline_variant(), md::on_surface_variant()),
    };
    div().px_3().py_1().rounded_full().text_size(px(11.)).font_weight(FontWeight::BOLD).bg(bg).text_color(fg).child(text.to_string())
}

enum BadgeKind {
    Good,
    Mid,
    Off,
}

/// M3E loading indicator の一肢(大きさと不透明度を bounce で脈動)。GPUI の Div は
/// transformation を持てないため、M3E の "capture attention" をサイズの脈動で表す。
fn loading_dot(id: &'static str, dur: f32, color: gpui::Rgba) -> gpui::AnimationElement<gpui::Div> {
    div()
        .size(px(7.))
        .rounded_full()
        .bg(color)
        .with_animation(
            id,
            Animation::new(Duration::from_secs_f32(dur)).repeat().with_easing(bounce(ease_in_out)),
            |el, delta| el.size(px(5.0 + 5.0 * delta)).opacity(0.35 + 0.65 * delta),
        )
}

fn loading(text: &str) -> gpui::Div {
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(div().flex().items_center().gap_1().child(loading_dot("ld0", 0.80, md::primary())).child(loading_dot("ld1", 0.95, md::primary())).child(loading_dot("ld2", 1.10, md::primary())))
        .child(div().text_size(px(12.)).text_color(md::on_surface_variant()).child(text.to_string()))
}

/// REC の点滅(error の点が bounce で脈動)。
fn rec_dot() -> gpui::AnimationElement<gpui::Div> {
    div()
        .size(px(10.))
        .rounded_full()
        .bg(md::error())
        .with_animation(
            "rec",
            Animation::new(Duration::from_millis(800)).repeat().with_easing(bounce(ease_in_out)),
            |el, delta| el.size(px(8.0 + 5.0 * delta)).opacity(0.55 + 0.45 * delta),
        )
}

/// M3 の linear progress: 溝 + 針 + 末端の停止ドット。
fn meter(level: f32, color: gpui::Rgba) -> gpui::Div {
    let w = (METER_W * level.clamp(0.0, 1.0)).max(3.0);
    div()
        .flex()
        .items_center()
        .gap_1()
        .child(div().w(px(METER_W)).h(px(6.)).rounded_full().bg(md::outline_variant()).child(div().w(px(w)).h_full().rounded_full().bg(color)))
        .child(div().size(px(3.)).rounded_full().bg(color))
}

/// 見出し(title-medium)。
fn heading(text: &str) -> gpui::Div {
    div().text_size(px(15.)).text_color(md::on_surface()).font_weight(FontWeight::SEMIBOLD).child(text.to_string())
}

/// 主役の数値(headline-emphasis・単位は label-small で薄く)。
fn stat(value: String, unit: &str, color: gpui::Rgba) -> gpui::Div {
    div()
        .flex()
        .items_baseline()
        .gap_1()
        .child(div().text_size(px(24.)).font_weight(FontWeight::BOLD).text_color(color).child(value))
        .child(div().text_size(px(10.5)).text_color(md::on_surface_variant()).child(unit.to_string()))
}

fn subtext(text: impl Into<SharedString>) -> gpui::Div {
    div().text_size(px(11.)).text_color(md::on_surface_variant()).child(text.into())
}

fn card() -> gpui::Div {
    div().bg(md::surface_low()).rounded(px(md::SHAPE_MD)).p_4().flex().flex_col().gap_2()
}

fn toggle_stop(player_slot: &mut Option<AudioPlayer>) {
    if let Some(p) = player_slot.take() {
        p.stop();
    }
}

fn play_path(player_slot: &mut Option<AudioPlayer>, path: &PathBuf) -> anyhow::Result<()> {
    let (wav, sr) = audio_playback::load_wav_mono(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let wav44 = audio_playback::resample_linear(&wav, sr);
    *player_slot = Some(AudioPlayer::play(wav44)?);
    Ok(())
}

impl Render for LightVcUi {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.pump();
        if self.needs_tick() {
            window.request_animation_frame();
        }

        let root = div().id("root").size_full().flex().flex_col().bg(md::surface());

        let tab = self.tab;
        let bar = div()
            .flex()
            .items_center()
            .gap_4()
            .px_5()
            .py_3()
            .bg(md::surface_low())
            .child(div().text_size(px(22.)).font_weight(FontWeight::BOLD).text_color(md::primary()).child("✦ LightVC"))
            .child(group_container().child(group_seg("tab_off", "Offline", tab == Tab::Offline).on_click(cx.listener(|this, _, _, cx| {
                this.tab = Tab::Offline;
                cx.notify();
            }))).child(group_seg("tab_rt", "Realtime", tab == Tab::Realtime).on_click(cx.listener(|this, _, _, cx| {
                this.tab = Tab::Realtime;
                cx.notify();
            }))).child(group_seg("tab_vo", "Voices", tab == Tab::Catalog).on_click(cx.listener(|this, _, _, cx| {
                this.tab = Tab::Catalog;
                cx.notify();
            }))))
            .child(div().flex_1())
            .child(btn("settings", "⚙ Devices", Emph::Outlined).on_click(cx.listener(|this, _, _, cx| {
                if !this.settings_open {
                    this.pick_settings();
                }
                this.settings_open = !this.settings_open;
                cx.notify();
            })));

        let (status_line, status_color, err) = {
            let s = self.state.lock().unwrap();
            (s.status.clone(), if s.error.is_some() { md::error() } else { md::tertiary() }, s.error.clone())
        };

        let bottom = div()
            .flex()
            .items_center()
            .gap_3()
            .px_5()
            .py_2()
            .bg(md::surface_low())
            .child(div().text_size(px(11.)).text_color(status_color).child(format!("● {}", err.unwrap_or(status_line))))
            .child(div().flex_1())
            .child(subtext("Out"))
            .child(meter(self.metrics.output_rms * 6.0, md::tertiary()));

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
            host_row = host_row.child(
                group_seg(SharedString::from(format!("host_{i}")), label.clone(), selected).on_click(cx.listener(move |this, _, _, cx| {
                    this.state.lock().unwrap().selected_host = pref;
                    this.pick_settings();
                    cx.notify();
                })),
            );
        }
        el = el.child(div().flex().items_center().gap_2().child(subtext("Host")).child(host_row));

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
                    .rounded(px(md::SHAPE_XS))
                    .text_size(px(11.))
                    .map(move |s| if sel { s.bg(md::primary_container()).text_color(md::on_primary_container()) } else { s.text_color(md::on_surface_variant()) })
                    .hover(|s| s.bg(md::surface_high()))
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
                    .rounded(px(md::SHAPE_XS))
                    .text_size(px(11.))
                    .map(move |s| if sel { s.bg(md::primary_container()).text_color(md::on_primary_container()) } else { s.text_color(md::on_surface_variant()) })
                    .hover(|s| s.bg(md::surface_high()))
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
                    .child(div().flex_1().text_size(px(13.)).text_color(md::on_surface()).child(src_label))
                    .child(btn("src_play", if src_playing { "Stop" } else { "Play" }, Emph::Tonal).on_click(cx.listener(|this, _, _, cx| {
                        if this.offline.source_preview.is_some() {
                            toggle_stop(&mut this.offline.source_preview);
                        } else if let Some(p) = this.offline.source.clone() {
                            match play_path(&mut this.offline.source_preview, &p) {
                                Ok(()) => this.state.lock().unwrap().status = format!("再生中: {}", p.display()),
                                Err(e) => this.state.lock().unwrap().error = Some(format!("再生できない: {e}")),
                            }
                        }
                        cx.notify();
                    })))
                    .child(btn("src_browse", "Browse", Emph::Outlined).on_click(cx.listener(|this, _, _, cx| {
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
                    .child(div().flex_1().text_size(px(13.)).text_color(md::on_surface()).child(ref_label))
                    .child(btn("ref_play", if ref_playing { "Stop" } else { "Play" }, Emph::Tonal).on_click(cx.listener(|this, _, _, cx| {
                        if this.offline.reference_preview.is_some() {
                            toggle_stop(&mut this.offline.reference_preview);
                        } else if let Some(p) = this.offline.reference.clone() {
                            match play_path(&mut this.offline.reference_preview, &p) {
                                Ok(()) => this.state.lock().unwrap().status = format!("再生中: {}", p.display()),
                                Err(e) => this.state.lock().unwrap().error = Some(format!("再生できない: {e}")),
                            }
                        }
                        cx.notify();
                    })))
                    .child(btn("ref_browse", "Browse", Emph::Outlined).on_click(cx.listener(|this, _, _, cx| {
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
                    chip(SharedString::from(format!("cat_{i}")), name.clone(), selected).on_click(cx.listener(move |this, _, _, cx| {
                        let p = this.state.lock().unwrap().voices.get(i).map(|v| v.path.clone());
                        if let Some(p) = p {
                            this.offline.reference = Some(p);
                        }
                        cx.notify();
                    })),
                );
            }
            el = el.child(row);
        }

        let mut pitch_group = group_container();
        for (i, (k, name)) in SEMITONES.iter().enumerate() {
            let selected = semitones == *k;
            let k = *k;
            pitch_group = pitch_group.child(
                group_seg(SharedString::from(format!("pitch_{i}")), *name, selected).on_click(cx.listener(move |this, _, _, cx| {
                    let mut s = this.state.lock().unwrap();
                    s.settings.semitones = k;
                    if let Some(ref tx) = s.rt_control_tx {
                        let _ = tx.send(RtControl::SetSemitones(k));
                    }
                    cx.notify();
                })),
            );
        }
        el = el.child(card().child(div().flex().flex_wrap().items_center().gap_3().child(subtext("Pitch (固定の半音)")).child(pitch_group)));

        let can = self.offline.source.is_some() && self.offline.reference.is_some() && has_operator && !self.offline.converting;
        el = el.child(
            div()
                .flex()
                .items_center()
                .gap_4()
                .child(
                    btn("convert", if self.offline.converting { "Converting…" } else { "Convert" }, if can { Emph::Filled } else { Emph::Outlined }).on_click(cx.listener(move |this, _, _, cx| {
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
                    }))
                )
                .when(self.offline.converting, |r| r.child(loading("変換中…"))),
        );

        if let Some(samples) = &self.offline.converted {
            let out_playing = self.offline.player.as_ref().is_some_and(|p| p.is_playing());
            let seconds = samples.len() as f32 / SR as f32;
            el = el.child(
                div()
                    .bg(md::surface_lowest())
                    .rounded(px(md::SHAPE_XS))
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(heading("Output"))
                    .child(stat(format!("{seconds:.1}"), "s・48 kHz", md::tertiary()))
                    .child(
                        div()
                            .flex()
                            .gap_3()
                            .child(btn("out_play", if out_playing { "Stop" } else { "Play Output" }, Emph::Tonal).on_click(cx.listener(|this, _, _, cx| {
                                if this.offline.player.is_some() {
                                    toggle_stop(&mut this.offline.player);
                                } else if let Some(samples) = this.offline.converted.clone() {
                                    let wav44 = audio_playback::resample_linear(&samples, SR);
                                    this.offline.player = AudioPlayer::play(wav44).ok();
                                }
                                cx.notify();
                            })))
                            .child(btn("out_save", "Save As…", Emph::Outlined).on_click(cx.listener(|this, _, _, cx| {
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
            el = el.child(div().text_size(px(12.)).text_color(md::error()).child("操作者の登録がありません(Realtime タブで追加)"));
        }
        el.into_any_element()
    }

    fn render_realtime(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let m = self.metrics.clone();
        let (has_engine, is_loading, cfg, voices, selected_voice, rec) = {
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

        // ===== 1. 運用(hero・最も明るい器・最も大きく)=====
        let (status_label, status_kind) = if self.rt_running {
            if self.rt_bypass || force_bypass { ("BYPASS", BadgeKind::Mid) } else { ("CONVERTING", BadgeKind::Good) }
        } else {
            ("STOPPED", BadgeKind::Off)
        };
        let mut hero = div()
            .bg(md::surface_high())
            .rounded(px(md::SHAPE_LG))
            .p_5()
            .flex()
            .flex_col()
            .gap_3();

        hero = hero.child(
            div().flex().flex_wrap().items_center().gap_3().child(badge(status_label, status_kind)).child(
                div().text_size(px(14.)).font_weight(FontWeight::SEMIBOLD).text_color(md::on_surface()).child(format!("> {}", ref_name.clone().unwrap_or_else(|| "no voice".into()))),
            ).when(self.rt_running && !self.rt_bypass && has_engine, |r| {
                r.child(div().flex_1()).child(div().text_size(px(11.)).font_weight(FontWeight::BOLD).text_color(if m.gate_open { md::tertiary() } else { md::on_surface_variant() }).child(if m.gate_open { "GATE" } else { "gate" }))
            }),
        );

        let measuring = m.blocks > 0 || m.compute_p99 > 0.0;
        let cc = |v: f32| if !measuring { md::on_surface_variant() } else if v >= 5.0 { md::error() } else { md::tertiary() };
        hero = hero.child(
            div()
                .flex()
                .flex_wrap()
                .items_baseline()
                .gap_6()
                .child(stat(format!("{:.0}", m.latency_ms), "ms E2E", md::on_surface()))
                .child(stat(if measuring { format!("{:.2}", m.compute_p99) } else { "—".into() }, "compute p99 ms", cc(m.compute_p99)))
                .child(stat(format!("{:.2}", m.rtf), "RTF", md::on_surface())),
        );
        hero = hero.child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_2()
                .child(subtext("p50"))
                .child(div().text_size(px(11.)).text_color(cc(m.compute_p50)).child(if measuring { format!("{:.2}", m.compute_p50) } else { "—".into() }))
                .child(subtext("p95"))
                .child(div().text_size(px(11.)).text_color(cc(m.compute_p95)).child(if measuring { format!("{:.2}", m.compute_p95) } else { "—".into() }))
                .child(div().text_size(px(11.)).text_color(md::on_surface_variant()).child(format!("host {}", if m.host.is_empty() { "-" } else { &m.host })))
                .when(m.overrun > 0 || m.underrun > 0, |r| {
                    r.child(div().text_size(px(11.)).text_color(md::error()).child(format!("in:{} out:{}", m.overrun, m.underrun)))
                }),
        );

        // 輸送バー: Pitch = connected group・Start = filled(画面で 1 個の高強調)・Stop = error filled
        let mut pitch_group = group_container();
        for (i, (k, name)) in SEMITONES.iter().enumerate() {
            let selected = cfg.semitones == *k;
            let k = *k;
            pitch_group = pitch_group.child(
                group_seg(SharedString::from(format!("rt_pitch_{i}")), *name, selected).on_click(cx.listener(move |this, _, _, cx| {
                    let mut s = this.state.lock().unwrap();
                    s.settings.semitones = k;
                    if let Some(ref tx) = s.rt_control_tx {
                        let _ = tx.send(RtControl::SetSemitones(k));
                    }
                    cx.notify();
                })),
            );
        }
        let mut transport = div().flex().flex_wrap().items_center().gap_2();
        if !force_bypass {
            transport = transport
                .child(btn("bypass", if self.rt_bypass { "BYPASS" } else { "Bypass" }, if self.rt_bypass { Emph::Filled } else { Emph::Outlined }).on_click(cx.listener(|this, _, _, cx| {
                    this.rt_bypass = !this.rt_bypass;
                    state::send_control(&this.state, RtControl::Bypass(this.rt_bypass));
                    cx.notify();
                })))
                .child(btn("mute", if self.rt_muted { "MUTED" } else { "Mute" }, if self.rt_muted { Emph::Filled } else { Emph::Outlined }).on_click(cx.listener(|this, _, _, cx| {
                    this.rt_muted = !this.rt_muted;
                    state::send_control(&this.state, RtControl::Mute(this.rt_muted));
                    cx.notify();
                })));
        }
        transport = transport.child(subtext("Pitch")).child(pitch_group).child(div().flex_1());
        if self.rt_running {
            transport = transport.child(btn("stop_btn", "■ Stop", Emph::Filled).on_click(cx.listener(|this, _, _, cx| {
                this.stop(cx);
            })));
        } else {
            transport = transport.child(btn("start_btn", "▶ Start", Emph::Filled).on_click(cx.listener(|this, _, _, cx| {
                this.start(cx);
            })));
        }
        hero = hero.child(transport);
        el = el.child(hero);

        // ===== 2. 登録(low)=====
        let mut model_card = card().child(heading("登録"));
        let mut ex_row = div().flex().flex_wrap().items_center().gap_2();
        for (id, label, cur) in [("front", "front", cfg.front.clone()), ("model", "E / C", cfg.model.clone()), ("voc", "voc", cfg.voc.clone())] {
            let short = std::path::Path::new(&cur).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(cur.clone());
            ex_row = ex_row.child(
                div().flex().items_center().gap_1().child(subtext(format!("{label}: {short}"))).child(
                    btn(SharedString::from(format!("pick_{id}")), "…", Emph::Text).on_click({
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
        let mut th_group = group_container();
        for n in [1usize, 2, 4] {
            let selected = cfg.threads == n;
            th_group = th_group.child(
                group_seg(SharedString::from(format!("th_{n}")), n.to_string(), selected).on_click(cx.listener(move |this, _, _, cx| {
                    this.state.lock().unwrap().settings.threads = n;
                    cx.notify();
                })),
            );
        }
        model_card = model_card.child(div().flex().flex_wrap().items_center().gap_3().child(subtext("threads")).child(th_group));
        model_card = model_card.child(subtext(format!("Operator: {} 本", cfg.operator.len())));
        let mut op_row = div().flex().flex_wrap().items_center().gap_2();
        op_row = op_row.child(btn("op_add", "Add operator WAV", Emph::Tonal).on_click(cx.listener(|this, _, _, cx| {
            this.picks.operator.open();
            cx.notify();
        })));
        if let Some(rem) = rec {
            op_row = op_row
                .child(div().flex().items_center().gap_2().child(rec_dot()).child(div().text_size(px(12.)).font_weight(FontWeight::BOLD).text_color(md::error()).child(format!("REC 残り {rem} s"))))
                .child(btn("op_cancel", "Cancel", Emph::Text).on_click(cx.listener(|this, _, _, cx| {
                    if let Some(r) = this.state.lock().unwrap().operator_rec.as_ref() {
                        r.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    cx.notify();
                })));
        } else {
            op_row = op_row.child(btn("op_rec", "Record operator (30 s)", Emph::Tonal).on_click(cx.listener(|this, _, _, cx| {
                if !this.rt_running {
                    state::record_operator(&this.state, 30);
                }
                cx.notify();
            })));
        }
        if !cfg.operator.is_empty() {
            op_row = op_row.child(btn("op_clear", "Clear", Emph::Text).on_click(cx.listener(|this, _, _, cx| {
                this.state.lock().unwrap().settings.operator.clear();
                cx.notify();
            })));
        }
        model_card = model_card.child(op_row);
        let can_load = !cfg.operator.is_empty() && ref_name.is_some() && !is_loading;
        model_card = model_card.child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_3()
                .child(btn("load", if has_engine { "Reload" } else { "Load" }, if can_load { Emph::Tonal } else { Emph::Outlined }).on_click(cx.listener(|this, _, _, cx| {
                    let ok = {
                        let s = this.state.lock().unwrap();
                        !s.settings.operator.is_empty() && s.selected_voice.is_some()
                    };
                    if ok {
                        state::load_engine(&this.state);
                    }
                    cx.notify();
                })))
                .when(is_loading, |r| r.child(loading("モデルを読み込み中…")))
                .when(ref_name.is_none(), |r| r.child(div().text_size(px(11.)).text_color(md::error()).child("Target で目標を選ぶ"))),
        );

        // Target(chips)
        let mut target_row = div().flex().flex_wrap().items_center().gap_2().child(subtext("Target"));
        if voices.is_empty() {
            target_row = target_row.child(
                btn("tgt_open", "Open Voices →", Emph::Tonal).on_click(cx.listener(|this, _, _, cx| {
                    this.tab = Tab::Catalog;
                    cx.notify();
                })),
            );
        } else {
            for (i, name) in voices.iter().enumerate() {
                let selected = selected_voice == Some(i);
                target_row = target_row.child(
                    chip(SharedString::from(format!("tgt_{i}")), name.clone(), selected).on_click(cx.listener(move |this, _, _, cx| {
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
            target_row = target_row.child(
                div()
                    .id("tgt_manage")
                    .px_2()
                    .text_size(px(11.))
                    .text_color(md::secondary())
                    .hover(|s| s.text_color(md::primary()))
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.tab = Tab::Catalog;
                        cx.notify();
                    }))
                    .child("管理…"),
            );
        }
        model_card = model_card.child(target_row);
        el = el.child(model_card);

        // ===== 3. 診断(白・四角 = tension)=====
        let mut diag = div().bg(md::surface_lowest()).rounded(px(md::SHAPE_XS)).p_4().flex().flex_col().gap_3();
        diag = diag.child(div().flex().flex_wrap().items_center().gap_4().child(subtext("Input")).child(meter(self.metrics.input_rms * 6.0, md::primary())));
        let mut buf_group = group_container();
        for n in BUFFERS {
            let selected = self.rt_buffer == n;
            buf_group = buf_group.child(
                group_seg(SharedString::from(format!("buf_{n}")), n.to_string(), selected).on_click(cx.listener(move |this, _, _, cx| {
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
        diag = diag.child(div().flex().flex_wrap().items_center().gap_3().child(subtext("Buffer")).child(buf_group));
        let buf_ms = self.rt_buffer as f32 * 1000.0 / 48_000.0;
        let resamp_ms = 3.0;
        let algo_ms = lightvc_core::snv_stream::SnvStream::LATENCY_SAMPLES as f32 / 48.0;
        let total = (buf_ms * 2.0 + resamp_ms * 2.0 + algo_ms).max(0.001);
        let segs: [(f32, gpui::Rgba); 5] = [(buf_ms, md::secondary()), (resamp_ms, md::tertiary()), (algo_ms, md::primary()), (resamp_ms, md::tertiary()), (buf_ms, md::secondary())];
        let mut seg_row = div().flex().w_full().h(px(8.));
        for (ms, c) in segs {
            seg_row = seg_row.child(div().flex_grow().h_full().bg(c).map(move |s| s.w(px(600.0 * ms / total))));
        }
        diag = diag.child(seg_row);
        diag = diag.child(div().flex().gap_4().child(subtext("buf")).child(subtext("resamp")).child(subtext("algo")).child(subtext("E2E の内訳")));
        el = el.child(diag);

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
                div().flex().items_center().gap_3().child(subtext("WAV を選ぶとファイル名で登録します")).child(btn("voice_add", "Browse…", Emph::Tonal).on_click(cx.listener(|this, _, _, cx| {
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
                        .rounded_full()
                        .map(move |s| if selected { s.bg(md::primary_container()) } else { s })
                        .hover(|s| s.bg(md::primary_container()))
                        .child(
                            div()
                                .id(SharedString::from(format!("voice_sel_{i}")))
                                .flex()
                                .flex_1()
                                .items_center()
                                .gap_3()
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
                                .child(div().text_size(px(13.)).font_weight(FontWeight::SEMIBOLD).text_color(if selected { md::on_primary_container() } else { md::on_surface() }).child(name.clone()))
                                .child(div().flex_1().text_size(px(10.5)).text_color(md::on_surface_variant()).child(path.display().to_string())),
                        )
                        .child(btn(SharedString::from(format!("voice_play_{i}")), if playing { "Stop" } else { "Play" }, Emph::Tonal).on_click({
                            let path = path2.clone();
                            cx.listener(move |this, _, _, cx| {
                                if this.catalog.playing == Some(i) && this.catalog.player.is_some() {
                                    toggle_stop(&mut this.catalog.player);
                                    this.catalog.playing = None;
                                } else {
                                    toggle_stop(&mut this.catalog.player);
                                    match play_path(&mut this.catalog.player, &path) {
                                        Ok(()) => {
                                            this.catalog.playing = Some(i);
                                            let mut s = this.state.lock().unwrap();
                                            s.error = None;
                                            s.status = format!("再生中: {}", path.display());
                                        }
                                        Err(e) => {
                                            this.catalog.playing = None;
                                            this.state.lock().unwrap().error = Some(format!("再生できない: {e}"));
                                        }
                                    }
                                }
                                cx.notify();
                            })
                        }))
                        .child(btn(SharedString::from(format!("voice_del_{i}")), "✕", Emph::Text).on_click(cx.listener(move |this, _, _, cx| {
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
