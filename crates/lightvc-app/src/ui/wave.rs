//! 音声の波形とシークバー(ファイル変換の変換元・変換結果)。
//!
//! 波形の列を押す・なぞると、その位置へ飛ぶ(止まっていればそこから鳴らす)。再生した部分は primary で塗る。
//! キーボード(フォーカス中): Space = 再生 / 停止・← → = 5 秒・Home = 先頭。

use std::path::PathBuf;
use std::sync::Arc;

use gpui::{div, prelude::*, px, AnyElement, Context, KeyDownEvent, MouseButton, SharedString};

use super::{supporting, LightVcUi};
use crate::audio_playback::{self, AudioPlayer};
use crate::m3::{self, c, Typo};

/// 波形の列の数。
const COLS: usize = 160;
/// ← → で飛ぶ秒。
const STEP_S: f32 = 5.0;

/// 鳴らせる音声と、その波形の列(列ごとのピーク・0〜1)。
pub struct Wave {
    pub path: Option<PathBuf>,
    pub peaks: Vec<f32>,
    pub secs: f32,
    pub samples44: Arc<Vec<f32>>,
}

impl Wave {
    pub fn new(path: Option<PathBuf>, x: &[f32], sr: u32) -> Self {
        let secs = x.len() as f32 / sr.max(1) as f32;
        let per = x.len().div_ceil(COLS).max(1);
        let mut peaks: Vec<f32> = x.chunks(per).map(|c| c.iter().fold(0f32, |a, v| a.max(v.abs()))).collect();
        peaks.resize(COLS, 0.0);
        let top = peaks.iter().fold(0f32, |a, &v| a.max(v)).max(1e-4);
        peaks.iter_mut().for_each(|v| *v = (*v / top).sqrt());
        Self { path, peaks, secs, samples44: Arc::new(audio_playback::resample_linear(x, sr)) }
    }

    pub fn load(path: PathBuf) -> anyhow::Result<Self> {
        let (x, sr) = audio_playback::load_audio_mono(&path)?;
        Ok(Self::new(Some(path), &x, sr))
    }
}

/// どの再生か。
#[derive(Clone, Copy, PartialEq)]
pub enum Slot {
    Source,
    Result,
}

pub fn clock(s: f32) -> String {
    let s = s.max(0.0).round() as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

impl LightVcUi {
    fn slot_parts(&mut self, slot: Slot) -> (Option<Arc<Wave>>, &mut Option<AudioPlayer>) {
        match slot {
            Slot::Source => (self.offline.source_wave.clone(), &mut self.offline.source_preview),
            Slot::Result => (self.offline.result_wave.clone(), &mut self.offline.player),
        }
    }

    /// 位置(0〜1)へ飛ぶ。止まっていればそこから鳴らす。
    pub(super) fn seek_slot(&mut self, slot: Slot, frac: f32, cx: &mut Context<Self>) {
        let (wave, player) = self.slot_parts(slot);
        let Some(wave) = wave else { return };
        match player.as_ref().filter(|p| p.is_playing()) {
            Some(p) => p.seek(frac),
            None => *player = AudioPlayer::play_from(wave.samples44.to_vec(), frac).ok(),
        }
        cx.notify();
    }

    pub(super) fn toggle_slot(&mut self, slot: Slot, cx: &mut Context<Self>) {
        let (wave, player) = self.slot_parts(slot);
        let Some(wave) = wave else { return };
        match player.take() {
            Some(p) if p.is_playing() => {
                p.stop();
                *player = Some(p);
            }
            last => {
                let from = last.map(|p| p.progress()).filter(|&f| f < 0.999).unwrap_or(0.0);
                *player = AudioPlayer::play_from(wave.samples44.to_vec(), from).ok();
            }
        }
        cx.notify();
    }

    fn slot_position(&self, slot: Slot) -> (bool, f32) {
        let p = match slot {
            Slot::Source => &self.offline.source_preview,
            Slot::Result => &self.offline.player,
        };
        p.as_ref().map_or((false, 0.0), |p| (p.is_playing(), p.progress()))
    }

    /// 再生 / 停止の button・波形(シークバー)・時刻。
    pub(super) fn wave_player(&self, slot: Slot, wave: &Wave, fg: gpui::Rgba, cx: &mut Context<Self>) -> AnyElement {
        let id = match slot {
            Slot::Source => "src",
            Slot::Result => "out",
        };
        let (playing, frac) = self.slot_position(slot);
        let cur = ((frac * COLS as f32) as usize).min(COLS - 1);
        let shown = playing || frac > 0.0;
        let mut bars = div().flex().items_center().h(px(56.)).flex_1().min_w_0();
        for (i, &v) in wave.peaks.iter().enumerate() {
            let played = shown && i <= cur;
            let color = if played { c::primary() } else { m3::alpha(fg, 0.32) };
            bars = bars.child(
                div().relative().flex_1().h_full().flex().items_center().justify_center()
                    .cursor_pointer()
                    .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, window, cx| {
                        window.focus(this.wave_focus(slot));
                        this.seek_slot(slot, i as f32 / COLS as f32, cx);
                    }))
                    .on_mouse_move(cx.listener(move |this, ev: &gpui::MouseMoveEvent, _, cx| {
                        if ev.dragging() {
                            this.seek_slot(slot, i as f32 / COLS as f32, cx);
                        }
                    }))
                    .child(div().w(relative_w()).h(px((v * 48.).max(2.))).rounded_full().bg(color))
                    .when(shown && i == cur, |s| s.child(div().absolute().top_0().bottom_0().left(px(0.)).w(px(2.)).rounded_full().bg(c::primary()))),
            );
        }
        let focus = self.wave_focus(slot).clone();
        div().id(SharedString::from(format!("{id}_wave")))
            .track_focus(&focus)
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let secs = match slot {
                    Slot::Source => this.offline.source_wave.as_ref().map(|w| w.secs),
                    Slot::Result => this.offline.result_wave.as_ref().map(|w| w.secs),
                }
                .unwrap_or(0.0)
                .max(0.1);
                let (_, at) = this.slot_position(slot);
                match ev.keystroke.key.as_str() {
                    "space" => this.toggle_slot(slot, cx),
                    "left" => this.seek_slot(slot, at - STEP_S / secs, cx),
                    "right" => this.seek_slot(slot, at + STEP_S / secs, cx),
                    "home" => this.seek_slot(slot, 0.0, cx),
                    _ => return,
                }
                cx.stop_propagation();
            }))
            .flex().items_center().gap_3().p_1().rounded(px(m3::shape::LG))
            .border_2().border_color(m3::alpha(c::secondary(), 0.0))
            .focus(|s| s.border_color(c::secondary()))
            .child(m3::icon_button(SharedString::from(format!("{id}_toggle")), if playing { "stop_fill1" } else { "play_arrow_fill1" }, true, c::on_secondary_container())
                .on_click(cx.listener(move |this, _, window, cx| {
                    window.focus(this.wave_focus(slot));
                    this.toggle_slot(slot, cx);
                })))
            .child(bars)
            .child(div().flex_none().w(px(88.)).flex().justify_end()
                .child(div().label_m().text_color(fg).child(format!("{} / {}", clock(frac * wave.secs), clock(wave.secs)))))
            .into_any_element()
    }

    fn wave_focus(&self, slot: Slot) -> &gpui::FocusHandle {
        match slot {
            Slot::Source => &self.offline_focus[0],
            Slot::Result => &self.offline_focus[1],
        }
    }

    /// 波形がまだ無いとき(読み込み中・読めない)の表示。
    pub(super) fn wave_placeholder(&self, text: &str) -> AnyElement {
        div().h(px(64.)).flex().items_center().child(supporting(text.to_string())).into_any_element()
    }
}

fn relative_w() -> gpui::DefiniteLength {
    gpui::relative(0.6)
}
