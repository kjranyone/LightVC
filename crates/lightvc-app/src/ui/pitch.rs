//! ピッチ(固定の半音・−12〜+12 の整数)の操作部。
//!
//! − / + の icon button と数値、リアルタイムでは 1 半音ごとの slider も出す(押す・なぞる)。
//! キーボード(フォーカス中): ← ↓ / → ↑ = 1 半音・PageDown / PageUp = 1 オクターブ・Home / End = 端・0 / Delete = 0。

use gpui::{div, prelude::*, px, Context, Div, KeyDownEvent, MouseButton, SharedString, Stateful};
use lightvc_core::snv_stream::SnvStream;

use super::{supporting, LightVcUi};
use crate::m3::{self, c, Typo};
use crate::state::RtControl;

const MIN: i32 = SnvStream::SEMITONES_MIN as i32;
const MAX: i32 = SnvStream::SEMITONES_MAX as i32;

/// 「+5」「0」「−7」(負は U+2212)。
pub fn text(k: i32) -> String {
    match k {
        0 => "0".to_string(),
        k if k > 0 => format!("+{k}"),
        k => format!("\u{2212}{}", -k),
    }
}

impl LightVcUi {
    pub(super) fn set_semitones(&mut self, k: i32, cx: &mut Context<Self>) {
        let k = k.clamp(MIN, MAX) as f32;
        let mut s = self.state.lock().unwrap();
        if s.settings.semitones == k {
            return;
        }
        s.settings.semitones = k;
        if let Some(ref tx) = s.rt_control_tx {
            let _ = tx.send(RtControl::SetSemitones(k));
        }
        cx.notify();
    }

    pub(super) fn pitch_control(&self, prefix: &'static str, semitones: f32, slider: bool, cx: &mut Context<Self>) -> Stateful<Div> {
        let k = (semitones.round() as i32).clamp(MIN, MAX);
        let step = |id: &str, icon_name: &'static str, d: i32, cx: &mut Context<Self>| {
            let enabled = (MIN..=MAX).contains(&(k + d));
            m3::icon_button(SharedString::from(format!("{prefix}_{id}")), icon_name, true, c::on_secondary_container())
                .when(!enabled, |b| b.opacity(0.38).cursor_default())
                .on_click(cx.listener(move |this, _, window, cx| {
                    window.focus(&this.pitch_focus);
                    this.set_semitones(k + d, cx);
                }))
        };
        let value = div().w(px(64.)).flex().flex_col().items_center().flex_none()
            .child(div().title_l().text_color(c::on_surface()).child(text(k)))
            .child(div().label_m().text_color(c::on_surface_variant()).child("半音"));
        let stepper = div().flex().items_center().gap_1().flex_none()
            .child(step("down", "remove", -1, cx))
            .child(value)
            .child(step("up", "add", 1, cx));
        let mut root = div().id(SharedString::from(format!("{prefix}_ctl")))
            .track_focus(&self.pitch_focus)
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let to = match ev.keystroke.key.as_str() {
                    "left" | "down" => k - 1,
                    "right" | "up" => k + 1,
                    "pagedown" => k - 12,
                    "pageup" => k + 12,
                    "home" => MIN,
                    "end" => MAX,
                    "0" | "delete" | "backspace" => 0,
                    _ => return,
                };
                this.set_semitones(to, cx);
                cx.stop_propagation();
            }))
            .flex().flex_wrap().items_center().gap_4().p_1().rounded(px(m3::shape::LG))
            .border_2().border_color(m3::alpha(c::secondary(), 0.0))
            .focus(|s| s.border_color(c::secondary()))
            .child(stepper);
        if slider {
            root = root.child(div().flex().flex_col().gap_1().flex_1().min_w(px(260.)).max_w(px(420.))
                .child(self.pitch_slider(prefix, k, cx))
                .child(div().flex().justify_between()
                    .child(supporting(text(MIN)))
                    .child(supporting("0(元の高さ)"))
                    .child(supporting(text(MAX)))));
        }
        root
    }

    /// 1 半音 = 1 区画の slider。0 から今の値までを塗り、今の値に縦の取っ手を置く。
    fn pitch_slider(&self, prefix: &'static str, k: i32, cx: &mut Context<Self>) -> Div {
        let mut track = div().flex().items_center().h(px(40.)).w_full();
        for v in MIN..=MAX {
            let on = (v != 0 || k == 0) && ((0 < v && v <= k) || (k <= v && v < 0));
            let bar = if v == k {
                div().w(px(4.)).h(px(32.)).rounded_full().bg(c::primary())
            } else {
                let fill = if on { c::primary() } else { c::secondary_container() };
                let l = if v == MIN || v == k + 1 { px(8.) } else { px(0.) };
                let r = if v == MAX || v == k - 1 { px(8.) } else { px(0.) };
                div().flex_1().h(px(16.)).bg(fill).rounded_tl(l).rounded_bl(l).rounded_tr(r).rounded_br(r)
                    .when(v == 0, |b| b.flex().items_center().justify_center()
                        .child(div().size(px(4.)).rounded_full().bg(if on { c::on_primary() } else { c::on_secondary_container() })))
            };
            track = track.child(
                div().id(SharedString::from(format!("{prefix}_s{v}"))).flex_1().h_full().flex().items_center().justify_center().px(px(1.))
                    .cursor_pointer()
                    .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, window, cx| {
                        window.focus(&this.pitch_focus);
                        this.set_semitones(v, cx);
                    }))
                    .on_mouse_move(cx.listener(move |this, ev: &gpui::MouseMoveEvent, _, cx| {
                        if ev.dragging() {
                            this.set_semitones(v, cx);
                        }
                    }))
                    .child(bar),
            );
        }
        track
    }
}
