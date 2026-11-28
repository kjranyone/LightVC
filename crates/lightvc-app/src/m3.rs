//! Material 3 Expressive のトークンと部品(正本は DESIGN.md)。
//!
//! 色は Material の HCT トーンパレットから生成した値(DESIGN.md §1)。寸法と形は M3E の部品の仕様。

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use gpui::{
    canvas, div, point, prelude::*, px, relative, svg, Animation, AnimationExt as _, AnyElement,
    AssetSource, Div, ElementId, FontWeight, PathBuilder, Rgba, SharedString, Stateful, Styled, Svg,
};

static DARK: AtomicBool = AtomicBool::new(false);

pub fn set_dark(dark: bool) {
    DARK.store(dark, Ordering::Relaxed);
}

/// M3 の色ロール(light / dark)。
pub mod c {
    use super::{Ordering, DARK};
    use gpui::{rgb, Rgba};

    macro_rules! role {
        ($name:ident, $light:expr, $dark:expr) => {
            pub fn $name() -> Rgba {
                if DARK.load(Ordering::Relaxed) { rgb($dark) } else { rgb($light) }
            }
        };
    }

    role!(primary, 0xA82F5F, 0xFFB1C6);
    role!(on_primary, 0xFFFFFF, 0x650031);
    role!(primary_container, 0xFFD9E1, 0x891447);
    role!(on_primary_container, 0x891447, 0xFFD9E1);
    role!(secondary, 0x63568F, 0xCDBDFF);
    role!(on_secondary, 0xFFFFFF, 0x34275E);
    role!(secondary_container, 0xE8DEFF, 0x4B3E76);
    role!(on_secondary_container, 0x4B3E76, 0xE8DEFF);
    role!(tertiary, 0x006C4F, 0x81D7B3);
    role!(on_tertiary, 0xFFFFFF, 0x003827);
    role!(tertiary_container, 0x9DF4CF, 0x00513B);
    role!(on_tertiary_container, 0x00513B, 0x9DF4CF);
    role!(error, 0xBA1A1A, 0xFFB4AB);
    role!(on_error_container, 0x93000A, 0xFFDAD6);
    role!(error_container, 0xFFDAD6, 0x93000A);
    role!(surface, 0xFDF7FF, 0x141318);
    role!(surface_container, 0xF2ECF4, 0x201F24);
    role!(surface_container_high, 0xECE6EE, 0x2B292F);
    role!(on_surface, 0x1C1B20, 0xE6E1E9);
    role!(on_surface_variant, 0x484551, 0xCAC4D2);
    role!(outline_variant, 0xCAC4D2, 0x484551);
    role!(inverse_surface, 0x312F35, 0xE6E1E9);
    role!(inverse_on_surface, 0xF4EFF7, 0x312F35);
    role!(inverse_primary, 0xFFB1C6, 0xA82F5F);
}

/// state layer: base の上に over を不透明度 a で重ねた色(hover 8 %・pressed 10 %)。
pub fn mix(base: Rgba, over: Rgba, a: f32) -> Rgba {
    Rgba {
        r: base.r + (over.r - base.r) * a,
        g: base.g + (over.g - base.g) * a,
        b: base.b + (over.b - base.b) * a,
        a: base.a,
    }
}

pub fn alpha(color: Rgba, a: f32) -> Rgba {
    Rgba { a, ..color }
}

const HOVER: f32 = 0.08;
const PRESSED: f32 = 0.12;

/// M3 の shape scale。
pub mod shape {
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 12.0;
    pub const LG: f32 = 16.0;
    pub const XL: f32 = 28.0;
}

/// M3 の type scale(size / line height / weight)。
pub trait Typo: Styled + Sized {
    fn ty(self, size: f32, line: f32, weight: FontWeight) -> Self {
        self.text_size(px(size)).line_height(px(line)).font_weight(weight)
    }
    fn display_l(self) -> Self { self.ty(57., 64., FontWeight::BOLD) }
    fn headline_s(self) -> Self { self.ty(24., 32., FontWeight::SEMIBOLD) }
    fn title_l(self) -> Self { self.ty(22., 28., FontWeight::MEDIUM) }
    fn title_m(self) -> Self { self.ty(16., 24., FontWeight::SEMIBOLD) }
    fn body_l(self) -> Self { self.ty(16., 24., FontWeight::NORMAL) }
    fn body_m(self) -> Self { self.ty(14., 20., FontWeight::NORMAL) }
    fn body_s(self) -> Self { self.ty(12., 16., FontWeight::NORMAL) }
    fn label_l(self) -> Self { self.ty(14., 20., FontWeight::MEDIUM) }
    fn label_m(self) -> Self { self.ty(12., 16., FontWeight::MEDIUM) }
}

impl<T: Styled> Typo for T {}

// ===== アイコン(Material Symbols Rounded・Apache-2.0)=====

pub struct Assets;

macro_rules! icons {
    ($($name:literal),* $(,)?) => {
        fn icon_bytes(name: &str) -> Option<&'static [u8]> {
            match name {
                $($name => Some(include_bytes!(concat!("../assets/icons/", $name, ".svg"))),)*
                _ => None,
            }
        }
    };
}

icons!(
    "graphic_eq", "graphic_eq_fill1", "audio_file", "audio_file_fill1", "record_voice_over",
    "record_voice_over_fill1", "settings", "settings_fill1", "play_arrow_fill1", "stop_fill1", "mic",
    "mic_off", "volume_up", "volume_off", "check", "close", "add", "folder_open", "delete", "download",
    "refresh", "person", "arrow_forward", "error", "alt_route", "auto_awesome_fill1", "info", "remove",
);

impl AssetSource for Assets {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        let name = path.strip_prefix("icons/").and_then(|p| p.strip_suffix(".svg")).unwrap_or(path);
        Ok(icon_bytes(name).map(Cow::Borrowed))
    }

    fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok(Vec::new())
    }
}

pub fn icon_exists(name: &str) -> bool {
    icon_bytes(name).is_some()
}

pub fn icon(name: &str, size: f32, color: Rgba) -> Svg {
    svg().path(format!("icons/{name}.svg")).size(px(size)).flex_none().text_color(color)
}

// ===== ボタン =====

#[derive(Copy, Clone, PartialEq)]
pub enum Btn {
    Filled,
    Tonal,
    Outlined,
    Text,
}

fn btn_colors(kind: Btn) -> (Option<Rgba>, Rgba) {
    match kind {
        Btn::Filled => (Some(c::primary()), c::on_primary()),
        Btn::Tonal => (Some(c::secondary_container()), c::on_secondary_container()),
        Btn::Outlined => (None, c::on_surface_variant()),
        Btn::Text => (None, c::primary()),
    }
}

/// M3E の button(size S・高さ 40・round)。押すと角が締まる。
pub fn button(id: impl Into<ElementId>, icon_name: Option<&str>, label: impl Into<SharedString>, kind: Btn, enabled: bool) -> Stateful<Div> {
    let (bg, fg) = btn_colors(kind);
    let (bg, fg) = if enabled {
        (bg, fg)
    } else {
        (bg.map(|_| alpha(c::on_surface(), 0.10)), alpha(c::on_surface(), 0.38))
    };
    let pad = if kind == Btn::Text { 12. } else { 16. };
    div()
        .id(id)
        .h(px(40.))
        .px(px(pad))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .gap_2()
        .rounded_full()
        .label_l()
        .whitespace_nowrap()
        .text_color(fg)
        .when_some(bg, |s, bg| s.bg(bg))
        .when(kind == Btn::Outlined, |s| s.border_1().border_color(c::outline_variant()))
        .when(enabled, |s| {
            s.cursor_pointer()
                .hover(|s| s.bg(bg.map(|b| mix(b, fg, HOVER)).unwrap_or(alpha(fg, HOVER))))
                .active(|s| s.bg(bg.map(|b| mix(b, fg, PRESSED)).unwrap_or(alpha(fg, PRESSED))).rounded(px(shape::SM)))
        })
        .when_some(icon_name, |s, name| s.child(icon(name, 20., fg)))
        .child(label.into())
}

/// M3E の button(size M・高さ 56)。画面の主役の行動に 1 つだけ使う。square = 選択済みの形。
pub fn button_m(id: impl Into<ElementId>, icon_name: &str, label: impl Into<SharedString>, bg: Rgba, fg: Rgba, square: bool, enabled: bool) -> Stateful<Div> {
    let (bg, fg) = if enabled { (bg, fg) } else { (alpha(c::on_surface(), 0.10), alpha(c::on_surface(), 0.38)) };
    div()
        .id(id)
        .h(px(56.))
        .pl(px(20.))
        .pr(px(24.))
        .flex()
        .flex_none()
        .items_center()
        .gap_2()
        .rounded(px(if square { shape::LG } else { 28. }))
        .title_m()
        .whitespace_nowrap()
        .bg(bg)
        .text_color(fg)
        .when(enabled, |s| {
            s.cursor_pointer()
                .shadow_sm()
                .hover(|s| s.bg(mix(bg, fg, HOVER)))
                .active(|s| s.bg(mix(bg, fg, PRESSED)).rounded(px(shape::MD)))
        })
        .child(icon(icon_name, 24., fg))
        .child(label.into())
}

/// M3E の toggle button(tonal)。未選択 = round・選択 = square で形も変わる。
pub fn toggle_button(id: impl Into<ElementId>, icon_name: &str, label: impl Into<SharedString>, selected: bool) -> Stateful<Div> {
    let (bg, fg) = if selected { (c::secondary(), c::on_secondary()) } else { (c::secondary_container(), c::on_secondary_container()) };
    div()
        .id(id)
        .h(px(40.))
        .px(px(16.))
        .flex()
        .flex_none()
        .items_center()
        .gap_2()
        .rounded(px(if selected { shape::MD } else { 20. }))
        .label_l()
        .whitespace_nowrap()
        .bg(bg)
        .text_color(fg)
        .cursor_pointer()
        .hover(|s| s.bg(mix(bg, fg, HOVER)))
        .active(|s| s.bg(mix(bg, fg, PRESSED)).rounded(px(shape::SM)))
        .child(icon(icon_name, 20., fg))
        .child(label.into())
}

/// M3 の icon button(standard / tonal・40 × 40)。
pub fn icon_button(id: impl Into<ElementId>, icon_name: &str, tonal: bool, fg: Rgba) -> Stateful<Div> {
    let bg = tonal.then(c::secondary_container);
    let fg = if tonal { c::on_secondary_container() } else { fg };
    div()
        .id(id)
        .size(px(40.))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .rounded_full()
        .cursor_pointer()
        .when_some(bg, |s, bg| s.bg(bg))
        .hover(|s| s.bg(bg.map(|b| mix(b, fg, HOVER)).unwrap_or(alpha(fg, HOVER))))
        .active(|s| s.bg(bg.map(|b| mix(b, fg, PRESSED)).unwrap_or(alpha(fg, PRESSED))).rounded(px(shape::MD)))
        .child(icon(icon_name, 24., fg))
}

/// M3E の connected button group の 1 区画。間隔 2・内側の角 8・外側と選択中は full。
pub fn segment(id: impl Into<ElementId>, label: impl Into<SharedString>, selected: bool, first: bool, last: bool) -> Stateful<Div> {
    let (bg, fg) = if selected { (c::secondary(), c::on_secondary()) } else { (c::secondary_container(), c::on_secondary_container()) };
    let outer = px(20.);
    let inner = px(shape::SM);
    let (l, r) = if selected { (outer, outer) } else { (if first { outer } else { inner }, if last { outer } else { inner }) };
    div()
        .id(id)
        .h(px(40.))
        .min_w(px(52.))
        .px(px(14.))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .rounded_tl(l)
        .rounded_bl(l)
        .rounded_tr(r)
        .rounded_br(r)
        .label_l()
        .whitespace_nowrap()
        .bg(bg)
        .text_color(fg)
        .cursor_pointer()
        .hover(|s| s.bg(mix(bg, fg, HOVER)))
        .active(|s| s.bg(mix(bg, fg, PRESSED)).rounded(px(shape::XS)))
        .child(label.into())
}

pub fn segments() -> Div {
    div().flex().flex_none().items_center().gap(px(2.))
}

/// M3 の filter chip(高さ 32・角 8)。
pub fn chip(id: impl Into<ElementId>, label: impl Into<SharedString>, selected: bool) -> Stateful<Div> {
    let fg = if selected { c::on_secondary_container() } else { c::on_surface_variant() };
    div()
        .id(id)
        .h(px(32.))
        .flex()
        .flex_none()
        .items_center()
        .gap_2()
        .rounded(px(shape::SM))
        .label_l()
        .whitespace_nowrap()
        .text_color(fg)
        .cursor_pointer()
        .map(|s| {
            if selected {
                s.pl(px(8.)).pr(px(16.)).bg(c::secondary_container()).hover(|s| s.bg(mix(c::secondary_container(), fg, HOVER))).child(icon("check", 18., fg))
            } else {
                s.px(px(15.)).border_1().border_color(c::outline_variant()).hover(|s| s.bg(alpha(fg, HOVER)))
            }
        })
        .child(label.into())
}

// ===== 容器 =====

/// 塗りの容器(登録・設定)。
pub fn card() -> Div {
    div().bg(c::surface_container()).rounded(px(shape::LG)).p_4().flex().flex_col().gap_3()
}

/// 縁の容器(低強調)。
pub fn card_outlined() -> Div {
    div().border_1().border_color(c::outline_variant()).rounded(px(shape::MD)).p_4().flex().flex_col().gap_3()
}

/// M3E の segmented list の 1 行。行ごとに面を持ち、間隔 2・端の行だけ外側の角が大きい。選択中は丸くなる。
pub fn list_item(id: impl Into<ElementId>, selected: bool, first: bool, last: bool) -> Stateful<Div> {
    let (bg, fg) = if selected { (c::secondary_container(), c::on_secondary_container()) } else { (c::surface_container(), c::on_surface()) };
    let big = px(shape::LG);
    let small = px(shape::XS);
    let (t, b) = if selected { (px(shape::XL), px(shape::XL)) } else { (if first { big } else { small }, if last { big } else { small }) };
    div()
        .id(id)
        .min_h(px(56.))
        .px_4()
        .py_2()
        .flex()
        .items_center()
        .gap_4()
        .rounded_tl(t)
        .rounded_tr(t)
        .rounded_bl(b)
        .rounded_br(b)
        .bg(bg)
        .text_color(fg)
        .hover(|s| s.bg(mix(bg, fg, HOVER)))
}

pub fn list() -> Div {
    div().flex().flex_col().gap(px(2.))
}

/// 丸い器に入れたアイコン(行の先頭・カードの見出し)。
pub fn avatar(icon_name: &str, bg: Rgba, fg: Rgba) -> Div {
    div().size(px(40.)).flex().flex_none().items_center().justify_center().rounded_full().bg(bg).child(icon(icon_name, 22., fg))
}

// ===== 指標 =====

/// M3E の linear indicator(太さ 8・針と溝の間 4・末端の停止点)。
pub fn meter(level: f32, active: Rgba, track: Rgba) -> Div {
    let level = level.clamp(0.0, 1.0);
    div()
        .w_full()
        .h(px(8.))
        .flex()
        .items_center()
        .gap(px(4.))
        .child(div().flex_none().h_full().min_w(px(8.)).w(relative(level * 0.94)).rounded_full().bg(active))
        .child(
            div().relative().flex_1().h_full().rounded_full().bg(track)
                .child(div().absolute().right(px(2.)).top(px(2.)).size(px(4.)).rounded_full().bg(active)),
        )
}

fn clock() -> f32 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f32()
}

/// オーバーシュートして落ち着く spring(M3E の spatial spring)。0 → 1。
pub fn spring(t: f32) -> f32 {
    1.0 - (-7.0 * t).exp() * (9.0 * t).cos()
}

/// M3E の loading indicator。形が spring で次々に変わりながら回る(描画は時刻から決める)。
pub fn loading_indicator(size: f32) -> impl IntoElement {
    const SHAPES: [(f32, f32); 6] = [(8.0, 0.09), (4.0, 0.17), (5.0, 0.07), (2.0, 0.24), (6.0, 0.12), (3.0, 0.15)];
    const STEP: f32 = 0.65;
    const POINTS: usize = 120;
    let t = clock();
    let color = c::primary();
    canvas(
        |_, _, _| {},
        move |bounds, _, window, _| {
            let k = (t / STEP) as usize;
            let f = spring((t / STEP).fract());
            let (n0, a0) = SHAPES[k % SHAPES.len()];
            let (n1, a1) = SHAPES[(k + 1) % SHAPES.len()];
            let center = bounds.center();
            let radius = f32::from(bounds.size.width) / 2.0;
            let rot = (k as f32 + f) * 0.9;
            let mut path = PathBuilder::fill();
            for i in 0..POINTS {
                let th = i as f32 / POINTS as f32 * std::f32::consts::TAU;
                let r0 = (1.0 + a0 * (n0 * th).cos()) / (1.0 + a0);
                let r1 = (1.0 + a1 * (n1 * th).cos()) / (1.0 + a1);
                let r = radius * (r0 + (r1 - r0) * f).min(1.0);
                let p = point(center.x + px(r * (th + rot).cos()), center.y + px(r * (th + rot).sin()));
                if i == 0 { path.move_to(p) } else { path.line_to(p) }
            }
            path.close();
            if let Ok(path) = path.build() {
                window.paint_path(path, color);
            }
        },
    )
    .size(px(size))
    .flex_none()
}

/// 録音中の点(脈動)。
pub fn rec_dot() -> impl IntoElement {
    div().size(px(12.)).flex_none().rounded_full().bg(c::error()).with_animation(
        "rec_dot",
        Animation::new(Duration::from_millis(900)).repeat(),
        |el, d| el.opacity(0.35 + 0.65 * (0.5 + 0.5 * (d * std::f32::consts::TAU).cos())),
    )
}

// ===== ナビゲーション =====

/// M3E の navigation rail の 1 項目。選択中は指示子(56 × 32)が spring で広がり、アイコンが塗りになる。
pub fn nav_item(key: &'static str, icon_name: &str, label: &str, selected: bool) -> Stateful<Div> {
    let fg = if selected { c::on_secondary_container() } else { c::on_surface_variant() };
    let filled = format!("{icon_name}_fill1");
    let glyph = if selected && icon_exists(&filled) { filled } else { icon_name.to_string() };
    let pill = div().h(px(32.)).flex().items_center().justify_center().rounded_full().child(icon(&glyph, 24., fg));
    let indicator: AnyElement = if selected {
        pill.bg(c::secondary_container())
            .with_animation(ElementId::Name(format!("{key}_on").into()), Animation::new(Duration::from_millis(450)), |el, d| {
                el.w(px(32.0 + 24.0 * spring(d)))
            })
            .into_any_element()
    } else {
        pill.w(px(56.)).group_hover(key, |s| s.bg(alpha(c::on_surface(), HOVER))).into_any_element()
    };
    div()
        .id(key)
        .group(key)
        .w(px(96.))
        .py(px(6.))
        .flex()
        .flex_col()
        .items_center()
        .gap_1()
        .cursor_pointer()
        .child(div().w(px(64.)).h(px(32.)).flex().items_center().justify_center().child(indicator))
        .child(
            div().label_m()
                .when(selected, |s| s.font_weight(FontWeight::BOLD))
                .text_color(if selected { c::secondary() } else { c::on_surface_variant() })
                .child(label.to_string()),
        )
}
