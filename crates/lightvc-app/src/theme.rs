//! Kawaii Future Bass theme for LightVC.
//!
//! Visual language:
//! - Milky lavender → pink gradient background (airy, not white-flat)
//! - Frosted glass cards with strong contrast text
//! - Cyan / mint / lavender accents give depth (not pink-monochrome)
//! - DAW-style meters: thick gradient bars, peak lines, dB readouts
//! - Neon-pink → mint gradient for active states (future-bass signature)
//! - Y2K sparkle kept subtle and functional
//!
//! Key principles:
//! 1. Readability first — text is deep plum, never pale pink
//! 2. Multi-hue depth — pink, cyan, mint, lavender all present
//! 3. Music-tool polish — meters and numeric cards are the hero

#![allow(dead_code)]

use eframe::egui;
use egui::Color32;

// ===========================================================================
// CJK font — load a system Japanese font so device names render correctly
// ===========================================================================

fn install_cjk_font(ctx: &egui::Context) {
    let candidates = [
        "C:\\Windows\\Fonts\\meiryo.ttc",
        "C:\\Windows\\Fonts\\YuGothM.ttc",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\msgothic.ttc",
    ];
    let font_data = candidates.iter().find_map(|p| std::fs::read(p).ok());
    let Some(font_data) = font_data else { return };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "cjk".into(),
        std::sync::Arc::new(egui::FontData::from_owned(font_data.into())),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push("cjk".into());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("cjk".into());
    ctx.set_fonts(fonts);
}

// ===========================================================================
// Design tokens
// ===========================================================================

pub mod space {
    pub const TIGHT: f32 = 2.0;
    pub const SMALL: f32 = 4.0;
    pub const MEDIUM: f32 = 8.0;
    pub const LARGE: f32 = 12.0;
    pub const XLARGE: f32 = 20.0;
}

pub const LABEL_WIDTH: f32 = 80.0;
pub const FIELD_HEIGHT: f32 = 28.0;
pub const BUTTON_HEIGHT: f32 = 34.0;
pub const CTA_HEIGHT: f32 = 46.0;
/// Card corner rounding — generous, candy-like.
pub const CARD_ROUNDING: f32 = 16.0;
/// Button corner rounding — pill-ish but not fully round.
pub const BUTTON_ROUNDING: f32 = 10.0;
pub const METER_HEIGHT: f32 = 22.0;

// ===========================================================================
// Color palette — Kawaii Future Bass / Neon Glassmorphism
// ===========================================================================

pub mod colors {
    use eframe::egui::Color32;

    // Backgrounds — milky lavender → pink gradient (airy, not white-flat)
    pub const BG_DEEP: Color32 = Color32::from_rgb(0xEE, 0xE4, 0xF2); // milky lavender
    pub const BG_DARK: Color32 = Color32::from_rgb(0xF2, 0xE4, 0xEC); // milky pink
    pub const BG_PANEL: Color32 = Color32::from_rgb(0xE4, 0xEC, 0xF2); // milky blue

    // Glass card tints — frosted white with hue shifts for depth
    pub const CARD_GLASS: Color32 = Color32::from_rgb(0xFD, 0xFC, 0xFE); // near-white, warm
    pub const CARD_GLASS_LIGHT: Color32 = Color32::from_rgb(0xF8, 0xF6, 0xFC); // lavender-tinted white
    pub const CARD_CYAN_TINT: Color32 = Color32::from_rgb(0xF2, 0xFA, 0xFC); // cyan-tinted white
    pub const CARD_MINT_TINT: Color32 = Color32::from_rgb(0xF2, 0xFC, 0xF8); // mint-tinted white

    // Primary accents — multi-hue for depth (not pink-monochrome)
    pub const PINK: Color32 = Color32::from_rgb(0xE0, 0x5A, 0x8A);
    pub const PINK_BRIGHT: Color32 = Color32::from_rgb(0xEC, 0x80, 0xA8);
    pub const PINK_DEEP: Color32 = Color32::from_rgb(0xC0, 0x3A, 0x6E);
    pub const LAVENDER: Color32 = Color32::from_rgb(0x88, 0x78, 0xC0);
    pub const LAVENDER_DEEP: Color32 = Color32::from_rgb(0x68, 0x58, 0xA8);
    pub const CYAN: Color32 = Color32::from_rgb(0x38, 0xA8, 0xC8);
    pub const CYAN_DEEP: Color32 = Color32::from_rgb(0x28, 0x88, 0xA8);
    pub const MINT: Color32 = Color32::from_rgb(0x38, 0xB8, 0x98);
    pub const MINT_DEEP: Color32 = Color32::from_rgb(0x28, 0x90, 0x78);
    pub const LEMON: Color32 = Color32::from_rgb(0xD8, 0xA8, 0x20);
    pub const LEMON_DEEP: Color32 = Color32::from_rgb(0xB0, 0x88, 0x10);
    pub const CORAL: Color32 = Color32::from_rgb(0xE0, 0x70, 0x50);
    pub const CORAL_DEEP: Color32 = Color32::from_rgb(0xC0, 0x50, 0x30);

    // Text — DEEP plum/charcoal for strong contrast on pastel bg
    pub const TEXT: Color32 = Color32::from_rgb(0x1E, 0x14, 0x32); // near-black plum
    pub const TEXT_DIM: Color32 = Color32::from_rgb(0x4A, 0x3C, 0x62); // medium plum
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x7A, 0x6C, 0x92); // muted plum

    // Borders — defined lavender/cyan for card edges
    pub const BORDER: Color32 = Color32::from_rgb(0xC8, 0xB8, 0xE4); // soft lavender border
    pub const BORDER_DEEP: Color32 = Color32::from_rgb(0xA8, 0x98, 0xD0); // stronger lavender border
    pub const BORDER_CYAN: Color32 = Color32::from_rgb(0xB8, 0xE0, 0xE8); // cyan border

    // Functional — state colors (vivid enough to read)
    pub const ERROR: Color32 = Color32::from_rgb(0xE8, 0x4A, 0x5C);
    pub const ERROR_DEEP: Color32 = Color32::from_rgb(0xC0, 0x2A, 0x3C);
    pub const WARN: Color32 = LEMON_DEEP;
    pub const ORANGE: Color32 = Color32::from_rgb(0xF8, 0x9E, 0x4E);

    // Status badge colors — Converting=mint/cyan, Bypass=lemon, Error=red
    pub const STATUS_CONVERTING: Color32 = MINT_DEEP;
    pub const STATUS_BYPASS: Color32 = LEMON_DEEP;
    pub const STATUS_STOPPED: Color32 = TEXT_MUTED;
    pub const STATUS_ERROR: Color32 = ERROR_DEEP;

    // Compatibility aliases (legacy names used in some call sites)
    pub const BG_PANEL_LIGHT: Color32 = CARD_GLASS_LIGHT;
    pub const YELLOW: Color32 = LEMON;
}

// ===========================================================================
// Gradient helpers
// ===========================================================================

/// A two-stop linear gradient as (top_color, bottom_color).
pub struct Gradient(pub Color32, pub Color32);

impl Gradient {
    #[allow(dead_code)]
    pub const PINK_CYAN: Gradient = Gradient(colors::PINK, colors::CYAN);
    #[allow(dead_code)]
    pub const LAVENDER_MINT: Gradient = Gradient(colors::LAVENDER, colors::MINT);
    #[allow(dead_code)]
    pub const CYAN_PINK: Gradient = Gradient(colors::CYAN, colors::PINK);
    #[allow(dead_code)]
    pub const LEMON_CORAL: Gradient = Gradient(colors::LEMON, colors::CORAL);

    /// Vertical gradient fill on a rect with given rounding.
    pub fn fill_vertical(&self, painter: &egui::Painter, rect: egui::Rect, rounding: f32) {
        let steps = 24;
        let step_h = rect.height() / steps as f32;
        for i in 0..steps {
            let t = i as f32 / (steps - 1) as f32;
            let c = lerp_color(self.0, self.1, t);
            let y = rect.min.y + step_h * i as f32;
            let r = egui::Rect::from_min_size(
                egui::pos2(rect.min.x, y),
                egui::vec2(rect.width(), step_h + 1.0),
            );
            painter.rect_filled(r, rounding, c);
        }
    }
}

pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
    )
}

pub fn with_alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_premultiplied(
        (c.r() as u16 * a as u16 / 255) as u8,
        (c.g() as u16 * a as u16 / 255) as u8,
        (c.b() as u16 * a as u16 / 255) as u8,
        a,
    )
}

/// Paint a kawaii future-bass background — a soft vertical gradient from
/// pale pink at the top through lavender to baby blue at the bottom,
/// overlaid with gentle pastel blooms. The vibe is airy and bright, not
/// dark club-night. Cards sit on this like frosted glass on candy-floss.
pub fn paint_background(painter: &egui::Painter, rect: egui::Rect) {
    use colors::*;
    painter.rect_filled(rect, 0.0, BG_DEEP);
}

// ===========================================================================
// Global theme setup
// ===========================================================================

pub fn apply_theme(ctx: &egui::Context) {
    install_cjk_font(ctx);
    let mut style = (*ctx.global_style()).clone();

    // Consistent rhythm — music-tool density with breathing room.
    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style.spacing.button_padding = egui::vec2(18.0, 8.0);
    style.spacing.interact_size = egui::vec2(0.0, BUTTON_HEIGHT);
    style.spacing.text_edit_width = 200.0;

    use colors::*;

    // LIGHT mode — kawaii pastel with strong contrast text
    style.visuals.dark_mode = false;
    style.visuals.panel_fill = BG_DEEP;
    style.visuals.extreme_bg_color = CARD_GLASS_LIGHT;
    style.visuals.faint_bg_color = CARD_GLASS_LIGHT;
    style.visuals.window_fill = CARD_GLASS;

    // Text override — deep plum for strong contrast
    style.visuals.override_text_color = Some(TEXT);

    // Noninteractive: frosted white panels with defined lavender border
    style.visuals.widgets.noninteractive.bg_fill = CARD_GLASS;
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.noninteractive.corner_radius = CARD_ROUNDING.into();
    style.visuals.widgets.noninteractive.expansion = 0.0;

    // Inactive: white card with lavender border (clear default state)
    style.visuals.widgets.inactive.bg_fill = CARD_GLASS;
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.5, BORDER_DEEP);
    style.visuals.widgets.inactive.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.inactive.expansion = 0.0;

    // Hovered: pink wash with deep pink border
    style.visuals.widgets.hovered.bg_fill = with_alpha(PINK, 100);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(2.0, PINK_DEEP);
    style.visuals.widgets.hovered.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.hovered.expansion = 0.0;

    // Active: cyan wash with deep cyan border (cool contrast to pink hover)
    style.visuals.widgets.active.bg_fill = with_alpha(CYAN, 120);
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(2.0, CYAN_DEEP);
    style.visuals.widgets.active.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.active.expansion = 0.0;

    // Open (expanded combos etc.) — mint tint
    style.visuals.widgets.open.bg_fill = with_alpha(MINT, 80);
    style.visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.5, MINT_DEEP);
    style.visuals.widgets.open.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.open.expansion = 0.0;

    // Selection — pink candy with deep edge
    style.visuals.selection.bg_fill = with_alpha(PINK, 180);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, PINK_DEEP);

    // Hyperlinks
    style.visuals.hyperlink_color = PINK_DEEP;

    ctx.set_global_style(style);
}

// ===========================================================================
// Decorations — kawaii future-bass accents (stars, dots, glows)
// ===========================================================================

/// Paint a small 4-point sparkle/star at `pos` with given radius and alpha.
pub fn paint_star(painter: &egui::Painter, pos: egui::Pos2, radius: f32, color: Color32) {
    let r = radius.max(1.0);
    // 4-point star via two crossed diamonds
    let pts = [
        egui::pos2(pos.x, pos.y - r),
        egui::pos2(pos.x + r * 0.28, pos.y - r * 0.28),
        egui::pos2(pos.x + r, pos.y),
        egui::pos2(pos.x + r * 0.28, pos.y + r * 0.28),
        egui::pos2(pos.x, pos.y + r),
        egui::pos2(pos.x - r * 0.28, pos.y + r * 0.28),
        egui::pos2(pos.x - r, pos.y),
        egui::pos2(pos.x - r * 0.28, pos.y - r * 0.28),
    ];
    painter.add(egui::Shape::convex_polygon(
        pts.to_vec(),
        color,
        egui::Stroke::NONE,
    ));
}

/// Paint a dot grid over a rect — Y2K pastel-cyber texture. Very faint.
pub fn paint_dot_grid(painter: &egui::Painter, rect: egui::Rect, step: f32, color: Color32) {
    let mut y = rect.top();
    while y < rect.bottom() {
        let mut x = rect.left();
        while x < rect.right() {
            painter.circle_filled(egui::pos2(x, y), 1.0, color);
            x += step;
        }
        y += step;
    }
}

/// Paint a diagonal light streak across a rect — future-bass light sweep.
pub fn paint_light_streak(painter: &egui::Painter, rect: egui::Rect, color: Color32) {
    let mid_y = rect.center().y;
    let p1 = egui::pos2(rect.left() - 20.0, mid_y - 40.0);
    let p2 = egui::pos2(rect.right() + 20.0, mid_y + 40.0);
    painter.line_segment([p1, p2], egui::Stroke::new(2.0, with_alpha(color, 16)));
    painter.line_segment([p1, p2], egui::Stroke::new(1.0, with_alpha(color, 30)));
}

/// Scatter a few faint stars across a rect for ambient kawaii sparkle.
pub fn paint_star_field(painter: &egui::Painter, rect: egui::Rect, seed: u64) {
    // Deterministic pseudo-random star placement.
    let mut s = seed.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(1);
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    let count = ((rect.width() * rect.height()) / 18000.0) as usize;
    for i in 0..count.min(40) {
        let x = rect.left() + next() * rect.width();
        let y = rect.top() + next() * rect.height();
        let r = 1.5 + next() * 3.0;
        let alpha = (40.0 + next() * 60.0) as u8;
        let c = if i % 3 == 0 {
            with_alpha(colors::PINK_BRIGHT, alpha)
        } else if i % 3 == 1 {
            with_alpha(colors::CYAN, alpha)
        } else {
            with_alpha(colors::LEMON, alpha)
        };
        paint_star(painter, egui::pos2(x, y), r, c);
    }
}

// ===========================================================================
// Typography helpers
// ===========================================================================

/// Screen / page title — large kawaii display.
pub fn page_title(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(26.0)
            .strong()
            .color(colors::TEXT),
    );
}

pub fn heading(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(18.0)
            .strong()
            .color(colors::TEXT),
    );
    ui.add_space(space::TIGHT);
}

pub fn subheading(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(13.0)
            .strong()
            .color(colors::TEXT_DIM),
    );
    ui.add_space(space::SMALL);
}

pub fn numeric(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(13.0)
            .color(colors::TEXT)
            .monospace(),
    );
}

// ===========================================================================
// Form helpers
// ===========================================================================

pub fn form_label(ui: &mut egui::Ui, text: &str) {
    ui.add_sized(
        [LABEL_WIDTH, FIELD_HEIGHT],
        egui::Label::new(egui::RichText::new(text).size(12.0).color(colors::TEXT_DIM)),
    );
}

pub fn path_text_edit(ui: &mut egui::Ui, buf: &mut String, hint: &str) {
    let w = (ui.available_width() * 0.7).max(120.0);
    ui.add_sized(
        [w, FIELD_HEIGHT],
        egui::TextEdit::singleline(buf).hint_text(hint),
    );
}

// ===========================================================================
// Buttons — candy style with subtle gradient + glow
// ===========================================================================

fn candy_button(
    text: &str,
    fill: Color32,
    text_color: Color32,
    min_size: egui::Vec2,
) -> egui::Button<'static> {
    egui::Button::new(
        egui::RichText::new(text)
            .size(13.0)
            .strong()
            .color(text_color),
    )
    .fill(fill)
    .corner_radius(BUTTON_ROUNDING)
    .min_size(min_size)
}

/// Standard pill button — pink candy when active, glass when idle.
pub fn pill_button(ui: &mut egui::Ui, text: &str, active: bool) -> bool {
    let (fill, text_color) = if active {
        (
            Color32::from_rgb(0x1E, 0x1E, 0x2E),
            Color32::from_rgb(0xFF, 0xFF, 0xFF),
        )
    } else {
        (
            with_alpha(Color32::from_rgb(0xFF, 0xFF, 0xFF), 80),
            colors::TEXT,
        )
    };
    ui.add(candy_button(
        text,
        fill,
        text_color,
        egui::vec2(80.0, BUTTON_HEIGHT),
    ))
    .clicked()
}

/// Icon + text button — same candy sizing as pill_button.
pub fn icon_button(
    ui: &mut egui::Ui,
    icon: &egui::TextureHandle,
    text: &str,
    active: bool,
) -> bool {
    let (fill, text_color) = if active {
        (
            Color32::from_rgb(0x1E, 0x1E, 0x2E),
            Color32::from_rgb(0xFF, 0xFF, 0xFF),
        )
    } else {
        (
            with_alpha(Color32::from_rgb(0xFF, 0xFF, 0xFF), 80),
            colors::TEXT,
        )
    };
    let btn = egui::Button::image_and_text(
        egui::Image::from_texture(icon).fit_to_exact_size(egui::vec2(14.0, 14.0)),
        egui::RichText::new(text)
            .size(12.0)
            .strong()
            .color(text_color),
    )
    .fill(fill)
    .corner_radius(BUTTON_ROUNDING)
    .min_size(egui::vec2(70.0, BUTTON_HEIGHT));
    ui.add(btn).clicked()
}

/// Primary CTA — bigger, pink gradient vibe.
pub fn primary_button(ui: &mut egui::Ui, text: &str, enabled: bool) -> bool {
    let btn = candy_button(
        text,
        if enabled {
            Color32::from_rgb(0x1E, 0x1E, 0x2E)
        } else {
            with_alpha(colors::TEXT_MUTED, 80)
        },
        if enabled {
            Color32::from_rgb(0xFF, 0xFF, 0xFF)
        } else {
            colors::TEXT_MUTED
        },
        egui::vec2(160.0, CTA_HEIGHT),
    )
    .corner_radius(CTA_HEIGHT * 0.5);
    ui.add_enabled(enabled, btn).clicked()
}

/// Tab button — pill, selected gets a vivid pink→mint gradient fill with
/// white text (future-bass signature). Unselected is frosted glass.
pub fn tab_button(ui: &mut egui::Ui, text: &str, selected: bool) -> bool {
    use colors::*;
    let btn_size = egui::vec2(86.0, BUTTON_HEIGHT + 2.0);
    let (rect, response) = ui.allocate_exact_size(btn_size, egui::Sense::click());
    let painter = ui.painter();
    let hovered = response.hovered();
    let rounding = BUTTON_HEIGHT * 0.5;

    if selected {
        painter.rect_filled(rect, rounding, Color32::from_rgb(0x1E, 0x1E, 0x2E));
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::proportional(13.0),
            Color32::from_rgb(0xFF, 0xFF, 0xFF),
        );
    } else {
        let fill = if hovered {
            with_alpha(Color32::from_rgb(0xFF, 0xFF, 0xFF), 80)
        } else {
            with_alpha(Color32::from_rgb(0xFF, 0xFF, 0xFF), 40)
        };
        painter.rect_filled(rect, rounding, fill);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::proportional(13.0),
            TEXT,
        );
    }

    response.clicked()
}

// ===========================================================================
// Status — glowing badge instead of bare dot
// ===========================================================================

/// Status badge with glow. Use for LIVE / BYPASS / STOPPED indicators.
pub fn status_dot(ui: &mut egui::Ui, active: bool, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
    let painter = ui.painter();
    let center = rect.center();

    if active {
        // Outer halo
        painter.circle_filled(center, 10.0, with_alpha(color, 40));
        painter.circle_filled(center, 7.0, with_alpha(color, 90));
    }
    painter.circle_filled(center, 4.5, if active { color } else { colors::TEXT_MUTED });
    if active {
        // Inner highlight — glossy candy
        painter.circle_filled(
            egui::pos2(center.x - 1.2, center.y - 1.2),
            1.5,
            Color32::from_rgba_premultiplied(255, 255, 255, 160),
        );
    }
}

/// A pill-shaped status badge with strong glow. Use for CONVERTING / BYPASS /
/// STOPPED / ERROR indicators. The fill is the accent color (not pale), with
/// a white leading dot, bright stroke, and deep-plum text for contrast.
pub fn status_badge(ui: &mut egui::Ui, text: &str, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(130.0, 30.0), egui::Sense::hover());
    let painter = ui.painter();

    painter.rect_filled(rect, 15.0, color);

    let dot_pos = egui::pos2(rect.min.x + 14.0, rect.center().y);
    painter.circle_filled(dot_pos, 3.0, Color32::from_rgb(0xFF, 0xFF, 0xFF));

    painter.text(
        egui::pos2(rect.center().x + 6.0, rect.center().y),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(11.0),
        Color32::from_rgb(0xFF, 0xFF, 0xFF),
    );
}

// ===========================================================================
// VU Meter — DAW-style horizontal level meter
// ===========================================================================

/// Meter channel — Input or Output. Both use the same future-bass gradient
/// (mint → cyan → pink → lemon) but Input leans pink and Output leans cyan.
#[derive(Copy, Clone)]
pub enum MeterKind {
    Input,
    Output,
}

/// Legacy compat — delegates to [`level_meter_kind`] with Input kind.
pub fn level_meter(ui: &mut egui::Ui, level: f32, label: &str) {
    level_meter_kind(ui, level, label, MeterKind::Input);
}

/// A DAW-style VU meter: thick horizontal bar with future-bass gradient
/// (mint → cyan → pink → lemon), tick marks, a peak line that holds, and
/// a right-aligned dB readout. The meter is the hero of the Signal card.
pub fn level_meter_kind(ui: &mut egui::Ui, level: f32, label: &str, kind: MeterKind) {
    use colors::*;

    ui.horizontal(|ui| {
        // Channel label — fixed width, deep color for contrast
        ui.add_space(2.0);
        ui.label(egui::RichText::new(label).size(12.0).strong().color(TEXT));
        ui.add_space(space::SMALL);

        let meter_width = (ui.available_width() - 72.0).max(80.0);
        let h = METER_HEIGHT + 6.0;
        let (rect, _) = ui.allocate_exact_size(egui::vec2(meter_width, h), egui::Sense::hover());
        let painter = ui.painter_at(rect);

        // Track — dark trough for the bar to sit in
        let track_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x, rect.center().y - METER_HEIGHT / 2.0),
            egui::vec2(rect.width(), METER_HEIGHT),
        );
        painter.rect_filled(
            track_rect,
            METER_HEIGHT * 0.5,
            Color32::from_rgb(0x2A, 0x24, 0x38),
        );
        painter.rect_stroke(
            track_rect,
            METER_HEIGHT * 0.5,
            egui::Stroke::new(1.0, BORDER_DEEP),
            egui::StrokeKind::Outside,
        );

        // Tick marks — every 5%, taller at 25/50/75/100%
        for i in 1..20 {
            let x = track_rect.min.x + track_rect.width() * (i as f32 / 20.0);
            let tick_h = if i % 5 == 0 {
                METER_HEIGHT - 3.0
            } else {
                METER_HEIGHT - 10.0
            };
            painter.line_segment(
                [
                    egui::pos2(x, track_rect.center().y - tick_h / 2.0),
                    egui::pos2(x, track_rect.center().y + tick_h / 2.0),
                ],
                egui::Stroke::new(
                    if i % 5 == 0 { 1.2 } else { 0.8 },
                    Color32::from_rgba_premultiplied(
                        255,
                        255,
                        255,
                        if i % 5 == 0 { 120 } else { 60 },
                    ),
                ),
            );
        }

        // Bar — future-bass gradient: mint → cyan → pink → lemon (horizontal).
        // Use ×5 so RMS 0.05-0.3 maps to 0.25-1.5.
        let bar_level = (level * 5.0).clamp(0.0, 1.0);
        let bar_width = track_rect.width() * bar_level;
        if bar_width > 1.0 {
            let bar_rect = egui::Rect::from_min_size(
                track_rect.min,
                egui::vec2(bar_width, track_rect.height()),
            );
            // Multi-stop horizontal gradient (mint@0 → cyan@0.4 → pink@0.7 → lemon@1.0)
            let stops = [
                (0.0, MINT),
                (0.35, CYAN),
                (0.65, PINK),
                (0.85, LEMON),
                (1.0, CORAL),
            ];
            paint_horizontal_gradient(&painter, bar_rect, METER_HEIGHT * 0.5, &stops);

            // Glossy top highlight — candy sheen
            let gloss_rect = egui::Rect::from_min_size(
                bar_rect.min,
                egui::vec2(bar_rect.width(), bar_rect.height() * 0.4),
            );
            painter.rect_filled(
                gloss_rect,
                METER_HEIGHT * 0.5,
                Color32::from_rgba_premultiplied(255, 255, 255, 50),
            );

            // Peak line — bright vertical edge at the bar tip
            let peak_x = bar_rect.max.x;
            painter.line_segment(
                [
                    egui::pos2(peak_x, track_rect.min.y + 1.0),
                    egui::pos2(peak_x, track_rect.max.y - 1.0),
                ],
                egui::Stroke::new(2.0, Color32::from_rgb(0xFF, 0xFF, 0xFF)),
            );
            painter.line_segment(
                [
                    egui::pos2(peak_x, track_rect.min.y + 1.0),
                    egui::pos2(peak_x, track_rect.max.y - 1.0),
                ],
                egui::Stroke::new(1.0, TEXT),
            );
        }

        // dB readout — right-aligned, large mono number
        let db = if level > 0.0 {
            20.0 * level.log10()
        } else {
            -99.0
        };
        let db_color = if db > -6.0 {
            CORAL_DEEP
        } else if db > -18.0 {
            LEMON_DEEP
        } else {
            TEXT
        };
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("{db:+.0}"))
                    .size(14.0)
                    .strong()
                    .color(db_color)
                    .monospace(),
            );
            ui.label(egui::RichText::new("dB").size(10.0).color(TEXT_MUTED));
        });
    });
}

pub fn level_meter_kind_compact(ui: &mut egui::Ui, level: f32, kind: MeterKind) {
    use colors::*;

    let meter_width = (ui.available_width() - 60.0).max(60.0);
    let h = 12.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(meter_width, h), egui::Sense::hover());
    let painter = ui.painter();

    let track_rect = egui::Rect::from_min_size(
        egui::pos2(rect.min.x, rect.center().y - h / 2.0),
        egui::vec2(rect.width(), h),
    );
    painter.rect_filled(track_rect, h * 0.5, Color32::from_rgb(0x2A, 0x24, 0x38));

    let bar_level = (level * 5.0).clamp(0.0, 1.0);
    let bar_width = track_rect.width() * bar_level;
    if bar_width > 1.0 {
        let bar_rect =
            egui::Rect::from_min_size(track_rect.min, egui::vec2(bar_width, track_rect.height()));
        let color = match kind {
            MeterKind::Input => PINK,
            MeterKind::Output => CYAN,
        };
        painter.rect_filled(bar_rect, h * 0.5, color);
    }

    let db = if level > 0.0 {
        20.0 * level.log10()
    } else {
        -99.0
    };
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.label(
            egui::RichText::new(format!("{db:+.0}"))
                .size(12.0)
                .strong()
                .color(TEXT)
                .monospace(),
        );
    });
}

/// Paint a multi-stop horizontal gradient into a rounded rect.
fn paint_horizontal_gradient(
    painter: &egui::Painter,
    rect: egui::Rect,
    rounding: f32,
    stops: &[(f32, Color32)],
) {
    let n = stops.len();
    if n < 2 {
        return;
    }
    // Solid rounded-rect base so corners stay clean (strips are flat).
    painter.rect_filled(rect, rounding, stops[0].1);

    // Flat gradient strips, clipped to an inset so square strip corners
    // never poke past the rounded base. Inset = ~40% of rounding horizontally.
    let inset_x = rounding * 0.4;
    let inner = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + inset_x, rect.min.y),
        egui::pos2(rect.max.x - inset_x, rect.max.y),
    );
    if inner.width() <= 0.0 || inner.height() <= 0.0 {
        return;
    }
    let clipped = painter.with_clip_rect(inner);
    let strips = (n - 1) * 16;
    let strip_w = inner.width() / strips as f32;
    for i in 0..strips {
        let t = i as f32 / (strips - 1) as f32;
        let mut c = stops[0].1;
        for seg in 0..(n - 1) {
            let (t0, c0) = stops[seg];
            let (t1, c1) = stops[seg + 1];
            if t >= t0 && t <= t1 {
                let local_t = (t - t0) / (t1 - t0).max(0.0001);
                c = lerp_color(c0, c1, local_t);
                break;
            }
        }
        let x = inner.min.x + strip_w * i as f32;
        clipped.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(x, inner.min.y),
                egui::vec2(strip_w + 1.0, inner.height()),
            ),
            0.0,
            c,
        );
    }
}

// ===========================================================================
// Info card — glassmorphism with translucent fill + soft border + glow
// ===========================================================================

/// A glass card: WHITE frosted panel with soft lavender border and a
/// pink-tinted drop shadow so it floats on the pastel background.
pub fn info_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    info_card_frame().show(ui, add_contents);
}

/// The frame used by [`info_card`]. Exposed so call sites can choose the
/// frame conditionally (e.g. switch to glow_card when live) without
/// duplicating the content closure.
pub fn info_card_frame() -> egui::Frame {
    use colors::*;
    egui::Frame::NONE
        .fill(CARD_GLASS)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(CARD_ROUNDING)
        .inner_margin(16)
        .shadow(egui::epaint::Shadow {
            offset: [0, 1],
            blur: 4,
            spread: 0,
            color: with_alpha(TEXT, 8),
        })
}

/// A highlighted card variant — used for the active/primary card.
/// Stronger pink glow + brighter stroke to draw focus.
pub fn glow_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    glow_card_frame().show(ui, add_contents);
}

/// The frame used by [`glow_card`]. See [`info_card_frame`] for rationale.
pub fn glow_card_frame() -> egui::Frame {
    use colors::*;
    egui::Frame::NONE
        .fill(CARD_GLASS_LIGHT)
        .stroke(egui::Stroke::new(1.0, BORDER_DEEP))
        .corner_radius(CARD_ROUNDING)
        .inner_margin(16)
        .shadow(egui::epaint::Shadow {
            offset: [0, 1],
            blur: 4,
            spread: 0,
            color: with_alpha(TEXT, 8),
        })
}

/// A cyan-accented card — for output/signal areas.
pub fn cyan_card_frame() -> egui::Frame {
    use colors::*;
    egui::Frame::NONE
        .fill(CARD_CYAN_TINT)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(CARD_ROUNDING)
        .inner_margin(16)
        .shadow(egui::epaint::Shadow {
            offset: [0, 1],
            blur: 4,
            spread: 0,
            color: with_alpha(TEXT, 8),
        })
}

// ===========================================================================
// Drop zone — large drag & drop file loader (music-app hero)
// ===========================================================================

/// A large, centered drag-and-drop file loader. Shows a big icon (upload
/// arrow when empty, checkmark when loaded), hint text centered, and a
/// Browse button below. This is the hero of the Model card — it should
/// feel like a real drop target, not a form field.
pub fn drop_zone(
    ui: &mut egui::Ui,
    hint: &str,
    file_label: Option<&str>,
    browse_text: &str,
) -> DropZoneOutput {
    use colors::*;
    let h = 110.0;
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    let has_file = file_label.is_some();
    let accent = if has_file { MINT_DEEP } else { LAVENDER_DEEP };

    // Fill — very light accent tint
    let fill = if has_file {
        with_alpha(MINT, 30)
    } else {
        with_alpha(LAVENDER, 30)
    };
    painter.rect_filled(rect, CARD_ROUNDING, fill);

    // Dashed border — draw as short segments to simulate a true dashed line
    paint_dashed_rect(&painter, rect, CARD_ROUNDING, accent, 6.0, 4.0);

    // Icon — centered, large. Upload arrow (empty) or checkmark (loaded).
    let icon_center = egui::pos2(rect.center().x, rect.min.y + 32.0);
    if has_file {
        // Checkmark in a mint circle
        painter.circle_filled(icon_center, 16.0, MINT_DEEP);
        painter.circle_filled(icon_center, 13.0, Color32::from_rgb(0xFF, 0xFF, 0xFF));
        // Check stroke
        let p1 = egui::pos2(icon_center.x - 6.0, icon_center.y);
        let p2 = egui::pos2(icon_center.x - 2.0, icon_center.y + 4.0);
        let p3 = egui::pos2(icon_center.x + 6.0, icon_center.y - 5.0);
        painter.line_segment([p1, p2], egui::Stroke::new(3.0, MINT_DEEP));
        painter.line_segment([p2, p3], egui::Stroke::new(3.0, MINT_DEEP));
    } else {
        // Upload arrow — arrow up + tray
        let arrow_top = egui::pos2(icon_center.x, icon_center.y - 10.0);
        let arrow_bot = egui::pos2(icon_center.x, icon_center.y + 4.0);
        painter.line_segment([arrow_bot, arrow_top], egui::Stroke::new(3.0, accent));
        // arrow head
        painter.line_segment(
            [
                egui::pos2(icon_center.x - 7.0, icon_center.y - 4.0),
                arrow_top,
            ],
            egui::Stroke::new(3.0, accent),
        );
        painter.line_segment(
            [
                egui::pos2(icon_center.x + 7.0, icon_center.y - 4.0),
                arrow_top,
            ],
            egui::Stroke::new(3.0, accent),
        );
        // tray (underline)
        let tray_y = icon_center.y + 12.0;
        painter.line_segment(
            [
                egui::pos2(icon_center.x - 12.0, tray_y),
                egui::pos2(icon_center.x + 12.0, tray_y),
            ],
            egui::Stroke::new(3.0, accent),
        );
    }

    // Hint text — centered, prominent
    let hint_text = file_label.unwrap_or(hint);
    let hint_color = if has_file { TEXT } else { TEXT_DIM };
    painter.text(
        egui::pos2(rect.center().x, rect.min.y + 62.0),
        egui::Align2::CENTER_CENTER,
        hint_text,
        egui::FontId::proportional(13.0),
        hint_color,
    );

    // "or Browse" subtext
    if !has_file {
        painter.text(
            egui::pos2(rect.center().x, rect.min.y + 80.0),
            egui::Align2::CENTER_CENTER,
            "or click Browse below",
            egui::FontId::proportional(10.0),
            TEXT_MUTED,
        );
    }

    // Browse button — centered at the bottom of the zone
    let btn_w = 100.0;
    let btn_rect = egui::Rect::from_min_size(
        egui::pos2(
            rect.right() - btn_w - 12.0,
            rect.center().y - BUTTON_HEIGHT / 2.0,
        ),
        egui::vec2(btn_w, BUTTON_HEIGHT),
    );
    let browse_response = ui.interact(
        btn_rect,
        ui.id().with("drop_zone_browse"),
        egui::Sense::click(),
    );
    let btn_fill = if browse_response.hovered() {
        PINK_DEEP
    } else {
        PINK
    };
    painter.rect_filled(btn_rect, BUTTON_ROUNDING, btn_fill);
    painter.rect_stroke(
        btn_rect,
        BUTTON_ROUNDING,
        egui::Stroke::new(1.0, PINK_DEEP),
        egui::StrokeKind::Outside,
    );
    painter.text(
        btn_rect.center(),
        egui::Align2::CENTER_CENTER,
        browse_text,
        egui::FontId::proportional(12.0),
        Color32::from_rgb(0xFF, 0xFF, 0xFF),
    );

    DropZoneOutput {
        browse_clicked: browse_response.clicked(),
    }
}

/// Paint a dashed rectangle border (simulates a drag-drop target outline).
fn paint_dashed_rect(
    painter: &egui::Painter,
    rect: egui::Rect,
    rounding: f32,
    color: Color32,
    dash_len: f32,
    gap_len: f32,
) {
    let step = dash_len + gap_len;
    // Top edge
    let mut x = rect.min.x + rounding;
    while x < rect.max.x - rounding {
        let x2 = (x + dash_len).min(rect.max.x - rounding);
        painter.line_segment(
            [egui::pos2(x, rect.min.y), egui::pos2(x2, rect.min.y)],
            egui::Stroke::new(2.0, color),
        );
        x += step;
    }
    // Bottom edge
    x = rect.min.x + rounding;
    while x < rect.max.x - rounding {
        let x2 = (x + dash_len).min(rect.max.x - rounding);
        painter.line_segment(
            [egui::pos2(x, rect.max.y), egui::pos2(x2, rect.max.y)],
            egui::Stroke::new(2.0, color),
        );
        x += step;
    }
    // Left edge
    let mut y = rect.min.y + rounding;
    while y < rect.max.y - rounding {
        let y2 = (y + dash_len).min(rect.max.y - rounding);
        painter.line_segment(
            [egui::pos2(rect.min.x, y), egui::pos2(rect.min.x, y2)],
            egui::Stroke::new(2.0, color),
        );
        y += step;
    }
    // Right edge
    y = rect.min.y + rounding;
    while y < rect.max.y - rounding {
        let y2 = (y + dash_len).min(rect.max.y - rounding);
        painter.line_segment(
            [egui::pos2(rect.max.x, y), egui::pos2(rect.max.x, y2)],
            egui::Stroke::new(2.0, color),
        );
        y += step;
    }
}

pub struct DropZoneOutput {
    pub browse_clicked: bool,
}

// ===========================================================================
// Numeric stat card — large value + small unit (latency / RTF / backend)
// ===========================================================================

/// A compact stat card: a big colored number with a small unit label below.
/// Used for latency, RTF, and backend info in the Status card. The number
/// is the hero; the unit is the caption.
pub fn stat_card(ui: &mut egui::Ui, value: &str, unit: &str, accent: Color32) {
    use colors::*;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(90.0, 54.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // Frosted card with accent-tinted border
    painter.rect_filled(rect, CARD_ROUNDING - 2.0, CARD_GLASS);
    painter.rect_stroke(
        rect,
        CARD_ROUNDING - 2.0,
        egui::Stroke::new(1.0, with_alpha(accent, 140)),
        egui::StrokeKind::Outside,
    );

    // Value — big, bold, accent color
    painter.text(
        egui::pos2(rect.center().x, rect.min.y + 18.0),
        egui::Align2::CENTER_CENTER,
        value,
        egui::FontId::proportional(20.0),
        TEXT,
    );
    // Unit — small, muted
    painter.text(
        egui::pos2(rect.center().x, rect.max.y - 12.0),
        egui::Align2::CENTER_CENTER,
        unit,
        egui::FontId::proportional(10.0),
        TEXT_MUTED,
    );
}

// ===========================================================================
// Operation button — large state-colored CTA (Start/Stop)
// ===========================================================================

/// Operation button state theme.
#[derive(Copy, Clone)]
pub enum OpKind {
    /// Start — pink/candy when stopped.
    Start,
    /// Stop — cyan/mint when running.
    Stop,
    /// Bypass toggle.
    Bypass,
}

/// A large rounded operation button with gradient fill and glow. The color
/// reflects the current operation state (Start=pink, Stop=cyan, Bypass=lemon).
/// Text is dark (light-mode kawaii palette).
pub fn operation_button(ui: &mut egui::Ui, text: &str, kind: OpKind, _active: bool) -> bool {
    let (base, hot) = match kind {
        OpKind::Start => (
            Color32::from_rgb(0x1E, 0x1E, 0x2E),
            Color32::from_rgb(0x2E, 0x2E, 0x3E),
        ),
        OpKind::Stop => (
            Color32::from_rgb(0xC0, 0x3A, 0x3A),
            Color32::from_rgb(0xD0, 0x4A, 0x4A),
        ),
        OpKind::Bypass => (
            Color32::from_rgb(0x88, 0x78, 0x10),
            Color32::from_rgb(0x98, 0x88, 0x20),
        ),
    };

    let size = egui::vec2(150.0, CTA_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter();
    let hovered = response.hovered();
    let rounding = CTA_HEIGHT * 0.5;

    let fill = if hovered { hot } else { base };
    painter.rect_filled(rect, rounding, fill);

    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(15.0),
        Color32::from_rgb(0xFF, 0xFF, 0xFF),
    );

    response.clicked()
}

// ===========================================================================
// Knob widget (unchanged — sprite-sheet based)
// ===========================================================================

const KNOB_FRAMES: usize = 12;
const KNOB_SIZE: f32 = 64.0;

pub fn knob(
    ui: &mut egui::Ui,
    knob_tex: &egui::TextureHandle,
    _id: egui::Id,
    value: f32,
    label: &str,
) -> Option<f32> {
    let size = egui::vec2(KNOB_SIZE, KNOB_SIZE);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(KNOB_SIZE, KNOB_SIZE + 20.0), egui::Sense::drag());
    let knob_rect = egui::Rect::from_min_size(rect.min, size);

    let mut new_value = None;
    if response.dragged() {
        let drag_delta = response.drag_delta().y;
        let delta = -drag_delta * 0.005;
        new_value = Some((value + delta).clamp(0.0, 1.0));
    }
    if response.double_clicked() {
        new_value = Some(0.5);
    }

    let display_value = new_value.unwrap_or(value);
    let frame_idx =
        ((display_value * (KNOB_FRAMES - 1) as f32).round() as usize).min(KNOB_FRAMES - 1);

    let frame_h = 1.0 / KNOB_FRAMES as f32;
    let frame_top = frame_idx as f32 / KNOB_FRAMES as f32;
    let uv = egui::Rect::from_min_max(
        egui::pos2(0.0, frame_top),
        egui::pos2(1.0, frame_top + frame_h),
    );

    let painter = ui.painter_at(knob_rect);
    painter.image(
        knob_tex.id(),
        knob_rect,
        uv,
        if response.dragged() {
            egui::Color32::from_rgba_premultiplied(255, 200, 240, 255)
        } else if response.hovered() {
            egui::Color32::from_rgba_premultiplied(255, 180, 220, 255)
        } else {
            egui::Color32::WHITE
        },
    );

    if response.dragged() {
        painter.circle_stroke(
            knob_rect.center(),
            KNOB_SIZE * 0.48,
            egui::Stroke::new(2.0, colors::PINK),
        );
    }

    let label_pos = egui::pos2(knob_rect.center().x, knob_rect.max.y + 6.0);
    painter.text(
        label_pos,
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(11.0),
        colors::TEXT_DIM,
    );

    new_value
}

#[allow(dead_code)]
pub fn knob_labeled(
    ui: &mut egui::Ui,
    knob_tex: &egui::TextureHandle,
    id_str: &str,
    value: f32,
    label: &str,
    value_text: &str,
) -> Option<f32> {
    ui.horizontal(|ui| {
        let id = ui.make_persistent_id(id_str);
        let result = knob(ui, knob_tex, id, value, label);
        ui.vertical(|ui| {
            ui.label(
                egui::RichText::new(label)
                    .size(12.0)
                    .color(colors::TEXT_DIM),
            );
            ui.label(
                egui::RichText::new(value_text)
                    .size(14.0)
                    .strong()
                    .color(colors::TEXT),
            );
        });
        result
    })
    .inner
}
