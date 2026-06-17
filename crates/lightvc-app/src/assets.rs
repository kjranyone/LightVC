//! Asset loading — embeds all images at compile time via include_bytes!.

use eframe::egui;

pub const ICON_256_PNG: &[u8] = include_bytes!("../assets/icons/icon_256.png");
pub const LOGO_PNG: &[u8] = include_bytes!("../assets/logo/logo_header.png");
#[allow(dead_code)] // Retina/HiDPI support not yet implemented (ASSETS_SPEC_V2 §Implementation Status)
pub const LOGO_2X_PNG: &[u8] = include_bytes!("../assets/logo/logo_header@2x.png");
pub const BG_TEXTURE_PNG: &[u8] = include_bytes!("../assets/textures/bg_texture.png");
pub const KNOB_FRAMES_PNG: &[u8] = include_bytes!("../assets/knobs/knob_64_frames.png");
pub const SPLASH_PNG: &[u8] = include_bytes!("../assets/splash/splash.png");

// UI icons (24×24 RGBA)
pub const ICON_FOLDER_PNG: &[u8] = include_bytes!("../assets/ui_icons/icon_folder.png");
pub const ICON_PLAY_PNG: &[u8] = include_bytes!("../assets/ui_icons/icon_play.png");
pub const ICON_STOP_PNG: &[u8] = include_bytes!("../assets/ui_icons/icon_stop.png");
pub const ICON_CONVERT_PNG: &[u8] = include_bytes!("../assets/ui_icons/icon_convert.png");
pub const ICON_TRASH_PNG: &[u8] = include_bytes!("../assets/ui_icons/icon_trash.png");
pub const ICON_MIC_PNG: &[u8] = include_bytes!("../assets/ui_icons/icon_mic.png");
pub const ICON_SPEAKER_PNG: &[u8] = include_bytes!("../assets/ui_icons/icon_speaker.png");

// Illustrations
pub const EMPTY_STARS_PNG: &[u8] = include_bytes!("../assets/illustrations/empty_stars.png");

/// Load RGBA image data from PNG bytes.
pub fn load_image_from_png(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    let img = image::load_from_memory(data).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some((w as usize, h as usize, rgba.into_raw()))
}

/// Load the app icon as egui::IconData.
pub fn load_icon() -> Option<egui::IconData> {
    let (w, h, rgba) = load_image_from_png(ICON_256_PNG)?;
    Some(egui::IconData {
        rgba,
        width: w as u32,
        height: h as u32,
    })
}

/// Cache for all texture handles (created once per egui context).
pub struct AssetCache {
    pub bg_texture: Option<egui::TextureHandle>,
    pub logo_texture: Option<egui::TextureHandle>,
    pub knob_texture: Option<egui::TextureHandle>,
    pub splash_texture: Option<egui::TextureHandle>,
    // UI icons
    pub icons: IconCache,
    // Empty state illustration
    pub empty_stars: Option<egui::TextureHandle>,
}

pub struct IconCache {
    pub folder: Option<egui::TextureHandle>,
    pub play: Option<egui::TextureHandle>,
    pub stop: Option<egui::TextureHandle>,
    pub convert: Option<egui::TextureHandle>,
    pub trash: Option<egui::TextureHandle>,
    pub mic: Option<egui::TextureHandle>,
    pub speaker: Option<egui::TextureHandle>,
}

impl Default for AssetCache {
    fn default() -> Self {
        Self {
            bg_texture: None,
            logo_texture: None,
            knob_texture: None,
            splash_texture: None,
            icons: IconCache {
                folder: None,
                play: None,
                stop: None,
                convert: None,
                trash: None,
                mic: None,
                speaker: None,
            },
            empty_stars: None,
        }
    }
}

impl AssetCache {
    fn make_texture(ctx: &egui::Context, id: &str, data: &[u8]) -> egui::TextureHandle {
        let (w, h, rgba) = load_image_from_png(data).unwrap_or_else(|| panic!("{id} is valid"));
        let pixels: Vec<egui::Color32> = rgba
            .chunks_exact(4)
            .map(|c| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
            .collect();
        ctx.load_texture(
            id,
            egui::ColorImage {
                size: [w, h],
                pixels,
                source_size: egui::Vec2::new(w as f32, h as f32),
            },
            egui::TextureOptions::LINEAR,
        )
    }

    pub fn bg(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.bg_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "bg_texture", BG_TEXTURE_PNG))
    }

    pub fn logo(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.logo_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "logo", LOGO_PNG))
    }

    pub fn knob(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.knob_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "knob_frames", KNOB_FRAMES_PNG))
    }

    pub fn splash(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.splash_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "splash", SPLASH_PNG))
    }

    pub fn empty_stars(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.empty_stars
            .get_or_insert_with(|| Self::make_texture(ctx, "empty_stars", EMPTY_STARS_PNG))
    }

    // Icon accessors
    pub fn icon_folder(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.icons
            .folder
            .get_or_insert_with(|| Self::make_texture(ctx, "icon_folder", ICON_FOLDER_PNG))
    }
    pub fn icon_play(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.icons
            .play
            .get_or_insert_with(|| Self::make_texture(ctx, "icon_play", ICON_PLAY_PNG))
    }
    pub fn icon_stop(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.icons
            .stop
            .get_or_insert_with(|| Self::make_texture(ctx, "icon_stop", ICON_STOP_PNG))
    }
    pub fn icon_convert(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.icons
            .convert
            .get_or_insert_with(|| Self::make_texture(ctx, "icon_convert", ICON_CONVERT_PNG))
    }
    pub fn icon_trash(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.icons
            .trash
            .get_or_insert_with(|| Self::make_texture(ctx, "icon_trash", ICON_TRASH_PNG))
    }
    pub fn icon_mic(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.icons
            .mic
            .get_or_insert_with(|| Self::make_texture(ctx, "icon_mic", ICON_MIC_PNG))
    }
    pub fn icon_speaker(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.icons
            .speaker
            .get_or_insert_with(|| Self::make_texture(ctx, "icon_speaker", ICON_SPEAKER_PNG))
    }
}
