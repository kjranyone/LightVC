//! Cross-platform file picker built on rfd (native dialogs).
//!
//! Replaces egui-file-dialog, which pulled in egui 0.31 alongside the app's
//! egui 0.34, causing a version split and broken dialogs (update(ctx) was
//! never called). rfd uses the OS native picker and runs off the UI thread,
//! so the egui loop is never blocked.
//!
//! Completion repaint: the background thread calls `ctx.request_repaint()`
//! when it stores the result, so non-Realtime tabs (which are event-driven
//! and would otherwise not poll `take()` until the next incidental repaint)
//! pick up the chosen path promptly.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use eframe::egui;

/// A handle to an in-flight or completed file selection.
///
/// Clone-cheap (Arc inside). Poll [`take`](Self::take) each frame; it returns
/// `Some(path)` once the user has chosen and the background thread finishes.
#[derive(Clone, Default)]
pub struct FilePick {
    result: Arc<Mutex<Option<PathBuf>>>,
}

impl FilePick {
    /// Start a file-open dialog in a background thread. When the user
    /// confirms a choice, `ctx.request_repaint()` is called so that any tab
    /// (not just the continuously-redrawing Realtime tab) wakes up to poll
    /// [`take`](Self::take).
    pub fn open(&self, ctx: &egui::Context) {
        let r = self.result.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let p = rfd::FileDialog::new().pick_file();
            *r.lock().unwrap() = p;
            // Wake the UI thread regardless of which tab is active.
            ctx.request_repaint();
        });
    }

    /// Returns the chosen path if ready, clearing the slot.
    pub fn take(&self) -> Option<PathBuf> {
        self.result.lock().unwrap().take()
    }
}
