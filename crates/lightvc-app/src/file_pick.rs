//! Cross-platform file picker built on rfd (native dialogs, off the UI thread).
//! UI は每フレーム(またはタイマー刻みで)[`take`](Self::take) を呼んで結果を受け取る。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A handle to an in-flight or completed file selection.
/// Clone-cheap (Arc inside). `pending()` tells the UI tick to keep repainting.
#[derive(Clone, Default)]
pub struct FilePick {
    result: Arc<Mutex<Option<PathBuf>>>,
    pending: Arc<AtomicBool>,
}

impl FilePick {
    /// Start a file-open dialog in a background thread.
    pub fn open(&self) {
        self.spawn(false);
    }

    /// 音声ファイル(WAV / FLAC / MP3)に絞った選択。
    pub fn open_audio(&self) {
        self.spawn(true);
    }

    fn spawn(&self, audio: bool) {
        if self.pending.load(Ordering::Relaxed) {
            return;
        }
        self.pending.store(true, Ordering::Relaxed);
        let r = self.result.clone();
        let pending = self.pending.clone();
        std::thread::spawn(move || {
            let mut d = rfd::FileDialog::new();
            if audio {
                d = d.add_filter("音声 (WAV / FLAC / MP3)", &crate::audio_playback::AUDIO_EXTENSIONS).add_filter("すべてのファイル", &["*"]);
            }
            let p = d.pick_file();
            *r.lock().unwrap() = p;
            pending.store(false, Ordering::Relaxed);
        });
    }

    /// True while the dialog is open (UI keeps ticking so the result lands promptly).
    pub fn pending(&self) -> bool {
        self.pending.load(Ordering::Relaxed)
    }

    /// 選んだことにする(画面写真・動作確認用の LVC_SOURCE / LVC_TARGET)。
    pub fn set(&self, p: PathBuf) {
        *self.result.lock().unwrap() = Some(p);
    }

    /// Returns the chosen path if ready, clearing the slot.
    pub fn take(&self) -> Option<PathBuf> {
        self.result.lock().unwrap().take()
    }
}
