//! LightVC desktop application.
//!
//! Subcommands:
//!   - `snv`:          paper_snv の製品経路のオフライン変換(WAV 48 kHz → WAV)
//!   - `realtime-snv`: 同じ経路のリアルタイム変換(マイク → スピーカー)
//!   - `gui`:          Launch the desktop GUI (3 tabs: offline/realtime/voices)

mod app;
mod assets;
mod audio_playback;
mod cli;
mod file_pick;
mod offline_tab;
mod realtime_tab;
mod theme;
mod voice_catalog;
mod widgets;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    lightvc_core::affinity::pin_current_to_perf();
    let args = cli::Cli::parse();
    let result = match args.command {
        cli::Command::Snv(cmd) => cli::run_snv(cmd),
        cli::Command::RealtimeSnv(cmd) => cli::run_realtime_snv(cmd),
        cli::Command::Gui(cmd) => cli::run_gui(cmd),
    };
    // Explicit exit to avoid hang on mmap/safetensors drop on Windows
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
