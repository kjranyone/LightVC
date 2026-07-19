//! LightVC desktop application.
//!
//! Subcommands:
//!   - `roundtrip`: Validate DAC encode/decode on a WAV file
//!   - `convert`:   Apply converter to a WAV file (offline)
//!   - `resynth`:   FreeVocoder resynthesis (WAV → mel → freeC vocoder → WAV)
//!   - `gui`:       Launch the desktop GUI (3 tabs: offline/realtime/catalog)

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
    let args = cli::Cli::parse();
    let result = match args.command {
        cli::Command::Roundtrip(cmd) => cli::run_roundtrip(cmd),
        cli::Command::Convert(cmd) => cli::run_convert(cmd),
        cli::Command::ConvertB1(cmd) => cli::run_convert_b1(cmd),
        cli::Command::Resynth(cmd) => cli::run_resynth(cmd),
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
