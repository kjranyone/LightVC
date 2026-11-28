//! LightVC desktop application.
//!
//! Subcommands:
//!   - `snv`:          paper_snv の製品経路のオフライン変換(WAV 48 kHz → WAV)
//!   - `realtime-snv`: 同じ経路のリアルタイム変換(マイク → スピーカー)
//!   - `devices`:      使えるホストと入出力装置の一覧
//!   - `gui`:          Launch the desktop GUI (3 tabs: offline/realtime/voices)

mod audio_playback;
mod cli;
mod denoise;
mod enroll;
mod file_pick;
mod m3;
mod offline_tab;
mod realtime_tab;
mod single_instance;
mod state;
mod ui;
mod voices;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    let result = match args.command {
        // GUI の描画の糸は E コアへ(P コアは音声の糸に空ける)。CLI の処理の糸は音声の糸なので P コア。
        cli::Command::Gui(cmd) => {
            lightvc_core::affinity::pin_current_to_efficiency();
            cli::run_gui(cmd)
        }
        cli::Command::Snv(cmd) => {
            lightvc_core::affinity::pin_audio_driver();
            cli::run_snv(cmd)
        }
        cli::Command::RealtimeSnv(cmd) => {
            lightvc_core::affinity::pin_audio_driver();
            cli::run_realtime_snv(cmd)
        }
        cli::Command::Devices => cli::run_devices(),
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
