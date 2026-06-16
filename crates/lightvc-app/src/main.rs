//! LightVC-X desktop application.
//!
//! CLI subcommands:
//!   - `roundtrip`: Validate DAC encode/decode on a WAV file
//!   - `convert`: Apply converter to a WAV file (offline)
//!   - `live`: Real-time streaming VC with egui UI

mod app;
mod cli;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    let result = match args.command {
        cli::Command::Roundtrip(cmd) => cli::run_roundtrip(cmd),
        cli::Command::Convert(cmd) => cli::run_convert(cmd),
        cli::Command::Live(cmd) => cli::run_live(cmd),
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
