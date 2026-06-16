//! LightVC build tasks.
//!
//! Usage:
//!   cargo xtask bundle         — Build release + create .clap and .vst3 bundles
//!   cargo xtask install        — Bundle + copy to system plugin directories
//!   cargo xtask clean          — Remove target/bundled

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PLUGIN_NAME: &str = "LightVC";
const DLL_NAME: &str = "lightvc_clap";

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "bundle" => bundle()?,
        "install" => install()?,
        "clean" => clean()?,
        "help" | _ => print_help(),
    }
    Ok(())
}

fn workspace_root() -> PathBuf {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let manifest = PathBuf::from(manifest);
    // manifest = crates/lightvc-xtask, go up 2 levels to workspace root
    manifest
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

fn target_dir() -> PathBuf {
    workspace_root().join("target").join("release")
}

fn bundled_dir() -> PathBuf {
    workspace_root().join("target").join("bundled")
}

fn run(cmd: &str, args: &[&str]) -> anyhow::Result<()> {
    let status = std::process::Command::new(cmd).args(args).status()?;
    if !status.success() {
        anyhow::bail!("`{cmd} {}` failed", args.join(" "));
    }
    Ok(())
}

fn build_release() -> anyhow::Result<()> {
    eprintln!("Building release...");
    run("cargo", &["build", "--release", "-p", "lightvc-clap"])?;
    Ok(())
}

fn bundle() -> anyhow::Result<()> {
    build_release()?;

    let dll_src = target_dir().join(format!("{DLL_NAME}.dll"));
    if !dll_src.exists() {
        anyhow::bail!("DLL not found: {}", dll_src.display());
    }

    let out = bundled_dir();
    if out.exists() {
        fs::remove_dir_all(&out)?;
    }

    if cfg!(target_os = "windows") {
        bundle_windows(&dll_src, &out)?;
    } else if cfg!(target_os = "macos") {
        bundle_macos(&dll_src, &out)?;
    } else {
        bundle_linux(&dll_src, &out)?;
    }

    eprintln!("\nBundles created in: {}", out.display());
    Ok(())
}

fn bundle_windows(dll: &Path, out: &Path) -> anyhow::Result<()> {
    // VST3 bundle: Plugin.vst3/Contents/x86_64-win/Plugin.vst3
    let vst3_dir = out
        .join(format!("{PLUGIN_NAME}.vst3"))
        .join("Contents")
        .join("x86_64-win");
    fs::create_dir_all(&vst3_dir)?;
    fs::copy(dll, vst3_dir.join(format!("{PLUGIN_NAME}.vst3")))?;
    eprintln!("  VST3: {}", vst3_dir.display());

    // CLAP bundle: Plugin.clap/contents/x86_64-win/Plugin.clap
    let clap_dir = out
        .join(format!("{PLUGIN_NAME}.clap"))
        .join("contents")
        .join("x86_64-win");
    fs::create_dir_all(&clap_dir)?;
    fs::copy(dll, clap_dir.join(format!("{PLUGIN_NAME}.clap")))?;
    eprintln!("  CLAP: {}", clap_dir.display());

    Ok(())
}

fn bundle_macos(dll: &Path, out: &Path) -> anyhow::Result<()> {
    // VST3 bundle: Plugin.vst3/Contents/MacOS/Plugin
    let vst3 = out.join(format!("{PLUGIN_NAME}.vst3"));
    fs::create_dir_all(vst3.join("Contents").join("MacOS"))?;
    fs::copy(dll, vst3.join("Contents").join("MacOS").join(PLUGIN_NAME))?;
    eprintln!("  VST3: {}", vst3.display());

    // CLAP bundle: Plugin.clap/Contents/MacOS/Plugin
    let clap = out.join(format!("{PLUGIN_NAME}.clap"));
    fs::create_dir_all(clap.join("Contents").join("MacOS"))?;
    fs::copy(dll, clap.join("Contents").join("MacOS").join(PLUGIN_NAME))?;
    eprintln!("  CLAP: {}", clap.display());

    Ok(())
}

fn bundle_linux(dll: &Path, out: &Path) -> anyhow::Result<()> {
    // VST3: Plugin.vst3/Contents/x86_64-linux/Plugin.so
    let vst3_dir = out
        .join(format!("{PLUGIN_NAME}.vst3"))
        .join("Contents")
        .join("x86_64-linux");
    fs::create_dir_all(&vst3_dir)?;
    fs::copy(dll, vst3_dir.join(format!("{PLUGIN_NAME}.so")))?;
    eprintln!("  VST3: {}", vst3_dir.display());

    // CLAP: Plugin.clap/Plugin.clap
    let clap_dir = out.join(format!("{PLUGIN_NAME}.clap"));
    fs::create_dir_all(&clap_dir)?;
    fs::copy(dll, clap_dir.join(format!("{PLUGIN_NAME}.clap")))?;
    eprintln!("  CLAP: {}", clap_dir.display());

    Ok(())
}

fn install() -> anyhow::Result<()> {
    bundle()?;

    let bundled = bundled_dir();
    let home = env::var("USERPROFILE")
        .or_else(|_| env::var("HOME"))
        .unwrap_or_else(|_| ".".into());

    if cfg!(target_os = "windows") {
        // VST3: C:\Program Files\Common Files\VST3\
        let vst3_dest = PathBuf::from(r"C:\Program Files\Common Files\VST3");
        if vst3_dest.exists() || vst3_dest.parent().map(|p| p.exists()).unwrap_or(false) {
            fs::create_dir_all(&vst3_dest)?;
            copy_dir(
                &bundled.join(format!("{PLUGIN_NAME}.vst3")),
                &vst3_dest.join(format!("{PLUGIN_NAME}.vst3")),
            )?;
            eprintln!("Installed VST3: {}", vst3_dest.display());
        }

        // CLAP: %LOCALAPPDATA%\Programs\Common\CLAP\
        let clap_dest = env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(&home))
            .join("Programs")
            .join("Common")
            .join("CLAP");
        fs::create_dir_all(&clap_dest)?;
        copy_dir(
            &bundled.join(format!("{PLUGIN_NAME}.clap")),
            &clap_dest.join(format!("{PLUGIN_NAME}.clap")),
        )?;
        eprintln!("Installed CLAP: {}", clap_dest.display());
    } else if cfg!(target_os = "macos") {
        let vst3 = PathBuf::from(format!("{}/Library/Audio/Plug-Ins/VST3", home));
        fs::create_dir_all(&vst3)?;
        copy_dir(
            &bundled.join(format!("{PLUGIN_NAME}.vst3")),
            &vst3.join(format!("{PLUGIN_NAME}.vst3")),
        )?;

        let clap = PathBuf::from(format!("{}/Library/Audio/Plug-Ins/CLAP", home));
        fs::create_dir_all(&clap)?;
        copy_dir(
            &bundled.join(format!("{PLUGIN_NAME}.clap")),
            &clap.join(format!("{PLUGIN_NAME}.clap")),
        )?;

        eprintln!("Installed to ~/Library/Audio/Plug-Ins/");
    } else {
        let vst3 = PathBuf::from(format!("{home}/.vst3"));
        let clap = PathBuf::from(format!("{home}/.clap"));
        fs::create_dir_all(&vst3)?;
        fs::create_dir_all(&clap)?;
        copy_dir(
            &bundled.join(format!("{PLUGIN_NAME}.vst3")),
            &vst3.join(format!("{PLUGIN_NAME}.vst3")),
        )?;
        copy_dir(
            &bundled.join(format!("{PLUGIN_NAME}.clap")),
            &clap.join(format!("{PLUGIN_NAME}.clap")),
        )?;
        eprintln!("Installed to ~/.vst3 and ~/.clap");
    }

    eprintln!("\nDone! Restart your DAW to pick up the plugin.");
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn clean() -> anyhow::Result<()> {
    let bundled = bundled_dir();
    if bundled.exists() {
        fs::remove_dir_all(&bundled)?;
        eprintln!("Removed {}", bundled.display());
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "LightVC build tasks

Usage: cargo xtask <COMMAND>

Commands:
  bundle   Build release and create .clap + .vst3 bundles in target/bundled/
  install  Bundle + copy to system plugin directories
  clean    Remove target/bundled/
  help     Show this message"
    );
}
