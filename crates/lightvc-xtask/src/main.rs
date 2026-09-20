//! LightVC build tasks.
//!
//! Usage:
//!   cargo xtask bundle         — Build release + create .clap and .vst3 bundles
//!   cargo xtask install        — Bundle + copy to system plugin directories
//!   cargo xtask clean          — Remove target/bundled
//!   cargo xtask package        — Bundle + zip + sha256 (G0-15)

use std::env;
use std::fs;
use std::process::Command;
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
        "package" => package()?,
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

/// G0-15: 配布 zip と sha256 を作る。
///
/// ⚠ 同梱するのは**バンドル・モデル・NOTICE・README・遅延台帳**のみ。
/// ASIO SDK はプロプライエタリで再配布禁止なので絶対に入れない（`CLAUDE.md`）。
fn package() -> anyhow::Result<()> {
    bundle()?;
    let root = workspace_root();
    let out = bundled_dir();
    let ver = env!("CARGO_PKG_VERSION");
    let os = if cfg!(target_os = "windows") { "windows" }
             else if cfg!(target_os = "macos") { "macos" } else { "linux" };
    let stage = root.join("target").join("package");
    if stage.exists() { fs::remove_dir_all(&stage)?; }
    fs::create_dir_all(&stage)?;

    copy_tree(&out, &stage)?;

    // モデルと配布に要る書類。**無ければ落とす**（黙って欠いた zip を出さない）。
    // CLI 本体（README が案内する `lightvc-app v2f` の実体）
    let app = if cfg!(target_os = "windows") { "lightvc-app.exe" } else { "lightvc-app" };
    let app_src = target_dir().join(app);
    if !app_src.exists() {
        anyhow::bail!("CLI が無い: {}（cargo build --release -p lightvc-app）", app_src.display());
    }
    fs::copy(&app_src, stage.join(app))?;

    let mut manifest = String::from("# LightVC 配布内容\n\n| ファイル | sha256 |\n|---|---|\n");
    // ⚠ models/ を丸ごと入れない——旧世代の重み（converter/DAC 系 1.3 GB）が
    //   同居している。**出荷する物を明示列挙**（欠品は fail-closed のまま）。
    for rel in ["models/v2f.bin", "models/v2f_v2f_prior_20260819_013905.json",
                "models/mel_fb_1024_80.bin", "models/mel2lin_W.bin",
                // バ美声変換段 (lightvc-app vc): E + voice cartridge (G)
                "models/e1.bin", "models/e1.json",
                "models/g1.bin", "models/g1.json",
                "NOTICE", "README.md", "results/z0/latency_decision.md"] {
        let src = root.join(rel);
        if !src.exists() {
            anyhow::bail!("配布物が無い: {rel}（G0-15 は欠品のまま zip を作らない）");
        }
        let dst = if rel.starts_with("models/") {
            let d = stage.join("models");
            fs::create_dir_all(&d)?;
            d.join(Path::new(rel).file_name().unwrap())
        } else {
            stage.join(Path::new(rel).file_name().unwrap())
        };
        if src.is_dir() { copy_tree(&src, &dst)?; } else { fs::copy(&src, &dst)?; }
    }
    for e in walk(&stage)? {
        let rel = e.strip_prefix(&stage)?.to_string_lossy().replace('\\', "/");
        manifest.push_str(&format!("| `{rel}` | `{}` |\n", sha256_file(&e)?));
    }
    fs::write(stage.join("MANIFEST.md"), &manifest)?;

    let zip = root.join("target").join(format!("LightVC-{ver}-{os}.zip"));
    let ok = Command::new("zip").arg("-qr").arg(&zip).arg(".").current_dir(&stage)
        .status().map(|s| s.success()).unwrap_or(false)
        || Command::new("python3")
            .args(["-c", concat!(
                "import sys, zipfile, os\n",
                "z = zipfile.ZipFile(sys.argv[1], 'w', zipfile.ZIP_DEFLATED)\n",
                "for r, _, fs in os.walk('.'):\n",
                "    for f in fs:\n",
                "        p = os.path.join(r, f)\n",
                "        z.write(p, os.path.relpath(p, '.'))\n",
                "z.close()\n")])
            .arg(&zip).current_dir(&stage)
            .status().map(|s| s.success()).unwrap_or(false);
    anyhow::ensure!(ok, "zip も python3(zipfile) も使えない");
    let sum = sha256_file(&zip)?;
    fs::write(zip.with_extension("zip.sha256"), format!("{sum}  {}\n",
        zip.file_name().unwrap().to_string_lossy()))?;
    eprintln!("\n配布 zip: {}\n  sha256: {sum}", zip.display());
    Ok(())
}

fn copy_tree(src: &Path, dst: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(dst)?;
    for e in fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() { copy_tree(&e.path(), &to)?; } else { fs::copy(e.path(), to)?; }
    }
    Ok(())
}

fn walk(d: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut v = Vec::new();
    for e in fs::read_dir(d)? {
        let e = e?;
        if e.file_type()?.is_dir() { v.extend(walk(&e.path())?); } else { v.push(e.path()); }
    }
    v.sort();
    Ok(v)
}

/// sha256（依存を足さずに済ませる。配布物の同一性を示すだけなので十分）。
fn sha256_file(p: &Path) -> anyhow::Result<String> {
    let out = Command::new("sha256sum").arg(p).output()?;
    if !out.status.success() {
        let out = Command::new("shasum").args(["-a", "256"]).arg(p).output()?;
        anyhow::ensure!(out.status.success(), "sha256sum / shasum が要る");
        return Ok(String::from_utf8_lossy(&out.stdout).split_whitespace().next()
            .unwrap_or("").to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).split_whitespace().next().unwrap_or("").to_string())
}

fn bundle() -> anyhow::Result<()> {
    build_release()?;

    // OS ごとの成果物名（旧版は Linux でも .dll を探して必ず落ちた）
    let dll_src = if cfg!(target_os = "windows") {
        target_dir().join(format!("{DLL_NAME}.dll"))
    } else if cfg!(target_os = "macos") {
        target_dir().join(format!("lib{DLL_NAME}.dylib"))
    } else {
        target_dir().join(format!("lib{DLL_NAME}.so"))
    };
    if !dll_src.exists() {
        anyhow::bail!("plugin artifact not found: {}", dll_src.display());
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
