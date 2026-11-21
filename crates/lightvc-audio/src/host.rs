//! オーディオホストの選択(WASAPI / ASIO)。全ての `cpal::default_host()` の代わりにここを通す。
//!
//! 順位: 明示的な指定(GUI・CLI)> 環境変数 LVC_HOST=wasapi|asio > 既定の方針
//! (asio feature 付きの Windows ビルドでは ASIO を先に試し、ドライバが無ければ WASAPI へ落とす)。

use anyhow::{anyhow, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum HostPref {
    #[default]
    Default,
    Wasapi,
    Asio,
}

impl HostPref {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "default" => Ok(Self::Default),
            #[cfg(windows)]
            "wasapi" => Ok(Self::Wasapi),
            #[cfg(all(windows, feature = "asio"))]
            "asio" => Ok(Self::Asio),
            other => Err(anyhow!("未知のホスト {other}(wasapi|asio)")),
        }
    }
}

/// この機械で今 使えるホストの名前(GUI の選択肢・昇順)。
pub fn available() -> Vec<&'static str> {
    let mut v = Vec::new();
    #[cfg(all(windows, feature = "asio"))]
    if cpal::host_from_id(cpal::HostId::Asio).is_ok() {
        v.push("asio");
    }
    #[cfg(windows)]
    if cpal::host_from_id(cpal::HostId::Wasapi).is_ok() {
        v.push("wasapi");
    }
    v
}

fn from_pref(pref: HostPref) -> Result<cpal::Host> {
    match pref {
        HostPref::Default => {
            #[cfg(all(windows, feature = "asio"))]
            if let Ok(h) = cpal::host_from_id(cpal::HostId::Asio) {
                return Ok(h);
            }
            Ok(cpal::default_host())
        }
        #[cfg(windows)]
        HostPref::Wasapi => cpal::host_from_id(cpal::HostId::Wasapi).map_err(|e| anyhow!("WASAPI が開けない: {e}")),
        #[cfg(not(windows))]
        HostPref::Wasapi => Ok(cpal::default_host()),
        #[cfg(all(windows, feature = "asio"))]
        HostPref::Asio => cpal::host_from_id(cpal::HostId::Asio).map_err(|e| anyhow!("ASIO が開けない(ドライバが無い?): {e}")),
        #[cfg(not(all(windows, feature = "asio")))]
        HostPref::Asio => Err(anyhow!("このビルドには ASIO が無い(--features asio)")),
    }
}

/// 使うホストを決めて開く(明示的な指定 > LVC_HOST > 既定の方針)。
pub fn pick(pref: HostPref) -> Result<cpal::Host> {
    let env = (pref == HostPref::Default).then(|| std::env::var("LVC_HOST").ok().filter(|s| !s.trim().is_empty())).flatten();
    from_pref(match env {
        Some(s) => HostPref::parse(&s)?,
        None => pref,
    })
}
