//! オーディオホストの選択(WASAPI / ASIO)。
//!
//! 順位: 明示的な指定(GUI・CLI) > 環境変数 LVC_HOST=wasapi|asio > 既定(WASAPI)。

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
            #[cfg(any(all(windows, target_arch = "x86_64"), all(windows, feature = "asio")))]
            "asio" => Ok(Self::Asio),
            other => Err(anyhow!("未知のホスト {other}(wasapi|asio)")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Wasapi => "wasapi",
            Self::Asio => "asio",
        }
    }
}

pub fn has_asio() -> bool {
    #[cfg(all(windows, target_arch = "x86_64"))]
    {
        crate::asio_win::has_drivers()
    }
    #[cfg(all(windows, not(target_arch = "x86_64"), feature = "asio"))]
    {
        cpal::host_from_id(cpal::HostId::Asio).is_ok()
    }
    #[cfg(not(any(all(windows, target_arch = "x86_64"), all(windows, feature = "asio"))))]
    {
        false
    }
}

/// この機械で今 使えるホストの名前(GUI の選択肢)。
pub fn available() -> Vec<&'static str> {
    let mut v = Vec::new();
    if has_asio() {
        v.push("asio");
    }
    #[cfg(windows)]
    if cpal::host_from_id(cpal::HostId::Wasapi).is_ok() {
        v.push("wasapi");
    }
    v
}

/// 実効ホストを解決する(Default のとき LVC_HOST を確認し、無ければ WASAPI)。
pub fn resolve(pref: HostPref) -> Result<HostPref> {
    let effective = if pref == HostPref::Default {
        match std::env::var("LVC_HOST").ok().filter(|s| !s.trim().is_empty()) {
            Some(s) => HostPref::parse(&s)?,
            None => HostPref::Default,
        }
    } else {
        pref
    };
    Ok(match effective {
        HostPref::Default => {
            #[cfg(windows)]
            {
                HostPref::Wasapi
            }
            #[cfg(not(windows))]
            {
                HostPref::Default
            }
        }
        other => other,
    })
}

/// cpal ホストを開く(WASAPI または非 Windows 既定)。
pub fn pick(pref: HostPref) -> Result<cpal::Host> {
    match resolve(pref)? {
        #[cfg(windows)]
        HostPref::Wasapi | HostPref::Default => {
            cpal::host_from_id(cpal::HostId::Wasapi).map_err(|e| anyhow!("WASAPI が開けない: {e}"))
        }
        #[cfg(not(windows))]
        HostPref::Wasapi | HostPref::Default => Ok(cpal::default_host()),
        #[cfg(all(windows, feature = "asio"))]
        HostPref::Asio => cpal::host_from_id(cpal::HostId::Asio).map_err(|e| anyhow!("ASIO が開けない: {e}")),
        #[cfg(not(all(windows, feature = "asio")))]
        HostPref::Asio => Err(anyhow!("cpal::Host としては WASAPI を開いてください(ASIO はネイティブ COM 経路を使用します)")),
    }
}
