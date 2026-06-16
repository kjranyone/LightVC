//! Weight loading utilities.
//!
//! Handles loading DAC and converter weights from safetensors files.

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

/// Load a VarBuilder from a mmaped safetensors file.
pub fn load_varbuilder<'a>(
    path: &'a Path,
    dtype: DType,
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    Ok(vb)
}

/// Download DAC weights from HuggingFace Hub if not present locally.
pub fn ensure_dac_weights(cache_dir: &Path) -> Result<std::path::PathBuf> {
    let local = cache_dir.join("dac_44khz.safetensors");
    if local.exists() {
        return Ok(local);
    }

    #[cfg(feature = "hf-hub")]
    {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .build()?;
        let repo = api.model("descript/dac_44khz".to_string());
        let path = repo.get("model.safetensors")?;
        return Ok(path);
    }

    #[cfg(not(feature = "hf-hub"))]
    {
        anyhow::bail!(
            "DAC weights not found at {}. Download model.safetensors from \
             https://huggingface.co/descript/dac_44khz and place it there.",
            local.display()
        )
    }
}
