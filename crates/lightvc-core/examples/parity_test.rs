use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use lightvc_core::soft_rvq::load_soft_rvq;
use lightvc_core::utte_adapter::load_adapter;
use std::path::PathBuf;

fn mse(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = (a - b)?;
    let s = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok(s as f64)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");

    let dac_quantizer = base.join("models/dac_quantizer.safetensors");
    let adapter_weights = base.join("models/utte_adapter_b1.safetensors");
    let ref_weights = base.join("models/rust_parity_ref.safetensors");

    for (name, path) in [
        ("DAC quantizer", &dac_quantizer),
        ("Adapter", &adapter_weights),
        ("Reference", &ref_weights),
    ] {
        if !path.exists() {
            anyhow::bail!("{} weights not found: {}", name, path.display());
        }
    }

    println!("=== Rust/Candle Parity Test ===\n");

    let rvb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&ref_weights], DType::F32, &device)? };
    let z_s = rvb.get((1, 1024, 298), "z_s")?;
    let q0_s = rvb.get((1, 1024, 298), "q0_s")?;
    let timbre = rvb.get((1, 192), "timbre")?;
    let z_q_ref = rvb.get((1, 1024, 298), "z_q_ref")?;
    let z_q_adapted_ref = rvb.get((1, 1024, 298), "z_q_adapted_ref")?;
    println!("Loaded reference tensors (T=298)");

    let soft_rvq = load_soft_rvq(&dac_quantizer, &device)?;
    println!("Soft RVQ loaded");

    let adapter = load_adapter(&adapter_weights, &device)?;
    println!("UTTE adapter loaded");

    let tau = 5.0f64;
    println!("\nRunning soft RVQ (tau={})...", tau);
    let z_q = soft_rvq.soft_requantize(&q0_s, &z_s, tau)?;
    let rvq_mse = mse(&z_q, &z_q_ref)?;
    println!(
        "  z_q MSE: {:.6}  (ref norm={:.1}, rust norm={:.1})",
        rvq_mse,
        z_q_ref.norm()?.to_scalar::<f32>()?,
        z_q.norm()?.to_scalar::<f32>()?,
    );

    println!("\nRunning UTTE adapter...");
    let z_q_adapted = adapter.forward(&z_q, &timbre)?;
    let adapter_mse = mse(&z_q_adapted, &z_q_adapted_ref)?;
    println!(
        "  z_q_adapted MSE: {:.6}  (ref norm={:.1}, rust norm={:.1})",
        adapter_mse,
        z_q_adapted_ref.norm()?.to_scalar::<f32>()?,
        z_q_adapted.norm()?.to_scalar::<f32>()?,
    );

    println!("\n=== Summary ===");
    let rvq_ok = rvq_mse < 1e-3;
    let adapter_ok = adapter_mse < 1e-3;
    println!(
        "Soft RVQ:     MSE={:.6}  {}",
        rvq_mse,
        if rvq_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "UTTE adapter: MSE={:.6}  {}",
        adapter_mse,
        if adapter_ok { "PASS" } else { "FAIL" }
    );

    if rvq_ok && adapter_ok {
        println!("\n✓ Python ↔ Rust parity confirmed");
    } else {
        println!("\n✗ Parity check failed — investigate divergence");
    }

    Ok(())
}
