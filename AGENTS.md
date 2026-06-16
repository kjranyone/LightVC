# AGENTS.md - Project Rules

## Environment

- **Python環境分離は必ず uv で行う**
- **conda は禁止**。`conda install` / `conda create` / `conda search` は一切使わない
- Intel加速は **IPEX (CPU) ではなく XPU (Intel GPU)** を使う
  - `import intel_extension_for_pytorch as ipex`
  - device は `xpu`
- Rust の lint/typecheck: `cargo check --workspace` / `cargo clippy`
