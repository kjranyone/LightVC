<p align="center">
  <img src="crates/lightvc-app/assets/splash/splash.png" alt="LightVC" width="600">
</p>

<h1 align="center">LightVC</h1>

<p align="center">
  Real-time voice conversion as a Rust desktop application and CLAP/VST3 plugin.<br>
  Research prototype for codec-space, low-latency voice conversion.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#features">Features</a> ·
  <a href="docs/MANUAL.md">Manual</a> ·
  <a href="ARCHITECTURE.md">Architecture</a> ·
  <a href="training/README.md">Training</a>
</p>

---

## Overview

LightVC is a research prototype for low-latency voice conversion in neural-codec space. The original continuous-DAC-latent flow-matching design is now **frozen** after experiments showed severe off-manifold and RVQ-cascade sensitivity.

Current research direction:

```
Mic input
→ DAC encode
→ preserve source RVQ depth 0 as content anchor
→ generate target-like residual trajectory
→ residual-chain-preserving re-quantization for depths 1..8
→ DAC decode
→ output
```

The best oracle so far is same-text/content-aligned `source depth 0 + target-like residual re-quantization`, reaching SECS `0.656-0.686` with CER `0.057-0.082` on 200 VCTK pairs. Cross-text/free-conversation generation remains unresolved.

## Features

- **Codec-space VC research** — avoids a heavy TTS/BigVGAN waveform-generation pipeline
- **Residual-chain-aware RVQ experiments** — naive token swaps are known to fail
- **Rust app/plugin shell** — standalone GUI, CLAP, and VST3 bundle support
- **DAC round-trip path** — native Rust/Candle DAC model work is preserved
- **Three modes** — Strict (0ms lookahead), Balanced (~40ms), Quality (~80ms)
- **Three form factors**:
  - Standalone GUI app (egui)
  - CLAP plugin (REAPER, Bitwig, etc.)
  - VST3 plugin (Ableton, FL Studio, etc.)
- **VC-teacher-free policy** — no synthetic parallel targets from another VC model
- **Pure Rust inference** — no Python runtime at deployment

## Quick Start

### One-command (Windows / PowerShell)

```powershell
.\dev.ps1
```

`dev.ps1` builds the release binary, auto-downloads the ~307 MB DAC
weights on first run (cached at `models/dac_44khz.safetensors`), then
launches the GUI. Converter weights are optional — load them later via
the Realtime tab once training completes.

```powershell
.\dev.ps1 -NoBuild                # skip build, use existing binary
.\dev.ps1 -Roundtrip -Input a.wav # validate DAC encode/decode
```

### Screenshots (demo mode)

The GUI has a `--demo-state` mode that renders with mock data — no model,
no audio devices needed. Used for documentation screenshots and visual
regression checks:

```powershell
cargo build --release -p lightvc-app
.\tools\capture-window.ps1 -DemoState realtime
.\tools\capture-window.ps1 -DemoState offline
.\tools\capture-window.ps1 -DemoState catalog
# → docs/screenshots/{realtime,offline,catalog}.png
```

### Manual build

### Prerequisites

- Rust 1.75+ (2024 edition)
- Python 3.10+ with [uv](https://github.com/astral-sh/uv) (training only)
- DAC weights: download `model.safetensors` from [descript/dac_44khz](https://huggingface.co/descript/dac_44khz) to `models/dac_44khz.safetensors`

### Build

```bash
# Standalone app
cargo build --release -p lightvc-app
./target/release/lightvc-app gui --dac-weights models/dac_44khz.safetensors

# CLAP + VST3 plugin bundle
cargo run -p lightvc-xtask -- bundle
# Output: target/bundled/LightVC.vst3 and LightVC.clap

# Install to system plugin directories
cargo run -p lightvc-xtask -- install
```

### CLI subcommands

```bash
# Validate DAC round-trip
lightvc-app roundtrip -i input.wav -o output.wav --dac-weights models/dac_44khz.safetensors

# Offline conversion
lightvc-app convert -i source.wav -r reference.wav -o converted.wav \
    --dac-weights models/dac_44khz.safetensors \
    --converter-weights models/converter.safetensors

# GUI (3 tabs: offline / realtime / voice catalog)
lightvc-app gui --dac-weights models/dac_44khz.safetensors
```

## Training

The historical Phase B/C training commands below belong to the frozen continuous-latent converter path. They are kept for reproducibility and code archaeology, not as the current recommended research route.

Current research notes:

- [plan/12_concept_v2.md](plan/12_concept_v2.md) — current codec-token / residual-chain plan
- [docs/research_report_2026-06-20.md](docs/research_report_2026-06-20.md) — experiment report through Phase 2
- [docs/literature_update_2026-06-21.md](docs/literature_update_2026-06-21.md) — updated literature and strategy notes

Historical pipeline:

```bash
cd training && uv sync

# 1. Generate TTS corpus (Edge TTS, 17 speakers)
uv run python generate_tts_corpus.py

# 2. Encode to DAC latents
uv run python encode_corpus.py --source ../data/tts_corpus --output data/latents

# 3. Phase B: bottleneck warm-start
uv run python train_warmstart.py --config configs/phase_b.yaml --data data/latents

# 4. Phase C: flow matching
uv run python train_flow.py --config configs/phase_c.yaml --data data/latents

# 5. Export to safetensors
uv run python export_weights.py --checkpoint checkpoints/phase_c/best.pt --output ../models/converter.safetensors
```

## Project Structure

```
LightVC/
├── crates/
│   ├── lightvc-core/      Core inference: DAC, converter, streaming, pipeline
│   ├── lightvc-audio/     Audio I/O: cpal, rubato resampling, ring buffers
│   ├── lightvc-app/       Standalone GUI (egui, 3 tabs)
│   ├── lightvc-clap/      CLAP/VST3 plugin (nice-plug + clap-wrapper)
│   └── lightvc-xtask/     Build automation (bundle, install)
├── training/              Python training pipeline (uv)
├── docs/                  ASIO setup guide
├── models/                Model weights (.safetensors, git-ignored)
└── samples/               Before/after audio samples
```

## Architecture

- **Codec**: DAC (Descript Audio Codec), 44.1kHz, MIT license
- **Current research converter**: target-like residual trajectory generator + residual-chain-preserving RVQ re-quantization
- **Frozen converter**: Causal Conv1d continuous-latent flow matching, ~10M params
- **Inference**: Candle (pure Rust), CPU/GPU/CUDA/Metal
- **Training**: PyTorch + Intel XPU (uv environment)
- **Plugin**: nice-plug (ISC) + clap-wrapper (MIT) → CLAP + VST3
- **Audio**: cpal (WASAPI/ASIO/CoreAudio/ALSA)

## License

MIT — all dependencies are MIT/ISC/Apache-2.0. No GPLv3.

## Documents

- [DESIGN.md](DESIGN.md) — High-level design and rationale
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture detail
- [MODEL_TRAINING.md](MODEL_TRAINING.md) — Historical continuous-latent training pipeline
- [RESEARCH.md](RESEARCH.md) — Literature survey and evidence base
- [plan/12_concept_v2.md](plan/12_concept_v2.md) — Current research plan
- [docs/ASIO_SETUP.md](docs/ASIO_SETUP.md) — Optional ASIO SDK setup
