# LightVC Architecture

Detailed system architecture for the Rust client and model components.

> **Status note (2026-06-21):** Several model sections below document the
> historical continuous-latent converter architecture. That path is frozen after
> experiments showed off-manifold decoding and RVQ-cascade sensitivity. The
> current research direction is residual-chain-preserving codec trajectory
> translation: preserve source RVQ depth 0, generate target-like residual
> trajectory, then re-quantize depths 1..8 or use a tolerant decoder adapter.
> See [plan/12_concept_v2.md](plan/12_concept_v2.md) for the active plan.

---

## 1. System Topology

### 1.1 Process Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LightVC Process                           │
│                                                                     │
│  Thread 1: Audio Capture (cpal callback)                           │
│    mic callback → ringbuf_capture (lock-free SPSC)                 │
│    Priority: real-time, no allocations                              │
│                                                                     │
│  Thread 2: Inference (Candle)                                       │
│    loop {                                                           │
│      ringbuf_capture.read(chunk)                                    │
│      → resample(device_sr → 44100)                                  │
│      → DAC.encode(chunk)           // latent + RVQ tokens           │
│      → Converter.forward(...)      // q0 anchor + residual path     │
│      → DAC.decode(tokens/latent)   // decoder or tolerant adapter   │
│      → resample(44100 → device_sr)                                  │
│      → ringbuf_playback.write(chunk)                                │
│    }                                                                │
│    Priority: high, owned by inference thread                        │
│                                                                     │
│  Thread 3: Audio Playback (cpal callback)                          │
│    ringbuf_playback → speaker callback                              │
│    Priority: real-time, no allocations                              │
│                                                                     │
│  Thread 4 (main): UI (egui/eframe)                                  │
│    device select / target voice load / mode toggle / meters         │
│    Communicates with Thread 2 via crossbeam channels               │
│    Priority: normal                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Thread Communication

| Channel | Direction | Payload | Mechanism |
|---------|-----------|---------|-----------|
| `ringbuf_capture` | Thread 1 → Thread 2 | `f32` PCM samples | `rtrb` lock-free SPSC ring buffer |
| `ringbuf_playback` | Thread 2 → Thread 3 | `f32` PCM samples | `rtrb` lock-free SPSC ring buffer |
| `control_tx` | Thread 4 → Thread 2 | Mode changes, target voice, params | `crossbeam_channel::unbounded` |
| `metrics_rx` | Thread 2 → Thread 4 | Input/output RMS, latency, RTF | `crossbeam_channel::unbounded` (non-blocking recv) |

### 1.3 Latency Budget

End-to-end latency = cpal buffers + resample + **algorithmic delay** +
resample + cpal buffers. The algorithmic delay is `chunk_size + lookahead`
(see §3.5), and is the mode-dependent term.

```
Component                        Latency
─────────────────────────────────────────
Capture buffer (cpal)            ~10 ms
Resample (device → 44100)         ~3 ms
Algorithmic delay (mode):
  Strict  (1 frame, 0 lookahead)  ~12 ms
  Balanced (4 frames + 4 FRC)     ~93 ms
  Quality (8 frames + 8 FRC)     ~186 ms
Resample (44100 → device)         ~3 ms
Playback buffer (cpal)            ~10 ms
─────────────────────────────────────────
Total (strict)                    ~38 ms
Total (balanced)                 ~119 ms
Total (quality)                  ~212 ms
```

> **Note ([08-8]):** An earlier version of this section reported quality
> total as ~141 ms with 80 ms lookahead. That was based on a design where
> lookahead excluded the chunk. The implementation (§3.5) defines
> algorithmic latency as `chunk + lookahead`, so quality is ~186 ms
> algorithmic / ~212 ms total. The numbers above match
> `realtime_tab.rs` and `pipeline::algorithmic_latency_ms()`.

---

## 2. Rust Crate Structure

```
lightvc/
├── Cargo.toml
├── crates/
│   ├── lightvc-core/           # Core inference (no UI, no audio I/O)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── codec/
│   │   │   │   ├── mod.rs          # DacCodec wrapper
│   │   │   │   ├── encoder.rs      # DAC encoder forward + streaming state
│   │   │   │   ├── decoder.rs      # DAC decoder forward + streaming state
│   │   │   │   └── streaming.rs    # Overlap-add, conv-state cache
│   │   │   ├── converter/
│   │   │   │   ├── mod.rs          # Converter model
│   │   │   │   ├── conv_block.rs   # Causal Conv1d blocks
│   │   │   │   ├── timbre.rs       # Universal Timbre Token Encoder
│   │   │   │   └── config.rs       # Model config structs
│   │   │   ├── pipeline.rs         # Full inference pipeline orchestrator
│   │   │   └── weights.rs          # Safetensors loading
│   │   └── Cargo.toml
│   │
│   ├── lightvc-audio/          # Audio I/O abstraction
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── device.rs           # cpal device enumeration
│   │   │   ├── stream.rs           # Duplex stream management
│   │   │   ├── ringbuf.rs          # rtrb wrappers
│   │   │   └── resample.rs         # rubato wrapper (RT-safe)
│   │   └── Cargo.toml
│   │
│   └── lightvc-app/            # Desktop application (egui)
│       ├── src/
│       │   ├── main.rs
│       │   ├── app.rs              # eframe App impl
│       │   ├── realtime_tab.rs     # Real-time VC tab (capture/convert/playback)
│       │   ├── offline_tab.rs      # Offline file conversion tab
│       │   ├── widgets.rs          # UI helpers (RMS, level meters)
│       │   ├── theme.rs            # Colors, styles, custom widgets
│       │   ├── voice_catalog.rs    # Target voice management
│       │   ├── audio_playback.rs   # Audio playback helpers
│       │   ├── assets.rs           # Embedded icons/textures
│       │   ├── cli.rs              # CLI subcommand handler
│       │   └── settings.rs         # Persisted settings (serde)
│       └── Cargo.toml
│
│   ├── lightvc-clap/           # CLAP/VST3 plugin (cdylib)
│   │   ├── src/
│   │   │   └── lib.rs              # CLAP plugin entry + process callback
│   │   ├── build.rs               # clap-wrapper bundle scaffolding
│   │   └── Cargo.toml
│   │
│   └── lightvc-xtask/          # Build automation (bundle / install)
│       ├── src/
│       │   └── main.rs             # xtask: cargo run -p lightvc-xtask -- bundle|install
│       └── Cargo.toml
│
├── models/                     # Model weights (git-lfs or download script)
│   ├── dac_44khz.safetensors   # Frozen DAC (~307 MB)
│   ├── converter_p1.safetensors # Phase 1 converter (~40 MB)
│   └── converter_p2.safetensors # Phase 2 converter (~120 MB)
│
├── training/                   # Python training pipeline
│   ├── README.md
│   ├── generate_pairs.py       # Synthetic parallel data generation
│   ├── train_converter.py      # Student converter training
│   ├── export_weights.py       # Export to .safetensors for Candle
│   └── configs/
│
└── docs/
    ├── DESIGN.md
    ├── ARCHITECTURE.md          ← this file
    ├── MODEL_TRAINING.md
    └── RESEARCH.md
```

### Key Dependencies

```toml
# lightvc-core
[dependencies]
candle-core = "0.10"
candle-nn = "0.10"
candle-transformers = "0.10"   # provides dac.rs
safetensors = "0.4"
hf-hub = "0.3"                  # download DAC weights from HF
anyhow = "1.0"

# lightvc-audio
[dependencies]
cpal = "0.18"
rubato = "0.16"                 # or 3.0 for RT-safe API
rtrb = "0.3"                    # lock-free ring buffer
crossbeam-channel = "0.5"

# lightvc-app
[dependencies]
eframe = "0.34"                 # egui
lightvc-core = { path = "../lightvc-core" }
lightvc-audio = { path = "../lightvc-audio" }
serde = { version = "1.0", features = ["derive"] }

# lightvc-clap (CLAP/VST3 plugin, cdylib)
[dependencies]
lightvc-core = { path = "../lightvc-core" }
nice-plug = "0.1"               # ISC — CLAP host abstraction
nice-plug-egui = "0.1"          # ISC — egui editor wrapper
egui = "0.34"
clap-wrapper = "0.3"            # MIT — CLAP→VST3/AUv2 wrapper
# NOTE: vst3-sys (GPLv3) is intentionally avoided. See AGENTS.md §Licensing.

# lightvc-xtask (build automation, not published)
[dependencies]
anyhow = "1.0"
```

---

## 3. DAC Streaming Implementation

### 3.1 The Challenge

The Candle `dac.rs` implementation has three gaps for our use case:

1. **No `encode()` method**: Only `decode_codes()` is exposed.
2. **No streaming**: `StreamingModule` trait not implemented.
3. **Non-causal**: Encoder convolutions use symmetric padding (require future samples).

### 3.2 Historical Solution: Continuous Latent Pipeline (frozen)

**Historical note**: The original converter operated on **continuous latents** and skipped the quantizer. This path is now frozen. Current experiments show that residual-chain-preserving RVQ re-quantization or a tolerant decoder is required for valid codec trajectories.

```
                    ┌─────────────────────────────┐
                    │   DAC Encoder (frozen)       │
  44.1kHz PCM ────►│   Conv1d → 4×EncoderBlock    │──── continuous latent
                    │   → Conv1d                   │     (1024-dim, 86 Hz)
                    └─────────────────────────────┘
                                                           │
                                                           ▼
                    ┌─────────────────────────────┐
                    │   Converter (trained)        │
                    │   Causal Conv1d stack        │
                    │   + timbre cross-attn        │
                    └─────────────────────────────┘
                                                           │
                                                           ▼
                    ┌─────────────────────────────┐
                    │   DAC Decoder (frozen)       │
  44.1kHz PCM ◄────│   Conv1d → 4×DecoderBlock    │◄── modified latent
                    │   (TConv) → Conv1d           │
                    └─────────────────────────────┘
```

This avoids the missing quantizer encode path entirely. The decoder's input is a continuous 1024-dim tensor — it does not require quantized codes.

### 3.3 Encoder Forward Implementation

The upstream `candle-transformers::models::dac` assumes PyTorch-original
safetensors key names and is decode-only. HuggingFace's `descript/dac_44khz`
uses transformers-style key names, so we reimplemented the full DAC
(encoder + decoder + Snake + ResidualUnit + blocks) natively in
`crates/lightvc-core/src/dac_model.rs` (~400 LOC) to match the HF weight
keys exactly. The encoder forward:

```rust
// lightvc-core/src/dac_model.rs — Encoder::forward

pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    // xs: [batch, 1, samples] at 44.1kHz
    let x = self.conv1.forward(xs)?;        // [B, 64, T]
    let x = self.block1.forward(&x)?;       // [B, 128, T/2]
    let x = self.block2.forward(&x)?;       // [B, 256, T/4]
    let x = self.block3.forward(&x)?;       // [B, 512, T/8]
    let x = self.block4.forward(&x)?;       // [B, 1024, T/8]
    let x = self.conv2.forward(&x)?;        // [B, latent_dim, T/512]
    Ok(x)  // [batch, 1024, frames] at 86 Hz
}
```

### 3.4 Streaming via Overlap-Add

Since DAC is non-causal, we use chunked processing with overlap:

```
Time →
input:  |--chunk_0--|--chunk_1--|--chunk_2--|
         ↕ overlap   ↕ overlap
encoded: [===latent_0===][===latent_1===]
                  ↑ cross-fade region
```

**Chunk size**: `chunk_samples = hop_length × N_frames` where `N_frames` is typically 4-8.

**Overlap region**: `padding_samples = delay` (DAC's receptive field radius, ~hundreds of samples).

**State management**: Cache the last `delay` samples of each conv layer's input. On next chunk, prepend the cached state. This gives causal behavior with bounded error at chunk boundaries.

```rust
// lightvc-core/src/codec/streaming.rs

pub struct StreamingDacEncoder {
    encoder: DacEncoder,
    /// Cached conv states: one per encoder block
    conv_states: Vec<ConvState>,
    /// Input tail buffer (last `delay` samples)
    input_tail: VecDeque<f32>,
    /// DAC algorithmic delay in samples
    delay: usize,
}

impl StreamingDacEncoder {
    /// Process one chunk of PCM → continuous latent
    pub fn encode_step(&mut self, chunk: &[f32]) -> Result<Tensor> {
        // 1. Prepend cached tail
        let mut buffer = self.input_tail.clone();
        buffer.extend(chunk);
        
        // 2. Encode with conv-state injection per layer
        let latent = self.encoder.forward_with_state(
            &buffer, &mut self.conv_states
        )?;
        
        // 3. Update tail cache
        self.input_tail = buffer[buffer.len() - self.delay..].into();
        
        // 4. Trim to new frames only
        let n_new_frames = chunk.len() / HOP_LENGTH;
        Ok(latent.narrow(2, latent.dim(2)? - n_new_frames, n_new_frames)?)
    }
    
    pub fn reset_state(&mut self) {
        self.conv_states.iter_mut().for_each(|s| s.reset());
        self.input_tail.clear();
    }
}
```

#### 3.4.1 Decoder overlap-add details

The encoder side (FRC + `input_tail`) was described above. The decoder
side mirrors it with a **linear cross-fade** over the boundary region,
implemented in `StreamingCodec::decode_step`:

```
chunk N-1 PCM :  [...=====tail=====]
chunk N   PCM :  [==overlap==|new===]
                            ↕ linear cross-fade, w = i/overlap_len
merged output :  [...=====faded=====|new===]
```

- **Cross-fade length** = `min(prev.len, cur.len, DAC_HOP_LENGTH)` =
  one latent frame (512 samples ≈ 11.6 ms at 44.1 kHz). This is a
  deliberate constant: the DAC decoder's receptive field radius is on
  the order of one hop, so a single-hop blend is sufficient to hide
  the discontinuity at the chunk seam.
- **Tail cache** = last `DAC_HOP_LENGTH` samples of the merged output,
  stored in `prev_output` and consumed by the next `decode_step`.
- **Returned PCM** = only the newly produced portion
  (`new_len = frames * DAC_HOP_LENGTH`), trimmed from the tail of the
  merged buffer so the caller receives exactly one chunk's worth of
  samples.
- **Empty-latent guard**: during FRC warmup `encode_step` returns a
  0-frame tensor; `decode_step` short-circuits to an empty `Vec`, so
  the realtime loop simply pushes nothing (silence) for that period.

> **Why not scale the cross-fade with chunk size?** The plan ([02-3])
> considered making the fade proportional to `samples_per_chunk` (e.g.
> longer for Quality's 8-frame chunks). The current single-hop fade is
  already inaudible in practice because FRC lookahead removes the
  boundary artefact at its source (the encoder), so the decoder only
  needs to smooth residual sample-level discontinuities. Scaling up
  the fade would increase latency without audible benefit; left as a
  tuning knob if future ABX tests ([02-4] acceptance) reveal issues.

#### 3.4.2 Converter left-context overlap

The streaming codec handles DAC's receptive field, but the **converter**
(`FlowConverter`) also has causal dilated convolutions (CausalResBlock
with dilations [1, 3, 9], kernel 7 — effective receptive field ~313
frames across 4 blocks). Without context, each chunk's first frames are
zero-padded, producing discontinuities.

`VcPipeline` caches the last N source-latent frames (`src_context`) and
prepends them to each chunk's encoded latent before conversion. Because
the converter is strictly causal when conditioned on a fixed reference,
feeding `[context | new]` and trimming to the last `n_new` frames
reproduces the non-chunked (offline) result near-exactly.

Context sizes: Strict = 16, Balanced = 32, Quality = 64 latent frames.

`process_full()` bypasses chunking entirely for offline file conversion
(encode → convert → decode in one pass), giving exact Python parity
(wave_corr > 0.997).

### 3.5 Latency / Quality Modes

| Mode | Lookahead | Chunk Size | Total algorithmic latency | Use Case |
|------|-----------|------------|---------------------------|----------|
| **Strict** | 0 ms | 1 frame (512 samples, 11.6ms) | ~12 ms | Minimum latency, boundary artifacts accepted |
| **Balanced** | ~46 ms (2048 samples) | 4 frames (2048 samples, 46ms) | ~93 ms | Good quality/latency tradeoff |
| **Quality** | ~93 ms (4096 samples) | 8 frames (4096 samples, 93ms) | ~186 ms | Best quality, streaming-safe |

> Implemented in `crates/lightvc-core/src/streaming.rs` via Future-Receptive
> Chunking (FRC). The encoder buffers `lookahead` samples of *future* audio
> before emitting the current chunk's latent, so DAC's symmetric-padded convs
> receive real context on both edges and chunk-boundary artifacts are
> eliminated (Balanced/Quality). Strict mode skips lookahead entirely.

The lookahead absorbs DAC's non-causal receptive field. In **strict mode**, we accept quality degradation at chunk boundaries (acceptable for communication); in **balanced/quality mode**, FRC provides real future context and the decoder uses overlap-add with cross-fade.

---

## 4. Converter Model Architecture

### 4.1 Phase 1: Causal Conv1d Latent Converter

```
Input: source latent z_src  [batch, 1024, T_frames]
       target speaker embedding s_tgt [batch, 256]

┌──────────────────────────────────────────────────┐
│  FiLM conditioning (per-frame)                    │
│    γ, β = MLP(s_tgt)  → [batch, 1024, 1]          │
│    z = γ * z_src + β                              │
├──────────────────────────────────────────────────┤
│  Residual Conv Block × 4                          │
│    Conv1d(1024 → 1024, k=7, dilation=1, causal)   │
│    Snake1d + Conv1d(1024 → 1024, k=7, d=3, causal)│
│    Snake1d + Conv1d(1024 → 1024, k=7, d=9, causal)│
│    + residual                                     │
├──────────────────────────────────────────────────┤
│  Output projection                                │
│    Conv1d(1024 → 1024, k=1)                       │
│    z_out = z_src + Δz  (residual prediction)      │
└──────────────────────────────────────────────────┘

Output: target latent z_out  [batch, 1024, T_frames]
Parameters: ~8-12M
```

**Design rationale**:
- **Residual prediction** (`z_out = z_src + Δz`): converter learns the *delta* to apply. Easier to train, preserves content by default.
- **Causal Conv1d**: left-pad only, no future context needed in converter (DAC handles lookahead separately).
- **Snake1d activation**: matches DAC's internal activation, ensures latent-space compatibility.
- **FiLM speaker injection**: AdaLN-style normalization per speaker. Cheap, effective (X-VC approach).

> Phase 1 `Converter` is selected by `model_type: "converter"` in the
> JSON config. It is the warm-start baseline and is kept for ABI smoke
> tests; production checkpoints use Phase C `FlowConverter` below.

### 4.1b Phase C: FlowConverter (mean-flow, 1-NFE)

`FlowConverter` (`converter.rs`, `model_type: "flow"`) is the core inference
model. It is a **mean-flow** network: trained to predict the *average*
velocity field of the linear flow `z_t = (1-t)·z_src + t·z_tgt`, so that a
single forward pass at `t=1` produces the target latent (1-NFE inference,
no teacher distillation).

```
Inputs : z_src   [B, 1024, T]    source latent (from DAC encode)
         z_ref   [B, 1024, T_ref] reference latent (target speaker)
         t       [B]              flow time ∈ [0,1]  (1.0 at inference)

┌──────────────────────────────────────────────────────────┐
│ BottleneckEncoder  Conv1d(1024→256)  content projection   │
│   content = bottleneck(z_src or z_t)                     │
├──────────────────────────────────────────────────────────┤
│ Conditioning (FiLM γ, β)                                  │
│   speaker_embed = SpeakerEncoder(z_ref)  [B, 256]         │
│   time_embed    = TimeEmbed(t)           [B, 128]         │
│   γ, β = CondMlp([speaker_embed ‖ time_embed])  ×2·1024   │
│   z = γ · content + β                                     │
├──────────────────────────────────────────────────────────┤
│ CausalResBlock × N (default N=4)                          │
│   Snake1d → Conv1d(1024→hidden, k=7, d=1, causal)         │
│   Snake1d → Conv1d(hidden→1024, k=7, d=3, causal)         │
│   + residual                                              │
│   (optional) CrossAttnBlock with TimbreTokenBank keys     │
├──────────────────────────────────────────────────────────┤
│ vel_proj  CausalConv1d(1024→1024, k=1)                    │
│   zero-initialized at training start (identity init)      │
└──────────────────────────────────────────────────────────┘

Training (Python `converter.py::FlowConverter`):
    v_pred = forward_velocity(z_t, t, z_ref)
    loss   = MSE(v_pred, z_tgt - z_src)   # flow-matching target
           + L1(z_src + v_pred, z_tgt)    # endpoint
           + (1 - cos(spk(z_src+v), spk(z_tgt)))   # speaker sim
           + L1(bottleneck(z_src+v).detach(),
                  bottleneck(z_src))      # content invariance

Inference (Rust `FlowConverter::convert`, 1-NFE):
    v   = forward_velocity(z_src, t=1, z_ref)
    z_out = z_src + velocity_scale * v

`velocity_scale` (>1 amplifies speaker-translation effect, analogous to
classifier-free guidance in diffusion models). Default 1.0, set via
`VcPipeline::velocity_scale`. At 1.0 the output matches the training
objective exactly.
```

`AnyConverter::new(config, vb)` dispatches on `config.model_type`:
`"flow"` → `FlowConverter`, anything else → Phase 1 `Converter`.
`export_weights.py` writes a sidecar `<model>_config.json` recording
`model_type`, `hidden_dim`, `enable_timbre`, etc., which the CLAP/app
loaders consume (see [06-1] config fallback chain).

**Design rationale** (additional):
- **Mean-flow / 1-NFE**: avoids the O(N) cost of Euler integration at
  inference. The network regresses the *time-averaged* velocity, so a
  single evaluation at `t=1` recovers the endpoint. Training is still
  plain flow-matching (no teacher).
- **TimeEmbed**: sinusoidal `exp(-log(10000)·i/d)` frequencies computed
  in f64 then cast to f32, matching PyTorch ([08-5]).
- **Zero-init `vel_proj`**: at step 0 the model is the identity
  (`z_out = z_src`), which stabilises early flow-matching training.
- **Optional UTTE cross-attention**: when `enable_timbre: true`, a
  `TimbreTokenBank` (K=32 learnable tokens) is queried by the speaker
  embedding and the resulting keys/values are injected via
  `CrossAttnBlock` after each residual block. Disabled by default
  ([03-2] pending validation).

### 4.2 Phase 2: Universal Timbre Token Encoder

```
Target reference audio (5-30s)
    │
    ▼
┌──────────────────────────────────────┐
│  Timbre Encoder (frozen DAC encode)   │
│  → reference latent [1024, T_ref]     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Global speaker embedding             │
│  → mean+std statistical pooling        │
│  → MLP → s [256]                      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Timbre Token Bank (K=32, learnable)  │
│  key_i = MLP_k(s)_i + tanh(prior_k_i) │
│  val_i = MLP_v(s)_i + tanh(prior_v_i) │
│  → tokens [32, 256]                   │
└──────────────┬───────────────────────┘
               │
               ▼  cross-attention
┌──────────────────────────────────────┐
│  Converter (Phase 1 + cross-attn)     │
│  At each Conv block, insert:          │
│    Q = proj_q(z_src) [latent→attn]    │
│    K = proj_k(tokens) [embed→attn]    │
│    V = proj_v(tokens) [embed→attn]    │
│    z += CrossAttn(Q, K, V) → proj_o   │
└──────────────────────────────────────┘

CrossAttnBlock uses separate q_dim (latent_dim=1024) and kv_dim
(embed_dim=256) projections, meeting at a shared attn_dim. This
resolves a dimension mismatch between the converter's latent-space
queries and the timbre token bank's embedding-space keys/values.

Additional parameters: ~8-15M (timbre encoder + cross-attn)
Total Phase 2: ~16-27M
```

**Design rationale** (from MeanVC2 UTTE):
- 32 learnable timbre prototypes (priors) shared across speakers — encode breathiness, nasality, brightness.
- Speaker embedding modulates prototypes, not replaces them → robust to low-quality references.
- Cross-attention lets each frame retrieve relevant timbre cues (fine-grained vs global).

### 4.3 Phase 3: Residual-Chain-Preserving RVQ Conversion

Naive RVQ depth control was experimentally rejected. Depths cannot be pasted independently because each codebook quantizes the residual left by previous depths.

```
DAC RVQ: 9 codebooks × 1024 entries

q0:     strongest speaker contribution, content mixed
q1:     strong speaker/timbre contribution
q2:     speaker auxiliary + phonetic detail
q3..8:  residual texture/detail

current best oracle:
  q0_hat = q0_source
  q1..8_hat = RVQ_requantize(z_target_like - q0_source)
```

**Note**: Phase 3 requires the DAC quantizer/re-quantizer encode path in Candle (nearest-neighbor codebook lookup + residual subtraction). Current open question: whether a decoder adapter/tolerant decoder can accept approximate target-like residual latents without hard RVQ cascade failure.

### 4.4 Phase 4 (Optional): Prosody/Rhythm Factorization

```
Source latent
    │
    ├── content path:  low-pass in latent space → linguistic content
    ├── prosody path:  latent residual → F0/energy contour
    └── rhythm path:   frame energy envelope → duration pattern

prosody_mode enum:
  PreserveSource  → keep source prosody, convert timbre only
  Blend           → interpolate prosody between source and target
  ImitateTarget   → replace prosody with target's
  FlattenPrivacy  → normalize prosody (anti-voice-print)
```

---

## 5. Audio Pipeline Detail

### 5.1 Capture → Inference → Playback

```rust
// lightvc-audio/src/stream.rs

pub struct DuplexStream {
    config: StreamConfig,
    capture_tx: Producer<f32>,      // → inference thread
    playback_rx: Consumer<f32>,     // ← inference thread
    capture_stream: cpal::Stream,
    playback_stream: cpal::Stream,
}

impl DuplexStream {
    pub fn start(
        input_device: &Device,
        output_device: &Device,
        sample_rate: u32,
        channels: u16,
        buffer_size: cpal::BufferSize,
    ) -> Result<Self> {
        let (capture_tx, capture_rx) = rtrb::RingBuffer::new(BUFFER_CAPACITY);
        let (playback_tx, playback_rx) = rtrb::RingBuffer::new(BUFFER_CAPACITY);
        
        // Input callback: write samples to ring buffer
        let input_data = move |data: &mut InputBuffer<f32>| {
            for sample in data.iter() {
                let _ = capture_tx.push(*sample);
            }
        };
        
        // Output callback: read samples from ring buffer
        let output_data = move |data: &mut OutputBuffer<f32>| {
            for sample in data.iter_mut() {
                *sample = playback_rx.pop().unwrap_or(0.0);
            }
        };
        
        // Build and play streams...
        Ok(Self { ... })
    }
}
```

### 5.2 Resampling

Device sample rates vary (44.1k, 48k, 96k). DAC requires exactly 44,100 Hz.

```rust
// lightvc-audio/src/resample.rs

pub struct RtResampler {
    input: Async<f32, FixedAsync::Input>,
    output: Async<f32, FixedAsync::Output>,
    // rubato 3.0: zero-allocation process_into_buffer
}

impl RtResampler {
    /// Resample device_sr → 44100 Hz (capture path)
    pub fn process_up(&mut self, input: &[f32], output: &mut [f32]) -> usize { ... }
    
    /// Resample 44100 Hz → device_sr (playback path)
    pub fn process_down(&mut self, input: &[f32], output: &mut [f32]) -> usize { ... }
}
```

### 5.3 Inference Loop

```rust
// lightvc-core/src/pipeline.rs

pub struct VcPipeline {
    encoder: StreamingDacEncoder,
    converter: Converter,
    decoder: StreamingDacDecoder,
    timbre_cache: Option<TimbreTokens>,
    mode: LatencyMode,
}

impl VcPipeline {
    /// Process one chunk of device-rate PCM → device-rate PCM
    pub fn process_chunk(&mut self, pcm_in: &[f32]) -> Result<Vec<f32>> {
        // 1. Resample to 44.1kHz (done by caller or here)
        // 2. Encode to continuous latent
        let latent = self.encoder.encode_step(pcm_in)?;
        
        // 3. Convert (one-step, no ODE loop)
        let converted = self.converter.forward(
            &latent,
            self.timbre_cache.as_ref(),
        )?;
        
        // 4. Decode back to PCM
        let pcm_out = self.decoder.decode_step(&converted)?;
        
        Ok(pcm_out)
    }
    
    /// Set target voice from reference audio
    pub fn set_target(&mut self, reference_pcm: &[f32]) -> Result<()> {
        let ref_latent = self.encoder.encode_full(reference_pcm)?;
        let tokens = self.converter.compute_timbre_tokens(&ref_latent)?;
        self.timbre_cache = Some(tokens);
        Ok(())
    }
}
```

---

## 6. Weight Loading

### 6.1 DAC Weights

```rust
// Download from HuggingFace Hub
let repo = "descript/dac_44khz";
let api = hf_hub::api::sync::Api::new()?;
let repo = api.model(repo.to_string());
let weights = repo.get("model.safetensors")?;
let config = repo.get("config.json")?;

// Load into Candle
let vb = candle_nn::VarBuilder::from_mmaped_safetensors(
    &[weights], candle::DType::F32, device
)?;

// NOTE: HF safetensors key naming may differ from Candle dac.rs expectations.
// May need key remapping. See Section 6.3.
let model = dac::Model::new(&config, vb)?;
```

### 6.2 Converter Weights

```rust
// Our trained converter
let weights = Path::new("models/converter_p2.safetensors");
let vb = candle_nn::VarBuilder::from_mmaped_safetensors(
    &[weights], candle::DType::F32, device
)?;
let converter = Converter::load(vb, &converter_config)?;
```

### 6.3 Known DAC Weight Key Issue

The HuggingFace `descript/dac_44khz` safetensors uses **transformers library naming** (e.g., `model.encoder.block.0...`), while Candle's `dac.rs` expects **original PyTorch naming** (e.g., `encoder.block.0.conv.weight_g...` with weight-norm decomposition).

**Options**:
1. **Key remapping**: Build a key mapping table and remap at load time. ~1 hour of work.
2. **Use original weights**: Download `weights.pth` from DAC GitHub releases, load via Candle's pickle reader (`candle_core::pickle`).
3. **Modify dac.rs**: Adjust Candle's key paths to match HF naming. Submit upstream PR.

**Recommended**: Option 2 (original weights) for fastest path. Option 1 for long-term HF compatibility.

---

## 7. Performance Considerations

### 7.1 CPU vs GPU

| Component | CPU (F32) | CPU (F16) | GPU (CUDA) | GPU (Metal) |
|-----------|-----------|-----------|------------|-------------|
| DAC encode (86Hz) | ~5 ms | ~3 ms | ~1 ms | ~2 ms |
| Converter (10M) | ~5 ms | ~3 ms | ~1 ms | ~1 ms |
| DAC decode (86Hz) | ~8 ms | ~5 ms | ~2 ms | ~3 ms |
| **Total** | **~18 ms** | **~11 ms** | **~4 ms** | **~6 ms** |

All well within real-time budget at 86 Hz frame rate.

### 7.2 Device Selection

```toml
# Feature flags for Candle backend
[features]
default = ["cpu"]
cpu = ["candle-core/mkl"]           # Intel MKL on x86
accelerate = ["candle-core/accelerate"]  # Apple Accelerate
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
```

Runtime selection:
```rust
let device = if cfg!(feature = "cuda") && cuda_available() {
    Device::new_cuda(0)?
} else if cfg!(feature = "metal") {
    Device::new_metal(0)?
} else {
    Device::Cpu  // with MKL/Accelerate BLAS
};
```

### 7.3 Memory

| Component | Memory |
|-----------|--------|
| DAC encoder weights | ~60 MB (F32) |
| DAC decoder weights | ~60 MB (F32) |
| DAC conv states (streaming) | ~2 MB |
| Converter (Phase 2, 20M) | ~80 MB (F32) |
| Timbre token cache | ~32 KB |
| Ring buffers | ~1 MB |
| **Total** | **~200 MB** |

Acceptable for a desktop application. Can halve with F16 weights.

---

## 8. Error Handling and Edge Cases

| Scenario | Handling |
|----------|----------|
| Ring buffer underrun (capture too fast) | Drop oldest samples, log warning |
| Ring buffer overrun (inference too slow) | Output silence for dropped chunk, log warning, auto-switch to lower quality mode |
| Target reference too short (<1s) | Reject with error message in UI |
| Silence detection (no input) | Skip encode/convert/decode, output silence (save CPU) |
| DAC decode NaN/Inf (divergence) | Clamp to [-1, 1], log warning |
| Device disconnection | Stop pipeline, return to device selection screen |
