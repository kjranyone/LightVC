# ASIO Setup (Optional)

ASIO is an optional low-latency audio driver for Windows. **Not required** — WASAPI works by default (10-30ms latency). ASIO achieves <5ms latency.

## License

ASIO support does **not** affect the project's MIT license:
- `cpal` crate: MIT
- `asio-sys` (cpal feature): uses Zlib-style bindings to Steinberg ASIO SDK headers
- ASIO SDK itself: proprietary (free download, no redistribution of headers)

Using ASIO does not make the project GPL. The VST3 SDK GPLv3 issue (via `vst3-sys`) has been resolved — we use `clap-wrapper` with the MIT-licensed Steinberg VST3 SDK (2025+).

## 1. Download the ASIO SDK

1. Visit https://www.steinberg.net/asiosdk
2. Download and extract, e.g., `C:\ASIOSDK`

## 2. Build with ASIO

```bash
# Set the SDK path
export CPAL_ASIO_DIR="C:/ASIOSDK"

# Build standalone app with ASIO
cargo build --release --features asio -p lightvc-app

# Run
./target/release/lightvc-app.exe gui --dac-weights models/dac_44khz.safetensors
# Realtime tab → Audio Devices → ASIO devices appear
```

## 3. Verify

In the GUI's **Realtime** tab, expand **Audio Devices**. ASIO interfaces (e.g., "ASIO4ALL", "Focusrite USB ASIO") appear alongside WASAPI devices when built with `--features asio`.

## Notes

- ASIO is **Windows-only** and **optional**.
- Do not commit the ASIO SDK to the repository (license prohibits redistribution).
- The ASIO SDK must not be bundled in release binaries.
