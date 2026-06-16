# ASIO Setup

ASIO is an optional low-latency audio driver for Windows.
To build with ASIO support, the Steinberg ASIO SDK is required.

## 1. Download the ASIO SDK

1. Visit https://www.steinberg.net/asiosdk
2. Download the ASIO SDK (free, but requires registration)
3. Extract to a local directory, e.g., `C:\ASIOSDK`

## 2. Build with ASIO

```bash
# Set the SDK path
export CPAL_ASIO_DIR="C:/ASIOSDK"

# Build with the asio feature
cargo build --release --features asio -p lightvc-app
```

## 3. Verify ASIO devices

```bash
./target/release/lightvc-app.exe gui --dac-weights models/dac_44khz.safetensors
# Realtime tab → Audio Devices → should show ASIO devices
```

## Notes

- ASIO is Windows-only and optional. Without `--features asio`, WASAPI is used.
- The ASIO SDK license prohibits redistribution of the SDK headers.
  Do not commit the SDK to the repository.
