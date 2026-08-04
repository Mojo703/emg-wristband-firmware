# EMG Wristband

A full-stack surface-EMG gesture system (capstone S1). It spans firmware for an
ESP32-S3 wristband, the host-side machine learning that trains and shrinks the
gesture model, a browser dashboard for inspecting the pipeline, and a
pose-inference service. The end goal is a wristband that reads forearm muscle
signals, works out which gesture you made, and acts on it. For now the action is
driving an iPhone's media controls over BLE.

Each subproject stands on its own. Every one has its own `Cargo.toml` with an
empty `[workspace]` table and its own toolchain, so team members can work in
parallel and integrate later. There is deliberately no shared Cargo workspace.

> Keep this tree at a path without spaces. ESP-IDF refuses to build under any path
> containing a space.

## Layout

Firmware (ESP32-S3, Xtensa toolchain):

| Path | What it is |
|------|------------|
| [`opal-firmware/`](opal-firmware) | The device. Two ADS1298s sample 16 channels on their own thread, the int8 model classifies each window, and the result streams to the dashboard over wifi or USB serial. `BRINGUP.md` is the bench procedure for new hardware. |
| [`drv2605l/`](drv2605l) | TI DRV2605L haptic driver over I2C, plus a bench binary that plays candidate wristband feedback patterns. The library is `embedded-hal` generic, so the firmware can take it without the bench setup. |
| [`ota-client/`](ota-client) | OTA update client. Pulls a firmware image over HTTP and reboots into it. Not yet folded into `opal-firmware`, which still boots from a single app partition. |
| [`ota-server/`](ota-server) | Dev-machine HTTP server that hosts firmware `.bin` images for the client. Plain host Rust, no ESP toolchain. |
| [`ble-media/`](ble-media) | BLE HID media remote. Bonds with an iPhone and sends media keys. Serial-driven for bring-up, gesture-driven later. |

Machine learning and tooling (host, plain Rust or Python):

| Path | What it is |
|------|------------|
| [`emg-tds/`](emg-tds) | The current gesture model: a depthwise-separable (TDS) conv encoder in Rust/candle, with swappable classifier and pose heads. Trains, evaluates, and exports. The `.npy` training/eval/pose windows live in its `data/` directory. |
| [`dashboard/`](dashboard) | Web dashboard. An axum backend relays CBOR frames between the wristband and a Svelte frontend, and runs the training-data collection game: import a Beat Saber map, play the falling-notes track, record labelled EMG. |
| [`pose-service/`](pose-service) | Python WebSocket service that turns EMG windows into 3-D hand pose. Runs a mock estimator or Meta's `emg2pose` model. |
| [`emg-runtime/`](emg-runtime) | The on-device inference path: int8 kernels with hand-written ESP32-S3 SIMD and a scalar fallback off target, plus the reject pipeline. Builds on the host too. `data/` holds the exported model blobs. |
| [`protocol/`](protocol) | A `no_std` crate of the CBOR frame types shared by the dashboard backend, the browser, and the firmware. |
| [`engineering-logs/`](engineering-logs) | Dated log of the ML and on-device optimisation work, one goal, method, measurement, and analysis per entry. Read it first to learn why the model is what it is. |

## Conventions

No shared workspace. Every Cargo project carries an empty `[workspace]` table so
cargo treats it as its own root and does not try to attach it to a parent. Build
each project from inside its own directory.

Two toolchains. The firmware projects (`ota-client`, `ble-media`, `drv2605l`,
`opal-firmware`) use
the Espressif Rust fork; their `rust-toolchain.toml` pins `channel = "esp"`, and
every build shell must first source `. ~/export-esp.sh`. The host projects build
with ordinary stable or nightly Rust.

The dashboard frontend uses pnpm, not npm. Identifiers are spelled out, with no
acronyms or abbreviations. Cross-process messages are CBOR frames defined once in
`protocol/`, and bulk EMG rides as a little-endian `i16` byte blob.

## Development environment

| Tool | Version |
|------|---------|
| OS | EndeavourOS (Arch Linux) |
| Kernel | 7.0.12-arch1-1 |
| CPU | AMD Ryzen 9 7900X |
| GCC / G++ | 16.1.1 |
| Clang | 22.1.6 |
| CMake | 4.3.3 |
| GNU Make | 4.4.1 |
| Rust (`rustc`) | 1.95.0-nightly |
| Cargo | 1.95.0-nightly |

## Target hardware

ESP32-S3-Zero (Waveshare): 4 MB flash, no PSRAM, native USB (USB-Serial-JTAG over
USB-C). Nothing here is board-pinned beyond the 4 MB partition tables, so a move to
a custom board later stays contained.

## Where to start

For firmware bring-up and OTA, read [`ota-client/README.md`](ota-client/README.md).
It has the one-time Xtensa toolchain setup that every firmware project shares,
including the Arch `libxml2` workaround, plus the end-to-end OTA verification.

For the model and its history, read [`engineering-logs/`](engineering-logs) for the
decisions and [`emg-tds/`](emg-tds) for the code.

To see it run without hardware, use [`dashboard/`](dashboard): run `./run.sh` and
open <http://localhost:8090>.
