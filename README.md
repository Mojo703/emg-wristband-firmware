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
| [`opal-firmware/`](opal-firmware) | The device. Two ADS1298s sample 16 channels, each chip's frame is clocked out inside its own data-ready interrupt, and the wearer-calibrated 64-feature linear classifier drives predictions and BLE commands. Wi-Fi support remains available behind an explicit future mode transition. `BRINGUP.md` is the bench procedure for new hardware. |
| [`drv2605l/`](drv2605l) | TI DRV2605L haptic driver over I2C, plus a named bench example that plays candidate wristband feedback patterns. The library is `embedded-hal` generic, so the firmware can take it without the bench setup. |
| [`ble-media/`](ble-media) | BLE HID media remote. The named bench example is serial-driven; `opal-firmware` dispatches calibrated gesture commits through the same library. |

Machine learning and tooling (host, plain Rust or Python):

| Path | What it is |
|------|------------|
| [`emg-tds/`](emg-tds) | Historical depthwise-separable convolution experiments and export tooling. The production firmware no longer runs this model. The `.npy` training/eval/pose windows live in its `data/` directory. |
| [`dashboard/`](dashboard) | Web dashboard. An axum backend relays CBOR frames between the wristband and a Svelte frontend, and runs the training-data collection game: import a Beat Saber map, play the falling-notes track, record labelled EMG. |
| [`pose-service/`](pose-service) | Python WebSocket service that turns EMG windows into 3-D hand pose. Runs a mock estimator or Meta's `emg2pose` model. |
| [`emg-runtime/`](emg-runtime) | Shared on-device classification arithmetic: filter-bank log-power features, wearer-model fitting/scoring, the reject pipeline, and the grid aligner. Historical int8 TDS kernels and blobs remain for reproducibility but are not linked into the production inference path. |
| [`protocol/`](protocol) | A `no_std` crate of the CBOR frame types shared by the dashboard backend, the browser, and the firmware. |
| [`engineering-logs/`](engineering-logs) | Dated log of the ML and on-device optimisation work, one goal, method, measurement, and analysis per entry. Read it first to learn why the model is what it is. |

Deferred experiments:

| Path | What it is |
|------|------------|
| [`experiments/ota/`](experiments/ota) | Verified dual-slot OTA client/server proof. It is archived for later integration and is not part of `opal-firmware`. |

## Conventions

No shared workspace. Every Cargo project carries an empty `[workspace]` table so
cargo treats it as its own root and does not try to attach it to a parent. Build
each project from inside its own directory.

Two toolchains. The active firmware projects (`ble-media`, `drv2605l`,
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
| Kernel | 7.1.4-arch1-1 |
| CPU | AMD Ryzen 9 7900X |
| GCC / G++ | 16.1.1 |
| Clang | 22.1.8 |
| CMake | 4.4.2 |
| GNU Make | 4.4.1 |
| Rust (`rustc`) | 1.96.0 stable, 1.96.0-nightly |
| Cargo | 1.96.0 stable, 1.96.0-nightly |

## Target hardware

ESP32-S3-Zero (Waveshare): 4 MB flash, no PSRAM, native USB (USB-Serial-JTAG over
USB-C). What ties the tree to this board is the 4 MB partition tables and, in
`opal-firmware`, the pin map in `src/main.rs` together with the two-board
`Board` enum the thread and interrupt placements in `src/cores.rs` are exhaustive
over. A move to a custom board goes through those and nothing else.

## Where to start

New to the project? [`ONBOARDING.md`](ONBOARDING.md) is the ordered path: the
dashboard, a track, the firmware toolchain, and how to run and check a recording
session. The rest of this section is the reference material it links to.

For the shared Xtensa toolchain and Arch `libxml2` workaround, read
[`FIRMWARE-SETUP.md`](FIRMWARE-SETUP.md). Opal bring-up is in
[`opal-firmware/BRINGUP.md`](opal-firmware/BRINGUP.md); the deferred OTA proof and
its end-to-end verification are in [`experiments/ota/`](experiments/ota).

For end-to-end hardware validation, follow
[`firmware-bench/TESTING.md`](firmware-bench/TESTING.md). Keep the one-page
[`firmware-bench/DONNING.md`](firmware-bench/DONNING.md) checklist beside the wearer.

For the current calibrated model and its history, read
[`firmware-bench/ARITHMETIC.md`](firmware-bench/ARITHMETIC.md) and
[`engineering-logs/`](engineering-logs). `emg-tds/` contains the superseded fixed-model work.

To see it run without hardware, use [`dashboard/`](dashboard): run `./run.sh` and
open <http://localhost:8090>.
