# CLAUDE.md

Guidance for working in this repository. Each subproject has its own `CLAUDE.md`
with the details specific to it.

## What this is

A full-stack surface-EMG gesture system: ESP32-S3 firmware, host-side ML that
trains and shrinks the gesture model, a browser dashboard, and a pose service.
`README.md` has the full layout.

## Repository structure

There is no shared Cargo workspace. Every Rust project has its own `Cargo.toml`
with an empty `[workspace]` table so cargo treats it as its own root. Always build
and run from inside the relevant subproject directory, never from the repo root.

The active subprojects are `ble-media`, `drv2605l`, and `opal-firmware` on the
firmware side; `emg-tds`, `dashboard`, `pose-service`, and `protocol` on the host
side; `emg-runtime`, which the firmware consumes but which also builds on the
host; and `engineering-logs` for the written record. The verified but deferred
OTA client/server proof lives under `experiments/ota/`. Prefer a module in an
existing active crate over a new crate.

## Two toolchains

The active firmware projects (`ble-media`, `drv2605l`, `opal-firmware`) target the
Xtensa ESP32-S3. Their `rust-toolchain.toml` pins `channel = "esp"`, and every
build shell must first run `. ~/export-esp.sh`. The one-time setup (espup,
espflash, ldproxy, and on Arch a `libxml2.so.2` compat symlink) is in
`FIRMWARE-SETUP.md`. These projects build for real hardware over
USB-Serial-JTAG, so you cannot run them on this host.

The active host projects (`emg-tds`, `dashboard`, `protocol`)
use ordinary stable or nightly Rust. `pose-service` is Python.

## Build and test commands

For a host Rust project, run `cargo build`, `cargo run`, or `cargo test` from its
directory. Unit tests live in `protocol` (`no_std` plus `alloc`; `cargo test` on
the host) and in `opal-firmware`; `cargo test-device` runs the firmware tests on
the board. From `emg-runtime`, `cargo +esp test-device` runs its device-only SIMD
example. Both device commands flash a libtest image and report through espflash. For ML training in `emg-tds` on GPU, build with
`--features cuda` and set `CUDARC_CUDA_VERSION=13020`, since CUDA 13.3 is
ABI-compatible with cudarc's 13.2 target but auto-detect rejects 13.3; CPU works
without the feature. Firmware uses `cargo run` to flash and open the serial
monitor. The dashboard frontend uses pnpm (`pnpm install`, `pnpm run check`,
`pnpm run build`); npm is not installed on this machine.

## Conventions

Spell things out. Identifiers carry no acronyms or abbreviations, a deliberate
past cleanup, so match it.

The wire protocol lives in `protocol/`. Cross-process messages are CBOR frames
defined once there (ciborium on Rust, cbor-x in the browser, cbor2 in Python), and
bulk EMG samples ride as a little-endian `i16` byte blob rather than a number
array. Change a frame in one place and update the other two consumers.

Keep the repo path free of spaces, which ESP-IDF requires.

No temporal or voting hacks in the model. Past work was ruined by leaning on
temporal voting as a crutch, so fix the model instead. The reject pipeline's
3-of-3 smoothing is a deliberate, documented decision spine and not an accuracy
patch, so keep that distinction.

The collected gesture set is a hardware workaround, not a design. The five
classes in `dashboard/config/collection.json` are large motions picked to clear
the current analog front end's noise floor. The product wants subtler gestures —
ones that separate from ordinary skiing motion, which these do not. When the
front end is fixed, revisit the set; do not treat the current one as settled.

## Where the reasoning lives

`engineering-logs/` records why the ML and on-device kernels turned out the way
they did. Before you change model architecture, the negative-class handling, the
augmentation, or the SIMD kernels, check whether a log already tried it.
