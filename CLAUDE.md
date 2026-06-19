# CLAUDE.md

Rust firmware for the sEMG gesture-recognition wristband (ENSC 405W capstone,
subsystem S1). This repo is the firmware side; the ML codebase lives separately
at `~/Documents/Projects/emg-gesture-class/`.

## Layout

Two **independent** cargo projects — no shared workspace, so two firmware members
can work in parallel and merge at an integration phase. Each has an empty
`[workspace]` table in its `Cargo.toml` so cargo doesn't adopt them into an
unrelated parent manifest.

- `ota-client/` — ESP32-S3 firmware (`std` / `esp-idf-svc`). WiFi → HTTP download
  → write inactive OTA slot → reboot. Modules: `main`, `wifi`, `ota`, `config`.
- `ota-server/` — minimal `axum` static host for firmware `.bin` images. Plain
  `x86_64` Rust, runs on the dev machine. No ESP toolchain needed.

The OTA system is the first firmware milestone: a tested update path so later
work ships faster. Build it modular and extensible. **Verified end-to-end on
hardware** (2026-06-19): a v1.0.0 image OTA-updated to v1.0.1 over WiFi, flipping
both the version banner and the active flash slot with no re-flash.

## How OTA works (short version)

`partitions.csv` splits the 4 MB flash into `otadata` + two equal ~1.875 MB app
slots (`ota_0`, `ota_1`), no `factory`. The bootloader boots the slot named in
`otadata`. On boot the client marks the running slot valid (rollback
protection), connects to WiFi, HTTP-GETs the app image from `ota-server`, streams
it via `EspOta` into the *inactive* slot, flips `otadata`, and reboots into it.
The served `.bin` is the app image only (`espflash save-image`); the bootloader
and partition table on the device are not touched by an update. Full detail and
the staged verification procedure live in `ota-client/README.md`.

Two known gaps to close before this is production-grade: the client updates
unconditionally every boot (no version gating, so it ping-pongs slots), and a
WiFi connect failure exits `app_main` instead of retrying.

## Hardware

ESP32-S3-Zero (Waveshare), 4 MB flash, native USB (USB-Serial-JTAG over USB-C),
2.4 GHz WiFi only. Moving to a custom board later; nothing is board-pinned beyond
`ota-client/partitions.csv` (sized for 4 MB).

## Building

`ota-server`: `cd ota-server && cargo run`.

`ota-client` needs the Xtensa toolchain — `. ~/export-esp.sh` in every build
shell, then `cargo run` (flashes + monitors) or `cargo build [--release]`. Full
setup and the OTA verification flow are in `ota-client/README.md`.

### Environment gotchas (hard-won; don't relearn them)

- **No spaces in the path.** ESP-IDF aborts on build dirs containing spaces. Keep
  this tree somewhere space-free.
- **Arch + libxml2.** Espressif's `esp-clang` tool wants `libxml2.so.2`; Arch
  ships `.so.16`. After the first build downloads `.embuild/`, symlink:
  `ln -sf /usr/lib/libxml2.so.16 ota-client/.embuild/espressif/tools/esp-clang/*/esp-clang/lib/libxml2.so.2`
  Needed once per machine (`.embuild/` is gitignored).
- **Custom partition table (build).** esp-idf-sys runs cmake from a generated
  dir, so the CSV path uses `$ENV{PROJECT_DIR}` (see
  `ota-client/sdkconfig.defaults`), not a relative path.
- **Custom partition table (flash).** The cargo runner passes
  `--partition-table partitions.csv`; without it espflash writes a default
  single-`factory` table and OTA has no slots to write to.
- **Host firewall.** The server listens on `0.0.0.0:8080`, but firewalld (active
  on this Arch box) drops inbound LAN connections, which the device sees as a
  connect timeout. Open it: `sudo firewall-cmd --add-port=8080/tcp`.
- **Version stack** (matches the current Xtensa toolchain; older `esp-idf-svc`
  fails with `c_char` errors): esp-idf-svc 0.52, esp-idf-sys 0.37,
  embedded-svc 0.29, embuild 0.33, ESP-IDF v5.2.2.

## Conventions

- `cfg.toml` in `ota-client/` holds WiFi credentials in plaintext — gitignored,
  never commit real secrets.
- Do **not** run `git commit`/`push` without being asked.
- Match the surrounding code's style; comments explain *why*, not *what*.
