# EMG Wristband — Firmware

Rust firmware for the sEMG gesture-recognition wristband (capstone S1). Each
project here is self-contained (no shared workspace) so team members can develop
in parallel and merge at the integration phase.

The first milestone is the OTA update system below, to make all later firmware
work faster to ship.

> Path note: keep this tree at a location **without spaces** in the path.
> ESP-IDF refuses to build under paths containing spaces.

## Projects

- **`ota-client/`** — ESP32-S3 firmware (std / `esp-idf-svc`). Connects to WiFi,
  pulls a firmware image over HTTP, writes it to the inactive OTA slot, and
  reboots into it. Uses the ESP-IDF `esp_https_ota`-style flow via `EspOta`.
- **`ota-server/`** — Minimal `axum` HTTP server (runs on your dev machine) that
  hosts firmware `.bin` images for the client to download.
- **`ble-media/`** — ESP32-S3 BLE HID media remote (`esp32-nimble`). Bonds with
  an iPhone and sends media keys (play/pause, next, volume); serial-console
  driven for bring-up, gesture-driven later. See `ble-media/README.md`.

## Target hardware

- ESP32-S3-Zero (Waveshare), 4 MB flash, native USB (USB-Serial-JTAG over USB-C).
- Will move to a custom board later; nothing here is board-pinned beyond the
  partition table (`ota-client/partitions.csv`) sized for 4 MB flash.

## How it works

The 4 MB flash holds two equal app slots (`ota_0`, `ota_1`) plus an `otadata`
region that records which slot to boot. The client connects to WiFi, downloads
an app image from `ota-server` over HTTP, writes it to whichever slot isn't
running, flips `otadata`, and reboots into it. The boot banner prints
`FW_VERSION` and the active slot, so a successful update shows both changing.
Full detail is in `ota-client/README.md`.

Status: **verified end-to-end on hardware** — v1.0.0 updated to v1.0.1 over WiFi,
slot `ota_0 → ota_1`, no re-flash.

## End-to-end verification (the goal of this first milestone)

Summary; the exact commands and expected logs are in `ota-client/README.md`.

1. Install the Xtensa toolchain and apply the Arch `libxml2` symlink
   (`ota-client/README.md`).
2. Start the server (`cd ota-server && cargo run`) with `firmware/` empty, and
   open port 8080 if a host firewall is running
   (`sudo firewall-cmd --add-port=8080/tcp`).
3. Fill in `ota-client/cfg.toml` (2.4 GHz SSID/PSK + `ota_url` with the dev
   machine's LAN IP).
4. Flash v1 (`cd ota-client && cargo run`). Empty server → device logs a clean
   `HTTP 404` and stays on v1.0.0: WiFi + HTTP + graceful-failure confirmed.
5. Bump `FW_VERSION`, `cargo build --release`, `espflash save-image` the `.bin`
   into `ota-server/firmware/`, then reset the board (don't re-flash). It updates
   over the air and reboots into the new version.
