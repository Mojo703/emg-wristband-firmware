# ota-client

ESP32-S3 OTA update client, built on `std` / `esp-idf-svc`.

On boot it: marks the running image valid (rollback protection), connects to
WiFi, downloads a firmware image over HTTP, writes it to the inactive OTA slot,
sets that slot as the boot partition, and reboots into it.

## Layout

- `src/main.rs`   — boot flow + `FW_VERSION` (bump to prove an update happened)
- `src/wifi.rs`   — WiFi station bring-up
- `src/ota.rs`    — HTTP download + flash-and-switch via `EspOta`
- `src/config.rs` — compile-time config from `cfg.toml`
- `partitions.csv`— dual OTA app slots for 4 MB flash
- `sdkconfig.defaults` — ESP-IDF options (custom partitions, rollback, stack)

## How it works

### Flash layout

`partitions.csv` carves the 4 MB flash into two equal app slots plus the
bookkeeping the bootloader needs:

| Partition | Offset | Size | Purpose |
|-----------|--------|------|---------|
| `nvs`      | 0x09000 | 24 KB | WiFi calibration / key-value store |
| `otadata`  | 0x0f000 | 8 KB  | which app slot the bootloader should boot |
| `phy_init` | 0x11000 | 4 KB  | RF data |
| `ota_0`    | 0x20000 | ~1.875 MB | app slot A |
| `ota_1`    | 0x200000 | ~1.875 MB | app slot B |

There is no `factory` partition. At first boot `otadata` is blank, so the
bootloader runs `ota_0`. Each update is written to whichever slot is *not*
running, then `otadata` is flipped to point at it. The current release app is
~1.1 MB, so it fits a slot with room to spare.

### Boot + update flow (`main.rs`)

1. `mark_running_slot_valid()` — confirms the image that's currently running so
   the bootloader's rollback protection won't revert it on the next reset.
2. Connect to WiFi (`wifi.rs`), block until an IP is assigned.
3. `ota::run_update()` (`ota.rs`):
   - HTTP `GET` the image from `ota_url`. Non-200 (e.g. 404) aborts cleanly and
     the device stays on the current firmware.
   - Stream the body in 4 KB chunks straight into `EspOta`'s update handle,
     which writes them to the inactive slot.
   - `complete()` finalizes the image and flips `otadata` to the new slot.
4. Reboot. The bootloader now loads the new slot; the boot banner prints the new
   `FW_VERSION` and `running from partition 'ota_1'` (or `ota_0`).

The image served by `ota-server` is the **app image only** (from `espflash
save-image`), not a full flash dump — `EspOta` drops it into an app partition,
so the bootloader and partition table on the device are untouched by an update.

### Current limitations (next hardening steps, not bugs)

- **Unconditional update.** The client downloads and applies on *every* boot, so
  once both slots hold the same version it ping-pongs between them on each reset.
  A version/manifest check (only update when the server is newer) is the fix.
- **WiFi failure exits.** `wifi::connect(...)?` propagates errors, so a failed
  association ends `app_main` rather than idling and retrying.

## One-time toolchain setup

The ESP32-S3 is an Xtensa core and needs the forked Rust toolchain. Run these
in your shell (not committed; they modify `~/.cargo` and `~/.rustup`):

```sh
cargo install espup --locked
espup install
cargo install espflash --locked
cargo install ldproxy --locked
. $HOME/export-esp.sh   # source in every shell that builds this project
```

### Arch Linux: libxml2 workaround

Espressif's bundled `esp-clang` (an ESP-IDF tool) links against the old
`libxml2.so.2` soname. Arch ships `libxml2.so.16`, so the IDF tool install
aborts with `libxml2.so.2: cannot open shared object file`. After the first
build downloads the toolchain into `.embuild/`, add a compat symlink (the ABI
difference is harmless for the version check `esp-clang` performs):

```sh
ln -sf /usr/lib/libxml2.so.16 \
  .embuild/espressif/tools/esp-clang/*/esp-clang/lib/libxml2.so.2
```

Then re-run `cargo build`. (`.embuild/` is gitignored and re-created on a fresh
checkout, so each machine needs this once after the initial download.)

## Version stack

Pinned to the current esp-rs line so it matches the freshly-installed Xtensa
toolchain (older `esp-idf-svc` hit `c_char` signedness errors against it):
`esp-idf-svc 0.52`, `esp-idf-sys 0.37`, `embedded-svc 0.29`, `embuild 0.33`,
ESP-IDF `v5.2.2`.

## Configure

Edit `cfg.toml`:
- `wifi_ssid` / `wifi_psk` — your 2.4 GHz network (ESP32 has no 5 GHz radio)
- `ota_url` — `http://<dev-machine-LAN-IP>:8080/firmware/ota-client.bin`

## Build, flash, monitor

Plug the board in over USB-C, then:

```sh
. $HOME/export-esp.sh
cargo run            # builds, flashes over USB-Serial-JTAG, opens monitor
```

The flash runner in `.cargo/config.toml` is
`espflash flash --monitor --partition-table partitions.csv`. The
`--partition-table` flag is **required**: without it espflash writes its own
default single-`factory` table and the device has no OTA slots (you'd see
`not found otadata` and `running from partition 'factory'`). In the monitor,
`CTRL+R` resets the chip and `CTRL+C` exits.

## Producing an image for OTA

The server serves a raw application image (not the ELF). Generate one with
`espflash`:

```sh
cargo build --release
espflash save-image --chip esp32s3 \
  target/xtensa-esp32s3-espidf/release/ota-client \
  ../ota-server/firmware/ota-client.bin
```

## End-to-end verification

This is the procedure that was used to confirm the system works on an
ESP32-S3-Zero. It builds up in stages so a failure points at one layer.

1. **Server.** From `../ota-server`, start it and leave `firmware/` empty:
   ```sh
   rm -f firmware/*.bin && cargo run        # binds 0.0.0.0:8080
   ```
   Open `8080` to the LAN if a host firewall is running (firewalld on Arch
   silently drops the inbound connection, which the device sees as a connect
   timeout, not a 404):
   ```sh
   sudo firewall-cmd --add-port=8080/tcp    # runtime rule, clears on reboot
   ```
   Sanity check from another machine on the WiFi: `curl http://<dev-ip>:8080/`.

2. **Flash v1 + check WiFi/HTTP.** Set `FW_VERSION = "1.0.0"`, `cargo run`. With
   `firmware/` empty you should see WiFi associate, get an IP, then
   `OTA failed: ... HTTP 404 ...; staying on v1.0.0`. That 404 (not a timeout)
   proves the device reaches the server, and that it survives a failed update.

3. **Publish v2 and update over the air.** Bump `FW_VERSION` to `"1.0.1"`, then:
   ```sh
   cargo build --release
   espflash save-image --chip esp32s3 \
     target/xtensa-esp32s3-espidf/release/ota-client \
     ../ota-server/firmware/ota-client.bin
   ```
   Do **not** `cargo run` — leave the board on v1.0.0. Reset it (`CTRL+R`). It
   downloads the image, writes the inactive slot, and reboots. The proof is the
   banner going `v1.0.0 → v1.0.1` with the slot flipping `ota_0 → ota_1`, all
   without re-flashing. Power off afterward (see "unconditional update" above).
