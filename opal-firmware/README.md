# opal-firmware

Firmware for the Opal EMG wristband (ESP32-S3): a fake provider replays embedded
Hyser windows, the int8 model classifies each window, the reject pipeline smooths
it into a wake-gate decision, and the frames stream to the dashboard over wifi or
USB serial. `src/main.rs` has the module-level docs.

## Building and flashing

This crate targets Xtensa and needs the esp toolchain; the one-time setup lives in
`../ota-client/README.md`. Every shell must source the exports first:

```sh
. ~/export-esp.sh
cargo run    # builds, flashes over USB-Serial-JTAG, opens the serial monitor
```

Compile-time defaults (wifi credentials, server address) come from `cfg.toml`,
which stays out of version control. Values persisted to NVS by the dashboard
override it.

## On-device tests

The unit tests (currently the `link_policy` module) run on the device, because the
crate only builds for Xtensa:

```sh
cargo test-device
```

This is an alias (see `.cargo/config.toml`) for `cargo test` with
`ESP_IDF_SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.test"`. The override
matters: the normal firmware disables the console because the USB channel carries
the framed CBOR dashboard link, so a plain `cargo test` runs the tests but shows
no output. `sdkconfig.test` re-enables the USB-Serial-JTAG console for the libtest
report.

What to expect:

- espflash flashes the test binary and opens the monitor; watch for
  `test result: ok`. The run does not exit on its own — Ctrl+C when done.
- The test binary replaces the firmware; `cargo run` afterwards restores it.
- Switching between test and normal builds regenerates the esp-idf config, which
  costs a partial rebuild (roughly half a minute) in each direction.
