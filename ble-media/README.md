# ble-media

An ESP32-S3 BLE media remote. It advertises as a BLE HID-over-GATT consumer-control
device, bonds with a host (iPhone, also macOS/Android/Windows), and sends media
keys: play/pause, next, previous, volume, mute. Built on `std` / `esp-idf-svc`
with the [`esp32-nimble`](https://github.com/taks/esp32-nimble) NimBLE wrapper.

The keys are driven from the serial console here. The wearer firmware drives the
same `Phone` state machine from its feedback thread, with a dashboard button in
place of the console toggle.

## How it works

- **`src/phone.rs` is the state machine, and it runs on a laptop.** The BLE stack
  comes up at boot and stays resident; the toggle only starts and stops
  advertising. Every boot pays NimBLE's footprint whether the phone is used or
  not, and that is the point: the one large allocation lands on a fresh heap
  rather than at a button press after hours of fragmentation, so a device that
  cannot afford it fails at boot, visibly, instead of mid-session. `Dormant` and
  `Standby` therefore mean *not advertising*, never *no stack*. Media keys go out
  only on an *encrypted* link, not merely a connected one: iOS reads the report
  map before it encrypts and discards anything sent in that window.
- **The single radio is not arbitrated yet.** A wifi-provisioned device refuses
  the toggle with a reason rather than half-enabling a phone that cannot work —
  standing the wifi dialer down never released the radio, since the station stays
  associated and only the dialling loop skips.
- `src/nimble.rs` is the only file that touches esp32-nimble, behind the
  `phone::Radio` trait. Its dependencies are gated on `cfg(target_os = "espidf")`,
  so `cargo test --manifest-path ble-media/Cargo.toml` (from the repo root) runs
  the state machine's tests on the host.
- A HID service (`0x1812`) exposes one Consumer Control input report carrying a
  16-bit usage code (`src/hid.rs`). Sending a usage = press; sending `0x0000` =
  release. The release rides a later `Phone::tick` rather than a 20 ms sleep, so
  no thread blocks on it.
- The device advertises with a HID-keyboard appearance and bonds on first
  connect. iOS only delivers HID input over an encrypted/bonded link, so bonding
  (`Bond | Sc`, "just works" pairing — the band has no display or keypad, so it
  does not ask for MITM protection it cannot deliver) is required. Bonding keys
  are persisted in NVS, so the pairing survives reboots. Anything in range can
  bond and nothing can forget a bonded phone yet.
- The media keys themselves are `protocol::MediaKey`: Play/Pause `0xCD`, Next
  `0xB5`, Prev `0xB6`, Vol+ `0xE9`, Vol− `0xEA`, Mute `0xE2`.

## Build, flash, monitor

This is firmware on an Xtensa core. It needs the one-time toolchain setup from
[`../ota-client/README.md`](../ota-client/README.md) (espup, espflash, and on Arch
the `libxml2` symlink). With the board on USB-C:

```sh
. ~/export-esp.sh
cargo run        # builds, flashes, opens the serial monitor
```

The flash runner passes `--partition-table partitions.csv` (required, same reason
as `ota-client`). Rename the advertised device in `cfg.toml` (`device_name`).

## Using it

1. Flash and open the monitor (`cargo run`).
2. On the iPhone: Settings → Bluetooth, tap "EMG Wristband", accept pairing.
   First-time pairing includes iOS reading the HID report map, so it can take a
   few seconds; reconnects are faster once bonded.
3. Open any media app (Music, YouTube), then type into the same espflash monitor
   window, a single keystroke with no Enter:
   `p` play/pause, `n` next, `b` back, `+`/`−` volume, `m` mute, `h` help.

The phone should respond. Typing before a host is connected is ignored with a
warning.

Console input arrives over USB-Serial-JTAG. The firmware installs that driver at
startup (`console::init`) because esp-idf's default `stdin` over USB-Serial-JTAG
delivers nothing.
