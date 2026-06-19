# ble-media

ESP32-S3 BLE media remote. Advertises as a BLE HID-over-GATT consumer-control
device, bonds with a host (iPhone, also macOS/Android/Windows), and sends media
keys (play/pause, next, previous, volume, mute). Built on `std` / `esp-idf-svc`
with the [`esp32-nimble`](https://github.com/taks/esp32-nimble) NimBLE wrapper.

For bring-up the keys are driven from the jtag serial console; later the gesture
classifier will call `MediaController::press` directly instead.

## How it works

- A HID service (`0x1812`) exposes one Consumer Control input report carrying a
  16-bit usage code (`media.rs`). Sending a usage = press, sending `0x0000` =
  release.
- The device advertises with a HID-keyboard appearance and bonds on first
  connect. iOS only delivers HID input over an encrypted/bonded link, so
  bonding (`AuthReq::all`, "just works" pairing) is required, and bonding keys
  are persisted in NVS so it stays paired across reboots.
- Media usages: Play/Pause `0xCD`, Next `0xB5`, Prev `0xB6`, Vol+ `0xE9`,
  Vol- `0xEA`, Mute `0xE2`.

## Build, flash, monitor

Needs the Xtensa toolchain (see `../ota-client/README.md` for one-time setup,
including the Arch `libxml2` symlink). Then, with the board on USB-C:

```sh
. ~/export-esp.sh
cargo run        # builds, flashes, opens the serial monitor
```

The flash runner passes `--partition-table partitions.csv` (required, same
reason as `ota-client`).

## Using it

1. Flash and open the monitor (`cargo run`).
2. On the iPhone: Settings>Bluetooth, tap "EMG Wristband", accept pairing.
   First-time pairing includes iOS reading the HID report map, so it can take a
   few seconds. Reconnects are faster once bonded.
3. Open any media (Music, YouTube, etc.), then type into the same espflash
   monitor window (a single keystroke, no Enter needed):
   - `p` play/pause, `n` next, `b` back, `+`/`-` volume, `m` mute, `h` help.
   The phone should respond. If you type before a host is connected, the command
   is ignored with a warning.

Console input arrives over USB-Serial-JTAG; the firmware installs that driver at
startup (`console::init`) because esp-idf's default `stdin` over USB-Serial-JTAG
delivers nothing.

Rename the advertised device in `cfg.toml` (`device_name`).
