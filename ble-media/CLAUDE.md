# CLAUDE.md: ble-media

ESP32-S3 BLE HID media remote (`std` / `esp-idf-svc` plus `esp32-nimble`). See
`README.md` for how the HID/bonding works and the pairing walkthrough.

Agent notes:

- Firmware. It builds and flashes to hardware; you cannot run it on this host.
- `MediaController::press` in `src/media.rs` is the seam: the serial console drives
  it now, and the gesture classifier will call it directly later. Keep that entry
  point stable.
- `src/console.rs` installs the USB-Serial-JTAG stdin driver at startup because
  esp-idf delivers nothing on that path by default. Don't remove it.
