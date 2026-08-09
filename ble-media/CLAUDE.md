# CLAUDE.md: ble-media

ESP32-S3 BLE HID media remote (`std` / `esp-idf-svc` plus `esp32-nimble`). See
`README.md` for how the HID/bonding works and the pairing walkthrough.

Agent notes:

- **`src/phone.rs` runs on this host.** It is the whole decision surface — the
  `Phone` state machine, the encryption gate on sending keys, the key-hold
  deadline — behind the `phone::Radio` seam, and its tests are the ones that must
  not need a board. Run them from the **repo root**:

  ```sh
  cargo test --manifest-path ble-media/Cargo.toml
  ```

  From inside the crate, `.cargo/config.toml` pins the Xtensa target and the esp
  toolchain instead, which is what `cargo run` (flash + monitor) wants.
- The radio dependencies are behind `[target.'cfg(target_os = "espidf")']` to keep
  that split working. New logic goes in `phone.rs`; only calls into esp32-nimble
  go in `nimble.rs`.
- Firmware otherwise: it flashes to hardware and you cannot run the binary here.
- `Phone::press` / `Phone::tick` are the seam the wearer firmware calls from its
  feedback thread. Keep those stable. `tick` must be called unconditionally, not
  only when there is something to send: it also restarts advertising after a peer
  leaves.
- `NimbleRadio::bring_up` runs once, at boot, and is deliberately not a `Radio`
  trait method — `BLEDevice::take()` is a singleton claim with no second chance.
  `Phone::state` is the per-tick question and must stay allocation-free;
  `Phone::status` owns a `String` and is for building a frame only.
- `src/console.rs` installs the USB-Serial-JTAG stdin driver at startup because
  esp-idf delivers nothing on that path by default. Don't remove it.
- Media key definitions live in `protocol::MediaKey`, not here. `src/hid.rs` has
  only the report descriptor around them.
