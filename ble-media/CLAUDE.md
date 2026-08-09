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
  toolchain. Build the standalone hardware bench without flashing via
  `. ~/export-esp.sh && cargo build --example ble-media-bench`. Flash and monitor
  it via `. ~/export-esp.sh && cargo run --example ble-media-bench`. Always name
  the example; this library package has no default binary.
- The radio dependencies are behind `[target.'cfg(target_os = "espidf")']` to keep
  that split working. New logic goes in `phone.rs`; only calls into esp32-nimble
  go in `nimble.rs`.
- Firmware otherwise: the example runs on hardware and cannot run on this host.
- `Phone::press` / `Phone::tick` are the seam the wearer firmware calls from its
  feedback thread. Keep those stable. `tick` must be called unconditionally, not
  only when there is something to send: it also restarts advertising after a peer
  leaves.
- `NimbleRadio::bring_up` and consuming `NimbleRadio::tear_down` bracket a radio
  lifecycle and deliberately are not `Radio` trait methods. Bring-up explicitly
  initializes NimBLE before taking its singleton, so it works again after full
  deinit. `Phone::state` is the per-tick question and must stay allocation-free;
  `Phone::status` owns a `String` and is for building a frame only.
- `src/console.rs` installs the USB-Serial-JTAG stdin driver at startup because
  esp-idf delivers nothing on that path by default. Don't remove it.
- Media key definitions live in `protocol::MediaKey`, not here. `src/hid.rs` has
  only the report descriptor around them.
