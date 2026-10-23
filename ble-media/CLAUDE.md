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
  the example; this library package has no default binary. `ble-gatt-bench` is
  the second example: the same remote plus the link bench service.
- The radio dependencies are behind `[target.'cfg(target_os = "espidf")']` to keep
  that split working. New logic goes in `phone.rs`; only calls into esp32-nimble
  go in `nimble.rs`.
- `src/link_bench.rs` is the second host-tested module: the byte layouts of the
  bench service (`ble-gatt-bench` example, driven from `../phone-app`). Its
  records are fixed little-endian, not CBOR, and `phone-app/src/lib/bench/
  protocol.ts` mirrors them by hand; change both. `NimbleRadio::attach_link_bench`
  must run before advertising starts, and it puts the service UUID in the scan
  response so the primary advertisement iOS bonds against is unchanged.
- `src/settings_service.rs` is the third host-tested module. It holds the byte
  layouts of the settings record (sensitivity and gesture-to-key bindings) and
  the gesture notification. `opal-firmware` serves them, and
  `phone-app/src/lib/band/protocol.ts` mirrors them by hand; change both.
  `NimbleRadio::attach_settings_service` must also run before advertising
  starts. It puts its own UUID in the scan response, so a firmware attaches
  either the settings service or the bench, never both.
- `src/emg_stream.rs` is the fourth host-tested module: the layout of the EMG
  characteristic in the settings service, one acquisition window per sequence
  number, chunked across notifications of at most 244 bytes.
  `phone-app/src/lib/band/emgStream.ts` mirrors it by hand; change both.
  `EmgStream::send_window` blocks the caller until every chunk is accepted.
- `Phone::tick` retries a refused advertising start every `ADVERTISING_RETRY`.
  The band has no toggle, so nothing else would retry it.
- Firmware otherwise: the example runs on hardware and cannot run on this host.
- `Phone::press` / `Phone::tick` are the seam the wearer firmware calls from its
  feedback thread. Keep those stable. `tick` must be called unconditionally, not
  only when there is something to send: it also restarts advertising after a peer
  leaves, and it asks the central for the band's link settings (2M PHY, long
  packets, 15 to 30 ms interval) once per encrypted link, through
  `Radio::request_fast_link`. The band owns those; the app never asks. It also
  keeps advertising while a central is connected, slowly, as long as NimBLE has
  room for another: iOS reclaims a bonded band as HID the moment it advertises,
  and a phone app discovers by scanning, so a silent band is unreachable.
  `Radio::peer` is the best link across every connection, read from the
  descriptors, never a flag a second central could overwrite.
- `NimbleRadio::bring_up` and consuming `NimbleRadio::tear_down` bracket a radio
  lifecycle and deliberately are not `Radio` trait methods. Bring-up explicitly
  initializes NimBLE before taking its singleton, so it works again after full
  deinit. `Phone::state` is the per-tick question and must stay allocation-free;
  `Phone::status` owns a `String` and is for building a frame only.
- `src/console.rs` installs the USB-Serial-JTAG stdin driver at startup because
  esp-idf delivers nothing on that path by default. Don't remove it.
- Media key definitions live in `protocol::MediaKey`, not here. `src/hid.rs` has
  only the report descriptor around them.
