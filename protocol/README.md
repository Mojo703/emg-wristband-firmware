# protocol

The CBOR wire-protocol types shared by the wristband firmware, the dashboard
backend, and the browser. The crate is `no_std` plus `alloc`, so the same
definitions compile for the ESP32-S3. `src/lib.rs` is the whole crate, and there is
one central enum, `Frame`. Plain host Rust: `cargo test` runs the round-trip
encode/decode tests (the repo's only unit tests), and `cargo build` cross-checks
the `no_std` build.

`Frame` is `#[serde(tag = "type")]` and covers all three links. The device sends
`DeviceHello` (identity + config) then `Emg`, `Prediction`, and `Event`. The backend
relays those to the browser and adds `Hello` (the view state: device list + selected
device + cosmetics) and `Pose`. The browser sends `SelectDevice`, `SetSensitivity`,
`SetKeymap`, and `SetWifi`, which the backend forwards to the selected device.
Supporting types are `Binding` and `MediaKey` for the keymap, `DeviceInfo` and
`DeviceConfig` for device identity/config, plus `SensitivityLevel`, `ClassInfo`,
`StateInfo`, and `WakeState` (`Idle`/`Arming`/`Active`).

The device is the source of functional truth and runs standalone, so it owns
`DeviceConfig` (gestures, keymap, sensitivity presets and their thresholds, the
active threshold, the streak goal). The backend owns only cosmetics the firmware has
no reason to carry — the colours in `ClassInfo`/`StateInfo` — and layers them on top
when projecting `DeviceConfig` into the browser `Hello`.

Two properties of the shapes are worth protecting. Bulk EMG samples ride as a raw
little-endian `i16` byte blob (`serde_bytes`): half the bytes of `f32`, far less
than a CBOR number array. And every frame is an internally-tagged map with string
keys, so a generic viewer shows `{type: "emg", ...}`.

## One source, three consumers

This crate is the single source of truth, but it is not code-generated for its
consumers. When you change a frame here, update the other two by hand: the browser
(cbor-x) in `../dashboard/web/src/lib/protocol.ts`, and the Python service (cbor2)
in `../pose-service`.
