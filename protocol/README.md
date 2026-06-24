# protocol

The CBOR wire-protocol types shared by the dashboard backend, the browser, and
(later) the wristband firmware. The crate is `no_std` plus `alloc`, so the same
definitions compile for the ESP32-S3. `src/lib.rs` is the whole crate, and there is
one central enum, `Frame`. Plain host Rust: `cargo test` runs the round-trip
encode/decode tests (the repo's only unit tests), and `cargo build` cross-checks
the `no_std` build.

`Frame` is `#[serde(tag = "type")]` and covers both directions of the dashboard
socket. The backend sends `Hello`, `Emg`, `Prediction`, `Event`, and `Pose`. The
browser sends `Replay`, `SetSensitivity`, `SetKeymap`, and `SetWifi`. Supporting
types are `Binding` and `MediaKey` for the keymap, plus `SensitivityLevel`,
`ClassInfo`, `StateInfo`, `WakeState` (`Idle`/`Arming`/`Active`), and
`ReplayAction`.

Two properties of the shapes are worth protecting. Bulk EMG samples ride as a raw
little-endian `i16` byte blob (`serde_bytes`): half the bytes of `f32`, far less
than a CBOR number array. And every frame is an internally-tagged map with string
keys, so a generic viewer shows `{type: "emg", ...}`. Display
descriptors (`ClassInfo`, `StateInfo`) carry labels and colours from the backend,
so the frontend hardcodes no palette and a new class or state kind needs no
frontend change.

## One source, three consumers

This crate is the single source of truth, but it is not code-generated for its
consumers. When you change a frame here, update the other two by hand: the browser
(cbor-x) in `../dashboard/web/src/lib/protocol.ts`, and the Python service (cbor2)
in `../pose-service`.
