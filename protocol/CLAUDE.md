# CLAUDE.md: protocol

The CBOR wire-protocol types (the `Frame` enum in `src/lib.rs`), `no_std` plus
`alloc` so they compile for the firmware. `cargo test` runs the round-trip tests.
See `README.md` for the full frame list.

Agent notes:

- This crate is the single source of truth, but it is not code-generated. Change a
  frame here and update the other two consumers by hand: the browser (cbor-x) in
  `../dashboard/web/src/lib/protocol.ts`, and the Python service (cbor2) in
  `../pose-service`.
- Preserve the shape decisions: bulk EMG stays a little-endian `i16` byte blob
  (`serde_bytes`), not a number list; every frame stays a tagged map with string
  keys; display descriptors (`ClassInfo`, `StateInfo`) carry labels and colours from
  the backend so the frontend hardcodes none.
