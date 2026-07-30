//! Training-data collection: the rhythm game's backend half.
//!
//! Collection is a consumer of the existing relay, not a change to it: the manager
//! subscribes to the selected device's frame broadcast exactly like a browser
//! session does, records the raw `Emg` stream to disk, drives the webcam capture,
//! and serves the beatmap. The browser stays a renderer; every recorded artifact
//! and every label-bearing timestamp originates here.
//!
//! Module layout (one implementation unit per file, built against
//! [`interfaces`]):
//! - `recorder`  — raw `emg.i16` + `events.jsonl` + `session.json` writing
//! - `video`     — ffmpeg child process: session video, placement photos, stall detection
//! - `beatmap`   — track catalog and note-schedule generation
//! - `manager`   — session state machine tying the above together (integration)

// Interfaces land before their implementations; drop this once the manager wires
// everything into the browser session loop.
#![allow(dead_code)]

pub mod interfaces;
