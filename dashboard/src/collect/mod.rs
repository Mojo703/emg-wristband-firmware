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
//! - `audio`     — track decode, cue clicks and metronome, and the exact playhead
//! - `recorder`  — raw `emg.i16` + `events.jsonl` + `session.json` writing
//! - `video`     — ffmpeg child process: session video, placement photos, stall detection
//! - `beatmap`   — track catalog and note-schedule generation
//! - `beatsaber` — Beat Saber map formats v2/v3/v4 → a cue schedule per level
//! - `import`    — map archives in (upload or BeatSaver), track directories out
//! - `manager`   — session state machine tying the above together (integration)
//! - `provenance` — what the host remembers per device and per subject
//!
//! [`provenance`]: crate::collect::provenance

pub mod audio;
pub mod beatmap;
pub mod beatsaber;
pub mod calibration_level;
pub mod import;
pub mod interfaces;
pub mod manager;
pub mod provenance;
pub mod recorder;
pub mod video;
