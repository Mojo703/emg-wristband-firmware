//! The backend's cosmetic layer — the only thing it owns that the device does not.
//! It projects a device's functional [`DeviceConfig`] into per-class and per-state
//! render hints (colours/labels) for the browser. It invents no functional values
//! and assumes no fixed gesture count: the palette is indexed modulo its length, so
//! any number of classes renders.

use protocol::{ClassInfo, DeviceConfig, MediaKey, StateInfo};

/// Per-class line/legend/band colours. Indexed `class % len`, so it never runs out.
const CLASS_PALETTE: [&str; 8] =
    ["#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ec4899", "#14b8a6", "#f97316", "#60a5fa"];

/// The palette colour for a class index — the single source for line, legend, band,
/// and commit-marker colour.
pub fn class_color(gesture: u8) -> &'static str {
    CLASS_PALETTE[gesture as usize % CLASS_PALETTE.len()]
}

/// Short human label for a media key, for class labels and event text.
pub fn media_label(key: MediaKey) -> &'static str {
    match key {
        MediaKey::PlayPause => "Play/Pause",
        MediaKey::NextTrack => "Next",
        MediaKey::PrevTrack => "Prev",
        MediaKey::VolumeUp => "Vol +",
        MediaKey::VolumeDown => "Vol −",
        MediaKey::Mute => "Mute",
    }
}

/// Render hints for each of a device's gesture classes, labelled from its keymap and
/// coloured from the palette. All classes are commands today; a reject/rest class
/// would append with `command: false`.
pub fn classes_for(config: &DeviceConfig) -> Vec<ClassInfo> {
    (0..config.gestures)
        .map(|gesture| {
            let key = config
                .keymap
                .iter()
                .find(|binding| binding.gesture == gesture)
                .map(|binding| media_label(binding.key));
            let label = match key {
                Some(name) => format!("C{gesture} · {name}"),
                None => format!("C{gesture}"),
            };
            ClassInfo { label, color: class_color(gesture).to_string(), command: true }
        })
        .collect()
}

/// Wake-gate state presentation: `intensity` is how strongly the band paints the
/// active command's colour, so states read by brightness while commands stay distinct
/// by hue. The names match `protocol::WakeState`'s snake_case.
pub fn states() -> Vec<StateInfo> {
    vec![
        StateInfo { name: "idle".into(), label: "idle".into(), color: "#6b7280".into(), intensity: 0.12 },
        StateInfo { name: "arming".into(), label: "arming".into(), color: "#f59e0b".into(), intensity: 0.45 },
        StateInfo { name: "active".into(), label: "active".into(), color: "#22c55e".into(), intensity: 0.9 },
    ]
}
