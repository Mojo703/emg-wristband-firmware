//! The deterministic core of a calibration run.
//!
//! Everything here is a function of the constants and the sample indices it is
//! handed. No clock is read, no hardware is touched, and nothing depends on how
//! long a pass took — so a collection that happened on a wrist replays
//! bit-for-bit on a host, which is the plan's first rule and the only reason a
//! failed field calibration is diagnosable at all.
//!
//! The device supplies three things this cannot compute for itself: the sample
//! index now, whether a labeled span was any good (which needs the band energy
//! and the front end's own flags), and what the model made of a held-out rep
//! (which needs the fitter). Everything else — when to prompt, which gesture,
//! which windows carry the label, when a round is done, whether the gate wants
//! more rounds, which pair is weak — is decided here.
//!
//! `no_std` plus `alloc`, so the firmware and the host both compile it.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use core::num::NonZeroU32;

mod anchored_fit;
mod anchored_recipe;
mod anchored_song;
mod gate;
mod grid;
mod machine;
mod validity;

pub use anchored_fit::{AnchoredFitPlan, AnchoredFitStage};
pub use anchored_recipe::{
    active_gesture_from_index, active_gesture_index, anchored_class_index, anchored_labeled_span,
    anchored_target_count, AnchoredClassCount, AnchoredRecipeProgress, ACTIVE_CALIBRATION_GESTURES,
    ACTIVE_GESTURE_COUNT, ANCHORED_CLASS_COUNT, ANCHORED_COMMAND_TARGET, ANCHORED_NO_OP_TARGET,
    CALIBRATION_MODEL_CLASS_COUNT,
};
pub use anchored_song::{
    AnchoredSong, AnchoredSongAction, AnchoredSongError, AnchoredSongIdentity,
    RetainedSongProgress, SongAnchor, SongInterruption, SongState, UploadEffect,
    HEARTBEAT_INTERVAL_MICROSECONDS, HEARTBEAT_TIMEOUT_MICROSECONDS, MAX_ANCHORED_SONG_CHUNK_CUES,
    MAX_ANCHORED_SONG_CUES, REQUIRED_CUE_HOLD_MILLISECONDS, REQUIRED_CUE_RECOVERY_MILLISECONDS,
    SONG_ANCHOR_LEAD_MICROSECONDS,
};
pub use gate::{ClassScore, GateVerdict, QualityGate};
pub use grid::{LabeledSpan, WindowGrid};
pub use machine::{
    Action, Elapsing, Ending, Erasing, Fitting, Flushing, Installing, Performing, Polishing,
    Prompting, RepOutcome, Run, RunOutcome,
};
pub use validity::RepEvidence;

/// Every number the flow is parameterized by.
///
/// The defaults are the plan's starting values, not measured ones: work package
/// V sweeps them against the golden fixtures and ships
/// `fixtures/calibration_constants.json`, which the host tool loads and the
/// firmware compiles in. Carried as one struct rather than as constants so a
/// swept value reaches every consumer at once and a test can vary one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Constants {
    pub sample_rate_hz: u32,
    /// Samples one feature window covers.
    pub window_samples: u32,
    /// Samples between the starts of consecutive windows.
    ///
    /// A quarter of the window, not the whole of it: inside a labeled span the
    /// feature pipeline emits a window every quarter, so a rep contributes
    /// overlapping rows rather than disjoint ones. The replay path outside a
    /// span still runs at 500 — every fourth sliding window is one of those,
    /// bit for bit — so the fixtures are untouched by this.
    pub hop_samples: u32,
    /// R: how long after a prompt the wearer is given before anything counts.
    pub prompt_delay_milliseconds: u32,
    /// How long the wearer is asked to hold the gesture.
    ///
    /// Longer than the labeled span, and deliberately. The span covers 1500
    /// samples of signal, but it starts at the first grid boundary at or after
    /// the hold-off, so where the prompt fell relative to the grid pushes the
    /// last labeled window as much as a stride later. Asking for exactly what
    /// is labeled would mean the wearer relaxing while the last window is
    /// still being taken, on whichever reps happened to land badly. This is the
    /// worst case with margin on top; nothing computes from it.
    pub prompt_hold_milliseconds: u32,
    /// W: grid windows one rep contributes, so every rep yields the same row
    /// count whatever phase the prompt landed on. At a quarter-window stride
    /// these overlap, and nine of them span 1500 samples of signal.
    pub labeled_windows: u32,
    /// The validated floor for the thumb-up block, in rounds. One rep per
    /// gesture per round, so this is also cues per class.
    pub thumb_up_round_floor: u32,
    /// The floor for the thumb-down block, which is larger.
    ///
    /// Not a symmetry anyone chose: V's interim found false fires only reach
    /// the golden 3.8% at twelve same-don thumb-down reps per class. The extra
    /// two rounds land in the pole-holding phase, which is also the phase a
    /// wearer is least able to hurry.
    pub thumb_down_round_floor: u32,
    /// The announced still phase: filter and amplitude settling, followed by
    /// reference-gain estimation. The target slot was erased at boot.
    pub settling_milliseconds: u32,
    /// How much of the still phase passes before the gain estimate starts
    /// accumulating. The filters and the electrode amplitudes are still moving
    /// before this, and a projection fitted through that transient measures the
    /// settling rather than the don.
    pub gain_settle_milliseconds: u32,
    /// How long the run waits between the two blocks for the pole to change
    /// hands. Fixed rather than acknowledged, because the device has no way to
    /// know a pole is in a hand and the panel is not required to be present.
    pub handover_milliseconds: u32,
    /// How long the reference-gain projection sums run for, after
    /// [`Self::gain_settle_milliseconds`] has passed.
    pub gain_window_milliseconds: u32,
    /// Tries one gesture gets in one round before the run gives up on it.
    pub rep_attempt_budget: u32,
    /// Reps per class the gate holds out of the checkpoint it scores against.
    pub held_out_reps_per_class: u32,
    /// Optimizer passes after each completed round, and after the last one.
    ///
    /// Here rather than in the firmware because they are V's numbers like every
    /// other field, and because the firmware's tests need a board — the
    /// striding addendum moved all three of these at once, and nothing would
    /// have failed if nobody had recompiled.
    ///
    /// Non-zero, and the type says so rather than a check somewhere: a
    /// checkpoint of no passes plans work it can never report finishing, and
    /// the driver that counts them down reaches zero by subtracting from zero.
    pub passes_per_round: NonZeroU32,
    pub final_passes: NonZeroU32,
    /// Visit every `prior_stride`-th prior row per pass, the starting offset
    /// rotating so consecutive passes cover the prior exactly once each.
    ///
    /// Two, not three. Strides three and four break misclassification almost
    /// everywhere once the sweep is scored on the shape the device really
    /// holds — 7,704 prior rows against 990 live ones under the nine-window
    /// labeling, rather than the fifteen-row labeling the earlier sweeps
    /// assumed. Halving the prior per pass is what this schedule can afford.
    pub prior_stride: usize,
    /// Per-class window accuracy, in permille, at or above which the report
    /// calls a class holding. The gate never changes the fixed round counts.
    pub holding_accuracy_permille: u32,
}

impl Constants {
    /// The plan's starting values, superseded by V's constants file.
    pub const DEFAULT: Self = Self {
        sample_rate_hz: 2000,
        window_samples: 500,
        hop_samples: 125,
        prompt_delay_milliseconds: 250,
        prompt_hold_milliseconds: 1500,
        labeled_windows: 9,
        thumb_up_round_floor: 10,
        thumb_down_round_floor: 12,
        settling_milliseconds: 60_000,
        gain_settle_milliseconds: 30_000,
        handover_milliseconds: 8_000,
        gain_window_milliseconds: 30_000,
        rep_attempt_budget: 4,
        held_out_reps_per_class: 2,
        passes_per_round: NonZeroU32::new(16).unwrap(),
        final_passes: NonZeroU32::new(10).unwrap(),
        prior_stride: 2,
        holding_accuracy_permille: 700,
    };

    /// Samples in `milliseconds` on the device's own grid. Integer throughout:
    /// a labeling boundary computed through a float would depend on the
    /// rounding mode, and the host replay would not be bit-exact.
    pub const fn samples_in(&self, milliseconds: u32) -> u64 {
        milliseconds as u64 * self.sample_rate_hz as u64 / 1000
    }

    pub const fn grid(&self) -> WindowGrid {
        WindowGrid::new(self.window_samples, self.hop_samples)
    }

    /// R, in samples.
    pub const fn prompt_delay_samples(&self) -> u64 {
        self.samples_in(self.prompt_delay_milliseconds)
    }

    /// The latest a labeled window can end, in milliseconds after the prompt.
    ///
    /// The hold the wearer is asked for has to cover this, or a badly aligned
    /// rep asks them to hold past what they were told. A test holds the two
    /// together.
    pub const fn latest_labeled_end_milliseconds(&self) -> u32 {
        let worst_alignment = self.hop_samples as u64 - 1;
        let span = (self.labeled_windows as u64 - 1) * self.hop_samples as u64
            + self.window_samples as u64;
        let samples = self.prompt_delay_samples() + worst_alignment + span;
        (samples * 1000 / self.sample_rate_hz as u64) as u32
    }

    /// Rows one accepted rep contributes: W windows, one row each.
    pub const fn rows_per_rep(&self) -> u32 {
        self.labeled_windows
    }
}

impl Default for Constants {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hold_asked_for_covers_the_last_labeled_window() {
        // The wearer is told to hold; the grid decides when the last window
        // closes. If the instruction were the shorter of the two, the reps that
        // landed worst against the grid would be the ones where the wearer
        // relaxed into the label — and nothing downstream could tell that from
        // a gesture performed badly.
        let constants = Constants::DEFAULT;
        assert!(
            constants.prompt_hold_milliseconds >= constants.latest_labeled_end_milliseconds(),
            "asking for {} ms but labeling as late as {} ms",
            constants.prompt_hold_milliseconds,
            constants.latest_labeled_end_milliseconds()
        );
    }

    #[test]
    fn durations_convert_on_the_device_grid() {
        let constants = Constants::DEFAULT;
        assert_eq!(constants.prompt_delay_samples(), 500);
        assert_eq!(constants.samples_in(0), 0);
        // A duration that is not a whole number of samples truncates rather
        // than rounding, so the host and the device agree without either
        // knowing the other's rounding mode.
        assert_eq!(constants.samples_in(1), 2);
        let odd = Constants {
            sample_rate_hz: 1999,
            ..constants
        };
        assert_eq!(odd.samples_in(500), 999);
    }

    /// V's constants file, which is the authority for every field the fixtures
    /// swept.
    fn shipped_constants() -> serde_json::Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures/calibration_constants.json"
        );
        let text =
            std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {path}: {error}"));
        serde_json::from_str(&text).expect("the constants fixture is valid JSON")
    }

    #[test]
    fn the_compiled_constants_are_the_ones_v_shipped() {
        // The failure this exists to catch is silent. `Constants::DEFAULT` is
        // compiled in, so a swept value that nobody recompiled changes nothing
        // and breaks nothing — the device simply runs a schedule no one
        // specified and reports timings for it. That nearly shipped once: the
        // striding addendum moved passes_per_round, final_passes and
        // prior_stride together, and the firmware was still fitting the
        // schedule from before it.
        //
        // Keyed by the fixture's own names, so a section V renames fails here
        // rather than silently stopping being checked.
        let shipped = shipped_constants();
        let compiled = Constants::DEFAULT;
        let number = |section: &str, key: &str| -> u64 {
            shipped[section][key]
                .as_u64()
                .unwrap_or_else(|| panic!("{section}.{key} is missing or not a whole number"))
        };

        assert_eq!(
            number("schedule", "passes_per_round"),
            compiled.passes_per_round.get() as u64
        );
        assert_eq!(
            number("schedule", "final_passes"),
            compiled.final_passes.get() as u64
        );
        assert_eq!(
            number("schedule", "prior_stride"),
            compiled.prior_stride as u64
        );

        assert_eq!(
            number("labeling", "hold_off_ms"),
            compiled.prompt_delay_milliseconds as u64
        );
        assert_eq!(
            number("labeling", "windows_per_rep"),
            compiled.labeled_windows as u64
        );
        assert_eq!(
            number("labeling", "stride_samples"),
            compiled.hop_samples as u64
        );
        assert_eq!(
            number("labeling", "window_samples"),
            compiled.window_samples as u64
        );

        assert_eq!(
            number("cue_floor", "thumb_up_per_class"),
            compiled.thumb_up_round_floor as u64
        );
        assert_eq!(
            number("cue_floor", "thumb_down_per_class"),
            compiled.thumb_down_round_floor as u64
        );

        // Seconds in the file, milliseconds here — the one place the two
        // vocabularies differ, so the conversion is asserted rather than
        // assumed.
        assert_eq!(
            number("reference_gains", "window_seconds") * 1000,
            compiled.gain_window_milliseconds as u64
        );
        assert_eq!(
            number("reference_gains", "settle_seconds") * 1000,
            compiled.gain_settle_milliseconds as u64
        );
        assert_eq!(
            number("reference_gains", "window_seconds") * 1000
                + number("reference_gains", "settle_seconds") * 1000,
            compiled.settling_milliseconds as u64,
            "the still phase has to cover the settle and the gain window"
        );
    }

    #[test]
    fn the_shipped_estimator_is_still_the_one_implemented() {
        // Not a number, and the reason it is checked anyway: the gain estimator
        // is marked provisional in the fixture, and the wording is what says
        // whether the mean is removed. An estimator swapped in prose while the
        // numbers stayed put would change what the firmware should compute and
        // move nothing this file could compare.
        let shipped = shipped_constants();
        let estimator = shipped["reference_gains"]["estimator"]
            .as_str()
            .expect("the fixture names its estimator");
        assert!(
            estimator.contains("removing each channel's window mean"),
            "the shipped estimator changed: {estimator}"
        );
    }
}
