//! The replay snapshot harness: drive a scripted calibration run through the
//! crate's public API and write down everything it did.
//!
//! This is the acceptance baseline for the firmware rewrite. The point is not
//! that any one number here is right — it is that the rewrite produces the same
//! numbers as the machine that shipped, and that a difference shows up as a
//! line in a text diff rather than as a bench run that aborts three phases
//! later.
//!
//! Determinism is the whole product, so:
//!
//! * Nothing here reads a clock, an environment variable that varies, or a
//!   random source. Every sample index is derived from the constants and the
//!   schedule.
//! * Nothing iterates a `HashMap`. The projection is an ordered `Vec` of
//!   fields and the field order is written down once, in [`Projection::of`].
//! * Nothing formats a float. The crate under test has none — every quantity
//!   it exposes is a `u32`, a `u64`, or an enum — and this harness must not
//!   introduce one, because a float would put the snapshot at the mercy of the
//!   host's formatting.
//! * State is written down only when it *changes*. A run is tens of thousands
//!   of samples long and the machine is polled every 125 of them; printing the
//!   projection each time would bury the two lines that matter. A changed field
//!   cannot be missed by this, which is what keeps it exhaustive.

#![allow(dead_code)]

use std::fmt::Write as _;
use std::num::NonZeroU32;
use std::path::PathBuf;

use calibration_flow::{
    Action, Constants, GateVerdict, LabeledSpan, RepEvidence, RepOutcome, Run, RunOutcome,
    ScheduledPrompt, ScriptedPoll, ScriptedSchedule, ScriptedWearer, THUMB_DOWN_BLOCK,
    THUMB_UP_BLOCK,
};
use protocol::{CalibrationGesture, CalibrationPhase};

/// Report a step against a run whose phase the caller already knows.
///
/// The machine's transitions consume the phase they belong to, which is what
/// makes a report against the wrong one a compile error. A driver holds the
/// phase and knows which it is; a test harness holds a `Run` and has just been
/// told by an action, so it says so here once rather than at every call site.
/// A wrong phase panics, which is the same answer the compiler gives a driver.
pub trait Reported {
    fn verified(self) -> Run;
    fn resolve_rep(self, evidence: RepEvidence) -> (Run, RepOutcome);
    fn flushed(self) -> Run;
    /// `passes` passes of `milliseconds` each, then the checkpoint.
    fn fitted(self, passes: NonZeroU32, milliseconds: u32) -> Run;
    fn polished(self, passes: NonZeroU32, milliseconds: u32) -> Run;
    fn installed(self) -> Run;
}

impl Reported for Run {
    fn verified(self) -> Run {
        match self {
            Run::Erasing(erasing) => erasing.verified(),
            other => panic!("not erasing, standing in {:?}", other.phase()),
        }
    }

    fn resolve_rep(self, evidence: RepEvidence) -> (Run, RepOutcome) {
        match self {
            Run::Performing(performing) => performing.resolve(evidence),
            other => panic!("no rep is open, standing in {:?}", other.phase()),
        }
    }

    fn flushed(self) -> Run {
        match self {
            Run::Flushing(flushing) => flushing.flushed(),
            other => panic!("not flushing, standing in {:?}", other.phase()),
        }
    }

    fn fitted(self, passes: NonZeroU32, milliseconds: u32) -> Run {
        match self {
            Run::Fitting(mut fitting) => {
                fitting.expect_passes(passes);
                for _ in 0..passes.get() {
                    fitting.pass_completed(milliseconds);
                }
                fitting.fitted()
            }
            other => panic!("not fitting, standing in {:?}", other.phase()),
        }
    }

    fn polished(self, passes: NonZeroU32, milliseconds: u32) -> Run {
        match self {
            Run::Polishing(mut polishing) => {
                polishing.expect_passes(passes);
                for _ in 0..passes.get() {
                    polishing.pass_completed(milliseconds);
                }
                polishing.polished()
            }
            other => panic!("not polishing, standing in {:?}", other.phase()),
        }
    }

    fn installed(self) -> Run {
        match self {
            Run::Installing(installing) => installing.installed(),
            other => panic!("not installing, standing in {:?}", other.phase()),
        }
    }
}

/// Samples one feature window covers, and the stride between window closes
/// inside a labeled span. The pipeline needs a whole window before its first
/// close exists, so windows close at 500, 625, 750, … and never at 125.
pub const WINDOW: u64 = 500;
pub const STRIDE: u64 = 125;

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// One line of snapshot text for a value the harness observed.
///
/// An extension trait rather than a pile of `fn render_x(x: &X)` helpers: each
/// of these has an obvious receiver, and the house rule is that a free function
/// with an obvious receiver is a method.
pub trait Rendered {
    fn rendered(&self) -> String;
}

impl Rendered for LabeledSpan {
    fn rendered(&self) -> String {
        format!(
            "samples[{},{}) windows[{},{})",
            self.first_sample,
            self.end_sample,
            self.first_window,
            self.first_window + self.window_count
        )
    }
}

impl Rendered for Action {
    fn rendered(&self) -> String {
        match self {
            Action::EraseSlot => "EraseSlot".to_string(),
            Action::Prompt {
                gesture,
                label,
                generation,
                span,
            } => format!(
                "Prompt gesture={gesture:?} label={label} generation={generation} {}",
                span.rendered()
            ),
            Action::FlushRows => "FlushRows".to_string(),
            Action::FitRound { round } => format!("FitRound round={round}"),
            Action::Polish => "Polish".to_string(),
            Action::Install => "Install".to_string(),
            Action::Finish { outcome } => format!("Finish outcome={outcome:?}"),
        }
    }
}

impl Rendered for RepEvidence {
    fn rendered(&self) -> String {
        format!(
            "windows={}/{} lead_off={} adc_settle={} flash={}",
            self.windows_present,
            self.windows_expected,
            self.lead_off_channels,
            self.adc_recovery_settle,
            self.flash_operation
        )
    }
}

impl Rendered for RepOutcome {
    fn rendered(&self) -> String {
        match self {
            RepOutcome::Accepted { label, span } => {
                format!("Accepted label={label} {}", span.rendered())
            }
            RepOutcome::Rejected(rejection) => format!("Rejected {rejection:?}"),
            RepOutcome::GestureExhausted { gesture, rejection } => {
                format!("GestureExhausted gesture={gesture:?} {rejection:?}")
            }
        }
    }
}

impl Rendered for GateVerdict {
    fn rendered(&self) -> String {
        let mut text = String::new();
        for class in &self.classes {
            let _ = write!(
                text,
                "{:?}={}/{}:{:?} ",
                class.gesture, class.correct, class.scored, class.status
            );
        }
        let _ = write!(text, "weak_pair={:?}", self.weak_pair);
        text
    }
}

impl Rendered for Constants {
    fn rendered(&self) -> String {
        let mut text = String::new();
        for (key, value) in [
            ("sample_rate_hz", self.sample_rate_hz as u64),
            ("window_samples", self.window_samples as u64),
            ("hop_samples", self.hop_samples as u64),
            (
                "prompt_delay_milliseconds",
                self.prompt_delay_milliseconds as u64,
            ),
            (
                "prompt_hold_milliseconds",
                self.prompt_hold_milliseconds as u64,
            ),
            ("labeled_windows", self.labeled_windows as u64),
            ("thumb_up_round_floor", self.thumb_up_round_floor as u64),
            ("thumb_down_round_floor", self.thumb_down_round_floor as u64),
            ("settling_milliseconds", self.settling_milliseconds as u64),
            (
                "gain_settle_milliseconds",
                self.gain_settle_milliseconds as u64,
            ),
            ("handover_milliseconds", self.handover_milliseconds as u64),
            (
                "gain_window_milliseconds",
                self.gain_window_milliseconds as u64,
            ),
            ("rep_attempt_budget", self.rep_attempt_budget as u64),
            (
                "held_out_reps_per_class",
                self.held_out_reps_per_class as u64,
            ),
            ("passes_per_round", self.passes_per_round.get() as u64),
            ("final_passes", self.final_passes.get() as u64),
            ("prior_stride", self.prior_stride as u64),
            (
                "holding_accuracy_permille",
                self.holding_accuracy_permille as u64,
            ),
        ] {
            let _ = writeln!(text, "#   {key} = {value}");
        }
        text
    }
}

// ---------------------------------------------------------------------------
// The projection
// ---------------------------------------------------------------------------

/// Everything [`Run`] will tell an observer, in a fixed order.
///
/// `elapsed_samples` is deliberately absent: it moves on every poll and would
/// make every one of the ~1500 polls in a run print a line. The sample a thing
/// happened at is already on the action and rep lines, and the run's total
/// elapsed span is in the summary.
struct Projection {
    fields: Vec<(&'static str, String)>,
}

impl Projection {
    fn of(run: &Run) -> Self {
        let (passes_done, passes_planned, pass_milliseconds) = run.fit_progress();
        let mut fields = vec![
            ("phase", format!("{:?}", run.phase())),
            ("round", run.round().to_string()),
            ("rounds_planned", run.rounds_planned().to_string()),
            ("round_floor", run.round_floor().to_string()),
            ("prompt", format!("{:?}", run.prompt())),
            ("next_gesture", format!("{:?}", run.next_gesture())),
            ("prompt_generation", run.prompt_generation().to_string()),
            ("notice_generation", run.notice_generation().to_string()),
            ("last_rejection", format!("{:?}", run.last_rejection())),
            (
                "last_rejection_at",
                format!("{:?}", run.last_rejection_at()),
            ),
            ("accepted_reps", run.accepted_reps().to_string()),
            ("rejected_reps", run.rejected_reps().to_string()),
        ];
        for gesture in CalibrationGesture::ALL {
            let (accepted, rejected) = run.reps_for(gesture);
            fields.push((
                Self::reps_field(gesture),
                format!("{accepted} accepted, {rejected} rejected"),
            ));
        }
        fields.extend([
            ("rows_stored", run.rows_stored().to_string()),
            ("flash_flushes", run.flash_flushes().to_string()),
            ("weak_pair", format!("{:?}", run.weak_pair())),
            (
                "verdict",
                run.verdict()
                    .map_or_else(|| "None".to_string(), Rendered::rendered),
            ),
            ("fit_passes_done", passes_done.to_string()),
            ("fit_passes_planned", passes_planned.to_string()),
            ("fit_pass_milliseconds", pass_milliseconds.to_string()),
            (
                "fit_wall_milliseconds",
                run.fit_wall_milliseconds().to_string(),
            ),
            ("outcome", format!("{:?}", run.outcome())),
        ]);
        Self { fields }
    }

    /// A stable per-gesture field name. `&'static str` so the field list has one
    /// type, which is what keeps the ordering obvious at the call site.
    fn reps_field(gesture: CalibrationGesture) -> &'static str {
        match gesture {
            CalibrationGesture::WristPronation => "reps.wrist_pronation",
            CalibrationGesture::WristSupination => "reps.wrist_supination",
            CalibrationGesture::WristRadialDeviation => "reps.wrist_radial_deviation",
            CalibrationGesture::WristUlnarDeviation => "reps.wrist_ulnar_deviation",
            CalibrationGesture::ThumbExtension => "reps.thumb_extension",
        }
    }

    /// The fields that differ from `previous`, or all of them when there is no
    /// previous projection.
    fn changes_from(&self, previous: Option<&Projection>) -> Vec<String> {
        let Some(previous) = previous else {
            return self
                .fields
                .iter()
                .map(|(key, value)| format!("{key} = {value}"))
                .collect();
        };
        self.fields
            .iter()
            .zip(&previous.fields)
            .filter(|((_, now), (_, before))| now != before)
            .map(|((key, now), (_, before))| format!("{key} = {before} -> {now}"))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The transcript
// ---------------------------------------------------------------------------

/// The text a scenario is pinned as.
pub struct Transcript {
    text: String,
    previous: Option<Projection>,
}

impl Transcript {
    pub fn new(scenario: &str, constants: &Constants) -> Self {
        let mut text = String::new();
        let _ = writeln!(text, "# scenario: {scenario}");
        let _ = writeln!(text, "# constants:");
        text.push_str(&constants.rendered());
        Self {
            text,
            previous: None,
        }
    }

    /// Start a fresh section. The projection baseline resets, so the first
    /// state line of the section lists every field — which is what a second run
    /// in one boot needs, since its `Run` is a different object entirely.
    pub fn begin(&mut self, label: &str) {
        let _ = writeln!(self.text, "\n=== {label}");
        self.previous = None;
    }

    pub fn note(&mut self, note: &str) {
        let _ = writeln!(self.text, "  note   {note}");
    }

    pub fn action(&mut self, at_sample: u64, action: &Action) {
        let _ = writeln!(self.text, "{at_sample:>9}  action {}", action.rendered());
    }

    /// What a second poll said, asked before the first was reported.
    pub fn again(&mut self, at_sample: u64, action: Option<&Action>) {
        let said = action.map_or_else(|| "nothing".to_string(), Rendered::rendered);
        let _ = writeln!(self.text, "{at_sample:>9}  again  {said}");
    }

    pub fn rep(
        &mut self,
        index: usize,
        at_sample: u64,
        gesture: CalibrationGesture,
        span: LabeledSpan,
        evidence: &RepEvidence,
        outcome: &RepOutcome,
    ) {
        let _ = writeln!(
            self.text,
            "{at_sample:>9}  rep#{index} gesture={gesture:?} {} evidence({}) -> {}",
            span.rendered(),
            evidence.rendered(),
            outcome.rendered()
        );
    }

    /// Write down whatever the run will now say about itself that it did not
    /// say last time.
    pub fn project(&mut self, run: &Run) {
        let projection = Projection::of(run);
        for change in projection.changes_from(self.previous.as_ref()) {
            let _ = writeln!(self.text, "           state  {change}");
        }
        self.previous = Some(projection);
    }

    pub fn summary(&mut self, run: &Run, windows_fed: u64, exhausted: bool) {
        let _ = writeln!(self.text, "  --- summary");
        for (key, value) in [
            ("phase", format!("{:?}", run.phase())),
            ("outcome", format!("{:?}", run.outcome())),
            ("accepted_reps", run.accepted_reps().to_string()),
            ("rejected_reps", run.rejected_reps().to_string()),
            ("rows_stored", run.rows_stored().to_string()),
            ("flash_flushes", run.flash_flushes().to_string()),
            ("elapsed_samples", run.elapsed_samples().to_string()),
            ("windows_fed", windows_fed.to_string()),
            ("schedule_exhausted", exhausted.to_string()),
        ] {
            let _ = writeln!(self.text, "  {key} = {value}");
        }
    }

    pub fn into_text(self) -> String {
        self.text
    }
}

// ---------------------------------------------------------------------------
// Golden files
// ---------------------------------------------------------------------------

/// A pinned transcript on disk.
///
/// Hand-rolled rather than `insta`, because the whole comparison is thirty
/// lines and the crate's dev-dependencies are deliberately one crate wide. The
/// file is plain text with no header, so a reviewer reads the snapshot itself
/// rather than a snapshot format.
pub struct Snapshot {
    name: &'static str,
}

impl Snapshot {
    pub fn named(name: &'static str) -> Self {
        Self { name }
    }

    fn path(&self) -> PathBuf {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots"))
            .join(format!("{}.txt", self.name))
    }

    /// Compare against the pinned file, failing with the first difference.
    ///
    /// `UPDATE_SNAPSHOTS=1 cargo test` rewrites them. Accepting a rewrite is
    /// the reviewable act: the diff in the commit is the behavior change.
    pub fn assert(&self, actual: &str) {
        let path = self.path();
        if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
            let directory = path.parent().expect("the snapshot path has a directory");
            std::fs::create_dir_all(directory)
                .unwrap_or_else(|error| panic!("create {}: {error}", directory.display()));
            std::fs::write(&path, actual)
                .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
            return;
        }
        let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!(
                "no pinned snapshot at {} ({error}). Re-run with UPDATE_SNAPSHOTS=1 to create it.",
                path.display()
            )
        });
        if expected == actual {
            return;
        }
        panic!("{}", self.difference(&expected, actual));
    }

    fn difference(&self, expected: &str, actual: &str) -> String {
        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = actual.lines().collect();
        let at = expected_lines
            .iter()
            .zip(&actual_lines)
            .position(|(left, right)| left != right)
            .unwrap_or(expected_lines.len().min(actual_lines.len()));

        let from = at.saturating_sub(4);
        let mut report = format!(
            "{} diverges from its pinned snapshot at line {}\n\
             (pinned {} lines, produced {} lines)\n\n",
            self.name,
            at + 1,
            expected_lines.len(),
            actual_lines.len()
        );
        for (offset, context) in expected_lines[from..at].iter().enumerate() {
            let _ = writeln!(report, "  {:>5} | {context}", from + offset + 1);
        }
        for line in at..(at + 6) {
            let pinned = expected_lines.get(line).unwrap_or(&"<end of file>");
            let produced = actual_lines.get(line).unwrap_or(&"<end of file>");
            if pinned == produced {
                let _ = writeln!(report, "  {:>5} | {pinned}", line + 1);
                continue;
            }
            let _ = writeln!(
                report,
                "  {:>5} | -{pinned}\n        | +{produced}",
                line + 1
            );
        }
        let _ = writeln!(
            report,
            "\nIf this change is intended, re-run with UPDATE_SNAPSHOTS=1 and review the diff."
        );
        report
    }
}

// ---------------------------------------------------------------------------
// Recordings the wearer is scripted from
// ---------------------------------------------------------------------------

/// A recording generated to the machine's own cue order.
///
/// The fixture sessions are the fidelity case and they cannot reach every
/// phase: the thumb-down session's cues are randomised, so a run driven from
/// them starves in the second block before it ever polishes or installs (which
/// `scripted_run.rs` pins as a property of the fixtures). A metronome recording
/// laid out in the canonical prompt order reaches every phase, which is what
/// the per-phase abort cases need.
///
/// Spares are whole extra round-robins appended after the rounds the block
/// needs. A rejection consumes the gesture's next cue, which shifts that
/// gesture's later rounds one round-robin along; putting the spares at the end
/// is what leaves room for that without disturbing the cue any other round is
/// answering.
pub struct SyntheticRecording {
    prompts: Vec<ScheduledPrompt>,
    first_cue: u64,
    end_sample: u64,
}

impl SyntheticRecording {
    /// Samples between the starts of consecutive cues. Twice the ~2000 samples
    /// a labeled span occupies, so a rep always closes before the next cue
    /// opens and the machine tracks the recording one cue at a time.
    pub const CUE_PERIOD: u64 = 4000;
    /// How long each cue is held. Longer than the span, shorter than the
    /// period, so a cue the machine reaches late is genuinely spent.
    pub const CUE_HOLD: u32 = 3000;

    /// A recording with `up_rounds` + `spares` thumb-up round-robins followed by
    /// `down_rounds` + `spares` thumb-down ones, the second block starting a
    /// cue period after the first ends.
    pub fn new(first_cue: u64, up_rounds: u32, down_rounds: u32, spares: u32) -> Self {
        let mut prompts = Vec::new();
        let mut at = first_cue;
        let push_block = |block: u8, rounds: u32, at: &mut u64, prompts: &mut Vec<_>| {
            for round in 0..(rounds + spares) {
                for gesture in CalibrationGesture::ALL {
                    prompts.push(ScheduledPrompt {
                        start_sample: *at,
                        sample_count: Self::CUE_HOLD,
                        gesture,
                        round: round as u8,
                        block,
                    });
                    *at += Self::CUE_PERIOD;
                }
            }
        };
        push_block(THUMB_UP_BLOCK, up_rounds, &mut at, &mut prompts);
        at += Self::CUE_PERIOD;
        push_block(THUMB_DOWN_BLOCK, down_rounds, &mut at, &mut prompts);
        Self {
            prompts,
            first_cue,
            end_sample: at,
        }
    }

    pub fn first_cue(&self) -> u64 {
        self.first_cue
    }

    /// One past the last cue's start, which is where a second run in the same
    /// boot can begin its own recording.
    pub fn end_sample(&self) -> u64 {
        self.end_sample
    }

    /// How many window steps it takes to stream the whole recording, plus a
    /// tail so the last rep can close.
    pub fn windows(&self) -> u64 {
        (self.end_sample - self.first_cue) / STRIDE + 64
    }

    pub fn schedule(&self) -> ScriptedSchedule {
        let mut schedule = ScriptedSchedule::new();
        schedule
            .extend(0, &ScriptedSchedule::encode(&self.prompts))
            .expect("a generated schedule is well-formed");
        schedule
    }
}

/// A recorded session on disk, and the cues its manifest names.
pub struct FixtureSession {
    path: &'static str,
}

/// The `class_id` a fixture cue carries, mapped the way `playback-host` maps
/// it. Copied deliberately rather than shared: if the tool's mapping drifts
/// from this, the harness should notice.
trait ClassId {
    fn gesture(&self) -> Option<CalibrationGesture>;
}

impl ClassId for str {
    fn gesture(&self) -> Option<CalibrationGesture> {
        match self.strip_prefix("thumb_up_").unwrap_or(self) {
            "wrist_pronation" | "pronation" => Some(CalibrationGesture::WristPronation),
            "wrist_supination" | "supination" => Some(CalibrationGesture::WristSupination),
            "wrist_radial_deviation" | "radial_deviation" => {
                Some(CalibrationGesture::WristRadialDeviation)
            }
            "wrist_ulnar_deviation" | "ulnar_deviation" => {
                Some(CalibrationGesture::WristUlnarDeviation)
            }
            "thumb_extension" | "hold" => Some(CalibrationGesture::ThumbExtension),
            _ => None,
        }
    }
}

impl FixtureSession {
    pub const THUMB_UP: Self = Self {
        path: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures/sessions/2026-08-07T22-08-47_Matthew/manifest.json"
        ),
    };
    pub const THUMB_DOWN: Self = Self {
        path: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures/sessions/2026-08-07T22-16-46_Matthew/manifest.json"
        ),
    };

    fn manifest(&self) -> serde_json::Value {
        let text = std::fs::read_to_string(self.path)
            .unwrap_or_else(|error| panic!("read {}: {error}", self.path));
        serde_json::from_str(&text).expect("the session manifest is valid JSON")
    }

    /// How many samples this session streams, which is the offset the next
    /// spliced session's cues are shifted by.
    pub fn samples(&self) -> u64 {
        let manifest = self.manifest();
        let raw = &manifest["raw_stream"];
        raw["records"].as_u64().expect("records")
            * raw["samples_per_record"]
                .as_u64()
                .expect("samples_per_record")
    }

    /// This session's cues, shifted by `offset` and tagged with `block`.
    pub fn prompts(&self, offset: u64, block: u8) -> Vec<ScheduledPrompt> {
        let manifest = self.manifest();
        let spans = manifest["cue_spans"].as_array().expect("cue_spans");
        let mut prompts = Vec::new();
        let mut per_gesture = [0u8; 5];
        for span in spans {
            let Some(gesture) = span["class_id"].as_str().expect("class_id").gesture() else {
                continue;
            };
            let start = span["start"].as_u64().expect("start") + offset;
            let stop = span["stop"].as_u64().expect("stop") + offset;
            prompts.push(ScheduledPrompt {
                start_sample: start,
                sample_count: (stop - start) as u32,
                gesture,
                round: per_gesture[gesture.index() as usize],
                block,
            });
            per_gesture[gesture.index() as usize] += 1;
        }
        prompts
    }
}

/// A schedule assembled from the fixture sessions, plus where its first cue is.
pub struct FixtureSchedule {
    prompts: Vec<ScheduledPrompt>,
}

impl FixtureSchedule {
    /// The thumb-up session alone.
    pub fn thumb_up() -> Self {
        Self {
            prompts: FixtureSession::THUMB_UP.prompts(0, THUMB_UP_BLOCK),
        }
    }

    /// Both sessions spliced into one sample space, exactly as
    /// `playback-host calibrate` sends them: the thumb-down session's cues
    /// shifted by everything the thumb-up session streamed, and tagged as the
    /// second block.
    pub fn spliced() -> Self {
        let mut prompts = FixtureSession::THUMB_UP.prompts(0, THUMB_UP_BLOCK);
        let offset = FixtureSession::THUMB_UP.samples();
        prompts.extend(FixtureSession::THUMB_DOWN.prompts(offset, THUMB_DOWN_BLOCK));
        Self { prompts }
    }

    pub fn first_cue(&self) -> u64 {
        self.prompts
            .first()
            .expect("the session has cues")
            .start_sample
    }

    pub fn schedule(&self) -> ScriptedSchedule {
        let mut schedule = ScriptedSchedule::new();
        schedule
            .extend(0, &ScriptedSchedule::encode(&self.prompts))
            .expect("the fixture schedule is well-formed");
        schedule
    }
}

// ---------------------------------------------------------------------------
// The replay
// ---------------------------------------------------------------------------

/// When the driver pulls the run down. Each corresponds to something the
/// device's own driver does: a host abort frame, a front end that stopped
/// answering, a storage failure noticed after a flush.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopWhen {
    /// The first poll at which the run stands in this phase.
    EnteringPhase(CalibrationPhase),
    /// After this many reps have resolved, accepted or not.
    AfterReps(usize),
    /// After this many rounds' rows have reached flash.
    AfterFlushes(u32),
}

impl StopWhen {
    fn reached(&self, run: &Run, reps_resolved: usize) -> bool {
        match self {
            StopWhen::EnteringPhase(phase) => run.phase() == *phase,
            StopWhen::AfterReps(count) => reps_resolved >= *count,
            StopWhen::AfterFlushes(count) => run.flash_flushes() >= *count,
        }
    }
}

/// One scripted run, driven exactly the way the firmware drives it: feed a
/// window, let the machine act, repeat.
pub struct Replay {
    constants: Constants,
    schedule: ScriptedSchedule,
    settle_until: u64,
    start_sample: u64,
    windows: u64,
    /// Rep indices — counting every rep the run resolves, rejected ones
    /// included — to force a front-end lead-off flag on. The one kind of
    /// evidence a recording cannot produce.
    reject_at: Vec<usize>,
    stop: Option<(StopWhen, RunOutcome)>,
    /// A gesture the scorer reports as another one, standing in for the fitter
    /// the flow does not contain. Without this the gate would score a clean
    /// sweep in every scenario and its verdict would pin nothing.
    confuse: Option<(CalibrationGesture, CalibrationGesture)>,
    /// Ask again, at the same sample, before reporting what the first answer
    /// asked for.
    polls_twice: bool,
}

impl Replay {
    pub fn new(constants: Constants, schedule: ScriptedSchedule, first_cue: u64) -> Self {
        Self {
            constants,
            schedule,
            // Two seconds of quiet head, the same margin `playback-host` leaves.
            settle_until: first_cue - constants.samples_in(2000),
            start_sample: 0,
            windows: 4000,
            reject_at: Vec::new(),
            stop: None,
            confuse: None,
            polls_twice: false,
        }
    }

    /// Begin the run at a later sample. A second calibration in one boot picks
    /// up where the first one left the stream.
    pub fn starting_at(mut self, start_sample: u64) -> Self {
        self.start_sample = start_sample;
        self
    }

    pub fn for_windows(mut self, windows: u64) -> Self {
        self.windows = windows;
        self
    }

    pub fn rejecting(mut self, reps: &[usize]) -> Self {
        self.reject_at = reps.to_vec();
        self
    }

    pub fn stopping(mut self, when: StopWhen, outcome: RunOutcome) -> Self {
        self.stop = Some((when, outcome));
        self
    }

    /// Have the held-out scorer report every `truth` window as `predicted`.
    pub fn confusing(mut self, truth: CalibrationGesture, predicted: CalibrationGesture) -> Self {
        self.confuse = Some((truth, predicted));
        self
    }

    /// Poll a second time, at the same sample, before reporting anything the
    /// first poll asked for.
    ///
    /// A driver is entitled to do this — the firmware polls once per feature
    /// window and once per serve-loop iteration, and a flash erase or a
    /// sixteen-pass fit is not finished by the time the next window lands. What
    /// the second poll says is the contract this pins.
    pub fn polling_twice(mut self) -> Self {
        self.polls_twice = true;
        self
    }

    /// Ask again at `at_sample`, writing down the answer.
    fn ask_again(&self, run: Run, transcript: &mut Transcript, at_sample: u64) -> Run {
        if !self.polls_twice {
            return run;
        }
        let (polled, action) = run.poll(at_sample);
        transcript.again(at_sample, action.as_ref());
        polled
    }

    /// The held-out windows one accepted rep contributes to the gate.
    ///
    /// The device scores its most recent reps against the checkpoint fitted at
    /// the end of the previous round. There is no fitter here, so the scorer is
    /// a fixed rule: every window is recovered correctly unless the scenario
    /// asked for a confusion. Fixed, because a gate verdict that varied would
    /// pin nothing.
    fn score_held_out(&self, run: &mut Run, gesture: CalibrationGesture) {
        let predicted = match self.confuse {
            Some((truth, predicted)) if truth == gesture => predicted,
            _ => gesture,
        };
        for _ in 0..self.constants.held_out_reps_per_class {
            run.record_held_out_window(gesture, Some(predicted));
        }
    }

    /// Drive it, writing everything down. Returns the finished run so a
    /// scenario can assert on it beyond what the snapshot pins.
    pub fn drive(self, transcript: &mut Transcript) -> Run {
        let constants = self.constants;
        // The firmware's own order: the driver knows the recording's first cue
        // before the erase has been issued, so the still phase is cut to it as
        // the run is constructed.
        let mut run = Run::settling_until(constants, self.start_sample, self.settle_until);
        let mut wearer = ScriptedWearer::new(self.schedule.clone());

        transcript.note(&format!(
            "start at {}, settling retimed to {}",
            self.start_sample, self.settle_until
        ));
        transcript.project(&run);

        let (polled, action) = run.poll(self.start_sample);
        if let Some(action) = action {
            transcript.action(self.start_sample, &action);
        }
        run = self.ask_again(polled, transcript, self.start_sample);
        run = run.verified();
        transcript.project(&run);

        let mut open: Option<(CalibrationGesture, LabeledSpan, u32)> = None;
        let mut reps = 0usize;
        let mut exhausted = false;
        let mut steps = 0u64;

        for step in 0..self.windows {
            steps = step + 1;
            let now = self.start_sample + WINDOW + step * STRIDE;

            if let Some((when, outcome)) = self.stop {
                if when.reached(&run, reps) {
                    transcript.note(&format!("the driver stops the run: {outcome:?}"));
                    run = run.stop(outcome);
                    transcript.project(&run);
                    let (polled, action) = run.poll(now);
                    run = polled;
                    if let Some(action) = action {
                        transcript.action(now, &action);
                    }
                    transcript.project(&run);
                    break;
                }
            }

            if let Some((gesture, span, present)) = open.as_mut() {
                if let Some(index) = constants.grid().window_ending_at(now) {
                    if span.covers_window(index) {
                        *present += 1;
                    }
                }
                if span.is_complete_at(now) {
                    let evidence = RepEvidence {
                        windows_present: *present,
                        windows_expected: constants.labeled_windows,
                        lead_off_channels: self.reject_at.contains(&reps),
                        ..RepEvidence::default()
                    };
                    let (gesture, span) = (*gesture, *span);
                    let (resolved, outcome) = run.resolve_rep(evidence);
                    run = resolved;
                    if matches!(outcome, RepOutcome::Accepted { .. }) {
                        self.score_held_out(&mut run, gesture);
                    }
                    transcript.rep(reps, now, gesture, span, &evidence, &outcome);
                    transcript.project(&run);
                    reps += 1;
                    open = None;
                }
            }

            let block = match run.phase() {
                CalibrationPhase::ThumbDownRounds => THUMB_DOWN_BLOCK,
                _ => THUMB_UP_BLOCK,
            };
            // Mid-rep the machine's own clock is the right one; asking the
            // schedule again would consume the next cue for a gesture already
            // being answered. The firmware guards this the same way.
            let decision = if open.is_some() {
                ScriptedPoll::At(now)
            } else {
                wearer.poll_sample(
                    run.next_gesture(),
                    block,
                    run.round(),
                    run.rounds_planned(),
                    now,
                )
            };
            let (action, at_sample) = match decision {
                ScriptedPoll::At(sample) => {
                    let (polled, action) = run.poll(sample);
                    run = polled;
                    (action, sample)
                }
                ScriptedPoll::Wait => (None, now),
                ScriptedPoll::Exhausted => {
                    transcript.note(&format!(
                        "the schedule is exhausted at sample {now}, \
                         phase {:?}, round {} of {}, waiting for {:?}",
                        run.phase(),
                        run.round(),
                        run.rounds_planned(),
                        run.next_gesture()
                    ));
                    exhausted = true;
                    break;
                }
            };
            let Some(action) = action else { continue };
            transcript.action(now, &action);
            run = self.ask_again(run, transcript, at_sample);
            if let Action::Prompt { gesture, span, .. } = action {
                open = Some((gesture, span, 0));
            }
            if matches!(action, Action::Finish { .. }) {
                transcript.project(&run);
                break;
            }
            // Do what the phase the run stands in is asking for. The action
            // named it; the phase is what carries the report.
            run = match run {
                Run::Erasing(erasing) => erasing.verified(),
                Run::Flushing(flushing) => flushing.flushed(),
                // One pass per serve-loop iteration on the device; the machine
                // cares about the count, not the wall time.
                Run::Fitting(fitting) => {
                    Run::Fitting(fitting).fitted(constants.passes_per_round, 600)
                }
                Run::Polishing(polishing) => {
                    Run::Polishing(polishing).polished(constants.final_passes, 700)
                }
                Run::Installing(installing) => installing.installed(),
                // The prompt just went out; the rep is the driver's now.
                Run::Performing(performing) => Run::Performing(performing),
                other => panic!(
                    "{:?} offered an action with nothing to report",
                    other.phase()
                ),
            };
            transcript.project(&run);
        }

        transcript.summary(&run, steps, exhausted);
        run
    }
}
