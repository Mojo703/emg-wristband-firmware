//! Offline quality report for one recorded collection session: is it usable for
//! training, and if not, what is the biggest single reason?
//!
//! The measurements are the live preflight panel's, run over a whole recording
//! instead of a two-second buffer — one estimator, in [`crate::signal_quality`],
//! so a session's floor and the number the operator saw before pressing record
//! cannot drift apart.

pub mod channels;
pub mod labels;
pub mod recording;
pub mod response;

use crate::signal_quality::{CHANNELS_PER_SOURCE, NOISE_FLOOR_LIMIT_MICROVOLTS};
use recording::Recording;
use std::fmt::Write;
use std::path::Path;

/// Family-wise significance a channel-by-class response has to clear.
const SIGNIFICANCE_LEVEL: f64 = 0.05;

/// Live channels a session needs before it is worth training on. Below this the
/// electrode placement is the finding, not the data.
const MINIMUM_USABLE_CHANNELS: usize = 4;

/// How far a played cue may sit from its authored note before the schedule
/// counts as a different one. The recorder logs cues from the anchored schedule,
/// so agreement is exact when nothing is wrong.
const SCHEDULE_TOLERANCE_MILLISECONDS: f64 = 5.0;

pub struct Report {
    pub recording: Recording,
    pub survey: channels::ChannelSurvey,
    pub labels: labels::LabelIntegrity,
    pub response: Option<response::ResponseFindings>,
}

pub fn analyze(session_directory: &Path, tracks_root: &Path) -> anyhow::Result<Report> {
    let recording = Recording::open(session_directory)?;
    let survey = channels::survey(&recording);
    let label_integrity = labels::check(&recording, tracks_root);
    let response = response::analyze(
        &recording,
        &survey.live_channels(),
        survey.mains_fundamental_hertz,
        SIGNIFICANCE_LEVEL,
    );
    Ok(Report {
        recording,
        survey,
        labels: label_integrity,
        response,
    })
}

/// What stops this session being training data, worst first. The first entry is
/// the headline.
fn problems(report: &Report) -> Vec<String> {
    let recording = &report.recording;
    let mut problems = Vec::new();

    if !recording.manifest.completed {
        problems.push(format!(
            "the session never completed — the manifest says completed: false, and it holds {:.1} s of EMG with {} cues",
            recording.integrity.recorded_seconds,
            recording.events.cues.len()
        ));
    }
    if recording.events.cues.is_empty() {
        problems.push("no cues were logged, so nothing in the recording is labelled".to_string());
    }

    let live = report.survey.live_channels();
    if live.len() < MINIMUM_USABLE_CHANNELS {
        problems.push(format!(
            "only {} of {} channels carry signal; {} are railed",
            live.len(),
            recording.channels,
            recording.channels - live.len()
        ));
    }
    let passing = report
        .survey
        .channels
        .iter()
        .filter(|quality| quality.passes())
        .count();
    if passing == 0 && !live.is_empty() {
        let (channel, floor) = quietest(report);
        problems.push(format!(
            "no channel meets the {NOISE_FLOOR_LIMIT_MICROVOLTS:.0} uV noise floor; the quietest is ch{channel} at {floor:.1} uV"
        ));
    }

    if report.labels.overlapping_holds > 0
        || report.labels.non_positive_holds > 0
        || !report.labels.duplicate_note_indices.is_empty()
        || report.labels.outside_recording > 0
    {
        problems.push("the cue timeline is not internally consistent".to_string());
    }
    if let Some(schedule) = &report.labels.schedule {
        let worst = |error: Option<f64>| error.map_or(0.0, f64::abs);
        if schedule.authored_notes != report.labels.cues
            || worst(schedule.worst_onset_error_milliseconds) > SCHEDULE_TOLERANCE_MILLISECONDS
            || worst(schedule.worst_hold_error_milliseconds) > SCHEDULE_TOLERANCE_MILLISECONDS
        {
            problems.push(format!(
                "the cues played do not match the authored {} schedule",
                schedule.difficulty
            ));
        }
    }

    let gap_steps: usize = recording
        .gaps
        .iter()
        .map(recording::SourceGaps::gap_steps)
        .sum();
    if gap_steps * 100 > recording.steps.max(1) {
        problems.push(format!(
            "more than 1% of the stream is placeholder: {gap_steps} gap steps"
        ));
    }
    if recording.integrity.windows_in_file != recording.integrity.windows_in_events
        || recording.integrity.trailing_bytes != 0
    {
        problems.push(
            "emg.i16 and events.jsonl disagree about how many windows were recorded".to_string(),
        );
    }

    match &report.response {
        None => problems.push(
            "no response test was possible: the session has no labelled, live data".to_string(),
        ),
        Some(findings) => {
            let responding: Vec<&str> = findings
                .class_ids
                .iter()
                .filter(|class_id| {
                    findings
                        .significant()
                        .iter()
                        .any(|cell| cell.class_id == **class_id)
                })
                .map(String::as_str)
                .collect();
            if responding.is_empty() {
                problems.push(format!(
                    "no gesture produced a cue-locked response above the detection limit ({:.0} uV of added activity on the best channel)",
                    findings
                        .limits
                        .iter()
                        .map(|limit| limit.smallest_resolvable_microvolts)
                        .fold(f64::INFINITY, f64::min)
                ));
            } else if responding.len() < findings.class_ids.len() {
                problems.push(format!(
                    "only {} of {} gestures responded ({}); the rest are indistinguishable from rest",
                    responding.len(),
                    findings.class_ids.len(),
                    responding.join(", ")
                ));
            }
        }
    }

    problems
}

fn quietest(report: &Report) -> (usize, f64) {
    report
        .survey
        .channels
        .iter()
        .filter(|quality| !quality.is_railed())
        .min_by(|left, right| {
            left.noise_floor_microvolts
                .partial_cmp(&right.noise_floor_microvolts)
                .expect("finite floors")
        })
        .map(|quality| (quality.channel, quality.noise_floor_microvolts))
        .unwrap_or((0, f64::NAN))
}

pub fn render(report: &Report) -> String {
    let mut out = String::new();
    render_verdict(report, &mut out);
    render_session(report, &mut out);
    render_integrity(report, &mut out);
    render_gaps(report, &mut out);
    render_channels(report, &mut out);
    render_labels(report, &mut out);
    render_response(report, &mut out);
    out
}

fn rule(out: &mut String) {
    out.push_str(&"=".repeat(78));
    out.push('\n');
}

fn heading(out: &mut String, title: &str) {
    let _ = write!(out, "\n{title}\n");
    out.push_str(&"-".repeat(78));
    out.push('\n');
}

fn render_verdict(report: &Report, out: &mut String) {
    let problems = problems(report);
    rule(out);
    let _ = writeln!(
        out,
        "SESSION QUALITY  {}",
        report.recording.manifest.session_id
    );
    rule(out);
    if problems.is_empty() {
        let _ = writeln!(out, "VERDICT: USABLE for training.");
    } else {
        let _ = writeln!(out, "VERDICT: NOT USABLE for training.");
        let _ = writeln!(out, "\n  Biggest problem: {}", problems[0]);
        for problem in problems.iter().skip(1) {
            let _ = writeln!(out, "  also: {problem}");
        }
    }
    if let Some(findings) = &report.response {
        let significant = findings.significant();
        if significant.is_empty() {
            let smallest = findings.limits.iter().min_by(|left, right| {
                left.smallest_resolvable_microvolts
                    .partial_cmp(&right.smallest_resolvable_microvolts)
                    .expect("finite limits")
            });
            if let Some(limit) = smallest {
                let _ = writeln!(
                    out,
                    "\n  Nothing detected. The recording could have resolved {:.0} uV RMS of added\n  activity on ch{}, so a response weaker than that would not show here.",
                    limit.smallest_resolvable_microvolts, limit.channel
                );
            }
        } else {
            let _ = writeln!(
                out,
                "\n  Detected, family-wise p < {:.2}:",
                findings.significance_level
            );
            for cell in significant {
                let sweep = findings
                    .sweeps
                    .iter()
                    .find(|sweep| sweep.channel == cell.channel && sweep.class_id == cell.class_id);
                let alignment = match sweep {
                    Some(sweep) if sweep.peaks_at_true_alignment() => "peaks at zero lag",
                    Some(_) => "DOES NOT peak at zero lag",
                    None => "no lag sweep",
                };
                let _ = writeln!(
                    out,
                    "    ch{:<3} {:<14} d = {:.2}, p = {:.4}, {}/{} repetitions, {}",
                    cell.channel,
                    cell.class_id,
                    cell.effect_size,
                    cell.family_wise_p,
                    cell.repetitions_above_rest,
                    cell.repetitions,
                    alignment
                );
            }
        }
    }
    out.push('\n');
}

fn render_session(report: &Report, out: &mut String) {
    let manifest = &report.recording.manifest;
    heading(out, "SESSION");
    let _ = writeln!(
        out,
        "  subject {}   arm {}   band offset {} mm, rotation {} deg",
        manifest.metadata.subject.as_deref().unwrap_or("-"),
        manifest.metadata.arm.as_deref().unwrap_or("-"),
        manifest
            .metadata
            .band_offset
            .map_or("-".to_string(), |value| value.to_string()),
        manifest
            .metadata
            .band_rotation
            .map_or("-".to_string(), |value| value.to_string()),
    );
    let _ = writeln!(
        out,
        "  device {}   {} channels at {} Hz, {:.4} uV per count",
        manifest.hardware.device_id,
        manifest.hardware.channels,
        manifest.hardware.sample_rate,
        manifest.hardware.scale_uv,
    );
    let _ = writeln!(
        out,
        "  track   {}   difficulty {}",
        manifest
            .track
            .as_ref()
            .map_or("-", |track| track.title.as_str()),
        manifest.difficulty.as_deref().unwrap_or("-"),
    );
    let _ = writeln!(
        out,
        "  classes {}",
        if manifest.class_ids.is_empty() {
            "-".to_string()
        } else {
            manifest.class_ids.join(", ")
        }
    );
    let _ = writeln!(out, "  completed: {}", manifest.completed);
}

fn render_integrity(report: &Report, out: &mut String) {
    let recording = &report.recording;
    let integrity = &recording.integrity;
    heading(out, "INTEGRITY");
    let _ = writeln!(
        out,
        "  emg.i16          {} bytes = {} channels x {} samples ({:.2} s at {} Hz){}",
        integrity.emg_bytes,
        recording.channels,
        recording.steps,
        integrity.recorded_seconds,
        recording.sample_rate,
        if integrity.trailing_bytes == 0 {
            String::new()
        } else {
            format!(
                ", plus {} trailing bytes that are not a whole window",
                integrity.trailing_bytes
            )
        }
    );
    let _ = writeln!(
        out,
        "  emg.missing      {} bytes, expected {}{}",
        integrity.missing_bytes,
        integrity.expected_missing_bytes,
        if integrity.missing_bytes == integrity.expected_missing_bytes {
            ""
        } else {
            "   MISMATCH"
        }
    );
    let _ = writeln!(
        out,
        "  window length    {} samples per channel (device clock says {}, file says {})",
        recording.samples_per_window,
        integrity
            .samples_per_window_from_clock
            .map_or("-".to_string(), |value| value.to_string()),
        integrity
            .samples_per_window_from_file
            .map_or("-".to_string(), |value| value.to_string()),
    );
    let _ = writeln!(
        out,
        "  windows          {} in emg.i16, {} in events.jsonl{}",
        integrity.windows_in_file,
        integrity.windows_in_events,
        if integrity.windows_in_file == integrity.windows_in_events {
            ""
        } else {
            "   MISMATCH"
        }
    );
    match integrity.wall_clock_seconds {
        Some(seconds) => {
            let _ = writeln!(
                out,
                "  wall clock       {seconds:.2} s of session against {:.2} s of samples ({:+.2} s)",
                integrity.recorded_seconds,
                integrity.recorded_seconds - seconds
            );
        }
        None => {
            let _ = writeln!(
                out,
                "  wall clock       no session_start to measure against"
            );
        }
    }
    let _ = writeln!(
        out,
        "  seq continuity   {} gaps{}, {} non-advancing{}",
        integrity.sequence_gaps.len(),
        if integrity.sequence_gaps.is_empty() {
            String::new()
        } else {
            let explained = integrity
                .sequence_gaps
                .iter()
                .filter(|gap| gap.during_pause)
                .count();
            format!(
                " (missing {} windows{})",
                integrity
                    .sequence_gaps
                    .iter()
                    .map(|gap| gap.missing)
                    .sum::<u64>(),
                if explained == 0 {
                    String::new()
                } else {
                    format!(", {explained} across a recorded pause")
                }
            )
        },
        integrity.sequence_backwards,
        if integrity.sequence_backwards_during_pause == 0 {
            String::new()
        } else {
            format!(
                " ({} a device that renumbered across a pause)",
                integrity.sequence_backwards_during_pause
            )
        }
    );
    render_pauses(recording, out);
    let step = &integrity.device_step_microseconds;
    let _ = writeln!(
        out,
        "  device clock     {} us nominal step, {} to {} us, worst deviation {:+} us",
        step.nominal, step.minimum, step.maximum, step.worst_deviation
    );
    match &integrity.drift_milliseconds {
        Some(drift) => {
            let _ = writeln!(
                out,
                "  backend v device {:+.1} to {:+.1} ms, ending {:+.1} ms ({:+.0} ppm)",
                drift.minimum, drift.maximum, drift.final_value, drift.parts_per_million
            );
        }
        None => {
            let _ = writeln!(out, "  backend v device too few windows to measure drift");
        }
    }
    if recording.events.unreadable_lines > 0 {
        let _ = writeln!(
            out,
            "  events.jsonl     {} unreadable lines",
            recording.events.unreadable_lines
        );
    }
}

/// Pauses are the reason a session can hold a legitimate discontinuity, so they
/// are listed with the continuity numbers they explain.
fn render_pauses(recording: &Recording, out: &mut String) {
    if recording.events.pauses.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "  pauses           {} recorded",
        recording.events.pauses.len()
    );
    let start = recording
        .events
        .session_start
        .unwrap_or(recording.manifest.created as f64);
    for pause in &recording.events.pauses {
        let _ = writeln!(
            out,
            "    {:.1} s in, at track {:.1} s: {} silent {:.1} s, {}",
            (pause.at - start) / 1000.0,
            pause.track_position / 1000.0,
            if pause.device_id.is_empty() {
                "the device"
            } else {
                &pause.device_id
            },
            pause.silent_for / 1000.0,
            match pause.milliseconds() {
                Some(length) => format!("resumed after {:.1} s", length / 1000.0),
                None => "never resumed".to_string(),
            }
        );
    }
}

fn render_gaps(report: &Report, out: &mut String) {
    let recording = &report.recording;
    heading(out, "GAPS");
    if recording.integrity.missing_bytes == 0 {
        let _ = writeln!(out, "  no emg.missing file; every step reads as data");
    }
    let mut rates = Vec::new();
    for (source, gaps) in recording.gaps.iter().enumerate() {
        let steps = gaps.gap_steps();
        let rate = steps as f64 / gaps.steps.max(1) as f64;
        rates.push(rate);
        let _ = writeln!(
            out,
            "  source {source} (ch{:>2}-{:<2})  {steps} gap steps of {} ({:.4}%), {} runs{}",
            source * CHANNELS_PER_SOURCE,
            (source + 1) * CHANNELS_PER_SOURCE - 1,
            gaps.steps,
            rate * 100.0,
            gaps.runs.len(),
            run_lengths(gaps),
        );
        if gaps.unflagged_zero_blocks > 0 {
            let _ = writeln!(
                out,
                "                    {} all-zero steps carry NO gap bit — genuine all-zero reads",
                gaps.unflagged_zero_blocks
            );
        }
        if gaps.flagged_but_nonzero > 0 {
            let _ = writeln!(
                out,
                "                    {} flagged steps are not placeholder zeros",
                gaps.flagged_but_nonzero
            );
        }
    }
    if recording.gaps.len() == 2 {
        let predicted = rates[0] * rates[1] * recording.steps as f64;
        let _ = writeln!(
            out,
            "  simultaneous     {} steps, against {:.1} if the two sources gapped independently",
            recording.simultaneous_gap_steps, predicted
        );
    }
    let unflagged: usize = recording
        .gaps
        .iter()
        .map(|gaps| gaps.unflagged_zero_blocks)
        .sum();
    if unflagged == 0 && recording.gaps.iter().all(|gaps| gaps.gap_steps() == 0) {
        let _ = writeln!(
            out,
            "  clean: no flagged gaps and no unflagged all-zero blocks"
        );
    }
}

fn run_lengths(gaps: &recording::SourceGaps) -> String {
    if gaps.runs.is_empty() {
        return String::new();
    }
    let mut lengths: Vec<usize> = gaps.runs.iter().map(|run| run.length).collect();
    lengths.sort_unstable();
    format!(
        "; run lengths {} to {} steps, median {}",
        lengths[0],
        lengths[lengths.len() - 1],
        lengths[lengths.len() / 2]
    )
}

fn render_channels(report: &Report, out: &mut String) {
    let recording = &report.recording;
    heading(out, "PER-CHANNEL SIGNAL");
    let _ = writeln!(
        out,
        "  mains fundamental {:.3} Hz, measured; {} blocks of 4096 samples averaged",
        report.survey.mains_fundamental_hertz, report.survey.blocks
    );
    let _ = writeln!(
        out,
        "  floor is interharmonic 20-450 Hz band power; the limit is {NOISE_FLOOR_LIMIT_MICROVOLTS:.0} uV RMS\n"
    );
    let _ = writeln!(
        out,
        "   ch   floor uV   mains uV   offset mV   headroom mV   at rail   verdict"
    );
    for (source, chunk) in report
        .survey
        .channels
        .chunks(CHANNELS_PER_SOURCE)
        .enumerate()
    {
        let _ = writeln!(
            out,
            "  -- chip {source} (channels {}-{}) --",
            source * CHANNELS_PER_SOURCE,
            (source + 1) * CHANNELS_PER_SOURCE - 1
        );
        for quality in chunk {
            let verdict = if quality.is_railed() {
                "RAILED"
            } else if quality.passes() {
                "pass"
            } else {
                "over limit"
            };
            let (floor, mains) = if quality.is_railed() {
                ("       -".to_string(), "       -".to_string())
            } else {
                (
                    format!("{:8.1}", quality.noise_floor_microvolts),
                    format!("{:8.0}", quality.mains_microvolts),
                )
            };
            let _ = writeln!(
                out,
                "  {:>4} {floor}   {mains}   {:9.1}   {:11.1}   {:6.1}%   {verdict}",
                quality.channel,
                quality.offset_millivolts,
                quality.headroom_millivolts,
                quality.saturated_fraction * 100.0,
            );
        }
        let live: Vec<&channels::ChannelQuality> = chunk
            .iter()
            .filter(|quality| !quality.is_railed())
            .collect();
        let median_floor = if live.is_empty() {
            "-".to_string()
        } else {
            let mut floors: Vec<f64> = live
                .iter()
                .map(|quality| quality.noise_floor_microvolts)
                .collect();
            floors.sort_by(|left, right| left.partial_cmp(right).expect("finite floors"));
            format!("{:.1} uV", floors[floors.len() / 2])
        };
        let _ = writeln!(
            out,
            "     chip {source}: {} of {} live, median floor {median_floor}",
            live.len(),
            chunk.len(),
        );
    }
    let _ = writeln!(
        out,
        "\n  {} of {} channels live, {} of those under the limit",
        report.survey.live_channels().len(),
        recording.channels,
        report
            .survey
            .channels
            .iter()
            .filter(|quality| quality.passes())
            .count()
    );
}

fn render_labels(report: &Report, out: &mut String) {
    let integrity = &report.labels;
    heading(out, "LABELS");
    let _ = writeln!(
        out,
        "  {} cues{}",
        integrity.cues,
        match (integrity.first_cue_seconds, integrity.last_release_seconds) {
            (Some(first), Some(last)) =>
                format!(", {first:.1} s to {last:.1} s after session_start"),
            _ => String::new(),
        }
    );
    for tally in &integrity.tallies {
        let _ = writeln!(out, "    {:<16} {} cues", tally.class_id, tally.cues);
    }
    if !integrity.classes_off_the_manifest.is_empty() {
        let _ = writeln!(
            out,
            "    cues name classes the manifest does not list: {}",
            integrity.classes_off_the_manifest.join(", ")
        );
    }
    let hold = if integrity.hold_milliseconds.len() == 1 {
        format!("{:.0} ms, every cue", integrity.hold_milliseconds[0])
    } else if integrity.hold_milliseconds.is_empty() {
        "-".to_string()
    } else {
        format!(
            "{:.0} to {:.0} ms across {} distinct values",
            integrity.hold_milliseconds[0],
            integrity.hold_milliseconds[integrity.hold_milliseconds.len() - 1],
            integrity.hold_milliseconds.len()
        )
    };
    let _ = writeln!(out, "  hold length      {hold}");
    let _ = writeln!(
        out,
        "  onset/release    {} non-positive holds, {} overlapping holds, {} out of time order",
        integrity.non_positive_holds, integrity.overlapping_holds, integrity.out_of_order
    );
    if integrity.interrupted_cues > 0 {
        let _ = writeln!(
            out,
            "  interrupted      {} cues cut short by a pause, excluded from the hold checks",
            integrity.interrupted_cues
        );
    }
    let _ = writeln!(
        out,
        "  note_index       {} duplicates, {} missing from 0..{}",
        integrity.duplicate_note_indices.len(),
        integrity.missing_note_indices.len(),
        integrity.cues.saturating_sub(1)
    );
    let _ = writeln!(
        out,
        "  inside recording {} cues fall outside the recorded samples",
        integrity.outside_recording
    );
    match (&integrity.schedule, &integrity.schedule_problem) {
        (Some(schedule), _) => {
            let agreement = if schedule.authored_notes == integrity.cues {
                "MATCHES".to_string()
            } else {
                format!(
                    "DISAGREES: {} notes authored, {} cues played",
                    schedule.authored_notes, integrity.cues
                )
            };
            let _ = writeln!(
                out,
                "  authored schedule {agreement} ({} notes at {})",
                schedule.authored_notes, schedule.difficulty
            );
            let _ = writeln!(
                out,
                "    worst onset error {} ms, worst hold error {} ms",
                schedule
                    .worst_onset_error_milliseconds
                    .map_or("-".to_string(), |error| format!("{error:+.0}")),
                schedule
                    .worst_hold_error_milliseconds
                    .map_or("-".to_string(), |error| format!("{error:+.0}")),
            );
            match schedule.class_rotation {
                Some(rotation) => {
                    let _ = writeln!(
                        out,
                        "    column-to-class binding is rotation {rotation}, {} cues bound elsewhere",
                        schedule.rotation_violations
                    );
                    if schedule.rotation_violations > 0 {
                        let _ = writeln!(
                            out,
                            "    (the session seed is not in the manifest, and a track re-imported\n     since the recording renumbers columns, so this is weaker evidence than\n     the timing above)"
                        );
                    }
                }
                None => {
                    let _ = writeln!(
                        out,
                        "    the track authors no column assignment for {} classes",
                        report.recording.manifest.class_ids.len()
                    );
                }
            }
        }
        (None, Some(problem)) => {
            let _ = writeln!(out, "  authored schedule not cross-checked: {problem}");
        }
        (None, None) => {}
    }
}

fn render_response(report: &Report, out: &mut String) {
    heading(out, "RESPONSE (hold versus rest, interharmonic envelope)");
    let Some(findings) = &report.response else {
        let _ = writeln!(
            out,
            "  not run: the session has no labelled, live, clock-bridged data to test"
        );
        return;
    };
    let _ = writeln!(
        out,
        "  {} envelope points at 32 ms; {} inside holds, {} at rest",
        findings.envelope_points, findings.hold_points, findings.rest_points
    );
    let _ = writeln!(
        out,
        "  family-wise null over the {}x{} grid: 95th percentile of max|d| is {:.2}\n",
        findings.channels.len(),
        findings.class_ids.len(),
        findings.null_ninety_fifth
    );

    let _ = write!(out, "   ch ");
    for class_id in &findings.class_ids {
        let _ = write!(out, "{class_id:>22}");
    }
    out.push('\n');
    for channel in &findings.channels {
        let _ = write!(out, "  {channel:>3} ");
        for class_id in &findings.class_ids {
            let cell = findings.cell(*channel, class_id);
            let mark = if cell.family_wise_p < findings.significance_level {
                '*'
            } else {
                ' '
            };
            let _ = write!(
                out,
                "{:8.2} (p={:.4}){mark}",
                cell.effect_size, cell.family_wise_p
            );
        }
        out.push('\n');
    }
    let _ = writeln!(
        out,
        "  d is Cohen's d of the log envelope, hold against rest; p is family-wise"
    );

    for sweep in &findings.sweeps {
        let _ = writeln!(
            out,
            "\n  lag sweep, ch{} {} ({})",
            sweep.channel,
            sweep.class_id,
            if sweep.peaks_at_true_alignment() {
                "peaks at true alignment"
            } else {
                "DOES NOT peak at true alignment — treat as an artefact"
            }
        );
        let _ = write!(out, "    lag ms ");
        for point in &sweep.points {
            let _ = write!(out, "{:>6.0}", point.lag_milliseconds);
        }
        let _ = write!(out, "\n    d      ");
        for point in &sweep.points {
            let _ = write!(out, "{:>6.2}", point.effect_size);
        }
        out.push('\n');
        let cell = findings.cell(sweep.channel, &sweep.class_id);
        let _ = writeln!(
            out,
            "    {}/{} repetitions above the rest median",
            cell.repetitions_above_rest, cell.repetitions
        );
    }

    heading(out, "DETECTION LIMIT");
    let _ = writeln!(
        out,
        "  The RMS of added, uncorrelated muscle activity, present throughout a hold,\n  that this recording could have resolved at family-wise p < {:.2}.\n",
        findings.significance_level
    );
    let _ = writeln!(out, "   ch   rest envelope uV   log spread   detectable uV");
    for limit in &findings.limits {
        let _ = writeln!(
            out,
            "  {:>3}   {:16.1}   {:10.3}   {:13.1}",
            limit.channel,
            limit.rest_envelope_microvolts,
            limit.log_envelope_spread,
            limit.smallest_resolvable_microvolts
        );
    }
    let _ = writeln!(
        out,
        "\n  Surface EMG on this band runs 14-20 uV RMS at a light pinch. A channel whose\n  detectable figure is above that could not have shown one."
    );
}
