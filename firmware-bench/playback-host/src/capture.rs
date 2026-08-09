//! Everything the device sends back, sorted into files.
//!
//! The orchestration runs this tool in a loop over sessions and fold models and
//! then compares the outputs against the host simulation, so the outputs are
//! written for a program to read, not a person: features as a flat little-endian
//! `f32` blob beside a JSON sidecar that says how to shape it, and every other
//! frame as JSON with its float fields carried as both the bit pattern and the
//! decoded value — the bits because that is what the comparison uses, the value
//! because that is what a person reading the file needs.

use anyhow::{Context, Result};
use protocol::{CalibrationOutcome, CalibrationPhase, Frame, BENCH_FEATURE_COUNT};
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct DecisionRecord {
    window: u32,
    command: u8,
    accepted: bool,
    reject_score_bits: u32,
    reject_score: f32,
}

#[derive(Serialize)]
struct StatusRecord {
    mode: String,
    session: String,
    samples_received: u64,
    windows_processed: u32,
    feature_minimum_microseconds: u32,
    feature_mean_microseconds: u32,
    feature_maximum_microseconds: u32,
    heap_free_bytes: u32,
    largest_free_block_bytes: u32,
    dropped_chunks: u32,
    sequence_gaps: u32,
    stored_rows: u32,
    flash_rows: u32,
}

#[derive(Serialize)]
struct ErrorRecord {
    stage: String,
    detail: String,
}

#[derive(Serialize)]
struct FitRecord {
    wall_milliseconds: u32,
    rows: u32,
    flash_rows: u32,
    flash_walk_microseconds: u32,
    class_count: u32,
    heap_free_before_bytes: u32,
    heap_free_after_bytes: u32,
    largest_free_block_before_bytes: u32,
    largest_free_block_after_bytes: u32,
    model_bytes: usize,
    model_path: String,
}

#[derive(Serialize)]
struct FeatureIndex {
    path: String,
    windows: u32,
    feature_count: usize,
    dtype: &'static str,
    layout: &'static str,
    /// Window indices in arrival order. Contiguous in a healthy run; a gap here
    /// means a `BenchFeatures` frame was lost and the blob's row `n` is not
    /// window `n`.
    windows_received: Vec<u32>,
}

/// Collects device output for one invocation.
pub struct Capture {
    directory: PathBuf,
    features: Vec<u8>,
    windows_received: Vec<u32>,
    decisions: Vec<DecisionRecord>,
    statuses: Vec<StatusRecord>,
    errors: Vec<ErrorRecord>,
    fit: Option<FitRecord>,
    fitted_model: Option<Vec<u8>>,
    /// Every calibration state frame in order, the probe, and the result. The
    /// state sequence is the run report: it is the only place the per-round
    /// pass timing appears, and that number decides the fit schedule's shape.
    calibration_states: Vec<Frame>,
    calibration_probe: Option<Frame>,
    calibration_result: Option<Frame>,
    calibration_result_count: usize,
    /// The most recent credit grant, which the streaming loop reads.
    pub credit: Option<(u32, u32)>,
}

impl Capture {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            features: Vec::new(),
            windows_received: Vec::new(),
            decisions: Vec::new(),
            statuses: Vec::new(),
            errors: Vec::new(),
            fit: None,
            fitted_model: None,
            calibration_states: Vec::new(),
            calibration_probe: None,
            calibration_result: None,
            calibration_result_count: 0,
            credit: None,
        }
    }

    /// Whether the device reported a refusal. The orchestration checks this
    /// instead of reading the log, because a run that produced numbers after an
    /// error produced the wrong numbers.
    pub fn failed(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn windows(&self) -> u32 {
        self.windows_received.len() as u32
    }

    /// Rows the device says it is holding, and chunks it says it refused, from
    /// the newest status. The two numbers are how a caller checks that what it
    /// sent is what arrived: the sample stream is credit-gated, but nothing
    /// paces calibration rows, and a batch the device's command queue refused
    /// would otherwise leave a fit quietly short of its training set.
    pub fn stored_rows(&self) -> Option<u32> {
        self.statuses.last().map(|status| status.stored_rows)
    }

    pub fn dropped_chunks(&self) -> Option<u32> {
        self.statuses.last().map(|status| status.dropped_chunks)
    }

    pub fn has_fit_result(&self) -> bool {
        self.fit.is_some()
    }

    /// Whether the device has reported a calibration ending. The calibrate step
    /// waits on this rather than on a timeout: the fit runs after the last
    /// sample, and how long it takes is exactly what the run is measuring.
    pub fn has_calibration_result(&self) -> bool {
        self.calibration_result.is_some()
    }

    /// The outcome a finished calibration reported, if one finished.
    ///
    /// A run that aborts, fails its fit, loses its front end or runs out of
    /// storage is a failed run, and the device says so in the result rather
    /// than in a `bench_error`. Reading only the error list called every one of
    /// those a success and exited zero, which is exactly the shape of failure
    /// an orchestration cannot see.
    pub fn calibration_outcome(&self) -> Option<CalibrationOutcome> {
        match self.calibration_result {
            Some(Frame::CalibrationResult { outcome, .. }) => Some(outcome),
            _ => None,
        }
    }

    /// Fit progress reports received from the firmware adapter, in order.
    pub fn calibration_fit_progress(
        &self,
    ) -> impl DoubleEndedIterator<Item = (CalibrationPhase, u32, u32)> + '_ {
        self.calibration_states
            .iter()
            .filter_map(|frame| match frame {
                Frame::CalibrationState {
                    phase,
                    fit_passes_done,
                    fit_passes_planned,
                    ..
                } => Some((*phase, *fit_passes_done, *fit_passes_planned)),
                _ => None,
            })
    }

    /// Require the terminal report produced by the abort-validation path.
    pub fn assert_aborted_calibration(&self) -> Result<()> {
        if self.calibration_result_count != 1 {
            anyhow::bail!(
                "expected exactly one calibration result, received {}",
                self.calibration_result_count
            );
        }
        match self.calibration_result.as_ref() {
            Some(Frame::CalibrationResult {
                outcome: CalibrationOutcome::Aborted,
                installed: None,
                previous_retained: true,
                ..
            }) => Ok(()),
            Some(Frame::CalibrationResult {
                outcome,
                installed,
                previous_retained,
                ..
            }) => anyhow::bail!(
                "abort validation reported outcome {outcome:?}, installed {installed:?}, \
                 previous_retained {previous_retained}"
            ),
            _ => anyhow::bail!("abort validation received no calibration result"),
        }
    }

    /// Route one frame. Frames from outside the bench — logs, telemetry, the
    /// device hello — are reported to the operator and not stored: they are
    /// useful while watching a run and meaningless to the comparison.
    pub fn accept(&mut self, frame: Frame) {
        match frame {
            Frame::PlaybackCredit {
                next_sequence,
                free_chunks,
            } => self.credit = Some((next_sequence, free_chunks)),
            Frame::BenchFeatures {
                first_window,
                window_count,
                features,
            } => {
                for offset in 0..window_count {
                    self.windows_received.push(first_window + offset);
                }
                self.features.extend_from_slice(&features);
            }
            Frame::BenchCommits { decisions } => {
                self.decisions
                    .extend(decisions.into_iter().map(|decision| DecisionRecord {
                        window: decision.window,
                        command: decision.command,
                        accepted: decision.accepted,
                        reject_score_bits: decision.reject_score_bits,
                        reject_score: f32::from_bits(decision.reject_score_bits),
                    }));
            }
            Frame::BenchFitResult {
                wall_milliseconds,
                rows,
                flash_rows,
                flash_walk_microseconds,
                class_count,
                heap_free_before_bytes,
                heap_free_after_bytes,
                largest_free_block_before_bytes,
                largest_free_block_after_bytes,
                model,
            } => {
                eprintln!(
                    "fit: {rows} live + {flash_rows} flash rows in {wall_milliseconds} ms \
                     ({flash_walk_microseconds} us per flash pass)"
                );
                self.fit = Some(FitRecord {
                    wall_milliseconds,
                    rows,
                    flash_rows,
                    flash_walk_microseconds,
                    class_count,
                    heap_free_before_bytes,
                    heap_free_after_bytes,
                    largest_free_block_before_bytes,
                    largest_free_block_after_bytes,
                    model_bytes: model.len(),
                    model_path: "fitted_model.f32".into(),
                });
                self.fitted_model = Some(model);
            }
            Frame::BenchStatus {
                mode,
                session,
                samples_received,
                windows_processed,
                feature_minimum_microseconds,
                feature_mean_microseconds,
                feature_maximum_microseconds,
                heap_free_bytes,
                largest_free_block_bytes,
                dropped_chunks,
                sequence_gaps,
                stored_rows,
                flash_rows,
            } => {
                eprintln!(
                    "status: {mode} {windows_processed} windows, feature {feature_mean_microseconds} us mean \
                     ({feature_minimum_microseconds}/{feature_maximum_microseconds}), heap {heap_free_bytes} \
                     free / {largest_free_block_bytes} largest block, {dropped_chunks} dropped, {sequence_gaps} gaps"
                );
                self.statuses.push(StatusRecord {
                    mode,
                    session,
                    samples_received,
                    windows_processed,
                    feature_minimum_microseconds,
                    feature_mean_microseconds,
                    feature_maximum_microseconds,
                    heap_free_bytes,
                    largest_free_block_bytes,
                    dropped_chunks,
                    sequence_gaps,
                    stored_rows,
                    flash_rows,
                });
            }
            Frame::BenchError { stage, detail } => {
                eprintln!("device refused {stage}: {detail}");
                self.errors.push(ErrorRecord { stage, detail });
            }
            frame @ Frame::CalibrationState { .. } => {
                if let Frame::CalibrationState {
                    phase,
                    round,
                    rounds_planned,
                    pass_milliseconds,
                    accepted_reps,
                    rejected_reps,
                    ..
                } = &frame
                {
                    eprintln!(
                        "calibration {phase:?}: round {round}/{rounds_planned},                          {accepted_reps} reps kept, {rejected_reps} rejected,                          last pass {pass_milliseconds} ms"
                    );
                }
                self.calibration_states.push(frame);
            }
            frame @ Frame::CalibrationProbe { .. } => self.calibration_probe = Some(frame),
            frame @ Frame::CalibrationResult { .. } => {
                if let Frame::CalibrationResult {
                    outcome,
                    installed,
                    fit_wall_milliseconds,
                    ..
                } = &frame
                {
                    eprintln!(
                        "calibration finished {outcome:?} (slot {installed:?},                          {fit_wall_milliseconds} ms of fitting)"
                    );
                }
                self.calibration_result_count += 1;
                self.calibration_result = Some(frame);
            }
            Frame::Log { level, message, .. } => eprintln!("device {level:?}: {message}"),
            _ => {}
        }
    }

    /// Write everything collected. Files with nothing in them are not written,
    /// so an orchestration step can test for a file rather than parse an empty
    /// one.
    pub fn write(&self) -> Result<()> {
        std::fs::create_dir_all(&self.directory)
            .with_context(|| format!("create {}", self.directory.display()))?;

        if !self.features.is_empty() {
            let path = self.directory.join("features.f32");
            write_bytes(&path, &self.features)?;
            self.write_json(
                "features.json",
                &FeatureIndex {
                    path: "features.f32".into(),
                    windows: self.windows(),
                    feature_count: BENCH_FEATURE_COUNT,
                    dtype: "little-endian float32",
                    layout: "window-major: window w feature f at element w * 64 + f, \
                             features band-major then channel per ARITHMETIC.md",
                    windows_received: self.windows_received.clone(),
                },
            )?;
        }
        if !self.decisions.is_empty() {
            self.write_json("commits.json", &self.decisions)?;
        }
        if !self.statuses.is_empty() {
            self.write_json("status.json", &self.statuses)?;
        }
        if !self.errors.is_empty() {
            self.write_json("errors.json", &self.errors)?;
        }
        if !self.calibration_states.is_empty() || self.calibration_result.is_some() {
            self.write_json(
                "calibration_run.json",
                &serde_json::json!({
                    "states": self.calibration_states,
                    "probe": self.calibration_probe,
                    "result": self.calibration_result,
                    "result_count": self.calibration_result_count,
                }),
            )?;
        }
        if let (Some(fit), Some(model)) = (self.fit.as_ref(), self.fitted_model.as_ref()) {
            write_bytes(&self.directory.join("fitted_model.f32"), model)?;
            self.write_json("fit_result.json", fit)?;
        }
        Ok(())
    }

    fn write_json(&self, name: &str, value: &impl Serialize) -> Result<()> {
        let path = self.directory.join(name);
        let text = serde_json::to_string_pretty(value).context("serialize capture")?;
        std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Capture;
    use protocol::{CalibrationOutcome, Frame, InstalledSlot};

    fn result(outcome: CalibrationOutcome, previous_retained: bool) -> Frame {
        Frame::CalibrationResult {
            outcome,
            installed: None,
            rounds_completed: 1,
            rows_stored: 45,
            accepted_reps: 5,
            rejected_reps: 0,
            quality: None,
            weak_pair: None,
            classes: Vec::new(),
            fit_wall_milliseconds: 600,
            previous_retained,
        }
    }

    #[test]
    fn abort_validation_requires_one_uninstalled_retained_result() {
        let mut capture = Capture::new("unused");
        capture.accept(result(CalibrationOutcome::Aborted, true));

        assert!(capture.assert_aborted_calibration().is_ok());
    }

    #[test]
    fn abort_validation_rejects_duplicate_results() {
        let mut capture = Capture::new("unused");
        capture.accept(result(CalibrationOutcome::Aborted, true));
        capture.accept(result(CalibrationOutcome::Aborted, true));

        assert!(capture.assert_aborted_calibration().is_err());
    }

    #[test]
    fn abort_validation_requires_previous_calibration_retention() {
        let mut capture = Capture::new("unused");
        capture.accept(result(CalibrationOutcome::Aborted, false));

        assert!(capture.assert_aborted_calibration().is_err());
    }

    #[test]
    fn abort_validation_rejects_an_installed_slot() {
        let mut frame = result(CalibrationOutcome::Aborted, true);
        let Frame::CalibrationResult { installed, .. } = &mut frame else {
            unreachable!()
        };
        *installed = Some(InstalledSlot {
            slot: 1,
            sequence: 2,
        });
        let mut capture = Capture::new("unused");
        capture.accept(frame);

        assert!(capture.assert_aborted_calibration().is_err());
    }
}
