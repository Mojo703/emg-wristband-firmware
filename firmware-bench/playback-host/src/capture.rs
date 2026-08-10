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
use protocol::{Frame, BENCH_FEATURE_COUNT};
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
    /// Anchored-song terminal event. Unlike the removed scripted-wearer result,
    /// an interruption keeps completed rows and checkpoints for Continue.
    calibration_song_terminal: Option<Frame>,
    resident_activation: Option<Frame>,
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
            calibration_song_terminal: None,
            resident_activation: None,
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

    pub fn has_calibration_song_terminal_event(&self) -> bool {
        self.calibration_song_terminal.is_some()
    }

    /// A resident acknowledgement is usable only when it carries the same
    /// numerical and storage validity firmware requires for activation.
    pub fn has_valid_resident_activation(&self) -> bool {
        matches!(
            self.resident_activation,
            Some(Frame::CalibrationResidentActivated { activation })
                if activation.validity.permits_activation()
        )
    }

    /// A completed song is a successful transport result even when its counts
    /// are short: deficits are deliberately permissive and Continue owns the
    /// next authored schedule. Interruption is distinct so callers cannot
    /// mistake lost liveness for a usable terminal result.
    pub fn assert_calibration_song_completed(&self) -> Result<()> {
        match self.calibration_song_terminal.as_ref() {
            Some(Frame::CalibrationSongResult { .. }) => Ok(()),
            Some(Frame::CalibrationSongInterrupted { interruption }) => anyhow::bail!(
                "calibration song interrupted {:?}; completed evidence is retained for Continue",
                interruption.reason
            ),
            _ => anyhow::bail!("no calibration song terminal event"),
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
            frame @ Frame::CalibrationSongResult { .. }
            | frame @ Frame::CalibrationSongInterrupted { .. } => {
                self.calibration_song_terminal = Some(frame);
            }
            frame @ Frame::CalibrationResidentActivated { .. } => {
                self.resident_activation = Some(frame);
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
        if self.calibration_song_terminal.is_some() || self.resident_activation.is_some() {
            self.write_json(
                "calibration_song.json",
                &serde_json::json!({
                    "terminal": self.calibration_song_terminal,
                    "resident_activation": self.resident_activation,
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
    use protocol::{
        CalibrationCandidateValidity, CalibrationResidentActivation, CalibrationRunId,
        CalibrationRunKey, CalibrationScheduleRevision, CalibrationSessionId, Frame,
    };

    fn activation(validity: CalibrationCandidateValidity) -> Frame {
        Frame::CalibrationResidentActivated {
            activation: CalibrationResidentActivation {
                run: CalibrationRunKey {
                    session_id: CalibrationSessionId::new(1).unwrap(),
                    run_id: CalibrationRunId::new(2).unwrap(),
                },
                schedule_revision: CalibrationScheduleRevision::new(3).unwrap(),
                validity,
                resident_sequence: 4,
            },
        }
    }

    #[test]
    fn save_waits_for_an_activation_with_numerical_and_crc_validity() {
        let mut capture = Capture::new("unused");
        capture.accept(activation(CalibrationCandidateValidity {
            model_numerically_valid: true,
            record_crc_valid: false,
        }));
        assert!(!capture.has_valid_resident_activation());

        capture.accept(activation(CalibrationCandidateValidity {
            model_numerically_valid: true,
            record_crc_valid: true,
        }));
        assert!(capture.has_valid_resident_activation());
    }
}
