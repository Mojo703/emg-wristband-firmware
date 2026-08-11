//! Unit 1: the file-backed [`SessionRecorder`].
//!
//! One [`FileSessionRecorder`] owns one session directory for that session's
//! lifetime. The three files it writes are described in
//! [`super::interfaces`]: `emg.i16` takes the raw sample blobs verbatim,
//! `events.jsonl` takes one JSON [`SessionEvent`] per line, and `session.json`
//! holds the manifest — written at the start with `completed: false` and
//! rewritten at the end with `completed: true`, so a crashed session is
//! recognizable by its manifest alone.
//!
//! Writes are unbuffered: every append is one write syscall, so the bytes are
//! the operating system's as soon as the call returns and
//! [`SessionRecorder::health`] can answer from the filesystem itself. A session
//! streams on the order of 64 kilobytes per second, which is nothing for the
//! page cache.
//!
//! # The one invariant worth ordering writes for
//!
//! `events.jsonl` must describe every byte of `emg.i16`, because it is the only
//! thing a windowing tool reads to label the stream. So a window's event line is
//! written *before* its samples. If the sample write then fails, the file is
//! short of what the log describes — a discrepancy a tool finds by comparing
//! expected bytes against the file length, and the recorder itself reports as
//! desynced from that moment on. The opposite order would leave trailing
//! samples no line accounts for, which nothing downstream can detect.

use super::interfaces::{EmgWindow, SessionEvent, SessionManifest, SessionRecorder};
use anyhow::Context;
use protocol::{CollectionSummary, FileReport, RecordedEmg, StreamProgress, UnixMilliseconds};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const EMG_FILE_NAME: &str = "emg.i16";
const EMG_MISSING_FILE_NAME: &str = "emg.missing";
const EVENTS_FILE_NAME: &str = "events.jsonl";
const MANIFEST_FILE_NAME: &str = "session.json";
/// Where the manifest is staged before being renamed over `session.json`.
const MANIFEST_TEMPORARY_FILE_NAME: &str = "session.json.tmp";

/// The current instant on the shared clock (the laptop's wall clock, which the
/// browser also reads through `Date.now()`).
fn now() -> UnixMilliseconds {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    UnixMilliseconds::new(since_epoch.as_millis() as u64)
}

/// Writes one session's `emg.i16`, `events.jsonl`, and `session.json` into its
/// own directory under the sessions root.
pub struct FileSessionRecorder {
    directory: PathBuf,
    manifest: SessionManifest,
    emg_file: File,
    emg_missing_file: File,
    events_file: File,
    /// Bytes of `emg.i16` that `events.jsonl` accounts for. `health()` checks
    /// the file's real length against this, so a counter that drifts from the
    /// disk is caught rather than believed.
    emg_bytes_described: u64,
    /// The byte count the previous `health()` reported, for `advancing`.
    emg_bytes_at_last_health_check: u64,
    /// Set once `emg.i16`'s length stops matching what the event log describes.
    /// Sticky: the tail is unaccounted for for the rest of the session.
    tail_desynced: bool,
    /// `seq` of the most recent window, the baseline for gap counting.
    last_sequence_number: Option<u32>,
    emg_gap_count: u32,
    event_line_count: u64,
}

impl FileSessionRecorder {
    /// Create `sessions_root/<session_id>/` and open the session's files.
    ///
    /// Fails if the directory already exists: a session id collision would
    /// otherwise append this session's samples onto another's. A failure part
    /// way through takes the half-built directory with it, so the same session
    /// id can be retried.
    pub fn begin(
        sessions_root: &Path,
        manifest: SessionManifest,
    ) -> anyhow::Result<FileSessionRecorder> {
        let directory = sessions_root.join(&manifest.session_id.0);
        if directory.exists() {
            anyhow::bail!("session directory already exists: {}", directory.display());
        }
        std::fs::create_dir_all(&directory)
            .with_context(|| format!("creating session directory {}", directory.display()))?;

        match Self::open_files_and_write_prologue(directory.clone(), manifest) {
            Ok(recorder) => Ok(recorder),
            Err(failure) => {
                let _ = std::fs::remove_dir_all(&directory);
                Err(failure)
            }
        }
    }

    /// Everything `begin` does after the directory exists, split out so one
    /// error path can clean the directory up.
    fn open_files_and_write_prologue(
        directory: PathBuf,
        manifest: SessionManifest,
    ) -> anyhow::Result<FileSessionRecorder> {
        let mut recorder = FileSessionRecorder {
            emg_file: create_new_file(&directory.join(EMG_FILE_NAME))?,
            emg_missing_file: create_new_file(&directory.join(EMG_MISSING_FILE_NAME))?,
            events_file: create_new_file(&directory.join(EVENTS_FILE_NAME))?,
            directory,
            manifest,
            emg_bytes_described: 0,
            emg_bytes_at_last_health_check: 0,
            tail_desynced: false,
            last_sequence_number: None,
            emg_gap_count: 0,
            event_line_count: 0,
        };

        recorder.write_manifest(false)?;
        recorder.append_event(&SessionEvent::SessionStart { at: now() })?;
        Ok(recorder)
    }

    /// The session's directory, for the units that write alongside the recorder
    /// (`video.mkv`, `placement.jpg`).
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Write `session.json` from the manifest with the given completion state.
    ///
    /// Written to a temporary file, synced, then renamed into place, so a crash
    /// during the rewrite leaves the previous manifest intact. `completed:
    /// false` is a signal the readers understand; a truncated manifest is not.
    fn write_manifest(&mut self, completed: bool) -> anyhow::Result<()> {
        self.manifest.completed = completed;
        let path = self.directory.join(MANIFEST_FILE_NAME);
        let temporary_path = self.directory.join(MANIFEST_TEMPORARY_FILE_NAME);
        let serialized = serde_json::to_string_pretty(&self.manifest)
            .context("serializing the session manifest")?;

        let mut file = File::create(&temporary_path)
            .with_context(|| format!("creating manifest {}", temporary_path.display()))?;
        file.write_all(serialized.as_bytes())
            .with_context(|| format!("writing manifest {}", temporary_path.display()))?;
        file.write_all(b"\n")
            .with_context(|| format!("writing manifest {}", temporary_path.display()))?;
        file.sync_all()
            .with_context(|| format!("syncing manifest {}", temporary_path.display()))?;
        drop(file);

        std::fs::rename(&temporary_path, &path)
            .with_context(|| format!("renaming manifest into {}", path.display()))?;
        Ok(())
    }

    /// Fold one window's `seq` into the gap count. A sequence number that goes
    /// backwards is a device reboot rather than a loss of some enormous number
    /// of windows, so it counts as a single gap and becomes the new baseline.
    fn account_for_sequence_number(&mut self, sequence_number: u32) {
        if let Some(previous) = self.last_sequence_number {
            if sequence_number > previous {
                self.emg_gap_count += sequence_number - previous - 1;
            } else {
                self.emg_gap_count += 1;
            }
        }
        self.last_sequence_number = Some(sequence_number);
    }

    /// The `detail` line for `emg.i16`. A desynced tail is named here, because
    /// the summary screen's files card is where someone would notice it.
    fn emg_detail(&self) -> String {
        let stream = format!(
            "{} sps, {} ch",
            self.manifest.hardware.sample_rate, self.manifest.hardware.channels
        );
        if self.tail_desynced {
            format!("{stream}, tail desynced")
        } else {
            stream
        }
    }

    /// `emg.i16`'s length as the filesystem reports it.
    fn emg_bytes_on_disk(&self) -> anyhow::Result<u64> {
        let path = self.directory.join(EMG_FILE_NAME);
        Ok(std::fs::metadata(&path)
            .with_context(|| format!("measuring {}", path.display()))?
            .len())
    }

    fn file_reports(&self) -> anyhow::Result<Vec<FileReport>> {
        let entries = [
            (EMG_FILE_NAME, self.emg_detail()),
            (
                EVENTS_FILE_NAME,
                format!("{} events", self.event_line_count),
            ),
            (MANIFEST_FILE_NAME, "tags + hardware identity".to_string()),
        ];
        entries
            .into_iter()
            .map(|(name, detail)| {
                let path = self.directory.join(name);
                let bytes = std::fs::metadata(&path)
                    .with_context(|| format!("measuring {}", path.display()))?
                    .len();
                Ok(FileReport {
                    name: name.to_string(),
                    bytes,
                    detail,
                })
            })
            .collect()
    }
}

/// Open a file that must not already exist, for appending.
fn create_new_file(path: &Path) -> anyhow::Result<File> {
    OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))
}

impl SessionRecorder for FileSessionRecorder {
    fn append_emg(&mut self, window: EmgWindow<'_>) -> anyhow::Result<()> {
        // The event line goes first: see the module doc. `emg.i16` may fall
        // short of the log, never the other way round.
        self.append_event(&SessionEvent::EmgWindow {
            seq: window.seq,
            t0_us: window.t0_us,
            at: now(),
        })?;
        self.account_for_sequence_number(window.seq);
        self.emg_bytes_described += window.samples.len() as u64;

        if let Err(failure) = self.emg_file.write_all(window.samples) {
            // The line describing these samples is already on disk, so the tail
            // is now short (or partially written). Say so from here on.
            self.tail_desynced = true;
            return Err(anyhow::Error::new(failure).context("appending EMG samples"));
        }
        // The mask rides in a sidecar file, same window order as `emg.i16`. A
        // failure here loses gap information, not data, so it does not desync
        // the sample tail.
        self.emg_missing_file
            .write_all(window.missing)
            .context("appending the EMG missing mask")?;
        Ok(())
    }

    fn append_event(&mut self, event: &SessionEvent) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(event).context("serializing a session event")?;
        line.push('\n');
        self.events_file
            .write_all(line.as_bytes())
            .context("appending a session event")?;
        self.event_line_count += 1;
        Ok(())
    }

    fn health(&mut self) -> StreamProgress {
        // Asked of the filesystem, not of a counter: appends are unbuffered, so
        // the length is already the truth, and a mismatch against what the event
        // log describes is exactly the desync worth reporting.
        let bytes_on_disk = match self.emg_bytes_on_disk() {
            Ok(bytes) => bytes,
            Err(failure) => {
                // Unable to see the file we are supposedly writing: not healthy,
                // and not something to paper over with the last known count.
                tracing::warn!("cannot measure {EMG_FILE_NAME}: {failure:#}");
                self.tail_desynced = true;
                return StreamProgress {
                    bytes_on_disk: self.emg_bytes_at_last_health_check,
                    advancing: false,
                };
            }
        };
        if bytes_on_disk != self.emg_bytes_described {
            self.tail_desynced = true;
        }
        let advancing = !self.tail_desynced && bytes_on_disk > self.emg_bytes_at_last_health_check;
        self.emg_bytes_at_last_health_check = bytes_on_disk;
        StreamProgress {
            bytes_on_disk,
            advancing,
        }
    }

    fn recorded_emg(&self) -> RecordedEmg {
        let channels = u64::from(self.manifest.hardware.channels).max(1);
        RecordedEmg {
            samples_per_channel: self.emg_bytes_described / 2 / channels,
            sample_rate: self.manifest.hardware.sample_rate,
        }
    }

    fn emg_gap_count(&self) -> u32 {
        self.emg_gap_count
    }

    fn finish(mut self: Box<Self>, summary: &CollectionSummary) -> anyhow::Result<Vec<FileReport>> {
        self.append_event(&SessionEvent::SessionEnd {
            at: now(),
            summary: summary.clone(),
        })?;
        self.write_manifest(true)?;
        self.emg_file.sync_all().context("syncing emg.i16")?;
        self.events_file
            .sync_all()
            .context("syncing events.jsonl")?;

        // Last word on the tail, in case health() was never called.
        if self.emg_bytes_on_disk()? != self.emg_bytes_described {
            self.tail_desynced = true;
        }
        self.file_reports()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{
        ActivityId, AnalogFrontEnd, Arm, BeatsPerMinute, BoardRevision, ClassId, Degrees,
        DeviceConfig, DeviceProvenance, DeviceTransport, DifficultyLevel, DurationMilliseconds,
        FirmwareBuild, Millimeters, RegisterReadback, SessionId, SessionMetadata, SubjectId,
        SweatId, TrackId, TrackInfo,
    };
    use std::collections::BTreeMap;
    use std::num::NonZeroU16;

    use crate::collect::interfaces::HardwareIdentity;

    fn sample_manifest(session_id: &str) -> SessionManifest {
        SessionManifest {
            session_id: SessionId(session_id.to_string()),
            created: UnixMilliseconds::new(1_784_000_000_000),
            metadata: SessionMetadata {
                subject: SubjectId("matthew".to_string()),
                arm: Arm::Left,
                gloves: false,
                skin_prep: true,
                band_offset: Millimeters(40),
                band_rotation: Degrees(-15),
                donned: UnixMilliseconds::new(1_783_999_000_000),
                activity: ActivityId("seated".to_string()),
                sweat: SweatId("dry".to_string()),
                note: Some("bench check".to_string()),
            },
            hardware: HardwareIdentity {
                device_id: "opal-01".to_string(),
                transport: DeviceTransport::Wifi,
                channels: 16,
                sample_rate: 2000,
                scale_uv: 0.1875,
                device_config: DeviceConfig {
                    gestures: 4,
                    keymap: Vec::new(),
                    wifi_ssid: Some("bench".to_string()),
                    sensitivity: "medium".to_string(),
                    sensitivity_levels: Vec::new(),
                    tau: 0.8,
                    needed: 3,
                },
                provenance: DeviceProvenance {
                    firmware: FirmwareBuild {
                        crate_version: "0.1.0".to_string(),
                        git_commit: "34370e7".to_string(),
                        working_tree_modified: false,
                        built_at: "2026-08-04T11:22:33Z".to_string(),
                    },
                    analog_front_ends: vec![AnalogFrontEnd {
                        chip: 0,
                        registers: vec![RegisterReadback {
                            name: "CONFIG1".to_string(),
                            address: 0x01,
                            value: Some(0xC4),
                        }],
                    }],
                },
                board_revision: Some(BoardRevision {
                    board: "rev A bodged".to_string(),
                    harness: "ribbon 2".to_string(),
                }),
            },
            don_count: 3,
            difficulty: DifficultyLevel::Medium,
            track: TrackInfo {
                id: TrackId("metronome".to_string()),
                title: "Metronome".to_string(),
                beats_per_minute: BeatsPerMinute(NonZeroU16::new(120).unwrap()),
                duration: DurationMilliseconds::new(60_000),
            },
            record_video: true,
            audio: crate::collect::interfaces::AudioPlayback {
                output: "silent".to_string(),
                sample_rate: 44_100,
            },
            class_ids: vec![
                ClassId("index_pinch".to_string()),
                ClassId("fist".to_string()),
            ],
            completed: false,
        }
    }

    fn sample_summary() -> CollectionSummary {
        let mut cues_per_class = BTreeMap::new();
        cues_per_class.insert(ClassId("index_pinch".to_string()), 20);
        cues_per_class.insert(ClassId("fist".to_string()), 20);
        CollectionSummary {
            duration: DurationMilliseconds::new(60_000),
            cues_per_class,
            activity_hits: 37,
            files: Vec::new(),
            emg_gap_count: 0,
            video_start_offset: None,
        }
    }

    /// A directory nobody else in this test run is using.
    fn unique_sessions_root(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("dashboard-recorder-{label}-{stamp}"));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn records_a_session_end_to_end() {
        let sessions_root = unique_sessions_root("end-to-end");
        let session_directory = sessions_root.join("2026-07-30T16-40-12_matthew");

        let mut recorder = FileSessionRecorder::begin(
            &sessions_root,
            sample_manifest("2026-07-30T16-40-12_matthew"),
        )
        .unwrap();

        // The start writes the manifest and the first event line, and leaves
        // emg.i16 empty.
        assert!(session_directory.join(EMG_FILE_NAME).exists());
        assert_eq!(
            std::fs::metadata(session_directory.join(EMG_FILE_NAME))
                .unwrap()
                .len(),
            0
        );

        // Nothing appended yet, so the first health check cannot be advancing.
        let idle = recorder.health();
        assert_eq!(idle.bytes_on_disk, 0);
        assert!(!idle.advancing);

        let blobs: [Vec<u8>; 4] = [
            vec![1, 0, 2, 0],
            vec![3, 0, 4, 0],
            vec![5, 0, 6, 0],
            vec![7, 0, 8, 0],
        ];
        // seq 7 → 8 is contiguous; 8 → 12 misses three windows; 12 → 3 is a
        // device reboot, one gap.
        let sequence_numbers = [7_u32, 8, 12, 3];

        for (index, (sequence_number, blob)) in
            sequence_numbers.iter().zip(blobs.iter()).enumerate()
        {
            recorder
                .append_emg(EmgWindow {
                    seq: *sequence_number,
                    t0_us: 1_000 * (index as u64 + 1),
                    samples: blob,
                    missing: &[0b0000_0001],
                })
                .unwrap();
        }

        assert_eq!(recorder.emg_gap_count(), 4);

        // Bytes grew since the last check, then stop growing. The reported count
        // is checked against the filesystem read independently, mid-session, so a
        // recorder that reported a bookkeeping number instead of the file's real
        // length would fail here.
        let length_on_disk = std::fs::metadata(session_directory.join(EMG_FILE_NAME))
            .unwrap()
            .len();
        assert_eq!(length_on_disk, 16);
        let growing = recorder.health();
        assert_eq!(growing.bytes_on_disk, length_on_disk);
        assert!(growing.advancing);
        let stalled = recorder.health();
        assert_eq!(stalled.bytes_on_disk, 16);
        assert!(!stalled.advancing);

        recorder
            .append_event(&SessionEvent::TrackStarted { at: now() })
            .unwrap();

        let summary = sample_summary();
        let reports = Box::new(recorder).finish(&summary).unwrap();

        // emg.i16 is exactly the blobs, concatenated, untouched.
        let recorded = std::fs::read(session_directory.join(EMG_FILE_NAME)).unwrap();
        let expected: Vec<u8> = blobs.iter().flatten().copied().collect();
        assert_eq!(recorded, expected);

        // events.jsonl parses back, one event per line, in the order written.
        let events_text =
            std::fs::read_to_string(session_directory.join(EVENTS_FILE_NAME)).unwrap();
        let events: Vec<SessionEvent> = events_text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 7);
        assert!(matches!(events[0], SessionEvent::SessionStart { .. }));
        for (index, event) in events[1..5].iter().enumerate() {
            match event {
                SessionEvent::EmgWindow { seq, t0_us, .. } => {
                    assert_eq!(*seq, sequence_numbers[index]);
                    assert_eq!(*t0_us, 1_000 * (index as u64 + 1));
                }
                other => panic!("expected an EmgWindow event, got {other:?}"),
            }
        }
        assert!(matches!(events[5], SessionEvent::TrackStarted { .. }));
        match &events[6] {
            SessionEvent::SessionEnd {
                summary: recorded_summary,
                ..
            } => assert_eq!(recorded_summary, &summary),
            other => panic!("expected a SessionEnd event, got {other:?}"),
        }

        // The manifest is now marked completed, and otherwise unchanged.
        let manifest_text =
            std::fs::read_to_string(session_directory.join(MANIFEST_FILE_NAME)).unwrap();
        let manifest: SessionManifest = serde_json::from_str(&manifest_text).unwrap();
        assert!(manifest.completed);
        assert_eq!(manifest.session_id.0, "2026-07-30T16-40-12_matthew");
        assert_eq!(manifest.hardware.channels, 16);

        // Three reports, sized from the filesystem, with the documented details.
        assert_eq!(reports.len(), 3);
        assert_eq!(reports[0].name, EMG_FILE_NAME);
        assert_eq!(reports[0].bytes, 16);
        assert_eq!(reports[0].detail, "2000 sps, 16 ch");
        assert_eq!(reports[1].name, EVENTS_FILE_NAME);
        assert_eq!(reports[1].detail, "7 events");
        assert_eq!(reports[1].bytes, events_text.len() as u64);
        assert_eq!(reports[2].name, MANIFEST_FILE_NAME);
        assert_eq!(reports[2].detail, "tags + hardware identity");
        assert_eq!(reports[2].bytes, manifest_text.len() as u64);

        std::fs::remove_dir_all(&sessions_root).unwrap();
    }

    /// `emg.i16` disagreeing with the event log is reported, not smoothed over:
    /// the stream stops counting as advancing and the files card says so. The
    /// disagreement is manufactured here by writing to the file behind the
    /// recorder's back, which is what a failed or partial sample write leaves.
    #[test]
    fn a_tail_that_disagrees_with_the_event_log_is_reported() {
        let sessions_root = unique_sessions_root("desync");
        let session_id = "2026-07-30T19-00-00_matthew";
        let mut recorder =
            FileSessionRecorder::begin(&sessions_root, sample_manifest(session_id)).unwrap();
        let emg_path = sessions_root.join(session_id).join(EMG_FILE_NAME);

        recorder
            .append_emg(EmgWindow {
                seq: 1,
                t0_us: 1_000,
                samples: &[1, 0, 2, 0],
                missing: &[],
            })
            .unwrap();
        let healthy = recorder.health();
        assert_eq!(healthy.bytes_on_disk, 4);
        assert!(healthy.advancing);

        // Bytes nothing in events.jsonl describes.
        OpenOptions::new()
            .append(true)
            .open(&emg_path)
            .unwrap()
            .write_all(&[9, 0])
            .unwrap();

        let desynced = recorder.health();
        assert_eq!(desynced.bytes_on_disk, 6);
        assert!(!desynced.advancing, "a desynced tail is not progress");

        // Sticky: appending correctly afterwards does not clear it.
        recorder
            .append_emg(EmgWindow {
                seq: 2,
                t0_us: 2_000,
                samples: &[3, 0, 4, 0],
                missing: &[],
            })
            .unwrap();
        assert!(!recorder.health().advancing);

        let reports = Box::new(recorder).finish(&sample_summary()).unwrap();
        assert_eq!(reports[0].name, EMG_FILE_NAME);
        assert_eq!(reports[0].detail, "2000 sps, 16 ch, tail desynced");

        std::fs::remove_dir_all(&sessions_root).unwrap();
    }

    #[test]
    fn refuses_an_existing_session_directory() {
        let sessions_root = unique_sessions_root("existing");
        let manifest = sample_manifest("2026-07-30T17-00-00_matthew");

        let first = FileSessionRecorder::begin(&sessions_root, manifest.clone()).unwrap();
        drop(first);

        let second = FileSessionRecorder::begin(&sessions_root, manifest);
        assert!(second.is_err());

        std::fs::remove_dir_all(&sessions_root).unwrap();
    }

    #[test]
    fn a_crashed_session_leaves_an_incomplete_manifest() {
        let sessions_root = unique_sessions_root("crashed");
        let session_id = "2026-07-30T18-00-00_matthew";
        let recorder =
            FileSessionRecorder::begin(&sessions_root, sample_manifest(session_id)).unwrap();
        drop(recorder);

        let manifest_text =
            std::fs::read_to_string(sessions_root.join(session_id).join(MANIFEST_FILE_NAME))
                .unwrap();
        let manifest: SessionManifest = serde_json::from_str(&manifest_text).unwrap();
        assert!(!manifest.completed);

        std::fs::remove_dir_all(&sessions_root).unwrap();
    }
}
