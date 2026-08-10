//! The collection session manager: the state machine tying the recorder, the
//! webcam capture, and the beatmap generator to the browser game.
//!
//! One manager exists for the whole backend. Browsers all see the same session:
//! collection frames fan out over a broadcast channel every browser session
//! forwards, and control frames from any browser funnel into the methods here.
//! The manager is the *only* writer of collection truth — the browser renders.
//!
//! Lifecycle: `Idle` → (`StartCollection`) → `Starting` (arming in flight) →
//! `Running` (a spawned session task owns the recorder, the device's EMG
//! subscription, and the audio playback; `Armed` until `StartTrack` begins the
//! track, then `Playing`) → `Reviewing` (track ended, or `FinishCollection`
//! ended it early; files finalized on disk either way) → (`StopCollection {
//! save }`) → `Idle`. Every ending goes through review: a take that reached
//! disk is never thrown away without the operator seeing its summary.
//!
//! # The clock
//!
//! The backend plays the audio, so the playhead is the one in
//! [`crate::collect::audio`] and nothing here reads a browser timestamp. A
//! session's `anchor` is the instant the subject heard audio t = 0, taken from
//! the mixer's own cursor, and a cue's wall-clock time is `anchor + note
//! position`. `PlaybackPosition` frames publish the same playhead to browsers,
//! which extrapolate it for a smooth playfield and own none of it.
//!
//! Obligations honored here (from the unit reviews):
//! - the recorder creates the session directory before video starts in it;
//! - every `VideoCapture` call that can block runs under `spawn_blocking`;
//! - health is polled at ~500 ms (which is also the video start-offset
//!   resolution);
//! - recorder and video file reports merge into one `CollectionSummary.files`;
//! - `generate` receives classes in catalog order (the lane order browsers see);
//! - a second `StartTrack` for a started session is ignored;
//! - `StartCollection` ids are validated against the live catalog.

use crate::collect::audio::{self, AudioOutput, Playback, Timeline};
use crate::collect::beatmap::{CatalogPaths, TrackCatalog};
use crate::collect::interfaces::{
    AudioPlayback, BeatmapGenerator, EmgWindow, HardwareIdentity, RecordingHandle, SessionEvent,
    SessionManifest, SessionRecorder, VideoCapture, VideoReport,
};
use crate::collect::provenance::{ProvenanceStore, StoredAudio};
use crate::collect::recorder::FileSessionRecorder;
use crate::collect::video::{CameraSettings, FfmpegVideoCapture};
use crate::registry::Registry;
use anyhow::Context;
use dashboard::guided_session::{
    CoordinatorError, DeviceConnectionIdentity, GuidedMode, GuidedModeAdapter,
    GuidedSessionBinding, GuidedSessionCoordinator, SessionExit, SessionLease,
};
use futures_util::FutureExt;
use protocol::{
    Beatmap, BoardRevision, ClassId, CollectionPause, CollectionPhase, CollectionSummary,
    DifficultyLevel, DurationMilliseconds, FileReport, Frame, LogLevel, NoteIndex, PauseCause,
    RecordingHealth, SessionId, SessionMetadata, StreamProgress, TrackId, TrackInfo,
    TrackMilliseconds, UnixMilliseconds,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{broadcast, mpsc, watch};

/// Fan-out capacity for collection frames toward browsers. State ticks are
/// small and periodic; a browser that lags simply sees the next tick.
const OUTBOUND_CAPACITY: usize = 64;

/// Lifecycle commands are sparse. Reaching this bound means the session task is
/// unhealthy; growing a stale backlog would make old Start/Pause/Finish intents
/// execute after it recovers.
const CONTROL_CAPACITY: usize = 8;

/// How long arming waits for the device's first EMG window (which carries the
/// hardware identity) before giving up.
const FIRST_WINDOW_TIMEOUT: Duration = Duration::from_secs(3);

/// An armed session nobody ever starts finalizes itself after this long and
/// goes to review like any other take. Generous, because the EMG it has been
/// recording since arming is real data and the operator decides its fate.
const ARMED_TIMEOUT: Duration = Duration::from_secs(600);

/// How long `StartTrack` and `ResumeTrack` wait for the mixer to publish the
/// playhead they just commanded. A device period is a couple of milliseconds.
const PLAYHEAD_TIMEOUT: Duration = Duration::from_millis(1_000);

/// Session-task cadence. Cue events resolve on this grid; health is published
/// every other tick (~500 ms), which is also the video start-offset resolution.
const TICK: Duration = Duration::from_millis(250);

/// The activity window around a cue: activity between `cue - 100 ms` and
/// `cue + 400 ms` counts toward that cue's hit verdict.
const ACTIVITY_WINDOW_BEFORE_MILLISECONDS: u64 = 100;
const ACTIVITY_WINDOW_AFTER_MILLISECONDS: u64 = 400;

/// A cue is a hit when its peak activity exceeds the rolling baseline by this
/// factor. Deliberately loose: the detector gates nothing, it only colours the
/// streak and counts `activity_hits`.
const ACTIVITY_HIT_FACTOR: f32 = 1.6;

/// Silence from the device that freezes the cue timeline. Windows arrive every
/// 250 ms; across the two sessions on disk the worst gap between consecutive
/// arrivals is 339 ms and the 95th percentile is 272 ms, so this is six
/// consecutive windows lost — a link that is down rather than one that stutters.
/// Checked on the tick, so a pause lands within 1.75 s of the last sample.
const STALL_THRESHOLD: Duration = Duration::from_millis(1_500);

/// Extra time after the track's end before the session finalizes itself.
const TRACK_END_SLACK: Duration = Duration::from_secs(1);

/// The pending placement photo's holding name inside the sessions root, before
/// a session adopts it.
const PENDING_PHOTO_NAME: &str = ".pending-placement.jpg";

fn now() -> UnixMilliseconds {
    let since_epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch");
    UnixMilliseconds::new(since_epoch.as_millis() as u64)
}

/// Control messages from browsers into a running session task.
enum SessionControl {
    /// Begin the track: audio playback and the cue timeline together.
    StartTrack,
    /// Freeze both where they stand.
    PauseTrack,
    /// Unfreeze both from where they froze.
    ResumeTrack,
    /// End the session now and move to review, mid-track or not.
    Finish,
}

/// A latest-wins safety interrupt, kept separate from operator commands.
///
/// Disconnect is idempotent, so multiple departures may coalesce. The
/// generation makes every departure observable even when the receiver has
/// already handled an earlier one. A watch channel cannot become full, which
/// means a saturated operator queue can never prevent an unattended recording
/// from being frozen.
#[derive(Clone, Copy, Default)]
struct BrowserDeparture(u64);

/// Latest-wins audio settings are state, not an event stream. A watch channel
/// prevents a dragged volume slider from starving lifecycle commands.
#[derive(Clone)]
struct LiveAudioSettings {
    output: AudioOutput,
    gain: f32,
}

enum Phase {
    Idle,
    /// An arming task is in flight; further starts are rejected.
    Starting {
        session_id: dashboard::guided_session::GuidedSessionId,
    },
    Running {
        session_id: dashboard::guided_session::GuidedSessionId,
        control: mpsc::Sender<SessionControl>,
        browser_departure: watch::Sender<BrowserDeparture>,
        audio: watch::Sender<LiveAudioSettings>,
    },
    /// The wire-facing session/summary data lives in the cached `latest_state`
    /// frame; the phase itself only needs what `StopCollection` acts on. A
    /// practice session has no directory — nothing was recorded.
    Reviewing {
        directory: Option<PathBuf>,
    },
}

/// How the game sounds. The volume is in thousandths because that is what
/// crosses the wire — a whole number survives the browser's CBOR encoder, where
/// a fraction would arrive as a float the backend refuses.
struct AudioSettings {
    output: AudioOutput,
    volume_permille: u32,
    /// The backend was started with `EMG_AUDIO_OUTPUT=silent`, so it stays
    /// silent whatever a browser asks for. Automated runs rely on this: a
    /// stored preference must not make a test audible to whoever is sitting at
    /// the machine.
    forced_silent: bool,
}

impl AudioSettings {
    fn gain(&self) -> f32 {
        self.volume_permille as f32 / 1000.0
    }

    /// The device name a browser should show as chosen, or `None` for the
    /// host's default.
    fn output_name(&self) -> Option<String> {
        match &self.output {
            AudioOutput::Named(name) => Some(name.clone()),
            AudioOutput::Default | AudioOutput::Silent => None,
        }
    }
}

/// A device stream claimed for a session, with the identity read off its first
/// EMG window.
struct AcquiredDevice {
    emg_receiver: broadcast::Receiver<Frame>,
    hardware: HardwareIdentity,
}

struct PendingPhoto {
    path: PathBuf,
    captured: UnixMilliseconds,
}

struct ManagerState {
    phase: Phase,
    placement_photo: Option<PendingPhoto>,
    /// The most recent state frame, replayed to browsers that connect mid
    /// session so they never wait for the next tick.
    latest_state: Option<Frame>,
    /// The running session's beatmap frame, replayed on connect for the same
    /// reason.
    latest_beatmap: Option<Frame>,
}

pub struct CollectionManager {
    /// Behind a lock because importing or deleting a track rebuilds it while
    /// the backend runs; every read here is short and uncontended.
    catalog: std::sync::RwLock<TrackCatalog>,
    catalog_paths: CatalogPaths,
    video: Arc<Mutex<FfmpegVideoCapture>>,
    registry: Arc<Registry>,
    sessions_root: PathBuf,
    outbound: broadcast::Sender<Frame>,
    /// What the host remembers between sessions: board revisions per device,
    /// don counts per subject and arm, and how the game sounds.
    provenance: ProvenanceStore,
    /// How the game sounds: which sink, and how loud the music is under the
    /// cues. Both are live — a running session is told about a change rather
    /// than waiting for the next one.
    audio_settings: Mutex<AudioSettings>,
    guided_sessions: GuidedSessionCoordinator,
    state: Mutex<ManagerState>,
}

impl CollectionManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        catalog: TrackCatalog,
        catalog_paths: CatalogPaths,
        camera_device: PathBuf,
        camera_settings: CameraSettings,
        registry: Arc<Registry>,
        sessions_root: PathBuf,
        provenance: ProvenanceStore,
        audio_output: AudioOutput,
        guided_sessions: GuidedSessionCoordinator,
    ) -> Arc<Self> {
        let (outbound, _) = broadcast::channel(OUTBOUND_CAPACITY);
        // The environment names the sink this run starts on; the stored
        // preference supplies one only when it does not. `silent` is the
        // exception in both directions — it is a promise that this backend
        // makes no sound, so nothing remembered or asked for can undo it.
        let forced_silent = audio_output == AudioOutput::Silent;
        let stored = provenance.audio();
        let audio_settings = AudioSettings {
            output: match (&audio_output, &stored.output) {
                (AudioOutput::Default, Some(name)) => AudioOutput::Named(name.clone()),
                _ => audio_output,
            },
            volume_permille: stored.volume_permille,
            forced_silent,
        };
        Arc::new(Self {
            catalog: std::sync::RwLock::new(catalog),
            catalog_paths,
            video: Arc::new(Mutex::new(FfmpegVideoCapture::new(
                camera_device,
                camera_settings,
            ))),
            registry,
            sessions_root,
            outbound,
            provenance,
            audio_settings: Mutex::new(audio_settings),
            guided_sessions,
            state: Mutex::new(ManagerState {
                phase: Phase::Idle,
                placement_photo: None,
                latest_state: None,
                latest_beatmap: None,
            }),
        })
    }

    /// The board and harness remembered for a device, for the browser's view.
    pub fn board_revision(&self, device_id: &str) -> Option<BoardRevision> {
        self.provenance.board_revision(device_id)
    }

    /// Remember a device's board and harness until someone says otherwise.
    pub fn set_board_revision(&self, device_id: &str, revision: BoardRevision) {
        self.provenance.set_board_revision(device_id, revision);
    }

    /// Subscribe a browser session to collection frames.
    pub fn subscribe(&self) -> broadcast::Receiver<Frame> {
        self.outbound.subscribe()
    }

    /// The frames a freshly connected browser needs to render the current
    /// collection reality: catalog, current state, and the running session's
    /// beatmap if one exists.
    pub fn connect_frames(&self) -> Vec<Frame> {
        // Enumerating devices talks to the sound server, so it happens before
        // the state lock rather than under it.
        let audio = self.audio_settings_frame();
        let state = self.state.lock().unwrap();
        let mut frames = vec![self.catalog_frame(), audio];
        frames.push(
            state
                .latest_state
                .clone()
                .unwrap_or_else(|| idle_state_frame(&state.placement_photo)),
        );
        if let Some(beatmap) = state.latest_beatmap.clone() {
            frames.push(beatmap);
        }
        frames
    }

    fn catalog_frame(&self) -> Frame {
        let catalog = self.catalog.read().unwrap();
        Frame::CollectionCatalog {
            subjects: catalog.subjects().to_vec(),
            tracks: catalog.tracks(),
            collection_classes: catalog.collection_classes().to_vec(),
            activities: catalog.activities().to_vec(),
            sweat_levels: catalog.sweat_levels().to_vec(),
        }
    }

    /// The track library's root, for the import and delete routes.
    pub fn tracks_root(&self) -> &std::path::Path {
        &self.catalog_paths.tracks_root
    }

    /// Tracks carrying a validated generated Calibration product. Kept outside
    /// the collection frame until the guided-session protocol owns this state.
    pub fn calibration_tracks(&self) -> Vec<crate::collect::beatmap::CalibrationTrack> {
        self.catalog.read().unwrap().calibration_tracks()
    }

    /// Open the exact imported audio for an authored guided-calibration
    /// schedule.  The playback object remains owned by the calibration adapter;
    /// this method only supplies the same mixer configuration as collection.
    pub fn open_calibration_playback(
        &self,
        track_id: &TrackId,
        entries: &[protocol::CalibrationScheduleEntry],
    ) -> anyhow::Result<audio::Playback> {
        let catalog = self.catalog.read().unwrap();
        let audio_path = catalog
            .audio_path(track_id)
            .ok_or_else(|| anyhow::anyhow!("unknown calibration track '{track_id}'"))?;
        let beat_times = catalog.beat_times(track_id).unwrap_or_default();
        let note_onsets = entries
            .iter()
            .map(|entry| entry.track_offset)
            .collect::<Vec<_>>();
        let settings = self.audio_settings.lock().unwrap();
        audio::Playback::open(
            &audio_path,
            &note_onsets,
            &beat_times,
            &settings.output,
            settings.gain(),
        )
    }

    /// Re-read the library from disk and tell every browser what it now holds.
    /// An import or a delete is only finished once this has run.
    pub fn reload_catalog(&self) -> anyhow::Result<()> {
        let reloaded = self.catalog_paths.load()?;
        *self.catalog.write().unwrap() = reloaded;
        self.publish(self.catalog_frame());
        Ok(())
    }

    /// Broadcast a frame, caching state/beatmap frames for later connectors.
    fn publish(&self, frame: Frame) {
        {
            let mut state = self.state.lock().unwrap();
            match &frame {
                Frame::CollectionState { .. } => state.latest_state = Some(frame.clone()),
                Frame::Beatmap { .. } => state.latest_beatmap = Some(frame.clone()),
                _ => {}
            }
        }
        let _ = self.outbound.send(frame);
    }

    /// Surface a human-readable failure in the dashboard's log panel; there is
    /// deliberately no dedicated error frame for a lab tool.
    fn publish_error(&self, message: String) {
        tracing::warn!("collection: {message}");
        let _ = self.outbound.send(Frame::Log {
            t_us: now().get() * 1000,
            level: LogLevel::Error,
            message: format!("collection: {message}"),
        });
    }

    fn publish_idle(&self) {
        let frame = {
            let mut state = self.state.lock().unwrap();
            state.latest_beatmap = None;
            idle_state_frame(&state.placement_photo)
        };
        self.publish(frame);
    }

    /// Browser → `StartCollection`. Validates against the catalog, then arms in
    /// a spawned task so the browser session loop never waits on ffmpeg or the
    /// device.
    pub fn start_collection(
        self: &Arc<Self>,
        metadata: SessionMetadata,
        track_id: TrackId,
        difficulty: DifficultyLevel,
        record_video: bool,
        device_id: Option<String>,
    ) {
        if let Err(reason) = self.validate_start(&metadata, &track_id) {
            self.publish_error(format!("rejected session start: {reason}"));
            return;
        }
        let device = match device_id.as_deref() {
            Some(device_id) => match self.registry.connection_identity(device_id) {
                Some(device) => Some(device),
                None => {
                    self.publish_error(format!(
                        "rejected session start: device '{device_id}' is not connected"
                    ));
                    return;
                }
            },
            None => None,
        };
        let lease = match self
            .guided_sessions
            .acquire_current(GuidedMode::Collection, device)
        {
            Ok(lease) => lease,
            Err(CoordinatorError::LeaseHeld { mode }) => {
                self.publish_error(format!(
                    "rejected session start: a {mode:?} guided session already exists"
                ));
                return;
            }
            Err(error) => {
                self.publish_error(format!("rejected session start: {error}"));
                return;
            }
        };
        {
            let mut state = self.state.lock().unwrap();
            if !matches!(state.phase, Phase::Idle) {
                drop(state);
                lease.finish(SessionExit::OperatorStopped);
                self.publish_error("rejected session start: a session already exists".into());
                return;
            }
            state.phase = Phase::Starting {
                session_id: lease.binding().session_id,
            };
        }
        let manager = Arc::clone(self);
        let guided_session_id = lease.binding().session_id;
        tokio::spawn(async move {
            let mut lease = Some(lease);
            let result = std::panic::AssertUnwindSafe(manager.clone().arm(
                metadata,
                track_id,
                difficulty,
                record_video,
                &mut lease,
            ))
            .catch_unwind()
            .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    if let Some(lease) = lease.take() {
                        lease.finish(SessionExit::DependencyFailed(format!("{error:#}")));
                    }
                    manager.publish_error(format!("session start failed: {error:#}"));
                    manager.restore_idle_after_task(guided_session_id);
                }
                Err(_) => {
                    if let Some(lease) = lease.take() {
                        lease.finish(SessionExit::TaskFailed(
                            "collection arming task panicked".into(),
                        ));
                    }
                    manager.publish_error("session start failed: arming task panicked".into());
                    manager.restore_idle_after_task(guided_session_id);
                }
            }
        });
    }

    /// Reject ids the current catalog does not offer — the browser should never
    /// send them, but the catalog is the authority, not the browser. The subject
    /// is the exception: the roster is a convenience, not a vocabulary, so any
    /// non-empty name is welcome (guests exist).
    fn validate_start(&self, metadata: &SessionMetadata, track_id: &TrackId) -> Result<(), String> {
        if metadata.subject.0.trim().is_empty() {
            return Err("empty subject".into());
        }
        let catalog = self.catalog.read().unwrap();
        if !catalog
            .activities()
            .iter()
            .any(|activity| activity.id == metadata.activity)
        {
            return Err(format!("unknown activity '{}'", metadata.activity));
        }
        if !catalog
            .sweat_levels()
            .iter()
            .any(|sweat| sweat.id == metadata.sweat)
        {
            return Err(format!("unknown sweat level '{}'", metadata.sweat));
        }
        if catalog.audio_path(track_id).is_none() {
            return Err(format!("unknown track '{track_id}'"));
        }
        Ok(())
    }

    /// Subscribe to a device's stream and read the hardware identity off its
    /// first EMG window.
    async fn acquire_device(
        &self,
        identity: &DeviceConnectionIdentity,
    ) -> anyhow::Result<AcquiredDevice> {
        let bound = self
            .registry
            .bind_connection(identity)
            .ok_or_else(|| anyhow::anyhow!("leased device connection is no longer current"))?;
        let mut emg_receiver = bound.frames;
        let (channels, sample_rate, scale_uv) = tokio::time::timeout(FIRST_WINDOW_TIMEOUT, async {
            loop {
                match emg_receiver.recv().await {
                    Ok(Frame::Emg {
                        channels,
                        sample_rate,
                        scale_uv,
                        ..
                    }) => break Ok((channels, sample_rate, scale_uv)),
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => {
                        break Err(anyhow::anyhow!("device stream closed"))
                    }
                }
            }
        })
        .await
        .map_err(|_| anyhow::anyhow!("no EMG from '{}' within 3 s", identity.device_id))??;
        Ok(AcquiredDevice {
            emg_receiver,
            hardware: HardwareIdentity {
                device_id: identity.device_id.clone(),
                transport: bound.transport,
                channels,
                sample_rate,
                scale_uv,
                device_config: bound.config,
                provenance: bound.provenance,
                board_revision: self.provenance.board_revision(&identity.device_id),
            },
        })
    }

    /// The arming sequence. Any error unwinds to `Idle` via the caller.
    ///
    /// With no device this is a *practice* session: the game runs on the real
    /// schedule, but nothing is recorded, no directory is created, and — with
    /// no EMG to detect activity in — every cue resolves as a miss.
    async fn arm(
        self: Arc<Self>,
        metadata: SessionMetadata,
        track_id: TrackId,
        difficulty: DifficultyLevel,
        record_video: bool,
        lease: &mut Option<SessionLease>,
    ) -> anyhow::Result<()> {
        // Everything device-independent happens first, and the order is
        // load-bearing: subscribing to the device starts buffering EMG, and
        // decoding a track takes seconds, so a subscription taken any earlier
        // hands the recorder seconds of windows that predate `session_start`.
        let (track, class_ids, beatmap, beat_times, audio_path, rest_label) = {
            let catalog = self.catalog.read().unwrap();
            let track = catalog
                .tracks()
                .into_iter()
                .find(|track| track.id == track_id)
                .ok_or_else(|| anyhow::anyhow!("unknown track '{track_id}'"))?;
            let rest_label = catalog.rest_label(&track_id);
            let class_ids = catalog.class_ids();
            let seed = now().get();
            let beatmap = catalog.generate(&track_id, &class_ids, difficulty, seed)?;
            let beat_times = catalog.beat_times(&track_id).unwrap_or_default();
            let audio_path = catalog
                .audio_path(&track_id)
                .ok_or_else(|| anyhow::anyhow!("track '{track_id}' has no audio"))?;
            (
                track, class_ids, beatmap, beat_times, audio_path, rest_label,
            )
        };

        // Decoding a whole track and opening a device both block, so this runs
        // off the runtime's threads. The stream is playing silence by the time
        // it returns, which is what makes the first cue's placement a
        // measurement of this device rather than a guess.
        let note_onsets: Vec<TrackMilliseconds> = beatmap.iter().map(|(_, note)| note.at).collect();
        let click_beats = beat_times.clone();
        let (output, gain) = {
            let settings = self.audio_settings.lock().unwrap();
            (settings.output.clone(), settings.gain())
        };
        let playback_audio = LiveAudioSettings {
            output: output.clone(),
            gain,
        };
        let playback = tokio::task::spawn_blocking(move || {
            Playback::open(&audio_path, &note_onsets, &click_beats, &output, gain)
        })
        .await?
        .context("opening the session's audio output")?;
        let audio = AudioPlayback {
            output: playback.output_name().to_string(),
            sample_rate: playback.output_sample_rate(),
        };
        tracing::info!(
            "collection audio on '{}' at {} Hz, output latency {} ms (already applied to every cue)",
            audio.output,
            audio.sample_rate,
            playback.output_latency().get()
        );

        let leased_device = lease
            .as_ref()
            .and_then(|lease| lease.binding().device.clone());
        let acquired = match leased_device {
            Some(identity) => Some(self.acquire_device(&identity).await?),
            None => None,
        };

        let created = now();
        let practice = acquired.is_none();
        let session_id = SessionId(format!(
            "{}_{}{}",
            chrono::Local::now().format("%Y-%m-%dT%H-%M-%S"),
            metadata.subject,
            if practice { "_practice" } else { "" },
        ));

        let mut capture = match acquired {
            Some(AcquiredDevice {
                emg_receiver,
                hardware,
            }) => {
                let device_id = hardware.device_id.clone();
                let don_count = self
                    .provenance
                    .next_don_count(&metadata.subject, metadata.arm);
                let manifest = SessionManifest {
                    session_id: session_id.clone(),
                    created,
                    metadata,
                    hardware,
                    don_count,
                    track: track.clone(),
                    difficulty,
                    class_ids,
                    record_video,
                    audio,
                    completed: false,
                };

                // Recorder first: it creates the session directory video records into.
                let mut recorder = FileSessionRecorder::begin(&self.sessions_root, manifest)?;
                let directory = recorder.directory().to_path_buf();

                // Adopt the pending placement photo, if one was captured at setup.
                let adopted_photo = {
                    let state = self.state.lock().unwrap();
                    state
                        .placement_photo
                        .as_ref()
                        .map(|photo| (photo.path.clone(), photo.captured))
                };
                if let Some((pending_path, captured)) = adopted_photo {
                    match std::fs::copy(&pending_path, directory.join("placement.jpg")) {
                        Ok(_) => {
                            recorder.append_event(&SessionEvent::PlacementPhoto { at: captured })?
                        }
                        Err(error) => tracing::warn!("placement photo not adopted: {error}"),
                    }
                }
                CaptureResources::Recording {
                    directory,
                    recorder: Box::new(recorder),
                    emg_receiver,
                    device_id,
                    video: None,
                }
            }
            None => CaptureResources::Practice,
        };

        // The operator asked for video or they did not. A camera that cannot
        // deliver it aborts the start — a session that silently records EMG
        // alone is the failure this whole path exists to prevent. Practice
        // sessions have no directory to record into, so no video either.
        if let (true, Some(directory)) = (record_video, capture.directory()) {
            let directory = directory.clone();
            let video = Arc::clone(&self.video);
            let video_output = directory.join("video.mkv");
            let requested_start = now();
            let started = tokio::task::spawn_blocking(move || {
                video
                    .lock()
                    .unwrap()
                    .start_recording(&video_output, requested_start)
            })
            .await?;
            match started {
                Ok(handle) => capture.attach_video(handle),
                Err(error) => {
                    // Nothing has been recorded yet, so the directory holds
                    // only the manifest; leaving it would look like a take.
                    let _ = std::fs::remove_dir_all(directory);
                    return Err(error.context("this session asked for video"));
                }
            }
        }

        let (control, control_receiver) = mpsc::channel(CONTROL_CAPACITY);
        let (browser_departure, browser_departure_receiver) =
            watch::channel(BrowserDeparture::default());
        let current_audio = {
            let settings = self.audio_settings.lock().unwrap();
            LiveAudioSettings {
                output: settings.output.clone(),
                gain: settings.gain(),
            }
        };
        let (audio, audio_receiver) = watch::channel(current_audio);
        let guided_session_id = lease
            .as_ref()
            .expect("arming owns the guided-session lease")
            .binding()
            .session_id;
        {
            let mut state = self.state.lock().unwrap();
            state.phase = Phase::Running {
                session_id: guided_session_id,
                control,
                browser_departure,
                audio,
            };
            // The photo now belongs to this session; a practice session leaves
            // it pending for the next real one.
            if !practice {
                state.placement_photo = None;
            }
        }
        self.publish(Frame::Beatmap {
            session_id: session_id.clone(),
            track: track.clone(),
            notes: beatmap.clone(),
            beat_times,
            lead_in: crate::collect::beatmap::LEAD_IN,
        });

        let session = RunningSession::new(
            Arc::clone(&self),
            session_id,
            capture,
            control_receiver,
            browser_departure_receiver,
            audio_receiver,
            playback_audio,
            beatmap,
            track,
            rest_label,
            playback,
            lease.take().expect("arming owns the guided-session lease"),
        );
        let manager = Arc::clone(&self);
        tokio::spawn(async move {
            if std::panic::AssertUnwindSafe(session.run())
                .catch_unwind()
                .await
                .is_err()
            {
                manager.publish_error("session failed: running task panicked".into());
                manager.restore_idle_after_task(guided_session_id);
            }
        });
        Ok(())
    }

    /// The audio settings frame browsers render the controls from.
    pub fn audio_settings_frame(&self) -> Frame {
        let settings = self.audio_settings.lock().unwrap();
        Frame::AudioSettings {
            devices: audio::output_devices(),
            output: settings.output_name(),
            volume_permille: settings.volume_permille,
        }
    }

    /// Browser → `SetAudioVolume`. A running session hears about it at once.
    pub fn set_audio_volume(&self, volume_permille: u32) {
        {
            let mut settings = self.audio_settings.lock().unwrap();
            settings.volume_permille = volume_permille.min(1000);
        }
        self.remember_audio();
        self.update_running_audio();
        self.publish(self.audio_settings_frame());
    }

    /// Browser → `SetAudioOutput`. A running session moves to the new device
    /// mid-track; anything else takes effect when the next one arms.
    pub fn set_audio_output(&self, output: Option<String>) {
        {
            let mut settings = self.audio_settings.lock().unwrap();
            if settings.forced_silent {
                tracing::info!("ignoring an output-device change: this backend runs silent");
                return;
            }
            settings.output = match output {
                Some(name) if !name.is_empty() => AudioOutput::Named(name),
                _ => AudioOutput::Default,
            };
        }
        self.remember_audio();
        self.update_running_audio();
        self.publish(self.audio_settings_frame());
    }

    fn remember_audio(&self) {
        let settings = self.audio_settings.lock().unwrap();
        self.provenance.set_audio(StoredAudio {
            output: settings.output_name(),
            volume_permille: settings.volume_permille,
        });
    }

    /// Browser → `StartTrack`.
    pub fn start_track(&self) {
        self.command(SessionControl::StartTrack);
    }

    /// Browser → `PauseTrack`.
    pub fn pause_track(&self) {
        self.command(SessionControl::PauseTrack);
    }

    /// Browser → `ResumeTrack`.
    pub fn resume_track(&self) {
        self.command(SessionControl::ResumeTrack);
    }

    fn command(&self, control: SessionControl) {
        let delivery = {
            let state = self.state.lock().unwrap();
            match &state.phase {
                Phase::Running {
                    control: sender, ..
                } => sender.try_send(control).map_err(|error| match error {
                    TrySendError::Full(_) => "collection control queue is full",
                    TrySendError::Closed(_) => "collection control task has stopped",
                }),
                Phase::Idle | Phase::Starting { .. } | Phase::Reviewing { .. } => return,
            }
        };
        if let Err(detail) = delivery {
            self.publish_error(detail.into());
        }
    }

    fn update_running_audio(&self) {
        let latest = {
            let settings = self.audio_settings.lock().unwrap();
            LiveAudioSettings {
                output: settings.output.clone(),
                gain: settings.gain(),
            }
        };
        let delivery = {
            let state = self.state.lock().unwrap();
            match &state.phase {
                Phase::Running { audio, .. } => audio.send(latest),
                Phase::Idle | Phase::Starting { .. } | Phase::Reviewing { .. } => return,
            }
        };
        if delivery.is_err() {
            self.publish_error("collection audio control task has stopped".into());
        }
    }

    fn restore_idle_after_task(&self, session_id: dashboard::guided_session::GuidedSessionId) {
        let restored = {
            let mut state = self.state.lock().unwrap();
            let matching_task = match state.phase {
                Phase::Starting {
                    session_id: current,
                }
                | Phase::Running {
                    session_id: current,
                    ..
                } => current == session_id,
                Phase::Idle | Phase::Reviewing { .. } => false,
            };
            if matching_task {
                state.phase = Phase::Idle;
            }
            matching_task
        };
        if restored {
            self.publish_idle();
        }
    }

    /// Browser → `FinishCollection`: end the running session now; recording is
    /// finalized as if the track had played out and the summary screen decides
    /// what happens to the partial take.
    pub fn finish_collection(&self) {
        self.command(SessionControl::Finish);
    }

    /// Browser → `StopCollection`: resolves the session under review;
    /// `save: false` deletes the directory.
    pub fn stop_collection(&self, save: bool) {
        let reviewed = {
            let mut state = self.state.lock().unwrap();
            match std::mem::replace(&mut state.phase, Phase::Idle) {
                Phase::Reviewing { directory } => Some(directory),
                phase @ (Phase::Idle | Phase::Starting { .. } | Phase::Running { .. }) => {
                    state.phase = phase;
                    None
                }
            }
        };
        if let Some(directory) = reviewed {
            if !save {
                // A practice session has nothing on disk to discard.
                if let Some(directory) = directory {
                    if let Err(error) = std::fs::remove_dir_all(&directory) {
                        self.publish_error(format!("failed to discard session: {error}"));
                    }
                }
            }
            self.publish_idle();
        }
    }

    /// Browser → `GET /collection/camera/preview`. Starts the setup preview and
    /// hands back its JPEG parts plus the hold that keeps it alive; the camera
    /// is released the moment the hold is dropped.
    pub async fn start_camera_preview(
        self: &Arc<Self>,
    ) -> anyhow::Result<(CameraPreviewHold, mpsc::Receiver<Vec<u8>>)> {
        let video = Arc::clone(&self.video);
        // start_preview waits for the first frame, so it must not run on a
        // runtime worker.
        let session =
            tokio::task::spawn_blocking(move || video.lock().unwrap().start_preview()).await??;
        Ok((
            CameraPreviewHold {
                video: Arc::clone(&self.video),
                token: session.token,
            },
            session.parts,
        ))
    }

    /// Browser → `CapturePlacementPhoto`. Idle only: the camera cannot be
    /// opened twice, and setup is the only place the photo makes sense.
    pub fn capture_placement_photo(self: &Arc<Self>) {
        if !matches!(self.state.lock().unwrap().phase, Phase::Idle) {
            self.publish_error("placement photo is only available before a session".into());
            return;
        }
        let manager = Arc::clone(self);
        let target = self.sessions_root.join(PENDING_PHOTO_NAME);
        tokio::spawn(async move {
            let video = Arc::clone(&manager.video);
            let path = target.clone();
            let captured = now();
            let result =
                tokio::task::spawn_blocking(move || video.lock().unwrap().capture_photo(&path))
                    .await;
            match result {
                Ok(Ok(())) => {
                    manager.state.lock().unwrap().placement_photo = Some(PendingPhoto {
                        path: target,
                        captured,
                    });
                    manager.publish_idle();
                }
                Ok(Err(error)) => {
                    manager.publish_error(format!("placement photo failed: {error:#}"))
                }
                Err(join_error) => {
                    manager.publish_error(format!("placement photo failed: {join_error}"))
                }
            }
        });
    }
}

impl GuidedModeAdapter for CollectionManager {
    fn pause_for_no_visible_views(&self, session: &GuidedSessionBinding) {
        let state = self.state.lock().unwrap();
        if let Phase::Running {
            session_id,
            browser_departure,
            ..
        } = &state.phase
        {
            if *session_id == session.session_id {
                browser_departure.send_modify(|departure| {
                    departure.0 = departure.0.wrapping_add(1);
                });
            }
        }
    }
}

/// One browser's claim on the setup preview, alive for as long as the HTTP
/// response streams. Dropping it releases the camera, so a browser that closes
/// its tab cannot leave the device held against the next session.
pub struct CameraPreviewHold {
    video: Arc<Mutex<FfmpegVideoCapture>>,
    token: crate::collect::video::PreviewToken,
}

impl Drop for CameraPreviewHold {
    fn drop(&mut self) {
        self.video.lock().unwrap().stop_preview(self.token);
    }
}

/// Next frame from the session's device stream, or pend forever for a practice
/// session (no device, so the EMG select arm simply never fires).
async fn next_emg(
    receiver: Option<&mut broadcast::Receiver<Frame>>,
) -> Result<Frame, broadcast::error::RecvError> {
    match receiver {
        Some(receiver) => receiver.recv().await,
        None => std::future::pending().await,
    }
}

fn idle_state_frame(placement_photo: &Option<PendingPhoto>) -> Frame {
    Frame::CollectionState {
        phase: CollectionPhase::Idle,
        placement_photo: placement_photo.as_ref().map(|photo| photo.captured),
    }
}

/// One cue's lifecycle inside the session task.
struct CueState {
    index: NoteIndex,
    class_id: ClassId,
    at_track: TrackMilliseconds,
    hold: protocol::DurationMilliseconds,
    /// Wall-clock hold transitions; known once the beat grid is anchored.
    at_wall: Option<UnixMilliseconds>,
    release_wall: Option<UnixMilliseconds>,
    logged: bool,
    resolved: bool,
    peak_activity: f32,
}

/// Rolling baseline of [`mean_absolute_deviation`] for the activity detector. Only
/// updated away from cues, so gesture bursts don't inflate it.
struct ActivityBaseline {
    value: Option<f32>,
}

/// One window's mean absolute deviation from each channel's own mean, in ADC counts,
/// or `None` for a window with no samples.
///
/// The mean has to come out per channel first. `Frame::Emg` carries raw counts, so a
/// channel's samples sit on whatever electrode offset that channel has — tens of
/// millivolts, hundreds of times the muscle signal, and different per channel. Mean
/// absolute *amplitude* would measure the offsets and barely move when a muscle
/// fires, which is what the ratio against [`ActivityBaseline`] depends on. Removing
/// the per-window mean also removes slow drift for free.
fn mean_absolute_deviation(channels: u16, samples: &[u8]) -> Option<f32> {
    let channels = usize::from(channels);
    let total = samples.len() / 2;
    if channels == 0 || total == 0 {
        return None;
    }
    // Channel-major: each channel's samples are one contiguous run.
    let per_channel = total / channels;
    if per_channel == 0 {
        return None;
    }
    let value_at = |index: usize| -> f32 {
        i16::from_le_bytes([samples[index * 2], samples[index * 2 + 1]]) as f32
    };
    let mut deviation_sum = 0.0f64;
    for channel in 0..channels {
        let start = channel * per_channel;
        let mut sum = 0.0f64;
        for index in start..start + per_channel {
            sum += value_at(index) as f64;
        }
        let mean = sum / per_channel as f64;
        for index in start..start + per_channel {
            deviation_sum += (value_at(index) as f64 - mean).abs();
        }
    }
    Some((deviation_sum / (channels * per_channel) as f64) as f32)
}

impl ActivityBaseline {
    fn update(&mut self, sample: f32) {
        self.value = Some(match self.value {
            None => sample,
            Some(current) => current * 0.95 + sample * 0.05,
        });
    }
}

/// Resources that exist together only when a physical device was acquired.
/// Keeping them in one variant makes a half-recording session unrepresentable:
/// practice cannot accidentally own a recorder, and recording cannot lose its
/// EMG stream or destination while retaining the others.
enum CaptureResources {
    Practice,
    Recording {
        directory: PathBuf,
        recorder: Box<FileSessionRecorder>,
        emg_receiver: broadcast::Receiver<Frame>,
        device_id: String,
        video: Option<RecordingHandle>,
    },
}

impl CaptureResources {
    fn recorder_mut(&mut self) -> Option<&mut FileSessionRecorder> {
        match self {
            Self::Practice => None,
            Self::Recording { recorder, .. } => Some(recorder.as_mut()),
        }
    }

    fn recorder(&self) -> Option<&FileSessionRecorder> {
        match self {
            Self::Practice => None,
            Self::Recording { recorder, .. } => Some(recorder.as_ref()),
        }
    }

    fn emg_receiver_mut(&mut self) -> Option<&mut broadcast::Receiver<Frame>> {
        match self {
            Self::Practice => None,
            Self::Recording { emg_receiver, .. } => Some(emg_receiver),
        }
    }

    fn device_id(&self) -> &str {
        match self {
            Self::Practice => "",
            Self::Recording { device_id, .. } => device_id,
        }
    }

    fn directory(&self) -> Option<&PathBuf> {
        match self {
            Self::Practice => None,
            Self::Recording { directory, .. } => Some(directory),
        }
    }

    fn attach_video(&mut self, handle: RecordingHandle) {
        match self {
            Self::Practice => unreachable!("practice sessions never open video"),
            Self::Recording { video, .. } => *video = Some(handle),
        }
    }

    fn is_recording(&self) -> bool {
        matches!(self, Self::Recording { .. })
    }
}

/// A live session.
struct RunningSession {
    manager: Arc<CollectionManager>,
    session_id: SessionId,
    capture: CaptureResources,
    control: mpsc::Receiver<SessionControl>,
    browser_departure: watch::Receiver<BrowserDeparture>,
    audio: watch::Receiver<LiveAudioSettings>,
    applied_audio: LiveAudioSettings,
    cues: Vec<CueState>,
    track: TrackInfo,
    /// `Some` on a rest track; the finalizer logs the played stretch.
    rest_label: Option<String>,
    playback: Playback,
    /// The instant the subject heard audio t = 0, for the segment now playing.
    /// Read off the mixer's cursor when the track starts and re-derived on every
    /// resume, so a cue's wall-clock time is always `anchor + note position` for
    /// the segment it falls in.
    anchor: Option<UnixMilliseconds>,
    /// When the most recent EMG window landed, the stall detector's baseline.
    last_window_at: Option<UnixMilliseconds>,
    paused: Option<CollectionPause>,
    armed_at: UnixMilliseconds,
    baseline: ActivityBaseline,
    activity_hits: u32,
    cues_per_class: BTreeMap<ClassId, u16>,
    latest_health: RecordingHealth,
    lease: Option<SessionLease>,
}

impl RunningSession {
    #[allow(clippy::too_many_arguments)]
    fn new(
        manager: Arc<CollectionManager>,
        session_id: SessionId,
        capture: CaptureResources,
        control: mpsc::Receiver<SessionControl>,
        browser_departure: watch::Receiver<BrowserDeparture>,
        audio: watch::Receiver<LiveAudioSettings>,
        applied_audio: LiveAudioSettings,
        beatmap: Beatmap,
        track: TrackInfo,
        rest_label: Option<String>,
        playback: Playback,
        lease: SessionLease,
    ) -> Self {
        let cues = beatmap
            .iter()
            .map(|(index, note)| CueState {
                index,
                class_id: note.class_id.clone(),
                at_track: note.at,
                hold: note.hold,
                at_wall: None,
                release_wall: None,
                logged: false,
                resolved: false,
                peak_activity: 0.0,
            })
            .collect();
        Self {
            manager,
            session_id,
            capture,
            control,
            browser_departure,
            audio,
            applied_audio,
            cues,
            track,
            rest_label,
            playback,
            anchor: None,
            last_window_at: None,
            paused: None,
            armed_at: now(),
            baseline: ActivityBaseline { value: None },
            activity_hits: 0,
            cues_per_class: BTreeMap::new(),
            latest_health: RecordingHealth {
                emg: StreamProgress {
                    bytes_on_disk: 0,
                    advancing: false,
                },
                video: None,
                recorded: None,
            },
            lease: Some(lease),
        }
    }

    async fn run(mut self) {
        // Settings may have changed while track decoding or device acquisition was
        // in flight. Reconcile the playback built above before accepting controls.
        let desired_audio = self.audio.borrow_and_update().clone();
        self.apply_audio_settings(desired_audio).await;
        let mut ticker = tokio::time::interval(TICK);
        let mut publish_health = false;
        let outcome = loop {
            tokio::select! {
                biased;
                departed = self.browser_departure.changed() => match departed {
                    Ok(()) => {
                        self.browser_departure.borrow_and_update();
                        self.browser_gone().await;
                    }
                    Err(_) => break SessionExit::TaskFailed(
                        "collection browser-safety channel closed".into(),
                    ),
                },
                message = self.control.recv() => match message {
                    Some(SessionControl::StartTrack) => self.start_track().await,
                    Some(SessionControl::PauseTrack) => self.pause_track().await,
                    Some(SessionControl::ResumeTrack) => self.resume_track().await,
                    Some(SessionControl::Finish) => break SessionExit::OperatorStopped,
                    None => break SessionExit::TaskFailed("collection control channel closed".into()),
                },
                changed = self.audio.changed() => match changed {
                    Ok(()) => {
                        let settings = self.audio.borrow_and_update().clone();
                        self.apply_audio_settings(settings).await;
                    }
                    Err(_) => break SessionExit::TaskFailed(
                        "collection audio control channel closed".into(),
                    ),
                },
                frame = next_emg(self.capture.emg_receiver_mut()) => match frame {
                    Ok(Frame::Emg { seq, t0_us, channels, samples, missing, .. }) => {
                        self.ingest_window(seq, t0_us, channels, &samples, &missing);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        let (report, exit) = recording_stream_failure(error);
                        self.manager.publish_error(report);
                        break exit;
                    }
                },
                _ = ticker.tick() => {
                    if self.tick().await {
                        break SessionExit::Completed;
                    }
                    self.publish_playback_position();
                    publish_health = !publish_health;
                    if publish_health {
                        self.poll_health();
                        self.publish_state();
                    }
                }
            }
        };
        self.finalize(outcome).await;
    }

    async fn apply_audio_settings(&mut self, settings: LiveAudioSettings) {
        if settings.gain != self.applied_audio.gain {
            self.playback.set_music_gain(settings.gain);
        }
        if settings.output != self.applied_audio.output {
            self.set_output(&settings.output).await;
        }
        self.applied_audio = settings;
    }

    /// The operator tapped Start: play the track, then take the anchor off the
    /// mixer's own cursor. Nothing here trusts a clock but the one the samples
    /// are leaving through.
    async fn start_track(&mut self) {
        // Defense in depth: a second start for a started session is ignored (it
        // would silently re-anchor and mislabel every later cue).
        if self.anchor.is_some() {
            tracing::warn!("ignoring a second start_track for {}", self.session_id);
            return;
        }
        self.playback.play();
        let Some(timeline) = self.playhead_after_command(true).await else {
            self.playback.pause();
            self.manager
                .publish_error("audio output did not start; the track has not begun".into());
            return;
        };
        let at = timeline.heard_at.before_track_position(timeline.position);
        self.anchor = Some(at);
        self.place_unlogged_cues(at);
        // Silence is measured from here rather than from arming, so a session
        // armed early does not start out looking stalled.
        self.last_window_at = Some(at);
        let armed_at = self.armed_at;
        self.log_event(&SessionEvent::ArmedPrefix {
            from: armed_at,
            to: at,
        });
        self.log_event(&SessionEvent::TrackStarted { at });
        self.publish_state();
        self.publish_playback_position();
    }

    /// Wait for the mixer to publish a playhead taken after playback was
    /// commanded, so the reading describes the command's effect rather than
    /// what came before it.
    async fn playhead_after_command(&self, playing: bool) -> Option<Timeline> {
        let commanded = now();
        let deadline = tokio::time::Instant::now() + PLAYHEAD_TIMEOUT;
        loop {
            let timeline = self.playback.timeline();
            if timeline.playing == playing && timeline.heard_at >= commanded {
                return Some(timeline);
            }
            if tokio::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// Every cue that has not been logged yet takes its wall-clock transitions
    /// from this anchor. Logged cues keep the times they were logged with: those
    /// are what the subject was actually shown.
    fn place_unlogged_cues(&mut self, anchor: UnixMilliseconds) {
        for cue in self.cues.iter_mut().filter(|cue| !cue.logged) {
            cue.at_wall = Some(anchor.at_track_position(cue.at_track));
            cue.release_wall = Some(anchor.at_track_position(cue.at_track.plus(cue.hold)));
        }
    }

    fn track_position(&self, at: UnixMilliseconds) -> TrackMilliseconds {
        let elapsed = match self.anchor {
            Some(anchor) => at.since(anchor).get().max(0) as u64,
            None => 0,
        };
        TrackMilliseconds::new(elapsed.min(u32::MAX as u64) as u32)
    }

    /// Move to another output device mid-session.
    ///
    /// The new device has its own latency, so every cue still ahead has to be
    /// re-placed against a fresh anchor — the same re-derivation a resume does,
    /// for the same reason. Cues already logged keep the instants they were
    /// logged with: those are when the subject actually heard them.
    async fn set_output(&mut self, output: &AudioOutput) {
        if let Err(error) = self.playback.switch_output(output) {
            self.manager.publish_error(format!(
                "could not switch audio output: {error:#}; still on {}",
                self.playback.output_name()
            ));
            return;
        }
        tracing::info!(
            "collection audio moved to '{}' at {} Hz",
            self.playback.output_name(),
            self.playback.output_sample_rate()
        );
        if self.anchor.is_none() || self.paused.is_some() {
            // Nothing is running against the anchor: a track that has not
            // started takes its anchor at the start, and a frozen one takes a
            // fresh one when it resumes.
            self.publish_playback_position();
            return;
        }
        if let Some(timeline) = self.playhead_after_command(true).await {
            let anchor = timeline.heard_at.before_track_position(timeline.position);
            self.anchor = Some(anchor);
            self.place_unlogged_cues(anchor);
        }
        self.publish_playback_position();
    }

    /// Browser → `PauseTrack`: the operator asked, so audio and the cue timeline
    /// stop together and the device is not implicated.
    async fn pause_track(&mut self) {
        if self.anchor.is_none() || self.paused.is_some() {
            return;
        }
        self.freeze(PauseCause::Operator, DurationMilliseconds::new(0))
            .await;
    }

    /// The last browser closed. An armed session has nothing to freeze — the
    /// track never began — and it is left alone to time out on its own.
    async fn browser_gone(&mut self) {
        if self.anchor.is_none() || self.paused.is_some() {
            return;
        }
        self.freeze(PauseCause::BrowserGone, DurationMilliseconds::new(0))
            .await;
    }

    /// Freeze audio and the cue timeline where they stand. Anything cued past
    /// here would be asked of a subject the music has stopped for, and — when
    /// the device is the reason — one nothing is being recorded from.
    ///
    /// The pause is stamped at the instant the *cursor* stopped, not at the
    /// instant the decision was taken. That is what makes the pause length the
    /// log records equal to the shift the resume puts into the beat grid, which
    /// is what a windowing tool subtracts to get back onto the track's timeline.
    async fn freeze(&mut self, cause: PauseCause, silent_for: DurationMilliseconds) {
        self.playback.pause();
        let track_position = match self.playhead_after_command(false).await {
            Some(timeline) => timeline.position,
            None => self.track_position(now()),
        };
        let Some(anchor) = self.anchor else { return };
        let at = anchor.at_track_position(track_position);
        let device_id = self.capture.device_id().to_owned();
        self.interrupt_cues_in_progress(at);
        self.log_event(&SessionEvent::Paused {
            at,
            track_position,
            silent_for,
            device_id: device_id.clone(),
            cause,
        });
        self.paused = Some(CollectionPause {
            cause,
            device_id,
            since: at,
            silent_for,
            track_position,
            // Only a silent device leaves anything to wait for; the other two
            // causes say nothing about the device.
            device_recovered: cause != PauseCause::DeviceSilent,
        });
        self.publish_state();
        self.publish_playback_position();
    }

    /// A cue whose hold straddles the pause is cut short here and never
    /// re-issued: the subject let go when the music stopped, so replaying the
    /// rest of the hold would label rest as a gesture.
    fn interrupt_cues_in_progress(&mut self, at: UnixMilliseconds) {
        for position in 0..self.cues.len() {
            let cue = &self.cues[position];
            if !cue.logged || cue.resolved {
                continue;
            }
            // A cue whose hold already ran out was shown in full; it is only
            // waiting on its activity window, so it resolves as usual.
            let held_through = cue.release_wall.is_some_and(|release| release > at);
            let index = cue.index;
            self.cues[position].resolved = true;
            let hit = self.cue_hit(position);
            if hit {
                self.activity_hits += 1;
            }
            if held_through {
                let event = SessionEvent::CueInterrupted {
                    note_index: index,
                    at,
                };
                if let Some(recorder) = self.capture.recorder_mut() {
                    if let Err(error) = recorder.append_event(&event) {
                        tracing::warn!("failed to log the interrupted cue: {error:#}");
                    }
                }
            }
            self.manager.publish(Frame::NoteResult {
                session_id: self.session_id.clone(),
                index,
                hit,
            });
        }
    }

    /// Unfreeze: play from where the pause left the cursor, then re-anchor on
    /// where the mixer says it actually restarted and place every cue still
    /// ahead of the playhead against that.
    async fn resume_track(&mut self) {
        if self.paused.is_none() {
            tracing::warn!("ignoring resume_track for a session that is not paused");
            return;
        }
        self.playback.play();
        let Some(timeline) = self.playhead_after_command(true).await else {
            self.playback.pause();
            self.manager
                .publish_error("audio output did not resume; the timeline is still frozen".into());
            return;
        };
        let pause = self.paused.take().expect("checked above");
        let (at, position) = (timeline.heard_at, timeline.position);
        let anchor = at.before_track_position(position);
        self.anchor = Some(anchor);
        // A cue whose onset is already behind the resumed playhead was never
        // shown — the pause froze the timeline just short of it. Retire it
        // silently rather than logging a gesture nobody was asked for.
        for cue in self.cues.iter_mut().filter(|cue| !cue.logged) {
            if cue.at_track < position {
                cue.logged = true;
                cue.resolved = true;
            }
        }
        self.place_unlogged_cues(anchor);
        // The stall clock restarts here, so resuming onto a device that is still
        // quiet takes the full threshold again rather than re-pausing at once.
        self.last_window_at = Some(at);
        let paused_for = DurationMilliseconds::new(
            at.since(pause.since).get().max(0).min(u32::MAX as i64) as u32,
        );
        self.log_event(&SessionEvent::Resumed {
            at,
            track_position: position,
            paused_for,
        });
        self.publish_state();
        self.publish_playback_position();
    }

    fn log_event(&mut self, event: &SessionEvent) {
        if let Some(recorder) = self.capture.recorder_mut() {
            if let Err(error) = recorder.append_event(event) {
                tracing::warn!("failed to log {event:?}: {error:#}");
            }
        }
    }

    /// Where the mixer's cursor stands, for the browsers drawing the playfield.
    fn publish_playback_position(&self) {
        let timeline = self.playback.timeline();
        self.manager.publish(Frame::PlaybackPosition {
            session_id: self.session_id.clone(),
            position_ms: timeline.position,
            at_unix_ms: timeline.heard_at,
            playing: timeline.playing && self.anchor.is_some(),
        });
    }

    fn ingest_window(
        &mut self,
        seq: u32,
        t0_us: u64,
        channels: u16,
        samples: &[u8],
        missing: &[u8],
    ) {
        let arrival = now();
        self.last_window_at = Some(arrival);
        let recovering = self
            .paused
            .as_ref()
            .is_some_and(|pause| !pause.device_recovered);
        if recovering {
            if let Some(pause) = &mut self.paused {
                pause.device_recovered = true;
            }
            self.publish_state();
        }
        if let Some(recorder) = self.capture.recorder_mut() {
            if let Err(error) = recorder.append_emg(EmgWindow {
                seq,
                t0_us,
                samples,
                missing,
            }) {
                tracing::warn!("EMG append failed: {error:#}");
            }
        }
        let Some(mean_absolute) = mean_absolute_deviation(channels, samples) else {
            return;
        };
        let arrival = arrival.get();
        let mut near_cue = false;
        for cue in &mut self.cues {
            let (Some(at_wall), Some(release_wall)) = (cue.at_wall, cue.release_wall) else {
                continue;
            };
            // The activity window spans the whole hold, with slop either side.
            let window_start = at_wall
                .get()
                .saturating_sub(ACTIVITY_WINDOW_BEFORE_MILLISECONDS);
            let window_end = release_wall.get() + ACTIVITY_WINDOW_AFTER_MILLISECONDS;
            if arrival >= window_start && arrival <= window_end {
                near_cue = true;
                if !cue.resolved {
                    cue.peak_activity = cue.peak_activity.max(mean_absolute);
                }
            }
        }
        if !near_cue {
            self.baseline.update(mean_absolute);
        }
    }

    /// Periodic work; true when the session has reached its own end.
    async fn tick(&mut self) -> bool {
        let current = now();
        match self.anchor {
            None => {
                if current.since(self.armed_at).get() > ARMED_TIMEOUT.as_millis() as i64 {
                    self.manager
                        .publish_error("armed session timed out; sending it to review".into());
                    return true;
                }
            }
            Some(anchor) => {
                // A frozen timeline resolves nothing and cannot reach the
                // track's end; only a resume restarts it.
                if self.paused.is_some() {
                    return false;
                }
                if let Some(silent_for) = self.silence_past_threshold(current) {
                    self.freeze(PauseCause::DeviceSilent, silent_for).await;
                    return false;
                }
                self.resolve_cues(current);
                let track_end = anchor
                    .at_track_position(TrackMilliseconds::new(self.track.duration.get()))
                    .get()
                    + TRACK_END_SLACK.as_millis() as u64;
                if current.get() >= track_end {
                    return true;
                }
            }
        }
        false
    }

    /// How long the device has been silent, once that exceeds
    /// [`STALL_THRESHOLD`]. Always `None` for a practice session, which has no
    /// device to fall silent.
    fn silence_past_threshold(&self, current: UnixMilliseconds) -> Option<DurationMilliseconds> {
        let last = self.last_window_at?;
        self.capture.is_recording().then_some(())?;
        let silent = current.since(last).get().max(0);
        (silent > STALL_THRESHOLD.as_millis() as i64)
            .then(|| DurationMilliseconds::new(silent.min(u32::MAX as i64) as u32))
    }

    /// Whether a cue's peak activity beat the rolling baseline.
    fn cue_hit(&self, position: usize) -> bool {
        match self.baseline.value {
            Some(baseline) if baseline > 0.0 => {
                self.cues[position].peak_activity > baseline * ACTIVITY_HIT_FACTOR
            }
            _ => false,
        }
    }

    fn resolve_cues(&mut self, current: UnixMilliseconds) {
        for position in 0..self.cues.len() {
            let cue = &mut self.cues[position];
            let (Some(at_wall), Some(release_wall)) = (cue.at_wall, cue.release_wall) else {
                continue;
            };
            if !cue.logged && current >= at_wall {
                cue.logged = true;
                let event = SessionEvent::Cue {
                    note_index: cue.index,
                    class_id: cue.class_id.clone(),
                    at: at_wall,
                    release: release_wall,
                };
                *self.cues_per_class.entry(cue.class_id.clone()).or_insert(0) += 1;
                if let Some(recorder) = self.capture.recorder_mut() {
                    if let Err(error) = recorder.append_event(&event) {
                        tracing::warn!("failed to log cue: {error:#}");
                    }
                }
            }
            let index = cue.index;
            if cue.logged
                && !cue.resolved
                && current.get() > release_wall.get() + ACTIVITY_WINDOW_AFTER_MILLISECONDS
            {
                cue.resolved = true;
                let hit = self.cue_hit(position);
                if hit {
                    self.activity_hits += 1;
                    let event = SessionEvent::ActivityHit {
                        note_index: index,
                        at: current,
                    };
                    if let Some(recorder) = self.capture.recorder_mut() {
                        if let Err(error) = recorder.append_event(&event) {
                            tracing::warn!("failed to log activity hit: {error:#}");
                        }
                    }
                }
                self.manager.publish(Frame::NoteResult {
                    session_id: self.session_id.clone(),
                    index,
                    hit,
                });
            }
        }
    }

    fn poll_health(&mut self) {
        // A practice session records nothing; its EMG "stream" is honestly
        // reported as never advancing.
        let emg = self.capture.recorder_mut().map_or(
            StreamProgress {
                bytes_on_disk: 0,
                advancing: false,
            },
            |recorder| recorder.health(),
        );
        let video = match &self.capture {
            CaptureResources::Practice => None,
            CaptureResources::Recording { video, .. } => video
                .as_ref()
                .map(|handle| self.manager.video.lock().unwrap().health(handle)),
        };
        self.latest_health = RecordingHealth {
            emg,
            video,
            recorded: self.capture.recorder().map(SessionRecorder::recorded_emg),
        };
    }

    fn publish_state(&self) {
        let phase = if self.anchor.is_some() {
            CollectionPhase::Playing {
                session_id: self.session_id.clone(),
                recording: self.latest_health,
                paused: self.paused.clone(),
            }
        } else {
            CollectionPhase::Armed {
                session_id: self.session_id.clone(),
                recording: self.latest_health,
            }
        };
        let placement_photo = self
            .capture
            .directory()
            .is_some_and(|directory| directory.join("placement.jpg").exists())
            .then(now);
        self.manager.publish(Frame::CollectionState {
            phase,
            placement_photo,
        });
    }

    /// Stop everything, write the files, and hand the take to review. Every
    /// ending arrives here: a take that reached disk is the operator's to keep
    /// or discard, never this function's.
    async fn finalize(mut self, outcome: SessionExit) {
        self.playback.pause();
        // A session whose track never started still recorded EMG from arming;
        // that whole stretch is the unlabelled prefix.
        if self.anchor.is_none() {
            let armed_at = self.armed_at;
            self.log_event(&SessionEvent::ArmedPrefix {
                from: armed_at,
                to: now(),
            });
        }

        if let (Some(label), Some(anchor)) = (self.rest_label.clone(), self.anchor) {
            let track_end =
                anchor.at_track_position(TrackMilliseconds::new(self.track.duration.get()));
            let to = track_end.min(now());
            self.log_event(&SessionEvent::Rest {
                label,
                from: anchor,
                to,
            });
        }

        let capture = std::mem::replace(&mut self.capture, CaptureResources::Practice);
        let (directory, recorder, video_handle) = match capture {
            CaptureResources::Practice => (None, None, None),
            CaptureResources::Recording {
                directory,
                recorder,
                video,
                ..
            } => (Some(directory), Some(recorder), video),
        };

        // Video first, off the runtime: stop_recording can block up to 5 s.
        let video_report: Option<VideoReport> = match video_handle {
            Some(handle) => {
                let video = Arc::clone(&self.manager.video);
                match tokio::task::spawn_blocking(move || {
                    video.lock().unwrap().stop_recording(handle)
                })
                .await
                {
                    Ok(Ok(report)) => Some(report),
                    Ok(Err(error)) => {
                        tracing::warn!("video finalize failed: {error:#}");
                        None
                    }
                    Err(join_error) => {
                        tracing::warn!("video finalize task failed: {join_error}");
                        None
                    }
                }
            }
            None => None,
        };

        let duration = match self.anchor {
            Some(anchor) => DurationMilliseconds::new(
                now().since(anchor).get().max(0).min(u32::MAX as i64) as u32,
            ),
            None => DurationMilliseconds::new(0),
        };
        let mut files = Vec::new();
        if let Some(report) = &video_report {
            files.push(FileReport {
                name: "video.mkv".into(),
                bytes: report.bytes,
                detail: report.detail.clone(),
            });
        }
        if let Some(directory) = &directory {
            if let Ok(metadata) = std::fs::metadata(directory.join("placement.jpg")) {
                files.push(FileReport {
                    name: "placement.jpg".into(),
                    bytes: metadata.len(),
                    detail: "band photo at don time".into(),
                });
            }
        }
        let mut summary = CollectionSummary {
            duration,
            cues_per_class: self.cues_per_class.clone(),
            activity_hits: self.activity_hits,
            files,
            emg_gap_count: recorder
                .as_ref()
                .map_or(0, |recorder| recorder.emg_gap_count()),
            video_start_offset: video_report.map(|report| report.start_offset),
        };

        // The recorder's own file reports only exist after finish(), so the
        // SessionEnd event's summary carries video/photo files; the frame the
        // summary screen renders carries all of them. A practice session has no
        // recorder and reports no files.
        if let Some(recorder) = recorder {
            match recorder.finish(&summary) {
                Ok(recorder_reports) => {
                    let mut all = recorder_reports;
                    all.append(&mut summary.files);
                    summary.files = all;
                }
                Err(error) => {
                    self.manager
                        .publish_error(format!("session finalize failed: {error:#}"));
                }
            }
        }

        {
            let mut state = self.manager.state.lock().unwrap();
            state.phase = Phase::Reviewing {
                directory: directory.clone(),
            };
        }
        self.manager.publish(Frame::CollectionState {
            phase: CollectionPhase::Reviewing {
                session_id: self.session_id,
                summary,
            },
            placement_photo: None,
        });
        self.lease
            .take()
            .expect("a running session retains its guided-session lease")
            .finish(outcome);
    }
}

/// Once recording has begun, a broadcast lag is data loss, not a routine
/// freshness event. Finalize the take immediately so its review state and
/// guided-session outcome cannot claim a continuous recording. Before arming,
/// `acquire_device` may still skip stale windows because no file exists yet.
fn recording_stream_failure(error: broadcast::error::RecvError) -> (String, SessionExit) {
    let detail = match error {
        broadcast::error::RecvError::Lagged(skipped) => {
            format!("device stream lost {skipped} frames mid-session; finalizing incomplete take")
        }
        broadcast::error::RecvError::Closed => {
            "device stream ended mid-session; finalizing incomplete take".into()
        }
    };
    (detail.clone(), SessionExit::DependencyFailed(detail))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A manager with nothing behind it but the pieces the attachment count
    /// touches. It never arms a session, so the catalog and the camera are
    /// never read.
    fn bare_manager(name: &str) -> (Arc<CollectionManager>, GuidedSessionCoordinator) {
        // Its own directory per test: these run in parallel, and a shared
        // config file is one test reading what another is still writing.
        let directory = std::env::temp_dir().join(format!("collection-attachment-{name}"));
        let paths = CatalogPaths {
            config_path: directory.join("collection.json"),
            tracks_root: directory.join("tracks"),
        };
        std::fs::create_dir_all(&paths.tracks_root).expect("the temp directory is writable");
        std::fs::write(
            &paths.config_path,
            r#"{"subjects":["subject"],
                "collection_classes":[{"id":"a","label":"A","color":"blue"}],
                "activities":[{"id":"seated","label":"Seated"}],
                "sweat_levels":[{"id":"dry","label":"Dry"}]}"#,
        )
        .expect("the temp directory is writable");
        let catalog = paths.load().expect("a track-less catalog loads");
        let guided_sessions = GuidedSessionCoordinator::new();
        let manager = CollectionManager::new(
            catalog,
            paths,
            PathBuf::from("/dev/null"),
            CameraSettings::default(),
            Arc::new(Registry::new()),
            directory.clone(),
            ProvenanceStore::load(directory.join("provenance.cbor")),
            AudioOutput::Silent,
            guided_sessions.clone(),
        );
        guided_sessions.set_mode_adapter(GuidedMode::Collection, manager.clone());
        (manager, guided_sessions)
    }

    /// Pretend a session is running, and hand back the end of the control
    /// channel a real session task would be reading.
    fn pretend_running(
        manager: &CollectionManager,
        session_id: dashboard::guided_session::GuidedSessionId,
    ) -> (
        mpsc::Receiver<SessionControl>,
        watch::Receiver<BrowserDeparture>,
        watch::Receiver<LiveAudioSettings>,
    ) {
        let (control, receiver) = mpsc::channel(CONTROL_CAPACITY);
        let (browser_departure, browser_departure_receiver) =
            watch::channel(BrowserDeparture::default());
        let settings = LiveAudioSettings {
            output: AudioOutput::Silent,
            gain: 0.0,
        };
        let (audio, audio_receiver) = watch::channel(settings);
        manager.state.lock().unwrap().phase = Phase::Running {
            session_id,
            control,
            browser_departure,
            audio,
        };
        (receiver, browser_departure_receiver, audio_receiver)
    }

    #[test]
    fn collection_pauses_only_when_the_last_visible_guided_view_leaves() {
        let (manager, guided_sessions) = bare_manager("last-leaves");
        let lease = guided_sessions
            .acquire_current(GuidedMode::Collection, None)
            .unwrap();
        let (mut control, mut browser_departure, _audio) =
            pretend_running(&manager, lease.binding().session_id);
        let generic = guided_sessions.connect_browser();
        let first = guided_sessions.connect_browser();
        let second = guided_sessions.connect_browser();
        first
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();
        second
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();

        drop(generic);
        drop(first);
        assert!(
            control.try_recv().is_err(),
            "a session with a browser still attached was told nobody is watching"
        );

        drop(second);
        assert!(browser_departure.has_changed().unwrap());
        assert_eq!(browser_departure.borrow_and_update().0, 1);
        lease.finish(SessionExit::Completed);
    }

    #[test]
    fn a_reconnected_visible_view_can_pause_collection_again() {
        let (manager, guided_sessions) = bare_manager("reconnecting");
        let lease = guided_sessions
            .acquire_current(GuidedMode::Collection, None)
            .unwrap();
        let (_control, mut browser_departure, _audio) =
            pretend_running(&manager, lease.binding().session_id);

        for _ in 0..2 {
            let connection = guided_sessions.connect_browser();
            connection
                .set_visible_mode(Some(GuidedMode::Collection))
                .unwrap();
            drop(connection);
            assert!(browser_departure.has_changed().unwrap());
            browser_departure.borrow_and_update();
        }
        lease.finish(SessionExit::Completed);
    }

    #[test]
    fn browser_departure_bypasses_a_saturated_operator_queue() {
        let (manager, guided_sessions) = bare_manager("departure-priority");
        let lease = guided_sessions
            .acquire_current(GuidedMode::Collection, None)
            .unwrap();
        let (mut control, mut browser_departure, _audio) =
            pretend_running(&manager, lease.binding().session_id);
        let browser = guided_sessions.connect_browser();
        browser
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();

        for _ in 0..CONTROL_CAPACITY {
            manager.start_track();
        }
        assert_eq!(control.len(), CONTROL_CAPACITY);

        drop(browser);

        assert!(browser_departure.has_changed().unwrap());
        assert_eq!(browser_departure.borrow_and_update().0, 1);
        assert!(matches!(control.try_recv(), Ok(SessionControl::StartTrack)));
        lease.finish(SessionExit::Completed);
    }

    #[test]
    fn stale_pause_callback_cannot_pause_a_replacement_session() {
        let (manager, guided_sessions) = bare_manager("stale-pause");
        let old = guided_sessions
            .acquire_current(GuidedMode::Collection, None)
            .unwrap();
        let old_binding = old.binding().clone();
        old.finish(SessionExit::Completed);
        let replacement = guided_sessions
            .acquire_current(GuidedMode::Collection, None)
            .unwrap();
        let (_control, browser_departure, _audio) =
            pretend_running(&manager, replacement.binding().session_id);

        manager.pause_for_no_visible_views(&old_binding);
        assert!(!browser_departure.has_changed().unwrap());
        manager.pause_for_no_visible_views(replacement.binding());
        assert!(browser_departure.has_changed().unwrap());
        replacement.finish(SessionExit::Completed);
    }

    #[test]
    fn rapid_audio_updates_coalesce_without_starving_lifecycle_control() {
        let (manager, guided_sessions) = bare_manager("audio-coalescing");
        let lease = guided_sessions
            .acquire_current(GuidedMode::Collection, None)
            .unwrap();
        let (mut control, _browser_departure, audio) =
            pretend_running(&manager, lease.binding().session_id);

        for volume in 0..=1000 {
            manager.audio_settings.lock().unwrap().volume_permille = volume;
            manager.update_running_audio();
        }
        manager.finish_collection();

        assert_eq!(audio.borrow().gain, 1.0);
        assert!(matches!(control.try_recv(), Ok(SessionControl::Finish)));
        assert!(control.try_recv().is_err());
        lease.finish(SessionExit::Completed);
    }

    #[tokio::test]
    async fn lagged_recording_stream_is_an_explicit_integrity_failure() {
        let (frames, mut receiver) = broadcast::channel(1);
        frames.send(Frame::Probe {}).unwrap();
        frames.send(Frame::Probe {}).unwrap();
        let error = receiver.recv().await.unwrap_err();

        let (report, exit) = recording_stream_failure(error);
        assert!(report.contains("lost 1 frames"));
        assert!(report.contains("incomplete take"));
        assert!(matches!(
            exit,
            SessionExit::DependencyFailed(detail)
                if detail.contains("lost 1 frames") && detail.contains("incomplete take")
        ));
    }

    #[test]
    fn closed_collection_control_task_is_reported() {
        let (manager, guided_sessions) = bare_manager("closed-control");
        let lease = guided_sessions
            .acquire_current(GuidedMode::Collection, None)
            .unwrap();
        let (control, browser_departure, audio) =
            pretend_running(&manager, lease.binding().session_id);
        drop(control);
        drop(browser_departure);
        drop(audio);
        let mut outbound = manager.subscribe();

        manager.finish_collection();

        assert!(matches!(
            outbound.try_recv(),
            Ok(Frame::Log { message, .. })
                if message.contains("collection control task has stopped")
        ));
        lease.finish(SessionExit::Completed);
    }

    #[tokio::test]
    async fn supervised_task_panic_restores_phase_and_releases_lease() {
        for running in [false, true] {
            let (manager, guided_sessions) = bare_manager(if running {
                "running-panic"
            } else {
                "arming-panic"
            });
            let lease = guided_sessions
                .acquire_current(GuidedMode::Collection, None)
                .unwrap();
            let session_id = lease.binding().session_id;
            if running {
                let (_control, _browser_departure, _audio) = pretend_running(&manager, session_id);
            } else {
                manager.state.lock().unwrap().phase = Phase::Starting { session_id };
            }

            let result = std::panic::AssertUnwindSafe(async {
                panic!("injected task panic");
            })
            .catch_unwind()
            .await;
            assert!(result.is_err());
            drop(lease);
            manager.restore_idle_after_task(session_id);

            assert!(matches!(manager.state.lock().unwrap().phase, Phase::Idle));
            assert!(guided_sessions.snapshot().active.is_none());
            assert!(matches!(
                guided_sessions.snapshot().failure,
                Some(dashboard::guided_session::GuidedFailure {
                    kind: dashboard::guided_session::GuidedFailureKind::TaskFailed,
                    ..
                })
            ));
        }
    }

    /// Channel-major blob from per-channel sample runs.
    fn blob(channels: &[&[i16]]) -> Vec<u8> {
        channels
            .iter()
            .flat_map(|channel| channel.iter().flat_map(|value| value.to_le_bytes()))
            .collect()
    }

    #[test]
    fn an_electrode_offset_does_not_change_the_activity_measure() {
        let quiet = [-2i16, 2, -2, 2];
        let offset: Vec<i16> = quiet.iter().map(|value| value + 20_000).collect();
        assert_eq!(
            mean_absolute_deviation(1, &blob(&[&quiet])),
            mean_absolute_deviation(1, &blob(&[&offset])),
        );
    }

    #[test]
    fn each_channels_own_offset_comes_off_separately() {
        // Two channels with the same 2-count swing on wildly different offsets: the
        // measure is the swing, not the difference between the channels.
        let low = [-2i16, 2, -2, 2];
        let high = [29_998i16, 30_002, 29_998, 30_002];
        assert_eq!(mean_absolute_deviation(2, &blob(&[&low, &high])), Some(2.0));
    }

    #[test]
    fn an_empty_or_channel_less_window_has_no_measure() {
        assert_eq!(mean_absolute_deviation(16, &[]), None);
        assert_eq!(mean_absolute_deviation(0, &blob(&[&[1i16, 2]])), None);
    }
}
