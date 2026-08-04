//! The collection session manager: the state machine tying the recorder, the
//! webcam capture, and the beatmap generator to the browser game.
//!
//! One manager exists for the whole backend. Browsers all see the same session:
//! collection frames fan out over a broadcast channel every browser session
//! forwards, and control frames from any browser funnel into the methods here.
//! The manager is the *only* writer of collection truth — the browser renders.
//!
//! Lifecycle: `Idle` → (`StartCollection`) → `Starting` (arming in flight) →
//! `Running` (a spawned session task owns the recorder and the device's EMG
//! subscription; `Armed` until `TrackStarted` anchors the beat grid, then
//! `Playing`) → `Reviewing` (track ended, or `FinishCollection` ended it
//! early; files finalized on disk either way) → (`StopCollection { save }`) →
//! `Idle`. Keep-or-discard is decided only in review, with the summary in
//! view; the sole self-resolving path is an armed session whose browser never
//! starts the track, which times out and discards itself.
//!
//! Obligations honored here (from the unit reviews):
//! - the recorder creates the session directory before video starts in it;
//! - every `VideoCapture` call that can block runs under `spawn_blocking`;
//! - health is polled at ~500 ms (which is also the video start-offset
//!   resolution);
//! - recorder and video file reports merge into one `CollectionSummary.files`;
//! - `generate` receives classes in catalog order (the lane order browsers see);
//! - a second `TrackStarted` for an anchored session is ignored;
//! - `StartCollection` ids are validated against the live catalog.

use crate::collect::beatmap::{CatalogPaths, TrackCatalog};
use crate::collect::interfaces::{
    BeatmapGenerator, EmgWindow, HardwareIdentity, RecordingHandle, SessionEvent, SessionManifest,
    SessionRecorder, VideoCapture, VideoReport,
};
use crate::collect::provenance::ProvenanceStore;
use crate::collect::recorder::FileSessionRecorder;
use crate::collect::video::{CameraSettings, FfmpegVideoCapture};
use crate::registry::Registry;
use protocol::{
    Beatmap, BoardRevision, ClassId, CollectionPause, CollectionPhase, CollectionSummary,
    DifficultyLevel, DurationMilliseconds, FileReport, Frame, LogLevel, NoteIndex, RecordingHealth,
    SessionId, SessionMetadata, StreamProgress, TrackId, TrackInfo, TrackMilliseconds,
    UnixMilliseconds,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};

/// Fan-out capacity for collection frames toward browsers. State ticks are
/// small and periodic; a browser that lags simply sees the next tick.
const OUTBOUND_CAPACITY: usize = 64;

/// How long arming waits for the device's first EMG window (which carries the
/// hardware identity) before giving up.
const FIRST_WINDOW_TIMEOUT: Duration = Duration::from_secs(3);

/// An armed session that never receives `TrackStarted` (browser closed mid
/// countdown) is aborted and discarded after this long.
const ARMED_TIMEOUT: Duration = Duration::from_secs(90);

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
    TrackStarted(UnixMilliseconds),
    /// Audio resumed after a pause, at this instant and this track position.
    TrackResumed(UnixMilliseconds, TrackMilliseconds),
    /// End the session now and move to review, mid-track or not.
    Finish,
}

enum Phase {
    Idle,
    /// An arming task is in flight; further starts are rejected.
    Starting,
    Running {
        control: mpsc::UnboundedSender<SessionControl>,
    },
    /// The wire-facing session/summary data lives in the cached `latest_state`
    /// frame; the phase itself only needs what `StopCollection` acts on. A
    /// practice session has no directory — nothing was recorded.
    Reviewing {
        directory: Option<PathBuf>,
    },
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
    /// don counts per subject and arm.
    provenance: ProvenanceStore,
    state: Mutex<ManagerState>,
}

impl CollectionManager {
    pub fn new(
        catalog: TrackCatalog,
        catalog_paths: CatalogPaths,
        camera_device: PathBuf,
        camera_settings: CameraSettings,
        registry: Arc<Registry>,
        sessions_root: PathBuf,
        provenance: ProvenanceStore,
    ) -> Arc<Self> {
        let (outbound, _) = broadcast::channel(OUTBOUND_CAPACITY);
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
        let state = self.state.lock().unwrap();
        let mut frames = vec![self.catalog_frame()];
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

    /// Re-read the library from disk and tell every browser what it now holds.
    /// An import or a delete is only finished once this has run.
    pub fn reload_catalog(&self) -> anyhow::Result<()> {
        let reloaded = self.catalog_paths.load()?;
        *self.catalog.write().unwrap() = reloaded;
        self.publish(self.catalog_frame());
        Ok(())
    }

    /// Filesystem path of a track's audio, for the HTTP audio route.
    pub fn audio_path(&self, track_id: &TrackId) -> Option<PathBuf> {
        self.catalog.read().unwrap().audio_path(track_id)
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
        {
            let mut state = self.state.lock().unwrap();
            if !matches!(state.phase, Phase::Idle) {
                drop(state);
                self.publish_error("rejected session start: a session already exists".into());
                return;
            }
            state.phase = Phase::Starting;
        }
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = manager
                .clone()
                .arm(metadata, track_id, difficulty, record_video, device_id)
                .await
            {
                manager.publish_error(format!("session start failed: {error:#}"));
                manager.state.lock().unwrap().phase = Phase::Idle;
                manager.publish_idle();
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
    async fn acquire_device(&self, device_id: &str) -> anyhow::Result<AcquiredDevice> {
        let mut emg_receiver = self
            .registry
            .subscribe(device_id)
            .ok_or_else(|| anyhow::anyhow!("device '{device_id}' is not connected"))?;
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
        .map_err(|_| anyhow::anyhow!("no EMG from '{device_id}' within 3 s"))??;

        let transport = self
            .registry
            .list()
            .into_iter()
            .find(|device| device.id == device_id)
            .map(|device| device.transport)
            .ok_or_else(|| anyhow::anyhow!("device '{device_id}' disappeared"))?;
        let device_config = self
            .registry
            .config_of(device_id)
            .ok_or_else(|| anyhow::anyhow!("device '{device_id}' has no config"))?;
        let provenance = self
            .registry
            .provenance_of(device_id)
            .ok_or_else(|| anyhow::anyhow!("device '{device_id}' reported no provenance"))?;
        Ok(AcquiredDevice {
            emg_receiver,
            hardware: HardwareIdentity {
                device_id: device_id.to_string(),
                transport,
                channels,
                sample_rate,
                scale_uv,
                device_config,
                provenance,
                board_revision: self.provenance.board_revision(device_id),
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
        device_id: Option<String>,
    ) -> anyhow::Result<()> {
        let acquired = match device_id {
            Some(device_id) => Some(self.acquire_device(&device_id).await?),
            None => None,
        };

        let (track, class_ids, beatmap, beat_times) = {
            let catalog = self.catalog.read().unwrap();
            let track = catalog
                .tracks()
                .into_iter()
                .find(|track| track.id == track_id)
                .ok_or_else(|| anyhow::anyhow!("unknown track '{track_id}'"))?;
            let class_ids = catalog.class_ids();
            let seed = now().get();
            let beatmap = catalog.generate(&track_id, &class_ids, difficulty, seed)?;
            let beat_times = catalog.beat_times(&track_id).unwrap_or_default();
            (track, class_ids, beatmap, beat_times)
        };

        let created = now();
        let practice = acquired.is_none();
        let session_device_id = acquired
            .as_ref()
            .map(|device| device.hardware.device_id.clone());
        let session_id = SessionId(format!(
            "{}_{}{}",
            chrono::Local::now().format("%Y-%m-%dT%H-%M-%S"),
            metadata.subject,
            if practice { "_practice" } else { "" },
        ));

        let (recorder, directory, emg_receiver) = match acquired {
            Some(AcquiredDevice {
                emg_receiver,
                hardware,
            }) => {
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
                (Some(recorder), Some(directory), Some(emg_receiver))
            }
            None => (None, None, None),
        };

        // The operator asked for video or they did not. A camera that cannot
        // deliver it aborts the start — a session that silently records EMG
        // alone is the failure this whole path exists to prevent. Practice
        // sessions have no directory to record into, so no video either.
        let recording_handle: Option<RecordingHandle> = match (record_video, &directory) {
            (true, Some(directory)) => {
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
                    Ok(handle) => Some(handle),
                    Err(error) => {
                        // Nothing has been recorded yet, so the directory holds
                        // only the manifest; leaving it would look like a take.
                        let _ = std::fs::remove_dir_all(directory);
                        return Err(error.context("this session asked for video"));
                    }
                }
            }
            (false, _) | (true, None) => None,
        };

        let (control, control_receiver) = mpsc::unbounded_channel();
        {
            let mut state = self.state.lock().unwrap();
            state.phase = Phase::Running { control };
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
            directory,
            recorder,
            emg_receiver,
            session_device_id,
            control_receiver,
            recording_handle,
            beatmap,
            track,
        );
        tokio::spawn(session.run());
        Ok(())
    }

    /// Browser → `TrackStarted`.
    pub fn track_started(&self, at: UnixMilliseconds) {
        let state = self.state.lock().unwrap();
        if let Phase::Running { control } = &state.phase {
            let _ = control.send(SessionControl::TrackStarted(at));
        }
    }

    /// Browser → `TrackResumed`.
    pub fn track_resumed(&self, at: UnixMilliseconds, position: TrackMilliseconds) {
        let state = self.state.lock().unwrap();
        if let Phase::Running { control } = &state.phase {
            let _ = control.send(SessionControl::TrackResumed(at, position));
        }
    }

    /// Browser → `FinishCollection`: end the running session now; recording is
    /// finalized as if the track had played out and the summary screen decides
    /// what happens to the partial take.
    pub fn finish_collection(&self) {
        let state = self.state.lock().unwrap();
        if let Phase::Running { control } = &state.phase {
            let _ = control.send(SessionControl::Finish);
        }
    }

    /// Browser → `StopCollection`: resolves the session under review;
    /// `save: false` deletes the directory.
    pub fn stop_collection(&self, save: bool) {
        let reviewed = {
            let mut state = self.state.lock().unwrap();
            match &state.phase {
                Phase::Reviewing { directory } => {
                    let directory = directory.clone();
                    state.phase = Phase::Idle;
                    Some(directory)
                }
                Phase::Idle | Phase::Starting | Phase::Running { .. } => None,
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
    receiver: &mut Option<broadcast::Receiver<Frame>>,
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

/// A live session. The four device-shaped `Option`s below are all `None`
/// together for a practice session: nothing records, nothing can stall, and
/// every cue misses.
struct RunningSession {
    manager: Arc<CollectionManager>,
    session_id: SessionId,
    directory: Option<PathBuf>,
    recorder: Option<FileSessionRecorder>,
    emg_receiver: Option<broadcast::Receiver<Frame>>,
    device_id: Option<String>,
    control: mpsc::UnboundedReceiver<SessionControl>,
    recording_handle: Option<RecordingHandle>,
    cues: Vec<CueState>,
    track: TrackInfo,
    /// The instant audio t = 0 was, for the segment now playing. `TrackStarted`
    /// sets it and every `TrackResumed` re-derives it, so a cue's wall-clock
    /// time is always `anchor + note position` for the segment it falls in.
    anchor: Option<UnixMilliseconds>,
    /// When the most recent EMG window landed, the stall detector's baseline.
    last_window_at: Option<UnixMilliseconds>,
    paused: Option<CollectionPause>,
    armed_at: UnixMilliseconds,
    baseline: ActivityBaseline,
    activity_hits: u32,
    cues_per_class: BTreeMap<ClassId, u16>,
    latest_health: RecordingHealth,
}

/// How the session ends: into review for a decision, or discarded outright.
enum Ending {
    /// Track finished, the browser finished it early, or the device died:
    /// finalize and enter review.
    Review,
    /// An armed session nobody started: finalize, delete, back to idle.
    Discard,
}

impl RunningSession {
    #[allow(clippy::too_many_arguments)]
    fn new(
        manager: Arc<CollectionManager>,
        session_id: SessionId,
        directory: Option<PathBuf>,
        recorder: Option<FileSessionRecorder>,
        emg_receiver: Option<broadcast::Receiver<Frame>>,
        device_id: Option<String>,
        control: mpsc::UnboundedReceiver<SessionControl>,
        recording_handle: Option<RecordingHandle>,
        beatmap: Beatmap,
        track: TrackInfo,
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
            directory,
            recorder,
            emg_receiver,
            device_id,
            control,
            recording_handle,
            cues,
            track,
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
        }
    }

    async fn run(mut self) {
        let mut ticker = tokio::time::interval(TICK);
        let mut publish_health = false;
        let ending = loop {
            tokio::select! {
                message = self.control.recv() => match message {
                    Some(SessionControl::TrackStarted(at)) => self.anchor_track(at),
                    Some(SessionControl::TrackResumed(at, position)) => self.resume_track(at, position),
                    Some(SessionControl::Finish) => break Ending::Review,
                    None => break Ending::Review, // manager dropped; shouldn't happen
                },
                frame = next_emg(&mut self.emg_receiver) => match frame {
                    Ok(Frame::Emg { seq, t0_us, channels, samples, missing, .. }) => {
                        self.ingest_window(seq, t0_us, channels, &samples, &missing);
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => {
                        self.manager.publish_error(
                            "device stream ended mid-session; finalizing".into(),
                        );
                        break Ending::Review;
                    }
                },
                _ = ticker.tick() => {
                    if let Some(ending) = self.tick() {
                        break ending;
                    }
                    publish_health = !publish_health;
                    if publish_health {
                        self.poll_health();
                        self.publish_state();
                    }
                }
            }
        };
        self.finalize(ending).await;
    }

    fn anchor_track(&mut self, at: UnixMilliseconds) {
        // Defense in depth: a second anchor for an anchored session is ignored
        // (a browser bug here would silently corrupt every later cue label).
        if self.anchor.is_some() {
            tracing::warn!("ignoring duplicate track_started for {}", self.session_id);
            return;
        }
        self.anchor = Some(at);
        self.place_unlogged_cues(at);
        // Silence is measured from here rather than from arming, so a session
        // armed early does not start out looking stalled.
        self.last_window_at = Some(at);
        if let Some(recorder) = &mut self.recorder {
            if let Err(error) = recorder.append_event(&SessionEvent::TrackStarted { at }) {
                tracing::warn!("failed to log track start: {error:#}");
            }
        }
        self.publish_state();
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

    /// Freeze the cue timeline: the device has gone quiet, so anything cued from
    /// here on would be asked of a subject nothing is being recorded from.
    fn pause_for_silence(&mut self, at: UnixMilliseconds, silent_for: DurationMilliseconds) {
        let track_position = self.track_position(at);
        let device_id = self.device_id.clone().unwrap_or_default();
        self.interrupt_cues_in_progress(at);
        let event = SessionEvent::Paused {
            at,
            track_position,
            silent_for,
            device_id: device_id.clone(),
        };
        if let Some(recorder) = &mut self.recorder {
            if let Err(error) = recorder.append_event(&event) {
                tracing::warn!("failed to log the pause: {error:#}");
            }
        }
        self.paused = Some(CollectionPause {
            device_id,
            since: at,
            silent_for,
            track_position,
            device_recovered: false,
        });
        self.publish_state();
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
                if let Some(recorder) = &mut self.recorder {
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

    /// Re-anchor on the browser's report of where audio actually restarted, and
    /// place every cue still ahead of the playhead against it.
    fn resume_track(&mut self, at: UnixMilliseconds, position: TrackMilliseconds) {
        let Some(pause) = self.paused.take() else {
            tracing::warn!("ignoring track_resumed for a session that is not paused");
            return;
        };
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
        if let Some(recorder) = &mut self.recorder {
            if let Err(error) = recorder.append_event(&SessionEvent::Resumed {
                at,
                track_position: position,
                paused_for,
            }) {
                tracing::warn!("failed to log the resume: {error:#}");
            }
        }
        self.publish_state();
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
        if let Some(recorder) = &mut self.recorder {
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

    /// Periodic work; returns an ending when the session concludes on its own.
    fn tick(&mut self) -> Option<Ending> {
        let current = now();
        match self.anchor {
            None => {
                if current.since(self.armed_at).get() > ARMED_TIMEOUT.as_millis() as i64 {
                    self.manager
                        .publish_error("armed session timed out; discarding".into());
                    return Some(Ending::Discard);
                }
            }
            Some(anchor) => {
                // A frozen timeline resolves nothing and cannot reach the
                // track's end; only the browser's resume restarts it.
                if self.paused.is_some() {
                    return None;
                }
                if let Some(silent_for) = self.silence_past_threshold(current) {
                    self.pause_for_silence(current, silent_for);
                    return None;
                }
                self.resolve_cues(current);
                let track_end = anchor
                    .at_track_position(TrackMilliseconds::new(self.track.duration.get()))
                    .get()
                    + TRACK_END_SLACK.as_millis() as u64;
                if current.get() >= track_end {
                    return Some(Ending::Review);
                }
            }
        }
        None
    }

    /// How long the device has been silent, once that exceeds
    /// [`STALL_THRESHOLD`]. Always `None` for a practice session, which has no
    /// device to fall silent.
    fn silence_past_threshold(&self, current: UnixMilliseconds) -> Option<DurationMilliseconds> {
        let last = self.last_window_at?;
        self.emg_receiver.as_ref()?;
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
                if let Some(recorder) = &mut self.recorder {
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
                    if let Some(recorder) = &mut self.recorder {
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
        let emg = self.recorder.as_mut().map_or(
            StreamProgress {
                bytes_on_disk: 0,
                advancing: false,
            },
            |recorder| recorder.health(),
        );
        let video = self
            .recording_handle
            .as_ref()
            .map(|handle| self.manager.video.lock().unwrap().health(handle));
        self.latest_health = RecordingHealth {
            emg,
            video,
            recorded: self.recorder.as_ref().map(SessionRecorder::recorded_emg),
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
            .directory
            .as_ref()
            .is_some_and(|directory| directory.join("placement.jpg").exists())
            .then(now);
        self.manager.publish(Frame::CollectionState {
            phase,
            placement_photo,
        });
    }

    async fn finalize(mut self, ending: Ending) {
        // Video first, off the runtime: stop_recording can block up to 5 s.
        let video_report: Option<VideoReport> = match self.recording_handle.take() {
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
        if let Some(directory) = &self.directory {
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
            emg_gap_count: self
                .recorder
                .as_ref()
                .map_or(0, |recorder| recorder.emg_gap_count()),
            video_start_offset: video_report.map(|report| report.start_offset),
        };

        // The recorder's own file reports only exist after finish(), so the
        // SessionEnd event's summary carries video/photo files; the frame the
        // summary screen renders carries all of them. A practice session has no
        // recorder and reports no files.
        if let Some(recorder) = self.recorder.take() {
            match Box::new(recorder).finish(&summary) {
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

        match ending {
            Ending::Review => {
                {
                    let mut state = self.manager.state.lock().unwrap();
                    state.phase = Phase::Reviewing {
                        directory: self.directory.clone(),
                    };
                }
                self.manager.publish(Frame::CollectionState {
                    phase: CollectionPhase::Reviewing {
                        session_id: self.session_id,
                        summary,
                    },
                    placement_photo: None,
                });
            }
            Ending::Discard => {
                self.manager.state.lock().unwrap().phase = Phase::Idle;
                // A practice session has nothing on disk to discard.
                if let Some(directory) = &self.directory {
                    if let Err(error) = std::fs::remove_dir_all(directory) {
                        self.manager
                            .publish_error(format!("failed to discard session: {error}"));
                    }
                }
                self.manager.publish_idle();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mean_absolute_deviation;

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
