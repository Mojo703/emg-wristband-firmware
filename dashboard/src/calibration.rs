//! Exact-connection adapter for backend-guided firmware calibration runs.

use crate::registry::{BoundDeviceHandle, ControlDeliveryError, Registry};
use dashboard::guided_session::{
    CoordinatorError, DeviceConnectionIdentity, GuidedIntentRequest, GuidedMode, GuidedModeAdapter,
    GuidedSessionBinding, GuidedSessionCoordinator, SessionExit, SessionLease,
};
use protocol::{
    CalibrationRunId, CalibrationRunKey, CalibrationScheduleRevision, CalibrationSessionId, Frame,
    GuidedCalibrationSnapshot, GuidedSessionAction,
};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};

/// The wire protocol permits up to 32 schedule entries, but the real USB CDC
/// endpoint has demonstrated reliable delivery only for smaller operational
/// units. This transport choice is not a protocol limit: firmware still
/// validates/reassembles chunks up to the public 32-entry maximum.
const OPERATIONAL_SCHEDULE_UPLOAD_ENTRIES: usize = 8;
const UPLOAD_ACK_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_CHUNK_ACK_RETRIES: u8 = 3;
const SCHEDULE_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(2);
const INTERRUPT_OR_DISCARD_DECISION_TIMEOUT: Duration = Duration::from_secs(5);
// Finalization includes queued checkpoint processing, fit/polish work, CRC,
// and an atomic resident-record promotion on the wristband.  It is not an
// interruption round trip, so preserve a realistic independent deadline.
const SAVE_DECISION_TIMEOUT: Duration = Duration::from_secs(120);
const RUN_ACTION_CAPACITY: usize = 8;

pub struct CalibrationModeAdapter {
    registry: Arc<Registry>,
    coordinator: GuidedSessionCoordinator,
    active: Mutex<Option<ActiveRun>>,
    tracks: Vec<crate::collect::beatmap::CalibrationTrack>,
    selected_track_id: Mutex<Option<String>>,
    collection: Mutex<Option<Arc<crate::collect::manager::CollectionManager>>>,
    timing: Mutex<Option<Arc<crate::timing::TimingService>>>,
    this: Weak<Self>,
}

struct ActiveRun {
    binding: GuidedSessionBinding,
    actions: mpsc::Sender<RunAction>,
    gate: tokio::sync::watch::Sender<RunGate>,
}

struct CalibrationRunContext {
    lease: SessionLease,
    device: DeviceConnectionIdentity,
    bound: BoundDeviceHandle,
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    track: crate::collect::beatmap::CalibrationTrack,
}

enum RunAction {
    /// Continue retains the completed evidence on the device and asks it to
    /// wait for a newly authored schedule.  The adapter owns the revision
    /// minting; the browser never talks to the device directly.
    Continue,
    /// A replacement song can be authored only at the between-songs boundary.
    /// The actor owns the transition so an in-flight schedule cannot be
    /// retargeted by a stale browser action.
    SelectNextTrack(crate::collect::beatmap::CalibrationTrack),
    /// Request candidate activation.  Firmware remains the authority for
    /// numerical validity and record CRC checks.
    Save,
    /// Drop the candidate while retaining the previous resident model.
    Discard,
    /// End the run from any phase. Unlike the between-songs Discard decision,
    /// a committed schedule must first reach its device-owned interruption
    /// boundary before its candidate can be discarded.
    Exit,
}

/// Latest requested heartbeat/playback gate. Before Commit, pause and resume
/// are idempotent state rather than an event backlog. After Commit, a pause is
/// sticky until the device reports the song boundary because the anchored
/// device timeline cannot be resumed in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunGate {
    Interrupted,
    Running,
}

fn gate_withholds_heartbeat(gate: RunGate, commit: &ScheduleCommitPhase) -> bool {
    matches!(gate, RunGate::Interrupted)
        && matches!(
            commit,
            ScheduleCommitPhase::Sent { .. } | ScheduleCommitPhase::Accepted
        )
}

/// Visibility may reopen a transaction only before Commit. Once Commit has
/// crossed the wire, the device schedule is tied to an absolute monotonic
/// anchor. Pausing host audio and then resuming it in place would move the song
/// while the device cues kept their original instants, silently mislabelling
/// every later window. That departure is therefore a sticky interruption until
/// the device reports the between-songs boundary.
fn effective_gate(requested: RunGate, commit: &ScheduleCommitPhase) -> RunGate {
    match (requested, commit) {
        (RunGate::Running, ScheduleCommitPhase::AwaitingDeviceReadiness) => RunGate::Running,
        (RunGate::Running, ScheduleCommitPhase::Sent { .. } | ScheduleCommitPhase::Accepted)
        | (RunGate::Interrupted, _) => RunGate::Interrupted,
    }
}

fn schedule_commit_is_authorized(
    gate: RunGate,
    upload: &UploadPhase,
    device_readiness: DeviceCommitReadiness,
    playback_ready: bool,
    commit: &ScheduleCommitPhase,
) -> bool {
    matches!(gate, RunGate::Running)
        && matches!(upload, UploadPhase::Complete)
        && matches!(device_readiness, DeviceCommitReadiness::Ready)
        && playback_ready
        && matches!(commit, ScheduleCommitPhase::AwaitingDeviceReadiness)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceCommitReadiness {
    AwaitingFreshStatus,
    Ready,
}

fn apply_commit_deferral(
    commit: &mut ScheduleCommitPhase,
    device_readiness: &mut DeviceCommitReadiness,
) {
    // A deferral invalidates the readiness observation that authorized the
    // rejected Commit.  Require a later device status to create fresh retry
    // authority; otherwise the actor's next turn immediately resends Commit
    // and can spin against the firmware's reliable deferral response.
    *commit = ScheduleCommitPhase::AwaitingDeviceReadiness;
    *device_readiness = DeviceCommitReadiness::AwaitingFreshStatus;
}

fn enqueue_run_action(
    actions: &mpsc::Sender<RunAction>,
    action: RunAction,
) -> Result<(), CoordinatorError> {
    actions.try_send(action).map_err(|error| match error {
        mpsc::error::TrySendError::Full(_) => CoordinatorError::AdapterTaskBusy,
        mpsc::error::TrySendError::Closed(_) => CoordinatorError::AdapterTaskUnavailable,
    })
}

/// Keep one physical queue slot reserved for the single-use Exit decision.
/// Ordinary actions may be bursty at a song boundary; they must never make it
/// impossible for the operator to stop the device-owned schedule.
fn enqueue_nonterminal_run_action(
    actions: &mpsc::Sender<RunAction>,
    action: RunAction,
) -> Result<(), CoordinatorError> {
    if actions.capacity() <= 1 {
        return Err(CoordinatorError::AdapterTaskBusy);
    }
    enqueue_run_action(actions, action)
}

/// The host actor's phase-local state.  These are deliberately not booleans:
/// a heartbeat cannot be active without the accepted revision it names, and
/// output cannot be both armed and playing.
enum ScheduleCommitPhase {
    AwaitingDeviceReadiness,
    Sent {
        acceptance_deadline: tokio::time::Instant,
    },
    /// The exact run/revision/identity was accepted. Late preparation
    /// narration can no longer project the UI back before this boundary.
    Accepted,
}

impl ScheduleCommitPhase {
    fn projects_preparation(&self) -> bool {
        matches!(self, Self::AwaitingDeviceReadiness | Self::Sent { .. })
    }

    fn sent_at(now: tokio::time::Instant) -> Self {
        Self::Sent {
            acceptance_deadline: now + SCHEDULE_ACCEPTANCE_TIMEOUT,
        }
    }

    fn acceptance_deadline(&self) -> Option<tokio::time::Instant> {
        match self {
            Self::Sent {
                acceptance_deadline,
            } => Some(*acceptance_deadline),
            Self::AwaitingDeviceReadiness | Self::Accepted => None,
        }
    }

    fn acceptance_timed_out(&self, now: tokio::time::Instant) -> bool {
        self.acceptance_deadline()
            .is_some_and(|deadline| now >= deadline)
    }
}

/// One upload exists only on the connection captured by the guided lease.  It
/// advances one device acknowledgement at a time; an enqueue into the host
/// writer is not evidence that a Begin or Chunk reached the device.
enum UploadPhase {
    AwaitingBeginAcknowledgement,
    AwaitingChunkAcknowledgement { first_entry: u32 },
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UploadTimeoutAction {
    FailBegin,
    RetryChunk,
    FailChunk,
    None,
}

fn upload_timeout_action(upload: &UploadPhase, chunk_retries: u8) -> UploadTimeoutAction {
    match upload {
        UploadPhase::AwaitingBeginAcknowledgement => UploadTimeoutAction::FailBegin,
        UploadPhase::AwaitingChunkAcknowledgement { .. }
            if chunk_retries >= MAX_CHUNK_ACK_RETRIES =>
        {
            UploadTimeoutAction::FailChunk
        }
        UploadPhase::AwaitingChunkAcknowledgement { .. } => UploadTimeoutAction::RetryChunk,
        UploadPhase::Complete => UploadTimeoutAction::None,
    }
}

enum HeartbeatMode {
    Withheld,
    Sending {
        schedule_revision: CalibrationScheduleRevision,
        next_sequence: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceInterruption {
    Available,
    Requested,
}

fn request_device_interruption(
    registry: &Registry,
    device: &DeviceConnectionIdentity,
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    state: &mut DeviceInterruption,
) -> Result<(), ControlDeliveryError> {
    if !matches!(state, DeviceInterruption::Available) {
        return Ok(());
    }
    registry.send_bound_control(
        device,
        Frame::CalibrationInterrupt {
            run,
            schedule_revision,
        },
    )?;
    *state = DeviceInterruption::Requested;
    Ok(())
}

enum PlaybackState {
    Dormant,
    Armed {
        playback: crate::collect::audio::Playback,
        audible_anchor: protocol::UnixMilliseconds,
    },
    Playing(crate::collect::audio::Playback),
}

fn enter_between_songs(
    evidence: &mut EvidenceState,
    completed_track_title: &str,
    heartbeat: &mut HeartbeatMode,
    playback: &mut PlaybackState,
) {
    *evidence = EvidenceState::Retained {
        completed_track_title: completed_track_title.to_owned(),
    };
    *heartbeat = HeartbeatMode::Withheld;
    *playback = PlaybackState::Dormant;
}

/// An opened song is affine authority for one accepted schedule identity.  A
/// `JoinHandle` is a future and may not be polled after it has completed; keeping
/// it loose beside the mutable track previously let Continue poll the first
/// song's completed handle a second time (and let a replacement track consume
/// the old song).  Consuming this enum before awaiting makes both states
/// unrepresentable.
enum PlaybackPreparation {
    Pending {
        content_identity: String,
        task: tokio::task::JoinHandle<anyhow::Result<crate::collect::audio::Playback>>,
    },
    Ready {
        content_identity: String,
        playback: crate::collect::audio::Playback,
    },
    Consumed,
}

impl PlaybackPreparation {
    /// Commit creates an absolute device-time anchor only three seconds ahead.
    /// The audio task must therefore have completed before Commit crosses the
    /// wire; awaiting an unfinished task after acceptance would suspend this
    /// actor's heartbeats and could miss that irrevocable anchor.
    fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }

    /// Resolve a finished background task before authorizing Commit. Task
    /// completion alone is not readiness: an audio decoder or output sink may
    /// have already returned an error, in which case the device must never be
    /// given an irrevocable schedule anchor.
    async fn resolve_if_finished(&mut self) -> Result<(), String> {
        let Self::Pending { task, .. } = self else {
            return Ok(());
        };
        if !task.is_finished() {
            return Ok(());
        }
        let Self::Pending {
            content_identity,
            task,
        } = core::mem::replace(self, Self::Consumed)
        else {
            unreachable!("checked pending playback preparation")
        };
        match task.await {
            Ok(Ok(playback)) => {
                *self = Self::Ready {
                    content_identity,
                    playback,
                };
                Ok(())
            }
            Ok(Err(error)) => Err(format!("calibration audio failed: {error:#}")),
            Err(error) => Err(format!("calibration audio task failed: {error}")),
        }
    }
}

fn prepare_playback(
    collection: Arc<crate::collect::manager::CollectionManager>,
    track: &crate::collect::beatmap::CalibrationTrack,
) -> PlaybackPreparation {
    let content_identity = track.content_identity.clone();
    #[cfg(test)]
    if content_identity == "test-content" {
        let task =
            tokio::task::spawn_blocking(|| Ok(crate::collect::audio::Playback::silent_fixture()));
        return PlaybackPreparation::Pending {
            content_identity,
            task,
        };
    }
    let track_id = track.id.clone();
    let entries = track.entries.clone();
    let task = tokio::task::spawn_blocking(move || {
        collection.open_calibration_playback(&track_id, &entries)
    });
    PlaybackPreparation::Pending {
        content_identity,
        task,
    }
}

fn consume_playback_preparation(
    preparation: &mut PlaybackPreparation,
    expected_content_identity: &str,
) -> Result<crate::collect::audio::Playback, String> {
    let PlaybackPreparation::Ready {
        content_identity,
        playback,
    } = core::mem::replace(preparation, PlaybackPreparation::Consumed)
    else {
        return Err("the accepted calibration schedule has no unused audio preparation".into());
    };
    if content_identity != expected_content_identity {
        return Err(format!(
            "calibration audio identity changed before acceptance (prepared {content_identity}, accepted {expected_content_identity})"
        ));
    }
    Ok(playback)
}

enum EvidenceState {
    Fresh,
    /// Evidence remains owned by the schedule that just ended even when the
    /// operator selects a different track for Continue. Keeping that identity
    /// in this variant prevents the projection from relabelling an old
    /// candidate as the newly selected song.
    Retained {
        completed_track_title: String,
    },
}

impl EvidenceState {
    fn is_retained(&self) -> bool {
        matches!(self, Self::Retained { .. })
    }

    fn completed_track_title(&self) -> Option<&str> {
        match self {
            Self::Fresh => None,
            Self::Retained {
                completed_track_title,
            } => Some(completed_track_title),
        }
    }
}

#[derive(Clone, Copy)]
enum CandidateState {
    Absent,
    /// A matching SongResult is sufficient authority to ask the wristband to
    /// build and atomically promote its candidate. The device deliberately
    /// does that work only after an explicit Save.
    Buildable,
    Present(protocol::CalibrationCandidateValidity),
}

/// Save and Discard are durable device operations, not fire-and-forget UI
/// events.  Keep the actor alive until the exact device acknowledgement
/// arrives, while making a duplicate/racing terminal action unrepresentable.
enum TerminalDecision {
    Open,
    AwaitingInterruption { deadline: tokio::time::Instant },
    AwaitingSave { deadline: tokio::time::Instant },
    AwaitingDiscard { deadline: tokio::time::Instant },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitRoute {
    DiscardNow,
    InterruptThenDiscard,
}

fn exit_route(commit: &ScheduleCommitPhase, evidence: &EvidenceState) -> ExitRoute {
    if matches!(commit, ScheduleCommitPhase::AwaitingDeviceReadiness) || evidence.is_retained() {
        ExitRoute::DiscardNow
    } else {
        ExitRoute::InterruptThenDiscard
    }
}

impl TerminalDecision {
    fn deadline(&self) -> Option<tokio::time::Instant> {
        match self {
            Self::Open => None,
            Self::AwaitingInterruption { deadline }
            | Self::AwaitingSave { deadline }
            | Self::AwaitingDiscard { deadline } => Some(*deadline),
        }
    }

    fn acknowledges(
        &self,
        frame: &Frame,
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
    ) -> bool {
        match (self, frame) {
            (Self::AwaitingSave { .. }, Frame::CalibrationResidentActivated { activation }) => {
                activation.run == run && activation.schedule_revision == schedule_revision
            }
            (Self::AwaitingDiscard { .. }, Frame::CalibrationCandidateStatus { candidate }) => {
                candidate.run == run
                    && candidate.schedule_revision == schedule_revision
                    && matches!(
                        candidate.presence,
                        protocol::CalibrationCandidatePresence::Absent
                    )
            }
            _ => false,
        }
    }
}

fn run_action_is_authorized(
    action: &RunAction,
    evidence: &EvidenceState,
    song_counts: &[protocol::CalibrationClassCounts],
    candidate: CandidateState,
    terminal: &TerminalDecision,
) -> bool {
    if !matches!(terminal, TerminalDecision::Open) {
        return false;
    }
    match action {
        RunAction::Continue => evidence.is_retained() && counts_have_deficits(song_counts),
        RunAction::SelectNextTrack(_) => {
            evidence.is_retained() && counts_have_deficits(song_counts)
        }
        RunAction::Save => evidence.is_retained() && candidate.permits_save(),
        RunAction::Discard => evidence.is_retained(),
        RunAction::Exit => true,
    }
}

impl CandidateState {
    fn from_presence(presence: protocol::CalibrationCandidatePresence) -> Self {
        match presence {
            protocol::CalibrationCandidatePresence::Absent => Self::Absent,
            protocol::CalibrationCandidatePresence::Present { validity, .. } => {
                Self::Present(validity)
            }
        }
    }

    fn preserves_buildable_after_status(
        self,
        presence: &protocol::CalibrationCandidatePresence,
    ) -> bool {
        matches!(self, Self::Buildable)
            && matches!(presence, protocol::CalibrationCandidatePresence::Absent)
    }

    fn permits_save(self) -> bool {
        match self {
            Self::Absent => false,
            Self::Buildable => true,
            Self::Present(validity) => validity.permits_activation(),
        }
    }
}

fn counts_have_deficits(counts: &[protocol::CalibrationClassCounts]) -> bool {
    counts.iter().any(|count| count.deficit_count > 0)
}

impl CalibrationModeAdapter {
    pub fn new(
        registry: Arc<Registry>,
        coordinator: GuidedSessionCoordinator,
        tracks: Vec<crate::collect::beatmap::CalibrationTrack>,
    ) -> Arc<Self> {
        coordinator
            .update_idle_calibration(setup_snapshot(&tracks, None))
            .expect("initial calibration projection uses the current coordinator revision");
        Arc::new_cyclic(|this| Self {
            registry,
            coordinator,
            active: Mutex::new(None),
            tracks,
            selected_track_id: Mutex::new(None),
            collection: Mutex::new(None),
            timing: Mutex::new(None),
            this: this.clone(),
        })
    }

    pub fn attach_collection(&self, collection: Arc<crate::collect::manager::CollectionManager>) {
        *self.collection.lock().unwrap() = Some(collection);
    }

    pub fn attach_timing(&self, timing: Arc<crate::timing::TimingService>) {
        *self.timing.lock().unwrap() = Some(timing);
    }

    fn start(
        &self,
        device: DeviceConnectionIdentity,
        request: &GuidedIntentRequest,
    ) -> Result<(), CoordinatorError> {
        if self.selected_track_id.lock().unwrap().is_none() {
            return Err(CoordinatorError::CalibrationTrackRequired);
        }
        let mut active = self.active.lock().unwrap();
        if active.is_some() {
            return Err(CoordinatorError::LeaseHeld {
                mode: GuidedMode::Calibration,
            });
        }

        let lease = self.coordinator.acquire_for_intent(
            request,
            GuidedMode::Calibration,
            Some(device.clone()),
        )?;
        let binding = lease.binding().clone();
        let bound = match self.registry.bind_connection(&device) {
            Some(bound) => bound,
            None => {
                lease.finish(SessionExit::DependencyFailed(
                    "the selected device connection changed before calibration began".into(),
                ));
                return Err(CoordinatorError::DeviceRequired);
            }
        };
        let run = run_key(&binding)?;
        let schedule_revision =
            CalibrationScheduleRevision::new(1).expect("the initial schedule revision is nonzero");
        let track_id = self
            .selected_track_id
            .lock()
            .unwrap()
            .clone()
            .expect("checked above");
        let track = self
            .tracks
            .iter()
            .find(|track| track.id.0 == track_id)
            .cloned()
            .ok_or(CoordinatorError::UnknownCalibrationTrack)?;
        let (actions, action_rx) = mpsc::channel(RUN_ACTION_CAPACITY);
        let (gate, gate_rx) = tokio::sync::watch::channel(RunGate::Running);
        *active = Some(ActiveRun {
            binding: binding.clone(),
            actions,
            gate,
        });
        let Some(adapter) = self.this.upgrade() else {
            *active = None;
            lease.finish(SessionExit::TaskFailed(
                "calibration adapter shut down before its run started".into(),
            ));
            return Err(CoordinatorError::AdapterTaskUnavailable);
        };
        tokio::spawn(async move {
            adapter
                .run(
                    CalibrationRunContext {
                        lease,
                        device,
                        bound,
                        run,
                        schedule_revision,
                        track,
                    },
                    action_rx,
                    gate_rx,
                )
                .await;
        });
        Ok(())
    }

    async fn run(
        self: Arc<Self>,
        context: CalibrationRunContext,
        mut actions: mpsc::Receiver<RunAction>,
        mut gate: tokio::sync::watch::Receiver<RunGate>,
    ) {
        let CalibrationRunContext {
            lease,
            device,
            mut bound,
            run,
            mut schedule_revision,
            mut track,
        } = context;
        // Decoding a full track and measuring an output sink are blocking
        // operations. Start them during the device-owned 30-second preparation
        // on the blocking pool, not in the actor branch that must keep its
        // 500-ms serial heartbeat alive through ScheduleAccepted.
        let Some(playback_collection) = self.collection.lock().unwrap().clone() else {
            lease.finish(SessionExit::TaskFailed(
                "calibration playback manager is unavailable".into(),
            ));
            return;
        };
        let collection_classes = playback_collection.collection_classes();
        let mut prepared_playback = prepare_playback(playback_collection.clone(), &track);
        let binding = lease.binding().clone();
        let mut heartbeat = tokio::time::interval(std::time::Duration::from_millis(u64::from(
            protocol::CALIBRATION_HEARTBEAT_INTERVAL_MILLISECONDS,
        )));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut commit_phase = ScheduleCommitPhase::AwaitingDeviceReadiness;
        let mut device_readiness = DeviceCommitReadiness::AwaitingFreshStatus;
        let mut heartbeat_mode = HeartbeatMode::Sending {
            schedule_revision,
            next_sequence: 0,
        };
        let mut device_interruption = DeviceInterruption::Available;
        let mut gate_state = *gate.borrow_and_update();
        let mut song_counts = Vec::new();
        let mut evidence = EvidenceState::Fresh;
        let mut candidate = CandidateState::Absent;
        let mut terminal = TerminalDecision::Open;
        // Opening an output sink may allocate or negotiate with the host. Do it
        // after the device acceptance, then start the already-opened playback
        // from its own deadline instead of from the 500 ms heartbeat cadence.
        let mut playback = PlaybackState::Dormant;
        let mut playback_alarm = Box::pin(tokio::time::sleep_until(
            tokio::time::Instant::now() + std::time::Duration::from_secs(365 * 24 * 60 * 60),
        ));
        let mut upload = match begin_upload(&self.registry, &device, run, schedule_revision, &track)
        {
            Ok(upload) => upload,
            Err(error) => {
                lease.finish(delivery_failure(error));
                return;
            }
        };
        let mut upload_ack_timeout = tokio::time::interval(UPLOAD_ACK_TIMEOUT);
        upload_ack_timeout.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Tokio intervals tick immediately. Consume that first tick so a
        // newly acknowledged Begin cannot instantly duplicate Chunk 0.
        upload_ack_timeout.tick().await;
        let mut chunk_ack_retries = 0u8;
        let outcome = loop {
            if let Err(detail) = prepared_playback.resolve_if_finished().await {
                break SessionExit::TaskFailed(detail);
            }
            // Playback readiness is independent of device traffic. Re-check it
            // on every actor turn (the heartbeat supplies a 500-ms upper bound)
            // and authorize Commit only once acceptance can be handled without
            // awaiting unfinished host work.
            if schedule_commit_is_authorized(
                gate_state,
                &upload,
                device_readiness,
                prepared_playback.is_ready(),
                &commit_phase,
            ) {
                if let Err(error) =
                    send_schedule_commit(&self.registry, &device, run, schedule_revision, &track)
                {
                    break delivery_failure(error);
                }
                commit_phase = ScheduleCommitPhase::sent_at(tokio::time::Instant::now());
            }
            let schedule_acceptance_deadline =
                commit_phase.acceptance_deadline().unwrap_or_else(|| {
                    tokio::time::Instant::now() + std::time::Duration::from_secs(365 * 24 * 60 * 60)
                });
            let terminal_deadline = terminal.deadline().unwrap_or_else(|| {
                tokio::time::Instant::now() + std::time::Duration::from_secs(365 * 24 * 60 * 60)
            });
            // Device EMG/log traffic can be continuously ready. An unbiased
            // select is required so that operator actions, deadlines, and the
            // playback alarm cannot starve behind the device broadcast.
            tokio::select! {
                changed = gate.changed() => match changed {
                    Ok(()) => {
                        gate_state = effective_gate(*gate.borrow_and_update(), &commit_phase);
                        if gate_withholds_heartbeat(gate_state, &commit_phase) {
                            // Before Commit, heartbeats keep the device-owned
                            // 30-second preparation transaction alive.  They
                            // do not authorize playback, so a hidden view can
                            // safely finish preparation while Commit remains
                            // gated.  Once Commit has crossed the wire,
                            // withholding heartbeat is the device's bounded
                            // interruption mechanism.
                            heartbeat_mode = HeartbeatMode::Withheld;
                            if let PlaybackState::Playing(opened) = &playback {
                                opened.pause();
                            }
                            if let Err(error) = request_device_interruption(
                                &self.registry,
                                &device,
                                run,
                                schedule_revision,
                                &mut device_interruption,
                            ) {
                                break delivery_failure(error);
                            }
                        }
                        if schedule_commit_is_authorized(
                            gate_state,
                            &upload,
                            device_readiness,
                            prepared_playback.is_ready(),
                            &commit_phase,
                        ) {
                            if let Err(error) = send_schedule_commit(
                                &self.registry, &device, run, schedule_revision, &track,
                            ) {
                                break delivery_failure(error);
                            }
                            commit_phase = ScheduleCommitPhase::sent_at(tokio::time::Instant::now());
                        }
                    },
                    Err(_) => break SessionExit::TaskFailed(
                        "calibration lifecycle gate closed".into(),
                    ),
                },
                _ = heartbeat.tick() => {
                    if let HeartbeatMode::Sending { schedule_revision: committed_revision, next_sequence } = &mut heartbeat_mode {
                        let frame = Frame::CalibrationHeartbeat {
                            heartbeat: protocol::CalibrationHeartbeat {
                                run,
                                schedule_revision: *committed_revision,
                                sequence: *next_sequence,
                            },
                        };
                        *next_sequence = next_sequence.wrapping_add(1);
                            if let Err(error) = self.registry.send_bound_control(&device, frame) {
                                break delivery_failure(error);
                            }
                            if let PlaybackState::Playing(playback) = &playback {
                                let timeline = playback.timeline();
                                let _ = self.coordinator.update_calibration(
                                    &binding,
                                    playing_snapshot(
                                        &track,
                                        CalibrationPlayhead {
                                            position_ms: u64::from(timeline.position.get()),
                                            observed_at: timeline.heard_at,
                                        },
                                        &collection_classes,
                                    ),
                                );
                            }
                    }
                }
                received = bound.frames.recv() => match received {
                    // Initial DeviceHello is consumed before this exact
                    // connection is registered. A later hello means the
                    // device link re-epoch'd underneath us; retained device
                    // evidence may survive, but this actor must never carry
                    // its upload/commit authority across that boundary.
                    Ok(Frame::DeviceHello { .. }) => {
                        break SessionExit::DependencyFailed(
                            "the device link restarted during calibration; restart explicitly before any retained evidence can be continued".into(),
                        );
                    }
                    Ok(frame) if terminal.acknowledges(&frame, run, schedule_revision) => {
                        break SessionExit::Completed;
                    }
                    Ok(Frame::CalibrationScheduleAccepted { accepted })
                        if accepted.run == run && accepted.schedule_revision == schedule_revision => {
                        if accepted.content_identity != track.content_identity
                            || !accepted.is_exactly_three_seconds_ahead()
                        {
                            break SessionExit::DependencyFailed(
                                "the device accepted a calibration schedule with a mismatched identity or anchor".into(),
                            );
                        }
                        if !matches!(commit_phase, ScheduleCommitPhase::Sent { .. }) {
                            break SessionExit::DependencyFailed(
                                "the device accepted a schedule that this host did not commit".into(),
                            );
                        }
                        commit_phase = ScheduleCommitPhase::Accepted;
                        heartbeat_mode = if gate_withholds_heartbeat(gate_state, &commit_phase) {
                            HeartbeatMode::Withheld
                        } else {
                            HeartbeatMode::Sending {
                                schedule_revision,
                                next_sequence: 0,
                            }
                        };
                        // Exit after Commit deliberately never opens or arms
                        // audio. The device owns the accepted anchor, so the
                        // actor waits for its heartbeat-timeout interruption
                        // before issuing the durable Discard.
                        if matches!(terminal, TerminalDecision::AwaitingInterruption { .. }) {
                            heartbeat_mode = HeartbeatMode::Withheld;
                            continue;
                        }
                        let host_anchor = self
                            .timing
                            .lock()
                            .unwrap()
                            .as_ref()
                            .and_then(|timing| timing.corrected_host_milliseconds(
                                &device.device_id,
                                accepted.anchor_device_monotonic_microseconds,
                            ));
                        let opened = match consume_playback_preparation(
                            &mut prepared_playback,
                            &track.content_identity,
                        ) {
                            Ok(opened) => opened,
                            Err(detail) => break SessionExit::TaskFailed(detail),
                        };
                        // The accepted anchor is when the subject/device must
                        // hear track position zero, not when the host begins
                        // filling the output queue. Collection derives this
                        // boundary from Timeline::heard_at. Calibration has an
                        // independently fixed device anchor, so start the
                        // already-warmed sink one measured output latency
                        // earlier to make its audible cursor meet that anchor.
                        let output_latency = opened.output_latency();
                        let deadline = match playback_deadline(
                            host_anchor,
                            output_latency,
                            unix_milliseconds(),
                            tokio::time::Instant::now(),
                        ) {
                            Ok(deadline) => deadline,
                            Err(detail) => break SessionExit::DependencyFailed(detail.into()),
                        };
                        tracing::info!(
                            device_id = %device.device_id,
                            device_anchor_microseconds = accepted.anchor_device_monotonic_microseconds,
                            host_audible_anchor_milliseconds = host_anchor.expect("a deadline requires a mapped host anchor"),
                            output_latency_milliseconds = output_latency.get(),
                            "calibration audio armed against the device's audible anchor"
                        );
                        let Ok(audible_anchor) = u64::try_from(
                            host_anchor.expect("a deadline requires a mapped host anchor"),
                        ) else {
                            break SessionExit::DependencyFailed(
                                "the mapped calibration anchor precedes the Unix epoch".into(),
                            );
                        };
                        playback = PlaybackState::Armed {
                            playback: opened,
                            audible_anchor: protocol::UnixMilliseconds::new(audible_anchor),
                        };
                        playback_alarm.as_mut().reset(deadline);
                    }
                    Ok(Frame::CalibrationScheduleUploadAcknowledged { acknowledgement })
                        if acknowledgement.run == run
                            && acknowledgement.schedule_revision == schedule_revision
                            && acknowledgement.content_identity == track.content_identity
                            && acknowledgement.total_count == track.entries.len() as u32
                            && expected_upload_fingerprint(
                                run,
                                schedule_revision,
                                &track,
                                &upload,
                            ) == Some(acknowledgement.operation) => {
                        match advance_upload(
                            &self.registry,
                            &device,
                            run,
                            schedule_revision,
                            &track,
                            &mut upload,
                            acknowledgement.operation,
                        ) {
                            Ok(()) => {}
                            Err(error) => break delivery_failure(error),
                        }
                        // A matching ACK authorizes the next distinct chunk;
                        // it also refreshes the deadline for that chunk.
                        chunk_ack_retries = 0;
                        upload_ack_timeout.reset();
                        if schedule_commit_is_authorized(
                            gate_state,
                            &upload,
                            device_readiness,
                            prepared_playback.is_ready(),
                            &commit_phase,
                        ) {
                            if let Err(error) = send_schedule_commit(
                                &self.registry, &device, run, schedule_revision, &track,
                            ) {
                                break delivery_failure(error);
                            }
                            commit_phase = ScheduleCommitPhase::sent_at(tokio::time::Instant::now());
                        }
                    }
                    Ok(Frame::CalibrationPreparationStatus { status })
                        if status.run == run
                            && status.schedule_revision == schedule_revision
                            && matches!(&evidence, EvidenceState::Fresh) => {
                        let ready_for_schedule = matches!(
                            &status.phase,
                            protocol::CalibrationPreparationPhase::ReadyForSchedule
                        );
                        match status.phase {
                            protocol::CalibrationPreparationPhase::Settling { .. }
                            | protocol::CalibrationPreparationPhase::EstimatingGains { .. }
                            | protocol::CalibrationPreparationPhase::ReadyForSchedule => {
                                if commit_phase.projects_preparation() {
                                    let _ = self.coordinator.update_calibration(
                                        &binding,
                                        preparing_snapshot_from_device(&track, status.phase),
                                    );
                                }
                            }
                            protocol::CalibrationPreparationPhase::Failed { detail } => {
                                break SessionExit::DependencyFailed(detail);
                            }
                        }
                        device_readiness = if ready_for_schedule {
                            DeviceCommitReadiness::Ready
                        } else {
                            DeviceCommitReadiness::AwaitingFreshStatus
                        };
                        if schedule_commit_is_authorized(
                            gate_state,
                            &upload,
                            device_readiness,
                            prepared_playback.is_ready(),
                            &commit_phase,
                        ) {
                            if let Err(error) = send_schedule_commit(
                                &self.registry, &device, run, schedule_revision, &track,
                            ) {
                                break delivery_failure(error);
                            }
                            commit_phase = ScheduleCommitPhase::sent_at(tokio::time::Instant::now());
                        }
                    }
                    Ok(Frame::CalibrationSongInterrupted { interruption })
                        if interruption.run == run
                            && interruption.schedule_revision == schedule_revision
                            && interruption.content_identity == track.content_identity => {
                        if matches!(terminal, TerminalDecision::AwaitingInterruption { .. }) {
                            if let Err(error) = self.registry.send_bound_control(
                                &device,
                                Frame::CalibrationDiscard { run },
                            ) {
                                break delivery_failure(error);
                            }
                            terminal = TerminalDecision::AwaitingDiscard {
                                deadline: tokio::time::Instant::now() + INTERRUPT_OR_DISCARD_DECISION_TIMEOUT,
                            };
                            continue;
                        }
                        enter_between_songs(&mut evidence, &track.title, &mut heartbeat_mode, &mut playback);
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            between_songs_snapshot(
                                &track,
                                &song_counts,
                                &evidence,
                                candidate,
                                &self.tracks,
                                &collection_classes,
                            ),
                        );
                    }
                    Ok(Frame::CalibrationSongResult { result })
                        if result.run == run
                            && result.schedule_revision == schedule_revision
                            && result.content_identity == track.content_identity => {
                        song_counts = result.counts;
                        // A completed, identity-matching song is the explicit
                        // boundary at which Save becomes available. Firmware
                        // defers fitting/polishing and resident promotion until
                        // it receives that Save request.
                        candidate = CandidateState::Buildable;
                        if matches!(terminal, TerminalDecision::AwaitingInterruption { .. }) {
                            heartbeat_mode = HeartbeatMode::Withheld;
                            playback = PlaybackState::Dormant;
                            if let Err(error) = self.registry.send_bound_control(
                                &device,
                                Frame::CalibrationDiscard { run },
                            ) {
                                break delivery_failure(error);
                            }
                            terminal = TerminalDecision::AwaitingDiscard {
                                deadline: tokio::time::Instant::now() + INTERRUPT_OR_DISCARD_DECISION_TIMEOUT,
                            };
                            continue;
                        }
                        // A normal song result is the same transport/output
                        // boundary as an explicit interruption.  Previously
                        // the UI entered BetweenSongs while the actor kept
                        // heartbeating and playing; SelectNextTrack therefore
                        // rejected the very state the UI advertised.
                        enter_between_songs(
                            &mut evidence,
                            &track.title,
                            &mut heartbeat_mode,
                            &mut playback,
                        );
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            between_songs_snapshot(
                                &track,
                                &song_counts,
                                &evidence,
                                candidate,
                                &self.tracks,
                                &collection_classes,
                            ),
                        );
                    }
                    Ok(Frame::CalibrationCandidateStatus { candidate: status })
                        if status.run == run
                            && status.schedule_revision == schedule_revision
                            && candidate_presence_matches_track(&status.presence, &track) => {
                        // The exact AwaitingDiscard/Absent pair was consumed
                        // before this arm. Any status received while Save or
                        // another durable operation is pending is progress
                        // noise, not authority to replace its projection.
                        if !matches!(terminal, TerminalDecision::Open) {
                            continue;
                        }
                        if !candidate.preserves_buildable_after_status(&status.presence) {
                            candidate = CandidateState::from_presence(status.presence);
                        }
                        if matches!(heartbeat_mode, HeartbeatMode::Withheld)
                            && evidence.is_retained() {
                            let _ = self.coordinator.update_calibration(
                                &binding,
                                between_songs_snapshot(
                                    &track,
                                    &song_counts,
                                    &evidence,
                                    candidate,
                                    &self.tracks,
                                    &collection_classes,
                                ),
                            );
                            }
                        }
                    Ok(Frame::CalibrationScheduleCommitDeferred { deferred })
                        if deferred.run == run
                            && deferred.schedule_revision == schedule_revision
                            && matches!(commit_phase, ScheduleCommitPhase::Sent { .. }) => {
                        // The preparation status is the retry authority.  A
                        // retryable refusal returns to its explicit ready
                        // phase; no backend stopwatch recreates that phase.
                        apply_commit_deferral(
                            &mut commit_phase,
                            &mut device_readiness,
                        );
                        heartbeat_mode = HeartbeatMode::Sending {
                            schedule_revision,
                            next_sequence: 0,
                        };
                    }
                    Ok(Frame::BenchError {
                        source: protocol::BenchErrorSource::Calibration,
                        detail,
                    }) => {
                        // Uncorrelated diagnostics are observable to the
                        // browser but cannot terminate an exact guided run.
                        // Firmware uses CalibrationRunFailed for that, so an
                        // old actor's delayed BenchError cannot kill a newer
                        // revision with the same device id.
                        tracing::warn!(%detail, "ignoring uncorrelated calibration diagnostic");
                    }
                    Ok(Frame::CalibrationRunFailed { failure })
                        if failure.run == run && failure.schedule_revision == schedule_revision => {
                        break SessionExit::DependencyFailed(failure.detail);
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        break SessionExit::DependencyFailed(
                            "calibration device acknowledgements were lost".into(),
                        );
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break SessionExit::DependencyFailed(
                            "the bound calibration device connection ended".into(),
                        );
                    }
                },
                _ = upload_ack_timeout.tick(),
                    if !matches!(upload, UploadPhase::Complete)
                        && matches!(terminal, TerminalDecision::Open) => {
                    match upload_timeout_action(&upload, chunk_ack_retries) {
                        UploadTimeoutAction::FailBegin => {
                            // Begin is not idempotent on the device: retrying it
                            // could replace a transaction whose ACK alone was
                            // lost. Fail explicitly and require a fresh run.
                            break SessionExit::DependencyFailed(
                                "calibration Begin acknowledgement timed out; restart before uploading a schedule".into(),
                            );
                        }
                        UploadTimeoutAction::FailChunk => {
                            break SessionExit::DependencyFailed(
                                "calibration Chunk acknowledgement timed out after exact idempotent retries".into(),
                            );
                        }
                        UploadTimeoutAction::RetryChunk => {
                            // Exact duplicate chunks are idempotent and ACKed by
                            // firmware, so only this phase may retry in place.
                            if let Err(error) = retry_upload_chunk(
                                &self.registry, &device, run, schedule_revision, &track, &upload,
                            ) {
                                break delivery_failure(error);
                            }
                            chunk_ack_retries = chunk_ack_retries.saturating_add(1);
                        }
                        UploadTimeoutAction::None => {}
                    }
                }
                _ = tokio::time::sleep_until(schedule_acceptance_deadline),
                    if matches!(commit_phase, ScheduleCommitPhase::Sent { .. })
                        && matches!(terminal, TerminalDecision::Open) => {
                    debug_assert!(commit_phase.acceptance_timed_out(tokio::time::Instant::now()));
                    // Commit is terminal and not idempotent. A lost acceptance
                    // cannot safely be reconstructed by the host, especially
                    // once its exact three-second playback anchor has passed.
                    break SessionExit::DependencyFailed(
                        "calibration schedule acceptance timed out after Commit; restart before playback".into(),
                    );
                }
                _ = tokio::time::sleep_until(terminal_deadline),
                    if !matches!(terminal, TerminalDecision::Open) => {
                    let operation = match terminal {
                        TerminalDecision::AwaitingInterruption { .. } => "Exit interruption",
                        TerminalDecision::AwaitingSave { .. } => "Save",
                        TerminalDecision::AwaitingDiscard { .. } => "Discard",
                        TerminalDecision::Open => unreachable!(),
                    };
                    break SessionExit::DependencyFailed(format!(
                        "calibration {operation} acknowledgement timed out; reconnect before making another durable decision"
                    ));
                }
                command = actions.recv() => {
                    let Some(command) = command else {
                        break SessionExit::TaskFailed("calibration actor action channel closed".into());
                    };
                    // The browser projection is not transaction authority.
                    // Re-check the actor-owned boundary so a racing or forged
                    // Continue cannot replace an upload that is still being
                    // prepared, acknowledged, or played.
                    if !run_action_is_authorized(
                        &command,
                        &evidence,
                        &song_counts,
                        candidate,
                        &terminal,
                    ) {
                        continue;
                    }
                    if matches!(&command, RunAction::Exit) {
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            GuidedCalibrationSnapshot::Exiting {
                                detail: if matches!(exit_route(&commit_phase, &evidence), ExitRoute::DiscardNow) {
                                    "Discarding calibration on the device…".into()
                                } else {
                                    "Stopping the accepted song before discarding calibration…".into()
                                },
                            },
                        );
                        gate_state = RunGate::Interrupted;
                        heartbeat_mode = HeartbeatMode::Withheld;
                        if let PlaybackState::Playing(opened) = &playback {
                            opened.pause();
                        }
                        if matches!(exit_route(&commit_phase, &evidence), ExitRoute::DiscardNow) {
                            playback = PlaybackState::Dormant;
                            if let Err(error) = self.registry.send_bound_control(
                                &device,
                                Frame::CalibrationDiscard { run },
                            ) {
                                break delivery_failure(error);
                            }
                            terminal = TerminalDecision::AwaitingDiscard {
                                deadline: tokio::time::Instant::now() + INTERRUPT_OR_DISCARD_DECISION_TIMEOUT,
                            };
                        } else {
                            if let Err(error) = request_device_interruption(
                                &self.registry,
                                &device,
                                run,
                                schedule_revision,
                                &mut device_interruption,
                            ) {
                                break delivery_failure(error);
                            }
                            terminal = TerminalDecision::AwaitingInterruption {
                                deadline: tokio::time::Instant::now() + INTERRUPT_OR_DISCARD_DECISION_TIMEOUT,
                            };
                        }
                        continue;
                    }
                    let save_requested = matches!(&command, RunAction::Save);
                    let frame = match command {
                        RunAction::Continue => {
                            let Some(next_revision) = next_schedule_revision(schedule_revision) else {
                                break SessionExit::TaskFailed("calibration schedule revision exhausted".into());
                            };
                            schedule_revision = next_revision;
                            if let Err(error) = self.registry.send_bound_control(
                                &device,
                                Frame::CalibrationContinue { run },
                            ) {
                                break delivery_failure(error);
                            }
                            upload = match begin_upload(
                                &self.registry, &device, run, schedule_revision, &track,
                            ) {
                                Ok(upload) => upload,
                                Err(error) => break delivery_failure(error),
                            };
                            // The upload timer is dormant between songs and its
                            // old deadline may be far in the past. Revision 2+
                            // gets a full Begin acknowledgement window.
                            upload_ack_timeout.reset();
                            commit_phase = ScheduleCommitPhase::AwaitingDeviceReadiness;
                            // A browser may have returned while the preceding
                            // committed song was waiting for its device-owned
                            // interruption edge. The new revision is a fresh
                            // pre-Commit transaction and may follow the latest
                            // visibility request again.
                            gate_state = effective_gate(*gate.borrow(), &commit_phase);
                            device_readiness = DeviceCommitReadiness::AwaitingFreshStatus;
                            evidence = EvidenceState::Fresh;
                            prepared_playback = prepare_playback(playback_collection.clone(), &track);
                            heartbeat_mode = HeartbeatMode::Sending {
                                schedule_revision,
                                next_sequence: 0,
                            };
                            device_interruption = DeviceInterruption::Available;
                            playback = PlaybackState::Dormant;
                            continue;
                        }
                        RunAction::SelectNextTrack(next_track) => {
                            // A live song owns its authored identity until its
                            // result/interruption is projected. Retargeting is
                            // deliberately restricted to that boundary.
                            if !evidence.is_retained()
                                || !matches!(heartbeat_mode, HeartbeatMode::Withheld) {
                                continue;
                            }
                            track = next_track;
                            *self.selected_track_id.lock().unwrap() = Some(track.id.0.clone());
                            let _ = self.coordinator.update_calibration(
                                &binding,
                                between_songs_snapshot(
                                    &track,
                                    &song_counts,
                                    &evidence,
                                    candidate,
                                    &self.tracks,
                                    &collection_classes,
                                ),
                            );
                            continue;
                        }
                        RunAction::Save => Frame::CalibrationSave { run },
                        RunAction::Discard => Frame::CalibrationDiscard { run },
                        RunAction::Exit => unreachable!("Exit is handled before song-end decisions"),
                    };
                    if save_requested {
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            GuidedCalibrationSnapshot::Finalizing {
                                detail: "Building, validating, and saving calibration on the wristband…".into(),
                            },
                        );
                    }
                    if let Err(error) = self.registry.send_bound_control(&device, frame) {
                        break delivery_failure(error);
                    }
                    terminal = if save_requested {
                        TerminalDecision::AwaitingSave {
                            deadline: tokio::time::Instant::now() + SAVE_DECISION_TIMEOUT,
                        }
                    } else {
                        TerminalDecision::AwaitingDiscard {
                            deadline: tokio::time::Instant::now() + INTERRUPT_OR_DISCARD_DECISION_TIMEOUT,
                        }
                    };
                }
                _ = &mut playback_alarm,
                    if matches!(gate_state, RunGate::Running)
                        && matches!(playback, PlaybackState::Armed { .. }) => {
                    let PlaybackState::Armed { playback: opened, audible_anchor } = core::mem::replace(&mut playback, PlaybackState::Dormant) else {
                        break SessionExit::TaskFailed("calibration playback deadline had no armed output".into());
                    };
                    opened.play();
                    playback = PlaybackState::Playing(opened);
                    let _ = self.coordinator.update_calibration(
                        &binding,
                        playing_snapshot(
                            &track,
                            CalibrationPlayhead {
                                position_ms: 0,
                                observed_at: audible_anchor,
                            },
                            &collection_classes,
                        ),
                    );
                }
            }
        };

        if matches!(outcome, SessionExit::Completed) {
            let selected_track_id = self.selected_track_id.lock().unwrap().clone();
            let _ = self
                .coordinator
                .update_calibration(&binding, setup_snapshot(&self.tracks, selected_track_id));
        }
        if matches!(
            outcome,
            SessionExit::DependencyFailed(_) | SessionExit::TaskFailed(_)
        ) {
            let detail = match &outcome {
                SessionExit::DependencyFailed(detail) | SessionExit::TaskFailed(detail) => {
                    detail.clone()
                }
                SessionExit::Completed | SessionExit::OperatorStopped => unreachable!(),
            };
            let _ = self.coordinator.update_calibration(
                &binding,
                GuidedCalibrationSnapshot::TechnicalFailure { detail },
            );
        }
        let mut active = self.active.lock().unwrap();
        if active
            .as_ref()
            .is_some_and(|active| active.binding == binding)
        {
            *active = None;
        }
        drop(active);
        lease.finish(outcome);
    }

    fn send_action(
        &self,
        binding: &GuidedSessionBinding,
        action: RunAction,
    ) -> Result<(), CoordinatorError> {
        let active = self.active.lock().unwrap();
        let actions = &active
            .as_ref()
            .filter(|active| &active.binding == binding)
            .ok_or(CoordinatorError::LeaseMismatch)?
            .actions;
        enqueue_nonterminal_run_action(actions, action)
    }

    fn set_gate(
        &self,
        binding: &GuidedSessionBinding,
        requested: RunGate,
    ) -> Result<(), CoordinatorError> {
        let active = self.active.lock().unwrap();
        let gate = &active
            .as_ref()
            .filter(|active| &active.binding == binding)
            .ok_or(CoordinatorError::LeaseMismatch)?
            .gate;
        gate.send(requested)
            .map_err(|_| CoordinatorError::AdapterTaskUnavailable)
    }

    fn request_exit(&self, binding: &GuidedSessionBinding) -> Result<(), CoordinatorError> {
        let active = self.active.lock().unwrap();
        let active = active
            .as_ref()
            .filter(|active| &active.binding == binding)
            .ok_or(CoordinatorError::LeaseMismatch)?;
        enqueue_run_action(&active.actions, RunAction::Exit)?;
        active
            .gate
            .send(RunGate::Interrupted)
            .map_err(|_| CoordinatorError::AdapterTaskUnavailable)
    }

    fn select_track(
        &self,
        track_id: String,
        request: &GuidedIntentRequest,
    ) -> Result<(), CoordinatorError> {
        if !self.tracks.iter().any(|track| track.id.0 == track_id) {
            return Err(CoordinatorError::UnknownCalibrationTrack);
        }
        // The coordinator revalidates authority inside `publish_calibration`.
        // Hold selection ownership across that check so a request which became
        // stale after the outer intent check cannot silently change the track
        // a later Start consumes while its browser projection was rejected.
        let mut selected_track_id = self.selected_track_id.lock().unwrap();
        self.coordinator.publish_calibration(
            request.authority,
            setup_snapshot(&self.tracks, Some(track_id.clone())),
        )?;
        *selected_track_id = Some(track_id);
        Ok(())
    }

    fn select_next_track(
        &self,
        binding: &GuidedSessionBinding,
        track_id: String,
    ) -> Result<(), CoordinatorError> {
        let track = self
            .tracks
            .iter()
            .find(|track| track.id.0 == track_id)
            .cloned()
            .ok_or(CoordinatorError::UnknownCalibrationTrack)?;
        self.require_between_songs(binding)?;
        self.send_action(binding, RunAction::SelectNextTrack(track))
    }

    fn require_between_songs(
        &self,
        binding: &GuidedSessionBinding,
    ) -> Result<(), CoordinatorError> {
        let snapshot = self.coordinator.snapshot();
        if snapshot.active() == Some(binding)
            && matches!(
                snapshot.calibration(),
                Some(GuidedCalibrationSnapshot::BetweenSongs { .. })
            )
        {
            Ok(())
        } else {
            Err(CoordinatorError::ActionUnavailable)
        }
    }

    fn send_between_songs_action(
        &self,
        binding: &GuidedSessionBinding,
        action: RunAction,
    ) -> Result<(), CoordinatorError> {
        self.require_between_songs(binding)?;
        self.send_action(binding, action)
    }

    fn exit_inactive(&self) -> Result<(), CoordinatorError> {
        let selected_track_id = self.selected_track_id.lock().unwrap().clone();
        self.coordinator
            .update_idle_calibration(setup_snapshot(&self.tracks, selected_track_id))?;
        Ok(())
    }
}

impl GuidedModeAdapter for CalibrationModeAdapter {
    fn pause_for_no_visible_views(&self, session: &GuidedSessionBinding) {
        let active = self.active.lock().unwrap();
        if let Some(active) = active.as_ref().filter(|active| &active.binding == session) {
            let _ = active.gate.send(RunGate::Interrupted);
        }
    }

    fn resume_for_visible_views(&self, session: &GuidedSessionBinding) {
        let active = self.active.lock().unwrap();
        if let Some(active) = active.as_ref().filter(|active| &active.binding == session) {
            let _ = active.gate.send(RunGate::Running);
        }
    }

    fn handle_intent(
        &self,
        session: Option<&GuidedSessionBinding>,
        device: Option<DeviceConnectionIdentity>,
        request: GuidedIntentRequest,
    ) -> Result<(), CoordinatorError> {
        match request.action.clone() {
            GuidedSessionAction::SelectCalibrationTrack { track_id } if session.is_none() => {
                self.select_track(track_id, &request)
            }
            GuidedSessionAction::SelectCalibrationTrack { track_id }
                if session.is_some_and(|binding| binding.mode == GuidedMode::Calibration) =>
            {
                self.select_next_track(session.expect("checked calibration binding"), track_id)
            }
            GuidedSessionAction::StartCalibration if session.is_none() => {
                self.start(device.ok_or(CoordinatorError::DeviceRequired)?, &request)
            }
            GuidedSessionAction::PauseCalibration => self.set_gate(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunGate::Interrupted,
            ),
            GuidedSessionAction::ResumeCalibration => self.set_gate(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunGate::Running,
            ),
            GuidedSessionAction::DiscardCalibration => self.send_between_songs_action(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunAction::Discard,
            ),
            GuidedSessionAction::ExitCalibration => match session {
                Some(binding) if binding.mode == GuidedMode::Calibration => {
                    self.request_exit(binding)
                }
                Some(_) => Err(CoordinatorError::LeaseMismatch),
                None => self.exit_inactive(),
            },
            GuidedSessionAction::SaveCalibration => self.send_between_songs_action(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunAction::Save,
            ),
            GuidedSessionAction::ContinueCalibration => self.send_between_songs_action(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunAction::Continue,
            ),
            GuidedSessionAction::SelectCalibrationTrack { .. }
            | GuidedSessionAction::StartCalibration => Ok(()),
        }
    }
}

fn run_key(binding: &GuidedSessionBinding) -> Result<CalibrationRunKey, CoordinatorError> {
    let run_id = u32::try_from(binding.run_revision.get())
        .ok()
        .and_then(CalibrationRunId::new)
        .ok_or(CoordinatorError::RevisionExhausted)?;
    let session_id = CalibrationSessionId::new(binding.session_id.get())
        .ok_or(CoordinatorError::RevisionExhausted)?;
    Ok(CalibrationRunKey { session_id, run_id })
}

fn guided_track(
    track: &crate::collect::beatmap::CalibrationTrack,
) -> protocol::GuidedCalibrationTrack {
    protocol::GuidedCalibrationTrack {
        id: track.id.0.clone(),
        title: track.title.clone(),
        beats_per_minute: track.beats_per_minute,
        duration_ms: u64::from(track.duration_ms),
        cue_count: u32::try_from(track.cue_count)
            .expect("calibration cue count fits the wire type"),
        content_identity: track.content_identity.clone(),
        cue_shortfall: u32::try_from(track.cue_shortfall)
            .expect("calibration cue shortfall fits the wire type"),
    }
}

fn setup_snapshot(
    tracks: &[crate::collect::beatmap::CalibrationTrack],
    selected_track_id: Option<String>,
) -> GuidedCalibrationSnapshot {
    GuidedCalibrationSnapshot::Setup {
        tracks: tracks.iter().map(guided_track).collect(),
        selected_track_id,
    }
}

fn candidate_presence_matches_track(
    presence: &protocol::CalibrationCandidatePresence,
    track: &crate::collect::beatmap::CalibrationTrack,
) -> bool {
    match presence {
        protocol::CalibrationCandidatePresence::Absent => true,
        protocol::CalibrationCandidatePresence::Present {
            content_identity,
            total_count,
            ..
        } => {
            content_identity == &track.content_identity
                && *total_count == track.entries.len() as u32
        }
    }
}

fn next_schedule_revision(
    revision: CalibrationScheduleRevision,
) -> Option<CalibrationScheduleRevision> {
    revision
        .get()
        .checked_add(1)
        .and_then(CalibrationScheduleRevision::new)
}

fn delivery_failure(error: ControlDeliveryError) -> SessionExit {
    SessionExit::DependencyFailed(error.calibration_message().into())
}

fn playback_deadline(
    host_anchor_milliseconds: Option<i64>,
    output_latency: protocol::DurationMilliseconds,
    now_wall_milliseconds: i64,
    now: tokio::time::Instant,
) -> Result<tokio::time::Instant, &'static str> {
    let host_anchor_milliseconds = host_anchor_milliseconds.ok_or(
        "calibration playback is withheld until production clock probes map the device anchor",
    )?;
    let playback_start_milliseconds = host_anchor_milliseconds
        .checked_sub(i64::from(output_latency.get()))
        .ok_or("the output-latency-compensated calibration anchor is already in the past")?;
    let delay_milliseconds = playback_start_milliseconds
        .checked_sub(now_wall_milliseconds)
        .ok_or("the output-latency-compensated calibration anchor is already in the past")?;
    let delay_milliseconds = u64::try_from(delay_milliseconds)
        .map_err(|_| "the output-latency-compensated calibration anchor is already in the past")?;
    if delay_milliseconds == 0 {
        return Err("the output-latency-compensated calibration anchor is already in the past");
    }
    Ok(now + std::time::Duration::from_millis(delay_milliseconds))
}

fn begin_upload(
    registry: &Registry,
    device: &DeviceConnectionIdentity,
    run: CalibrationRunKey,
    revision: CalibrationScheduleRevision,
    track: &crate::collect::beatmap::CalibrationTrack,
) -> Result<UploadPhase, ControlDeliveryError> {
    let total_count = u32::try_from(track.entries.len()).unwrap_or(u32::MAX);
    registry.send_bound_control(
        device,
        Frame::CalibrationScheduleBegin {
            run,
            schedule_revision: revision,
            content_identity: track.content_identity.clone(),
            total_count,
        },
    )?;
    Ok(UploadPhase::AwaitingBeginAcknowledgement)
}

/// Apply one exact acknowledgement and send the one operation it authorizes.
/// Any wrong/duplicate/out-of-order acknowledgement is deliberately ignored:
/// only the currently awaited operation can advance this connection-bound
/// transaction.
fn advance_upload(
    registry: &Registry,
    device: &DeviceConnectionIdentity,
    run: CalibrationRunKey,
    revision: CalibrationScheduleRevision,
    track: &crate::collect::beatmap::CalibrationTrack,
    upload: &mut UploadPhase,
    acknowledged_operation: protocol::CalibrationScheduleUploadOperationAcknowledgement,
) -> Result<(), ControlDeliveryError> {
    if expected_upload_fingerprint(run, revision, track, upload) != Some(acknowledged_operation) {
        return Ok(());
    }
    let next_first_entry = match upload {
        UploadPhase::AwaitingBeginAcknowledgement => 0,
        UploadPhase::AwaitingChunkAcknowledgement { first_entry } => {
            first_entry.saturating_add(OPERATIONAL_SCHEDULE_UPLOAD_ENTRIES as u32)
        }
        UploadPhase::Complete => return Ok(()),
    };
    if next_first_entry as usize >= track.entries.len() {
        *upload = UploadPhase::Complete;
        return Ok(());
    }
    let end =
        (next_first_entry as usize + OPERATIONAL_SCHEDULE_UPLOAD_ENTRIES).min(track.entries.len());
    send_schedule_chunk(
        registry,
        device,
        run,
        revision,
        track,
        next_first_entry,
        end,
    )?;
    *upload = UploadPhase::AwaitingChunkAcknowledgement {
        first_entry: next_first_entry,
    };
    Ok(())
}

fn expected_upload_fingerprint(
    run: CalibrationRunKey,
    revision: CalibrationScheduleRevision,
    track: &crate::collect::beatmap::CalibrationTrack,
    upload: &UploadPhase,
) -> Option<protocol::CalibrationScheduleUploadOperationAcknowledgement> {
    match upload {
        UploadPhase::AwaitingBeginAcknowledgement => Some(
            protocol::CalibrationScheduleUploadOperationAcknowledgement::Begin {
                operation_fingerprint: protocol::calibration_schedule_begin_fingerprint(
                    run,
                    revision,
                    &track.content_identity,
                    track.entries.len() as u32,
                ),
            },
        ),
        UploadPhase::AwaitingChunkAcknowledgement { first_entry } => {
            let start = *first_entry as usize;
            let end = (start + OPERATIONAL_SCHEDULE_UPLOAD_ENTRIES).min(track.entries.len());
            let entries = track.entries.get(start..end)?;
            Some(
                protocol::CalibrationScheduleUploadOperationAcknowledgement::Chunk {
                    first_entry: *first_entry,
                    operation_fingerprint: protocol::calibration_schedule_chunk_fingerprint(
                        run,
                        revision,
                        &track.content_identity,
                        track.entries.len() as u32,
                        *first_entry,
                        entries,
                    ),
                },
            )
        }
        UploadPhase::Complete => None,
    }
}

fn send_schedule_chunk(
    registry: &Registry,
    device: &DeviceConnectionIdentity,
    run: CalibrationRunKey,
    revision: CalibrationScheduleRevision,
    track: &crate::collect::beatmap::CalibrationTrack,
    first_entry: u32,
    end: usize,
) -> Result<(), ControlDeliveryError> {
    registry.send_bound_control(
        device,
        Frame::CalibrationScheduleChunk {
            run,
            schedule_revision: revision,
            content_identity: track.content_identity.clone(),
            total_count: track.entries.len() as u32,
            first_entry,
            entries: track.entries[first_entry as usize..end].to_vec(),
        },
    )
}

fn retry_upload_chunk(
    registry: &Registry,
    device: &DeviceConnectionIdentity,
    run: CalibrationRunKey,
    revision: CalibrationScheduleRevision,
    track: &crate::collect::beatmap::CalibrationTrack,
    upload: &UploadPhase,
) -> Result<(), ControlDeliveryError> {
    let UploadPhase::AwaitingChunkAcknowledgement { first_entry } = upload else {
        return Ok(());
    };
    let end =
        (*first_entry as usize + OPERATIONAL_SCHEDULE_UPLOAD_ENTRIES).min(track.entries.len());
    send_schedule_chunk(registry, device, run, revision, track, *first_entry, end)
}

fn send_schedule_commit(
    registry: &Registry,
    device: &DeviceConnectionIdentity,
    run: CalibrationRunKey,
    revision: CalibrationScheduleRevision,
    track: &crate::collect::beatmap::CalibrationTrack,
) -> Result<(), ControlDeliveryError> {
    registry.send_bound_control(
        device,
        Frame::CalibrationScheduleCommit {
            run,
            schedule_revision: revision,
            content_identity: track.content_identity.clone(),
            total_count: track.entries.len() as u32,
        },
    )
}

#[derive(Clone, Copy)]
struct CalibrationPlayhead {
    position_ms: u64,
    observed_at: protocol::UnixMilliseconds,
}

fn playing_snapshot(
    track: &crate::collect::beatmap::CalibrationTrack,
    playhead: CalibrationPlayhead,
    collection_classes: &[protocol::CollectionClass],
) -> GuidedCalibrationSnapshot {
    let lanes = calibration_lanes(collection_classes);
    let cues = track
        .entries
        .iter()
        .map(|entry| protocol::GuidedCalibrationCue {
            visual_lane: entry.gesture.index(),
            at: u64::from(entry.track_offset.get()),
            hold: u64::from(entry.hold.get()),
            thumb_variant: match entry.modifier {
                protocol::CalibrationModifier::ThumbUp => protocol::GuidedThumbVariant::Up,
                protocol::CalibrationModifier::ThumbDown => protocol::GuidedThumbVariant::Down,
            },
        })
        .collect();
    GuidedCalibrationSnapshot::Playing {
        track: protocol::GuidedCalibrationTrack {
            id: track.id.0.clone(),
            title: track.title.clone(),
            beats_per_minute: track.beats_per_minute,
            duration_ms: u64::from(track.duration_ms),
            cue_count: track.cue_count as u32,
            content_identity: track.content_identity.clone(),
            cue_shortfall: track.cue_shortfall as u32,
        },
        lanes,
        cues,
        position_ms: playhead.position_ms,
        position_observed_at_unix_ms: playhead.observed_at,
        valid_reps: 0,
        invalid_reps: 0,
        paused_reason: None,
        counts: Vec::new(),
    }
}

fn calibration_lanes(
    collection_classes: &[protocol::CollectionClass],
) -> Vec<protocol::GuidedCalibrationLane> {
    protocol::CalibrationGesture::ALL
        .into_iter()
        .map(|gesture| {
            let id = calibration_gesture_id(gesture);
            let descriptor = collection_classes
                .iter()
                .find(|descriptor| descriptor.id.0 == id);
            protocol::GuidedCalibrationLane {
                visual_lane: gesture.index(),
                id: id.into(),
                label: descriptor.map_or_else(|| id.replace('_', " "), |value| value.label.clone()),
                color_name: descriptor.map_or_else(|| "gray".into(), |value| value.color.clone()),
                motion: descriptor.and_then(|value| value.motion.clone()),
            }
        })
        .collect()
}

fn calibration_gesture_id(gesture: protocol::CalibrationGesture) -> &'static str {
    match gesture {
        protocol::CalibrationGesture::WristPronation => "wrist_pronation",
        protocol::CalibrationGesture::WristSupination => "wrist_supination",
        protocol::CalibrationGesture::WristRadialDeviation => "wrist_radial_deviation",
        protocol::CalibrationGesture::WristUlnarDeviation => "wrist_ulnar_deviation",
        protocol::CalibrationGesture::ThumbExtension => "thumb_extension",
    }
}

fn calibration_count_label(
    count: &protocol::CalibrationClassCounts,
    collection_classes: &[protocol::CollectionClass],
) -> String {
    let id = calibration_gesture_id(count.gesture);
    let gesture = collection_classes
        .iter()
        .find(|descriptor| descriptor.id.0 == id)
        .map_or_else(
            || id.replace('_', " "),
            |descriptor| descriptor.label.clone(),
        );
    let variant = match count.modifier {
        protocol::CalibrationModifier::ThumbUp => "command",
        protocol::CalibrationModifier::ThumbDown => "no-op",
    };
    format!("{gesture}, {variant}")
}

fn preparing_snapshot_from_device(
    track: &crate::collect::beatmap::CalibrationTrack,
    phase: protocol::CalibrationPreparationPhase,
) -> GuidedCalibrationSnapshot {
    let (stage, elapsed_milliseconds, remaining_milliseconds) = match phase {
        protocol::CalibrationPreparationPhase::Settling {
            elapsed_milliseconds,
            remaining_milliseconds,
        } => (
            protocol::GuidedCalibrationPreparationStage::Stillness,
            elapsed_milliseconds,
            remaining_milliseconds,
        ),
        protocol::CalibrationPreparationPhase::EstimatingGains {
            elapsed_milliseconds,
            remaining_milliseconds,
        } => (
            protocol::GuidedCalibrationPreparationStage::GainEstimation,
            elapsed_milliseconds,
            remaining_milliseconds,
        ),
        protocol::CalibrationPreparationPhase::ReadyForSchedule => (
            protocol::GuidedCalibrationPreparationStage::ReadyForSchedule,
            0,
            0,
        ),
        protocol::CalibrationPreparationPhase::Failed { detail } => {
            return GuidedCalibrationSnapshot::TechnicalFailure { detail };
        }
    };
    GuidedCalibrationSnapshot::Preparing {
        track: guided_track(track),
        stage,
        elapsed_milliseconds,
        remaining_milliseconds,
    }
}

fn between_songs_snapshot(
    track: &crate::collect::beatmap::CalibrationTrack,
    counts: &[protocol::CalibrationClassCounts],
    evidence: &EvidenceState,
    candidate: CandidateState,
    tracks: &[crate::collect::beatmap::CalibrationTrack],
    collection_classes: &[protocol::CollectionClass],
) -> GuidedCalibrationSnapshot {
    let valid_reps = counts.iter().map(|count| count.accepted_count).sum();
    let invalid_reps = counts.iter().map(|count| count.rejected_count).sum();
    let deficits = counts
        .iter()
        .filter(|count| count.deficit_count > 0)
        .map(|count| {
            format!(
                "{}: {} short",
                calibration_count_label(count, collection_classes),
                count.deficit_count,
            )
        })
        .collect();
    GuidedCalibrationSnapshot::BetweenSongs {
        track_title: evidence
            .completed_track_title()
            .expect("BetweenSongs requires retained evidence")
            .to_owned(),
        tracks: tracks.iter().map(guided_track).collect(),
        selected_track_id: Some(track.id.0.clone()),
        candidate_available: candidate.permits_save(),
        continue_available: evidence.is_retained() && counts_have_deficits(counts),
        valid_reps,
        invalid_reps,
        deficits,
    }
}

fn unix_milliseconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_millis().min(i64::MAX as u128) as i64
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{DeviceConfig, DeviceProvenance, DeviceTransport, FirmwareBuild};
    use std::path::PathBuf;

    #[test]
    fn between_songs_is_one_actor_owned_transport_and_action_boundary() {
        let revision = CalibrationScheduleRevision::new(1).unwrap();
        let mut evidence = EvidenceState::Fresh;
        let mut heartbeat = HeartbeatMode::Sending {
            schedule_revision: revision,
            next_sequence: 7,
        };
        let mut playback = PlaybackState::Dormant;
        let absent = CandidateState::Absent;
        let buildable = CandidateState::Buildable;
        let valid = CandidateState::Present(protocol::CalibrationCandidateValidity {
            model_numerically_valid: true,
            record_crc_valid: true,
        });
        let with_deficit = [protocol::CalibrationClassCounts {
            gesture: protocol::CalibrationGesture::WristPronation,
            modifier: protocol::CalibrationModifier::ThumbUp,
            accepted_count: 9,
            rejected_count: 0,
            target_count: 10,
            deficit_count: 1,
        }];

        assert!(!run_action_is_authorized(
            &RunAction::Continue,
            &evidence,
            &with_deficit,
            absent,
            &TerminalDecision::Open,
        ));
        assert!(!run_action_is_authorized(
            &RunAction::Save,
            &evidence,
            &[],
            valid,
            &TerminalDecision::Open,
        ));
        assert!(!run_action_is_authorized(
            &RunAction::Discard,
            &evidence,
            &[],
            absent,
            &TerminalDecision::Open,
        ));
        assert!(run_action_is_authorized(
            &RunAction::Exit,
            &evidence,
            &[],
            absent,
            &TerminalDecision::Open,
        ));

        enter_between_songs(
            &mut evidence,
            "Completed Track",
            &mut heartbeat,
            &mut playback,
        );

        assert!(matches!(evidence, EvidenceState::Retained { .. }));
        assert_eq!(evidence.completed_track_title(), Some("Completed Track"));
        assert!(matches!(heartbeat, HeartbeatMode::Withheld));
        assert!(matches!(playback, PlaybackState::Dormant));
        assert!(run_action_is_authorized(
            &RunAction::Continue,
            &evidence,
            &with_deficit,
            absent,
            &TerminalDecision::Open,
        ));
        assert!(run_action_is_authorized(
            &RunAction::SelectNextTrack(track()),
            &evidence,
            &with_deficit,
            absent,
            &TerminalDecision::Open,
        ));
        assert!(!run_action_is_authorized(
            &RunAction::Continue,
            &evidence,
            &[],
            absent,
            &TerminalDecision::Open,
        ));
        assert!(!run_action_is_authorized(
            &RunAction::SelectNextTrack(track()),
            &evidence,
            &[],
            absent,
            &TerminalDecision::Open,
        ));
        assert!(run_action_is_authorized(
            &RunAction::Save,
            &evidence,
            &[],
            buildable,
            &TerminalDecision::Open,
        ));
        assert!(run_action_is_authorized(
            &RunAction::Save,
            &evidence,
            &[],
            valid,
            &TerminalDecision::Open,
        ));
        assert!(run_action_is_authorized(
            &RunAction::Discard,
            &evidence,
            &[],
            absent,
            &TerminalDecision::Open,
        ));
        assert!(!run_action_is_authorized(
            &RunAction::Discard,
            &evidence,
            &[],
            valid,
            &TerminalDecision::AwaitingSave {
                deadline: tokio::time::Instant::now(),
            },
        ));
    }

    #[test]
    fn run_action_queue_is_bounded_and_distinguishes_full_from_closed() {
        let (actions, mut receiver) = mpsc::channel(2);
        enqueue_nonterminal_run_action(&actions, RunAction::Continue).unwrap();
        assert_eq!(
            enqueue_nonterminal_run_action(&actions, RunAction::Save),
            Err(CoordinatorError::AdapterTaskBusy)
        );
        enqueue_run_action(&actions, RunAction::Exit)
            .expect("the reserved terminal slot remains available");

        assert!(matches!(receiver.try_recv(), Ok(RunAction::Continue)));
        assert!(matches!(receiver.try_recv(), Ok(RunAction::Exit)));
        drop(receiver);
        assert_eq!(
            enqueue_run_action(&actions, RunAction::Discard),
            Err(CoordinatorError::AdapterTaskUnavailable)
        );
    }

    #[test]
    fn durable_terminal_decisions_require_their_exact_device_acknowledgement() {
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(8).unwrap(),
            run_id: CalibrationRunId::new(3).unwrap(),
        };
        let revision = CalibrationScheduleRevision::new(2).unwrap();
        let deadline = tokio::time::Instant::now() + INTERRUPT_OR_DISCARD_DECISION_TIMEOUT;
        let discarded = Frame::CalibrationCandidateStatus {
            candidate: protocol::CalibrationCandidateStatus {
                run,
                schedule_revision: revision,
                presence: protocol::CalibrationCandidatePresence::Absent,
            },
        };
        let activated = Frame::CalibrationResidentActivated {
            activation: protocol::CalibrationResidentActivation {
                run,
                schedule_revision: revision,
                validity: protocol::CalibrationCandidateValidity {
                    model_numerically_valid: true,
                    record_crc_valid: true,
                },
                resident_sequence: 4,
            },
        };

        assert!(
            TerminalDecision::AwaitingDiscard { deadline }.acknowledges(&discarded, run, revision)
        );
        assert!(
            !TerminalDecision::AwaitingDiscard { deadline }.acknowledges(&activated, run, revision)
        );
        assert!(TerminalDecision::AwaitingSave { deadline }.acknowledges(&activated, run, revision));
        assert!(
            !TerminalDecision::AwaitingSave { deadline }.acknowledges(&discarded, run, revision)
        );
        let next_revision = CalibrationScheduleRevision::new(3).unwrap();
        assert!(
            !TerminalDecision::AwaitingDiscard { deadline }.acknowledges(
                &discarded,
                run,
                next_revision
            )
        );
    }

    #[test]
    fn song_result_buildability_survives_an_unrelated_absent_candidate_status() {
        let mut candidate = CandidateState::Buildable;
        let absent = protocol::CalibrationCandidatePresence::Absent;

        assert!(candidate.preserves_buildable_after_status(&absent));
        if !candidate.preserves_buildable_after_status(&absent) {
            candidate = CandidateState::from_presence(absent.clone());
        }
        assert!(candidate.permits_save());
        assert_eq!(SAVE_DECISION_TIMEOUT, Duration::from_secs(120));
        assert_eq!(
            INTERRUPT_OR_DISCARD_DECISION_TIMEOUT,
            Duration::from_secs(5)
        );
    }

    #[test]
    fn interrupt_supersedes_a_stale_resume_even_when_actions_are_saturated() {
        let (actions, receiver) = mpsc::channel(1);
        enqueue_run_action(&actions, RunAction::Continue).unwrap();
        let (gate, mut gate_receiver) = tokio::sync::watch::channel(RunGate::Interrupted);

        gate.send(RunGate::Running).unwrap();
        gate.send(RunGate::Interrupted).unwrap();

        assert_eq!(*gate_receiver.borrow_and_update(), RunGate::Interrupted);
        assert_eq!(receiver.len(), 1);
    }

    #[test]
    fn browser_interrupt_delivery_survives_action_queue_saturation() {
        let (actions, receiver) = mpsc::channel(1);
        enqueue_run_action(&actions, RunAction::Continue).unwrap();
        let (gate, mut gate_receiver) = tokio::sync::watch::channel(RunGate::Running);

        gate.send(RunGate::Interrupted).unwrap();

        assert!(gate_receiver.has_changed().unwrap());
        assert_eq!(*gate_receiver.borrow_and_update(), RunGate::Interrupted);
        assert_eq!(receiver.len(), 1);
    }

    #[test]
    fn hidden_preparation_keeps_its_lease_but_cannot_commit() {
        let awaiting = ScheduleCommitPhase::AwaitingDeviceReadiness;
        assert!(!gate_withholds_heartbeat(RunGate::Interrupted, &awaiting));
        assert!(!schedule_commit_is_authorized(
            RunGate::Interrupted,
            &UploadPhase::Complete,
            DeviceCommitReadiness::Ready,
            true,
            &awaiting,
        ));
        assert!(schedule_commit_is_authorized(
            RunGate::Running,
            &UploadPhase::Complete,
            DeviceCommitReadiness::Ready,
            true,
            &awaiting,
        ));
    }

    #[test]
    fn commit_waits_for_host_playback_preparation() {
        let awaiting = ScheduleCommitPhase::AwaitingDeviceReadiness;
        assert!(!schedule_commit_is_authorized(
            RunGate::Running,
            &UploadPhase::Complete,
            DeviceCommitReadiness::Ready,
            false,
            &awaiting,
        ));
    }

    #[test]
    fn deferred_commit_requires_a_fresh_device_readiness_observation() {
        let mut commit = ScheduleCommitPhase::sent_at(tokio::time::Instant::now());
        let mut device_readiness = DeviceCommitReadiness::Ready;

        apply_commit_deferral(&mut commit, &mut device_readiness);

        assert!(matches!(
            commit,
            ScheduleCommitPhase::AwaitingDeviceReadiness
        ));
        assert!(!schedule_commit_is_authorized(
            RunGate::Running,
            &UploadPhase::Complete,
            device_readiness,
            true,
            &commit,
        ));
    }

    #[test]
    fn interruption_after_commit_is_sticky_across_a_quick_browser_return() {
        let sent = ScheduleCommitPhase::sent_at(tokio::time::Instant::now());
        assert!(gate_withholds_heartbeat(RunGate::Interrupted, &sent));
        assert!(gate_withholds_heartbeat(
            RunGate::Interrupted,
            &ScheduleCommitPhase::Accepted,
        ));
        assert_eq!(
            effective_gate(RunGate::Running, &sent),
            RunGate::Interrupted
        );
        assert_eq!(
            effective_gate(RunGate::Running, &ScheduleCommitPhase::Accepted),
            RunGate::Interrupted
        );
        assert_eq!(
            effective_gate(
                RunGate::Running,
                &ScheduleCommitPhase::AwaitingDeviceReadiness
            ),
            RunGate::Running,
            "the next revision may use the latest browser presence again"
        );
    }

    #[test]
    fn collapsed_pause_then_resume_after_commit_still_withholds_heartbeat() {
        let (gate, mut observed_gate) = tokio::sync::watch::channel(RunGate::Running);
        let committed = ScheduleCommitPhase::sent_at(tokio::time::Instant::now());

        gate.send(RunGate::Interrupted).unwrap();
        gate.send(RunGate::Running).unwrap();

        // A watch receiver can observe only the latest Running value. The
        // actor must nevertheless convert it through the committed boundary
        // and execute the same interruption effects as a visible pause.
        let effective = effective_gate(*observed_gate.borrow_and_update(), &committed);
        assert_eq!(effective, RunGate::Interrupted);
        assert!(gate_withholds_heartbeat(effective, &committed));
    }

    fn register_device(registry: &Registry) -> crate::registry::DeviceHandle {
        registry.register(
            "opal-test".into(),
            "Test device".into(),
            DeviceTransport::Serial,
            DeviceConfig {
                gestures: 0,
                keymap: Vec::new(),
                wifi_ssid: None,
                sensitivity: String::new(),
                sensitivity_levels: Vec::new(),
                tau: 0.0,
                needed: 0,
            },
            DeviceProvenance {
                firmware: FirmwareBuild {
                    crate_version: String::new(),
                    git_commit: String::new(),
                    working_tree_modified: false,
                    built_at: String::new(),
                },
                analog_front_ends: Vec::new(),
            },
        )
    }

    fn track() -> crate::collect::beatmap::CalibrationTrack {
        crate::collect::beatmap::CalibrationTrack {
            id: protocol::TrackId("calibration-track".into()),
            title: "Calibration Track".into(),
            beats_per_minute: 120,
            duration_ms: 60_000,
            cue_count: 1,
            content_identity: "test-content".into(),
            cue_shortfall: 129,
            entries: vec![protocol::CalibrationScheduleEntry {
                cue_id: protocol::CalibrationCueId::new(1).unwrap(),
                gesture: protocol::CalibrationGesture::WristPronation,
                modifier: protocol::CalibrationModifier::ThumbUp,
                track_offset: protocol::TrackMilliseconds::new(0),
                hold: protocol::DurationMilliseconds::new(1_500),
            }],
        }
    }

    fn attach_test_collection(
        adapter: &CalibrationModeAdapter,
        registry: Arc<Registry>,
        coordinator: GuidedSessionCoordinator,
    ) {
        let directory = std::env::temp_dir().join(format!(
            "calibration-playback-fixture-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let paths = crate::collect::beatmap::CatalogPaths {
            config_path: directory.join("collection.json"),
            tracks_root: directory.join("tracks"),
        };
        std::fs::create_dir_all(&paths.tracks_root).expect("fixture track root is writable");
        std::fs::write(
            &paths.config_path,
            r#"{"subjects":["subject"],
                "collection_classes":[{"id":"a","label":"A","color":"blue"}],
                "activities":[{"id":"seated","label":"Seated"}],
                "sweat_levels":[{"id":"dry","label":"Dry"}]}"#,
        )
        .expect("fixture catalog is writable");
        let catalog = paths.load().expect("fixture catalog loads");
        let collection = crate::collect::manager::CollectionManager::new(
            catalog,
            paths,
            PathBuf::from("/dev/null"),
            crate::collect::video::CameraSettings::default(),
            registry,
            directory.clone(),
            crate::collect::provenance::ProvenanceStore::load(directory.join("provenance.cbor")),
            crate::collect::audio::AudioOutput::Silent,
            coordinator,
        );
        adapter.attach_collection(collection);
    }

    async fn receive_control(device: &mut crate::registry::DeviceHandle) -> Frame {
        tokio::time::timeout(std::time::Duration::from_secs(1), device.control_rx.recv())
            .await
            .expect("calibration control delivery timed out")
            .expect("calibration control channel closed")
    }

    fn attach_test_timing(adapter: &CalibrationModeAdapter, identity: &DeviceConnectionIdentity) {
        let timing = Arc::new(crate::timing::TimingService::new());
        let epoch = timing.begin_epoch(&identity.device_id, identity.connection_token);
        let host_send_nanoseconds = crate::timing::host_monotonic_nanoseconds();
        let device_receive_microseconds = host_send_nanoseconds / 1_000;
        timing.record_clock_probe(
            &epoch,
            host_send_nanoseconds,
            device_receive_microseconds,
            device_receive_microseconds.saturating_add(100),
            host_send_nanoseconds.saturating_add(100_000),
        );
        adapter.attach_timing(timing);
    }

    async fn acknowledge_test_schedule(
        device: &mut crate::registry::DeviceHandle,
        run: CalibrationRunKey,
        revision: CalibrationScheduleRevision,
        track: &crate::collect::beatmap::CalibrationTrack,
    ) {
        loop {
            if let Frame::CalibrationScheduleBegin {
                run: received_run,
                schedule_revision,
                content_identity,
                total_count,
            } = receive_control(device).await
            {
                if received_run == run && schedule_revision == revision {
                    assert_eq!(content_identity, track.content_identity);
                    assert_eq!(total_count, track.entries.len() as u32);
                    break;
                }
            }
        }
        device
            .frames
            .send(Frame::CalibrationScheduleUploadAcknowledged {
                acknowledgement: protocol::CalibrationScheduleUploadAcknowledgement {
                    run,
                    schedule_revision: revision,
                    content_identity: track.content_identity.clone(),
                    total_count: track.entries.len() as u32,
                    operation: protocol::CalibrationScheduleUploadOperationAcknowledgement::Begin {
                        operation_fingerprint: protocol::calibration_schedule_begin_fingerprint(
                            run,
                            revision,
                            &track.content_identity,
                            track.entries.len() as u32,
                        ),
                    },
                },
            })
            .unwrap();

        let mut acknowledged_entries = 0usize;
        while acknowledged_entries < track.entries.len() {
            let Frame::CalibrationScheduleChunk {
                run: received_run,
                schedule_revision,
                content_identity,
                total_count,
                first_entry,
                entries,
            } = receive_control(device).await
            else {
                continue;
            };
            if received_run != run || schedule_revision != revision {
                continue;
            }
            assert_eq!(content_identity, track.content_identity);
            assert_eq!(total_count, track.entries.len() as u32);
            assert_eq!(first_entry as usize, acknowledged_entries);
            assert_eq!(
                entries,
                track.entries[acknowledged_entries..][..entries.len()]
            );
            device
                .frames
                .send(Frame::CalibrationScheduleUploadAcknowledged {
                    acknowledgement: protocol::CalibrationScheduleUploadAcknowledgement {
                        run,
                        schedule_revision: revision,
                        content_identity: track.content_identity.clone(),
                        total_count: track.entries.len() as u32,
                        operation:
                            protocol::CalibrationScheduleUploadOperationAcknowledgement::Chunk {
                                first_entry,
                                operation_fingerprint:
                                    protocol::calibration_schedule_chunk_fingerprint(
                                        run,
                                        revision,
                                        &track.content_identity,
                                        track.entries.len() as u32,
                                        first_entry,
                                        &entries,
                                    ),
                            },
                    },
                })
                .unwrap();
            acknowledged_entries += entries.len();
        }

        device
            .frames
            .send(Frame::CalibrationPreparationStatus {
                status: protocol::CalibrationPreparationStatus {
                    run,
                    schedule_revision: revision,
                    phase: protocol::CalibrationPreparationPhase::ReadyForSchedule,
                },
            })
            .unwrap();
        loop {
            if matches!(
                receive_control(device).await,
                Frame::CalibrationScheduleCommit {
                    run: committed_run,
                    schedule_revision,
                    ref content_identity,
                    total_count,
                } if committed_run == run
                    && schedule_revision == revision
                    && content_identity == &track.content_identity
                    && total_count == track.entries.len() as u32
            ) {
                break;
            }
        }

        let acknowledged_device_monotonic_microseconds =
            crate::timing::host_monotonic_nanoseconds() / 1_000;
        device
            .frames
            .send(Frame::CalibrationScheduleAccepted {
                accepted: protocol::CalibrationScheduleAccepted {
                    run,
                    schedule_revision: revision,
                    content_identity: track.content_identity.clone(),
                    acknowledged_device_monotonic_microseconds,
                    anchor_device_monotonic_microseconds: acknowledged_device_monotonic_microseconds
                        + 3_000_000,
                    acquisition_sample: 10_000,
                },
            })
            .unwrap();
    }

    #[test]
    fn exit_route_covers_every_actor_owned_schedule_boundary() {
        let fresh = EvidenceState::Fresh;
        let retained = EvidenceState::Retained {
            completed_track_title: "finished".into(),
        };
        assert_eq!(
            exit_route(&ScheduleCommitPhase::AwaitingDeviceReadiness, &fresh),
            ExitRoute::DiscardNow
        );
        assert_eq!(
            exit_route(
                &ScheduleCommitPhase::sent_at(tokio::time::Instant::now()),
                &fresh
            ),
            ExitRoute::InterruptThenDiscard
        );
        assert_eq!(
            exit_route(&ScheduleCommitPhase::Accepted, &fresh),
            ExitRoute::InterruptThenDiscard
        );
        assert_eq!(
            exit_route(&ScheduleCommitPhase::Accepted, &retained),
            ExitRoute::DiscardNow,
            "a song boundary is already a safe discard boundary"
        );
    }

    #[test]
    fn operator_interruption_is_an_explicit_exact_once_device_control() {
        let registry = Registry::new();
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(8).unwrap(),
            run_id: CalibrationRunId::new(3).unwrap(),
        };
        let revision = CalibrationScheduleRevision::new(2).unwrap();
        let mut state = DeviceInterruption::Available;

        request_device_interruption(&registry, &identity, run, revision, &mut state).unwrap();
        request_device_interruption(&registry, &identity, run, revision, &mut state).unwrap();

        assert_eq!(state, DeviceInterruption::Requested);
        assert!(matches!(
            device.control_rx.try_recv(),
            Ok(Frame::CalibrationInterrupt {
                run: received_run,
                schedule_revision: received_revision,
            }) if received_run == run && received_revision == revision
        ));
        assert!(device.control_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn exit_during_preparation_waits_for_exact_absent_ack_and_is_single_use() {
        let registry = Arc::new(Registry::new());
        let coordinator = GuidedSessionCoordinator::new();
        let adapter =
            CalibrationModeAdapter::new(registry.clone(), coordinator.clone(), vec![track()]);
        attach_test_collection(&adapter, registry.clone(), coordinator.clone());
        coordinator.set_mode_adapter(GuidedMode::Calibration, adapter.clone());
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();

        let setup = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: setup.action_authority,
                    action: GuidedSessionAction::SelectCalibrationTrack {
                        track_id: "calibration-track".into(),
                    },
                },
                None,
            )
            .unwrap();
        let selected = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: selected.action_authority,
                    action: GuidedSessionAction::StartCalibration,
                },
                Some(identity),
            )
            .unwrap();
        let exit_snapshot = coordinator.snapshot();
        let exit_request = GuidedIntentRequest {
            authority: exit_snapshot.action_authority,
            action: GuidedSessionAction::ExitCalibration,
        };
        coordinator
            .handle_intent_for_device(exit_request.clone(), None)
            .unwrap();
        assert!(matches!(
            coordinator.handle_intent_for_device(exit_request, None),
            Err(CoordinatorError::StaleActionPhase { .. })
        ));

        let discard = loop {
            let frame = receive_control(&mut device).await;
            if matches!(frame, Frame::CalibrationDiscard { .. }) {
                break frame;
            }
        };
        assert!(matches!(discard, Frame::CalibrationDiscard { .. }));
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Exiting { .. })
        ));

        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Setup { .. })
        ));
        assert!(coordinator.snapshot().active().is_none());
    }

    #[tokio::test]
    async fn save_finalizes_a_buildable_song_and_waits_for_resident_promotion() {
        let registry = Arc::new(Registry::new());
        let coordinator = GuidedSessionCoordinator::new();
        let adapter =
            CalibrationModeAdapter::new(registry.clone(), coordinator.clone(), vec![track()]);
        attach_test_collection(&adapter, registry.clone(), coordinator.clone());
        coordinator.set_mode_adapter(GuidedMode::Calibration, adapter.clone());
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();

        let setup = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: setup.action_authority,
                    action: GuidedSessionAction::SelectCalibrationTrack {
                        track_id: "calibration-track".into(),
                    },
                },
                None,
            )
            .unwrap();
        let selected = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: selected.action_authority,
                    action: GuidedSessionAction::StartCalibration,
                },
                Some(identity),
            )
            .unwrap();

        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let schedule_revision = CalibrationScheduleRevision::new(1).unwrap();
        device
            .frames
            .send(Frame::CalibrationSongResult {
                result: protocol::CalibrationSongResult {
                    run,
                    schedule_revision,
                    content_identity: "test-content".into(),
                    // Save remains available even with no deficits; the
                    // wristband owns the final queued checkpoint/fit work.
                    counts: vec![protocol::CalibrationClassCounts {
                        gesture: protocol::CalibrationGesture::WristPronation,
                        modifier: protocol::CalibrationModifier::ThumbUp,
                        accepted_count: 10,
                        rejected_count: 0,
                        target_count: 10,
                        deficit_count: 0,
                    }],
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: false,
                        record_crc_valid: false,
                    },
                },
            })
            .unwrap();
        // Firmware can still report its old absence before it receives Save.
        // That status must not withdraw the freshly earned Save authority.
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision,
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::BetweenSongs {
                candidate_available: true,
                continue_available: false,
                ..
            })
        ));

        let save_snapshot = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: save_snapshot.action_authority,
                    action: GuidedSessionAction::SaveCalibration,
                },
                None,
            )
            .unwrap();
        loop {
            if matches!(
                receive_control(&mut device).await,
                Frame::CalibrationSave { run: saved_run } if saved_run == run
            ) {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Finalizing { .. })
        ));

        // A normal status during finalization must not make the browser offer
        // Save/Continue/Discard again while the durable operation is pending.
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision,
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Finalizing { .. })
        ));

        device
            .frames
            .send(Frame::CalibrationResidentActivated {
                activation: protocol::CalibrationResidentActivation {
                    run,
                    schedule_revision,
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: true,
                        record_crc_valid: true,
                    },
                    resident_sequence: 1,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Setup { .. })
        ));
        assert!(coordinator.snapshot().active().is_none());
    }

    #[tokio::test]
    async fn two_song_actor_lifecycle_continues_then_saves_exact_revision_two() {
        let registry = Arc::new(Registry::new());
        let coordinator = GuidedSessionCoordinator::new();
        let first_track = track();
        let mut second_track = first_track.clone();
        second_track.id = protocol::TrackId("second-calibration-track".into());
        second_track.title = "Second Calibration Track".into();
        second_track.cue_count = 2;
        second_track.cue_shortfall = 128;
        second_track
            .entries
            .push(protocol::CalibrationScheduleEntry {
                cue_id: protocol::CalibrationCueId::new(2).unwrap(),
                gesture: protocol::CalibrationGesture::WristSupination,
                modifier: protocol::CalibrationModifier::ThumbDown,
                track_offset: protocol::TrackMilliseconds::new(2_000),
                hold: protocol::DurationMilliseconds::new(1_500),
            });
        let adapter = CalibrationModeAdapter::new(
            registry.clone(),
            coordinator.clone(),
            vec![first_track.clone(), second_track.clone()],
        );
        attach_test_collection(&adapter, registry.clone(), coordinator.clone());
        coordinator.set_mode_adapter(GuidedMode::Calibration, adapter.clone());
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();
        attach_test_timing(&adapter, &identity);

        let setup = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: setup.action_authority,
                    action: GuidedSessionAction::SelectCalibrationTrack {
                        track_id: first_track.id.0.clone(),
                    },
                },
                None,
            )
            .unwrap();
        let selected = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: selected.action_authority,
                    action: GuidedSessionAction::StartCalibration,
                },
                Some(identity),
            )
            .unwrap();

        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let first_revision = CalibrationScheduleRevision::new(1).unwrap();
        acknowledge_test_schedule(&mut device, run, first_revision, &first_track).await;

        let first_counts = vec![protocol::CalibrationClassCounts {
            gesture: protocol::CalibrationGesture::WristPronation,
            modifier: protocol::CalibrationModifier::ThumbUp,
            accepted_count: 9,
            rejected_count: 1,
            target_count: 10,
            deficit_count: 1,
        }];
        device
            .frames
            .send(Frame::CalibrationSongResult {
                result: protocol::CalibrationSongResult {
                    run,
                    schedule_revision: first_revision,
                    content_identity: first_track.content_identity.clone(),
                    counts: first_counts,
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: false,
                        record_crc_valid: false,
                    },
                },
            })
            .unwrap();
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision: first_revision,
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::BetweenSongs {
                track_title,
                candidate_available: true,
                continue_available: true,
                valid_reps: 9,
                invalid_reps: 1,
                ..
            }) if track_title == &first_track.title
        ));

        // Select the replacement track through the same public intent path a
        // browser uses. A delayed status for song one must neither relabel its
        // evidence nor withdraw Save before Continue consumes the selection.
        let between_songs = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: between_songs.action_authority,
                    action: GuidedSessionAction::SelectCalibrationTrack {
                        track_id: second_track.id.0.clone(),
                    },
                },
                None,
            )
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision: first_revision,
                    presence: protocol::CalibrationCandidatePresence::Present {
                        content_identity: first_track.content_identity.clone(),
                        total_count: first_track.entries.len() as u32,
                        validity: protocol::CalibrationCandidateValidity {
                            model_numerically_valid: false,
                            record_crc_valid: false,
                        },
                    },
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::BetweenSongs {
                track_title,
                selected_track_id: Some(selected),
                candidate_available: true,
                continue_available: true,
                ..
            }) if track_title == &first_track.title && selected == &second_track.id.0
        ));

        let continue_snapshot = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: continue_snapshot.action_authority,
                    action: GuidedSessionAction::ContinueCalibration,
                },
                None,
            )
            .unwrap();
        let second_revision = CalibrationScheduleRevision::new(2).unwrap();
        let mut saw_continue = false;
        loop {
            match receive_control(&mut device).await {
                Frame::CalibrationContinue { run: continued } if continued == run => {
                    saw_continue = true;
                }
                Frame::CalibrationScheduleBegin {
                    run: begun,
                    schedule_revision,
                    content_identity,
                    total_count,
                } if begun == run && schedule_revision == second_revision => {
                    assert!(saw_continue, "Continue must precede revision-2 Begin");
                    assert_eq!(content_identity, second_track.content_identity);
                    assert_eq!(total_count, second_track.entries.len() as u32);
                    break;
                }
                _ => {}
            }
        }
        device
            .frames
            .send(Frame::CalibrationScheduleUploadAcknowledged {
                acknowledgement: protocol::CalibrationScheduleUploadAcknowledgement {
                    run,
                    schedule_revision: second_revision,
                    content_identity: second_track.content_identity.clone(),
                    total_count: second_track.entries.len() as u32,
                    operation: protocol::CalibrationScheduleUploadOperationAcknowledgement::Begin {
                        operation_fingerprint: protocol::calibration_schedule_begin_fingerprint(
                            run,
                            second_revision,
                            &second_track.content_identity,
                            second_track.entries.len() as u32,
                        ),
                    },
                },
            })
            .unwrap();
        let chunk = loop {
            let frame = receive_control(&mut device).await;
            if matches!(
                frame,
                Frame::CalibrationScheduleChunk {
                    run: chunk_run,
                    schedule_revision,
                    ..
                } if chunk_run == run && schedule_revision == second_revision
            ) {
                break frame;
            }
        };
        let Frame::CalibrationScheduleChunk {
            first_entry,
            entries,
            ..
        } = chunk
        else {
            unreachable!("matched revision-2 chunk")
        };
        device
            .frames
            .send(Frame::CalibrationScheduleUploadAcknowledged {
                acknowledgement: protocol::CalibrationScheduleUploadAcknowledgement {
                    run,
                    schedule_revision: second_revision,
                    content_identity: second_track.content_identity.clone(),
                    total_count: second_track.entries.len() as u32,
                    operation: protocol::CalibrationScheduleUploadOperationAcknowledgement::Chunk {
                        first_entry,
                        operation_fingerprint: protocol::calibration_schedule_chunk_fingerprint(
                            run,
                            second_revision,
                            &second_track.content_identity,
                            second_track.entries.len() as u32,
                            first_entry,
                            &entries,
                        ),
                    },
                },
            })
            .unwrap();
        device
            .frames
            .send(Frame::CalibrationPreparationStatus {
                status: protocol::CalibrationPreparationStatus {
                    run,
                    schedule_revision: second_revision,
                    phase: protocol::CalibrationPreparationPhase::ReadyForSchedule,
                },
            })
            .unwrap();
        loop {
            if matches!(
                receive_control(&mut device).await,
                Frame::CalibrationScheduleCommit {
                    run: committed_run,
                    schedule_revision,
                    ref content_identity,
                    ..
                } if committed_run == run
                    && schedule_revision == second_revision
                    && content_identity == &second_track.content_identity
            ) {
                break;
            }
        }
        let acknowledged_device_monotonic_microseconds =
            crate::timing::host_monotonic_nanoseconds() / 1_000;
        device
            .frames
            .send(Frame::CalibrationScheduleAccepted {
                accepted: protocol::CalibrationScheduleAccepted {
                    run,
                    schedule_revision: second_revision,
                    content_identity: second_track.content_identity.clone(),
                    acknowledged_device_monotonic_microseconds,
                    anchor_device_monotonic_microseconds: acknowledged_device_monotonic_microseconds
                        + 3_000_000,
                    acquisition_sample: 20_000,
                },
            })
            .unwrap();

        let second_counts = protocol::CalibrationGesture::ALL
            .into_iter()
            .flat_map(|gesture| {
                [
                    protocol::CalibrationModifier::ThumbUp,
                    protocol::CalibrationModifier::ThumbDown,
                ]
                .into_iter()
                .map(move |modifier| (gesture, modifier))
            })
            .enumerate()
            .map(|(index, (gesture, modifier))| {
                let (accepted_count, target_count) = match modifier {
                    protocol::CalibrationModifier::ThumbUp => (12, 10),
                    protocol::CalibrationModifier::ThumbDown if index == 9 => (16, 16),
                    protocol::CalibrationModifier::ThumbDown => (17, 16),
                };
                protocol::CalibrationClassCounts {
                    gesture,
                    modifier,
                    accepted_count,
                    rejected_count: if index == 9 { 0 } else { 4 },
                    target_count,
                    deficit_count: 0,
                }
            })
            .collect();
        device
            .frames
            .send(Frame::CalibrationSongResult {
                result: protocol::CalibrationSongResult {
                    run,
                    schedule_revision: second_revision,
                    content_identity: second_track.content_identity.clone(),
                    counts: second_counts,
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: false,
                        record_crc_valid: false,
                    },
                },
            })
            .unwrap();
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision: second_revision,
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::BetweenSongs {
                track_title,
                candidate_available: true,
                continue_available: false,
                valid_reps: 144,
                invalid_reps: 36,
                ref deficits,
                ..
            }) if track_title == &second_track.title && deficits.is_empty()
        ));

        // Neither a prior revision's terminal nor the current pre-Save Absent
        // status may revoke the exact song result's Save authority.
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision: first_revision,
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision: second_revision,
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::BetweenSongs {
                candidate_available: true,
                continue_available: false,
                ..
            })
        ));

        let save_snapshot = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: save_snapshot.action_authority,
                    action: GuidedSessionAction::SaveCalibration,
                },
                None,
            )
            .unwrap();
        loop {
            if matches!(
                receive_control(&mut device).await,
                Frame::CalibrationSave { run: saved_run } if saved_run == run
            ) {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Finalizing { .. })
        ));

        device
            .frames
            .send(Frame::CalibrationResidentActivated {
                activation: protocol::CalibrationResidentActivation {
                    run,
                    schedule_revision: first_revision,
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: true,
                        record_crc_valid: true,
                    },
                    resident_sequence: 8,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Finalizing { .. })
        ));
        device
            .frames
            .send(Frame::CalibrationResidentActivated {
                activation: protocol::CalibrationResidentActivation {
                    run,
                    schedule_revision: second_revision,
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: true,
                        record_crc_valid: true,
                    },
                    resident_sequence: 9,
                },
            })
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Setup { .. })
        ));
        assert!(coordinator.snapshot().active().is_none());
    }

    #[tokio::test]
    async fn failed_playback_preparation_never_becomes_ready() {
        let task = tokio::spawn(async {
            Err(anyhow::anyhow!("fixture audio failed"))
                as anyhow::Result<crate::collect::audio::Playback>
        });
        let mut preparation = PlaybackPreparation::Pending {
            content_identity: "first-song".into(),
            task,
        };
        while matches!(&preparation, PlaybackPreparation::Pending { task, .. } if !task.is_finished())
        {
            tokio::task::yield_now().await;
        }
        let failure = preparation.resolve_if_finished().await.unwrap_err();
        assert!(failure.contains("fixture audio failed"));
        assert!(!preparation.is_ready());
        assert!(matches!(preparation, PlaybackPreparation::Consumed));
    }

    #[tokio::test]
    async fn panicked_playback_preparation_never_becomes_ready() {
        let task = tokio::spawn(async { panic!("fixture audio task panicked") });
        let mut preparation = PlaybackPreparation::Pending {
            content_identity: "song".into(),
            task,
        };
        while matches!(&preparation, PlaybackPreparation::Pending { task, .. } if !task.is_finished())
        {
            tokio::task::yield_now().await;
        }
        let failure = preparation.resolve_if_finished().await.unwrap_err();
        assert!(failure.contains("audio task failed"));
        assert!(!preparation.is_ready());
        assert!(matches!(preparation, PlaybackPreparation::Consumed));
    }

    #[tokio::test]
    async fn song_result_survives_candidate_not_yet_ready_for_continue() {
        let registry = Arc::new(Registry::new());
        let coordinator = GuidedSessionCoordinator::new();
        let adapter =
            CalibrationModeAdapter::new(registry.clone(), coordinator.clone(), vec![track()]);
        attach_test_collection(&adapter, registry.clone(), coordinator.clone());
        coordinator.set_mode_adapter(GuidedMode::Calibration, adapter.clone());
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();
        let snapshot = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: snapshot.action_authority,
                    action: GuidedSessionAction::SelectCalibrationTrack {
                        track_id: "calibration-track".into(),
                    },
                },
                None,
            )
            .unwrap();
        let snapshot = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: snapshot.action_authority,
                    action: GuidedSessionAction::StartCalibration,
                },
                Some(identity),
            )
            .unwrap();
        let first = receive_control(&mut device).await;
        let second = receive_control(&mut device).await;
        assert!(matches!(
            (&first, &second),
            (
                Frame::CalibrationScheduleBegin { .. },
                Frame::CalibrationHeartbeat { .. }
            ) | (
                Frame::CalibrationHeartbeat { .. },
                Frame::CalibrationScheduleBegin { .. }
            )
        ));
        let preparing = coordinator.snapshot();
        assert_eq!(
            coordinator.handle_intent_for_device(
                GuidedIntentRequest {
                    authority: preparing.action_authority,
                    action: GuidedSessionAction::ContinueCalibration,
                },
                None,
            ),
            Err(CoordinatorError::ActionUnavailable),
            "Continue must not replace a preparation/upload transaction"
        );
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        device
            .frames
            .send(Frame::CalibrationPreparationStatus {
                status: protocol::CalibrationPreparationStatus {
                    run,
                    schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
                    phase: protocol::CalibrationPreparationPhase::Settling {
                        elapsed_milliseconds: 750,
                        remaining_milliseconds: 9_250,
                    },
                },
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Preparing {
                stage: protocol::GuidedCalibrationPreparationStage::Stillness,
                elapsed_milliseconds: 750,
                remaining_milliseconds: 9_250,
                ..
            })
        ));
        device
            .frames
            .send(Frame::CalibrationSongResult {
                result: protocol::CalibrationSongResult {
                    run,
                    schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
                    content_identity: "test-content".into(),
                    counts: vec![protocol::CalibrationClassCounts {
                        gesture: protocol::CalibrationGesture::WristPronation,
                        modifier: protocol::CalibrationModifier::ThumbUp,
                        accepted_count: 3,
                        rejected_count: 1,
                        target_count: 10,
                        deficit_count: 7,
                    }],
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: false,
                        record_crc_valid: true,
                    },
                },
            })
            .unwrap();
        device
            .frames
            .send(Frame::CalibrationCandidateStatus {
                candidate: protocol::CalibrationCandidateStatus {
                    run,
                    schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
                    presence: protocol::CalibrationCandidatePresence::Absent,
                },
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::BetweenSongs {
                candidate_available: true,
                continue_available: true,
                valid_reps: 3,
                invalid_reps: 1,
                ..
            })
        ));
        // A delayed device progress packet belongs to the pre-anchor phase;
        // it must not regress an already-projected song boundary.
        device
            .frames
            .send(Frame::CalibrationPreparationStatus {
                status: protocol::CalibrationPreparationStatus {
                    run,
                    schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
                    phase: protocol::CalibrationPreparationPhase::EstimatingGains {
                        elapsed_milliseconds: 1_000,
                        remaining_milliseconds: 19_000,
                    },
                },
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::BetweenSongs { .. })
        ));
        let snapshot = coordinator.snapshot();
        coordinator
            .handle_intent_for_device(
                GuidedIntentRequest {
                    authority: snapshot.action_authority,
                    action: GuidedSessionAction::ContinueCalibration,
                },
                None,
            )
            .unwrap();
        // Continue and the new Begin share the reliable writer but heartbeats
        // may interleave, so observe the revision-2 Begin without assuming an
        // incidental scheduling order.
        loop {
            if matches!(
                receive_control(&mut device).await,
                Frame::CalibrationScheduleBegin {
                    schedule_revision,
                    ..
                } if schedule_revision == CalibrationScheduleRevision::new(2).unwrap()
            ) {
                break;
            }
        }
        device
            .frames
            .send(Frame::CalibrationPreparationStatus {
                status: protocol::CalibrationPreparationStatus {
                    run,
                    schedule_revision: CalibrationScheduleRevision::new(2).unwrap(),
                    phase: protocol::CalibrationPreparationPhase::Settling {
                        elapsed_milliseconds: 250,
                        remaining_milliseconds: 9_750,
                    },
                },
            })
            .unwrap();
        // A terminal frame for song 1 may still have been buffered when song 2
        // began. Run identity alone is insufficient; it must not withhold song
        // 2's heartbeat or replace its preparation projection.
        device
            .frames
            .send(Frame::CalibrationSongResult {
                result: protocol::CalibrationSongResult {
                    run,
                    schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
                    content_identity: "test-content".into(),
                    counts: Vec::new(),
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: true,
                        record_crc_valid: true,
                    },
                },
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Preparing {
                elapsed_milliseconds: 250,
                ..
            })
        ));
        // A new DeviceHello is a link epoch, even when the underlying serial
        // file stayed open. The actor must fail explicitly rather than use the
        // former connection's fully-acknowledged upload authority.
        device
            .frames
            .send(Frame::DeviceHello {
                device_id: "opal-test".into(),
                config: DeviceConfig {
                    gestures: 0,
                    keymap: Vec::new(),
                    wifi_ssid: None,
                    sensitivity: String::new(),
                    sensitivity_levels: Vec::new(),
                    tau: 0.0,
                    needed: 0,
                },
                provenance: DeviceProvenance {
                    firmware: FirmwareBuild {
                        crate_version: String::new(),
                        git_commit: String::new(),
                        working_tree_modified: false,
                        built_at: String::new(),
                    },
                    analog_front_ends: Vec::new(),
                },
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::TechnicalFailure { .. })
        ));
    }

    #[test]
    fn acknowledged_schedule_upload_uses_operational_eight_entry_chunks_for_130_entries() {
        let registry = Registry::new();
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();
        let mut track = track();
        track.entries = (0..130)
            .map(|index| protocol::CalibrationScheduleEntry {
                cue_id: protocol::CalibrationCueId::new(index + 1).unwrap(),
                gesture: protocol::CalibrationGesture::ALL[index as usize % 5],
                modifier: if index % 2 == 0 {
                    protocol::CalibrationModifier::ThumbUp
                } else {
                    protocol::CalibrationModifier::ThumbDown
                },
                track_offset: protocol::TrackMilliseconds::new(index * 2_000),
                hold: protocol::DurationMilliseconds::new(1_500),
            })
            .collect();
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let mut upload = begin_upload(
            &registry,
            &identity,
            run,
            CalibrationScheduleRevision::new(1).unwrap(),
            &track,
        )
        .unwrap();
        // Begin is delivered first, and every next chunk is released only by
        // the matching device acknowledgement.
        let acknowledgement = expected_upload_fingerprint(
            run,
            CalibrationScheduleRevision::new(1).unwrap(),
            &track,
            &upload,
        )
        .unwrap();
        advance_upload(
            &registry,
            &identity,
            run,
            CalibrationScheduleRevision::new(1).unwrap(),
            &track,
            &mut upload,
            acknowledgement,
        )
        .unwrap();
        for _ in (0..130).step_by(OPERATIONAL_SCHEDULE_UPLOAD_ENTRIES) {
            let acknowledgement = expected_upload_fingerprint(
                run,
                CalibrationScheduleRevision::new(1).unwrap(),
                &track,
                &upload,
            )
            .unwrap();
            advance_upload(
                &registry,
                &identity,
                run,
                CalibrationScheduleRevision::new(1).unwrap(),
                &track,
                &mut upload,
                acknowledgement,
            )
            .unwrap();
        }
        assert!(matches!(upload, UploadPhase::Complete));
        let mut sizes = Vec::new();
        while let Ok(frame) = device.control_rx.try_recv() {
            if let Frame::CalibrationScheduleChunk { entries, .. } = frame {
                sizes.push(entries.len());
            }
        }
        assert_eq!(
            sizes,
            vec![8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2]
        );
    }

    #[test]
    fn chunk_ack_retry_replays_the_same_operational_chunk_exactly() {
        let registry = Registry::new();
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();
        let track = track();
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let revision = CalibrationScheduleRevision::new(1).unwrap();
        let mut upload = begin_upload(&registry, &identity, run, revision, &track).unwrap();
        let acknowledgement = expected_upload_fingerprint(run, revision, &track, &upload).unwrap();
        advance_upload(
            &registry,
            &identity,
            run,
            revision,
            &track,
            &mut upload,
            acknowledgement,
        )
        .unwrap();
        let _begin = device.control_rx.try_recv().unwrap();
        let first = device.control_rx.try_recv().unwrap();
        retry_upload_chunk(&registry, &identity, run, revision, &track, &upload).unwrap();
        let retry = device.control_rx.try_recv().unwrap();
        assert_eq!(format!("{first:?}"), format!("{retry:?}"));
        assert!(matches!(
            first,
            Frame::CalibrationScheduleChunk { first_entry: 0, entries, .. }
                if entries.len() == track.entries.len().min(OPERATIONAL_SCHEDULE_UPLOAD_ENTRIES)
        ));
    }

    #[test]
    fn lost_begin_ack_fails_instead_of_retrying_a_non_idempotent_begin() {
        assert_eq!(
            upload_timeout_action(&UploadPhase::AwaitingBeginAcknowledgement, 0),
            UploadTimeoutAction::FailBegin
        );
        assert_eq!(
            upload_timeout_action(
                &UploadPhase::AwaitingChunkAcknowledgement { first_entry: 0 },
                0,
            ),
            UploadTimeoutAction::RetryChunk
        );
        assert_eq!(
            upload_timeout_action(
                &UploadPhase::AwaitingChunkAcknowledgement { first_entry: 0 },
                MAX_CHUNK_ACK_RETRIES,
            ),
            UploadTimeoutAction::FailChunk
        );
        assert_eq!(
            upload_timeout_action(&UploadPhase::Complete, 0),
            UploadTimeoutAction::None
        );
    }

    #[test]
    fn lost_commit_acceptance_has_a_phase_owned_deadline() {
        let sent_at = tokio::time::Instant::now();
        let phase = ScheduleCommitPhase::sent_at(sent_at);
        assert_eq!(
            phase.acceptance_deadline(),
            Some(sent_at + SCHEDULE_ACCEPTANCE_TIMEOUT)
        );
        assert!(!phase
            .acceptance_timed_out(sent_at + SCHEDULE_ACCEPTANCE_TIMEOUT - Duration::from_nanos(1)));
        assert!(phase.acceptance_timed_out(sent_at + SCHEDULE_ACCEPTANCE_TIMEOUT));
        assert_eq!(
            ScheduleCommitPhase::AwaitingDeviceReadiness.acceptance_deadline(),
            None
        );
        assert_eq!(ScheduleCommitPhase::Accepted.acceptance_deadline(), None);
    }

    #[test]
    fn awaited_ack_fingerprint_names_the_exact_begin_or_chunk_payload() {
        let track = track();
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let revision = CalibrationScheduleRevision::new(1).unwrap();
        let begin = expected_upload_fingerprint(
            run,
            revision,
            &track,
            &UploadPhase::AwaitingBeginAcknowledgement,
        )
        .unwrap();
        let chunk = expected_upload_fingerprint(
            run,
            revision,
            &track,
            &UploadPhase::AwaitingChunkAcknowledgement { first_entry: 0 },
        )
        .unwrap();
        assert_ne!(begin, chunk);

        let mut changed = track.clone();
        changed.entries[0].track_offset = protocol::TrackMilliseconds::new(1);
        assert_ne!(
            chunk,
            expected_upload_fingerprint(
                run,
                revision,
                &changed,
                &UploadPhase::AwaitingChunkAcknowledgement { first_entry: 0 },
            )
            .unwrap()
        );
        assert_eq!(
            expected_upload_fingerprint(run, revision, &track, &UploadPhase::Complete),
            None
        );
    }

    #[test]
    fn upload_cannot_reach_commit_before_every_chunk_acknowledgement() {
        let registry = Registry::new();
        let mut device = register_device(&registry);
        let identity = registry.connection_identity("opal-test").unwrap();
        let track = track();
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let mut upload = begin_upload(
            &registry,
            &identity,
            run,
            CalibrationScheduleRevision::new(1).unwrap(),
            &track,
        )
        .unwrap();
        // A chunk acknowledgement before Begin is ignored, as is an old index.
        advance_upload(
            &registry,
            &identity,
            run,
            CalibrationScheduleRevision::new(1).unwrap(),
            &track,
            &mut upload,
            protocol::CalibrationScheduleUploadOperationAcknowledgement::Chunk {
                first_entry: 0,
                operation_fingerprint: 0,
            },
        )
        .unwrap();
        assert!(matches!(upload, UploadPhase::AwaitingBeginAcknowledgement));
        assert!(matches!(
            device.control_rx.try_recv().unwrap(),
            Frame::CalibrationScheduleBegin { .. }
        ));
        assert!(device.control_rx.try_recv().is_err());
    }

    #[test]
    fn playback_deadline_requires_a_future_production_clock_map() {
        let now = tokio::time::Instant::now();
        let output_latency = protocol::DurationMilliseconds::new(20);
        assert_eq!(
            playback_deadline(None, output_latency, 1_000, now),
            Err("calibration playback is withheld until production clock probes map the device anchor")
        );
        assert_eq!(
            playback_deadline(Some(1_000), output_latency, 1_000, now),
            Err("the output-latency-compensated calibration anchor is already in the past")
        );
        assert_eq!(
            playback_deadline(Some(1_050), output_latency, 1_000, now).unwrap(),
            now + std::time::Duration::from_millis(30)
        );
    }

    #[test]
    fn calibration_lanes_reuse_collect_presentation_descriptors() {
        let classes = [
            ("wrist_pronation", "Tilt Out", "blue"),
            ("wrist_supination", "Tilt In", "amber"),
            ("wrist_radial_deviation", "Tilt Forward", "green"),
            ("wrist_ulnar_deviation", "Tilt Back", "purple"),
            ("thumb_extension", "Tip center", "pink"),
        ]
        .map(|(id, label, color)| protocol::CollectionClass {
            id: protocol::ClassId(id.into()),
            label: label.into(),
            color: color.into(),
            motion: Some(protocol::GestureMotion {
                arrow: protocol::MotionArrow::Left,
                hint: format!("{label} hint"),
            }),
        });

        let lanes = calibration_lanes(&classes);
        assert_eq!(lanes.len(), 5);
        for (lane, class) in lanes.iter().zip(&classes) {
            assert_eq!(lane.id, class.id.0);
            assert_eq!(lane.label, class.label);
            assert_eq!(lane.color_name, class.color);
            assert_eq!(lane.motion, class.motion);
        }
        assert!(lanes.iter().all(|lane| !lane.label.starts_with("Wrist")));

        let observed_at = protocol::UnixMilliseconds::new(1_800_000_012_345);
        let snapshot = playing_snapshot(
            &track(),
            CalibrationPlayhead {
                position_ms: 12_345,
                observed_at,
            },
            &classes,
        );
        assert!(matches!(
            snapshot,
            GuidedCalibrationSnapshot::Playing {
                position_ms: 12_345,
                position_observed_at_unix_ms,
                ..
            } if position_observed_at_unix_ms == observed_at
        ));
    }

    #[test]
    fn accepted_schedule_is_a_one_way_boundary_for_preparation_projection() {
        assert!(ScheduleCommitPhase::AwaitingDeviceReadiness.projects_preparation());
        assert!(ScheduleCommitPhase::sent_at(tokio::time::Instant::now()).projects_preparation());
        assert!(!ScheduleCommitPhase::Accepted.projects_preparation());
    }

    #[test]
    fn candidate_presence_is_bound_to_the_exact_current_track_identity() {
        let track = track();
        let valid = protocol::CalibrationCandidateValidity {
            model_numerically_valid: true,
            record_crc_valid: true,
        };
        assert!(candidate_presence_matches_track(
            &protocol::CalibrationCandidatePresence::Absent,
            &track,
        ));
        assert!(candidate_presence_matches_track(
            &protocol::CalibrationCandidatePresence::Present {
                content_identity: track.content_identity.clone(),
                total_count: track.entries.len() as u32,
                validity: valid,
            },
            &track,
        ));
        assert!(!candidate_presence_matches_track(
            &protocol::CalibrationCandidatePresence::Present {
                content_identity: "different-content".into(),
                total_count: track.entries.len() as u32,
                validity: valid,
            },
            &track,
        ));
    }

    #[test]
    fn selecting_a_continue_track_does_not_relabel_completed_evidence() {
        let completed = track();
        let mut next = track();
        next.id = protocol::TrackId("next-track".into());
        next.title = "Next Track".into();
        next.content_identity = "next-content".into();
        let evidence = EvidenceState::Retained {
            completed_track_title: completed.title.clone(),
        };
        let candidate = CandidateState::Present(protocol::CalibrationCandidateValidity {
            model_numerically_valid: true,
            record_crc_valid: true,
        });

        let snapshot = between_songs_snapshot(
            &next,
            &[],
            &evidence,
            candidate,
            &[completed.clone(), next.clone()],
            &[],
        );

        assert!(matches!(
            snapshot,
            GuidedCalibrationSnapshot::BetweenSongs {
                track_title,
                selected_track_id: Some(selected),
                candidate_available: true,
                continue_available: false,
                ..
            } if track_title == completed.title && selected == next.id.0
        ));
    }

    #[test]
    fn song_deficits_use_collect_labels_and_user_facing_variants() {
        let completed = track();
        let evidence = EvidenceState::Retained {
            completed_track_title: completed.title.clone(),
        };
        let classes = [protocol::CollectionClass {
            id: protocol::ClassId("wrist_pronation".into()),
            label: "Tilt in".into(),
            color: "blue".into(),
            motion: None,
        }];
        let counts = [
            protocol::CalibrationClassCounts {
                gesture: protocol::CalibrationGesture::WristPronation,
                modifier: protocol::CalibrationModifier::ThumbUp,
                accepted_count: 2,
                rejected_count: 1,
                target_count: 10,
                deficit_count: 8,
            },
            protocol::CalibrationClassCounts {
                gesture: protocol::CalibrationGesture::WristPronation,
                modifier: protocol::CalibrationModifier::ThumbDown,
                accepted_count: 3,
                rejected_count: 0,
                target_count: 16,
                deficit_count: 13,
            },
        ];

        let snapshot = between_songs_snapshot(
            &completed,
            &counts,
            &evidence,
            CandidateState::Absent,
            core::slice::from_ref(&completed),
            &classes,
        );

        assert!(matches!(
            snapshot,
            GuidedCalibrationSnapshot::BetweenSongs {
                deficits,
                continue_available: true,
                ..
            } if deficits == ["Tilt in, command: 8 short", "Tilt in, no-op: 13 short"]
        ));
    }

    #[test]
    fn terminal_setup_retains_selection_without_retaining_song_actions() {
        let track = track();
        let snapshot = setup_snapshot(core::slice::from_ref(&track), Some(track.id.0.clone()));
        assert!(matches!(
            snapshot,
            GuidedCalibrationSnapshot::Setup {
                selected_track_id: Some(selected),
                tracks,
            } if selected == track.id.0 && tracks.len() == 1
        ));
    }

    #[test]
    fn stale_setup_selection_cannot_change_the_track_a_later_start_consumes() {
        let registry = Arc::new(Registry::new());
        let coordinator = GuidedSessionCoordinator::new();
        let first = track();
        let mut second = first.clone();
        second.id = protocol::TrackId("second-track".into());
        second.title = "Second Track".into();
        let adapter = CalibrationModeAdapter::new(
            registry,
            coordinator.clone(),
            vec![first.clone(), second.clone()],
        );

        let setup = coordinator.snapshot();
        let first_request = GuidedIntentRequest {
            authority: setup.action_authority,
            action: GuidedSessionAction::SelectCalibrationTrack {
                track_id: first.id.0.clone(),
            },
        };
        adapter
            .select_track(first.id.0.clone(), &first_request)
            .unwrap();

        let stale_request = GuidedIntentRequest {
            authority: setup.action_authority,
            action: GuidedSessionAction::SelectCalibrationTrack {
                track_id: second.id.0.clone(),
            },
        };
        assert!(matches!(
            adapter.select_track(second.id.0.clone(), &stale_request),
            Err(CoordinatorError::StaleActionPhase { .. })
        ));
        assert_eq!(
            adapter.selected_track_id.lock().unwrap().as_deref(),
            Some(first.id.0.as_str())
        );
        assert!(matches!(
            coordinator.snapshot().calibration(),
            Some(GuidedCalibrationSnapshot::Setup {
                selected_track_id: Some(selected),
                ..
            }) if selected == &first.id.0
        ));
    }
}
