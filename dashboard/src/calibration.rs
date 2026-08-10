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
const CHUNK_ACK_RETRY_AFTER: Duration = Duration::from_secs(2);
const MAX_CHUNK_ACK_RETRIES: u8 = 3;
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
}

/// Latest requested heartbeat/playback gate. Pause and resume are idempotent
/// state, not an event backlog. A watch channel bounds storage at one value and
/// makes the browser-departure interrupt impossible to reject under load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunGate {
    Interrupted,
    Running,
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

/// The host actor's phase-local state.  These are deliberately not booleans:
/// a heartbeat cannot be active without the accepted revision it names, and
/// output cannot be both armed and playing.
enum ScheduleCommitPhase {
    AwaitingDeviceReadiness,
    Sent,
    /// The exact run/revision/identity was accepted. Late preparation
    /// narration can no longer project the UI back before this boundary.
    Accepted,
}

impl ScheduleCommitPhase {
    fn projects_preparation(&self) -> bool {
        matches!(self, Self::AwaitingDeviceReadiness | Self::Sent)
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

enum HeartbeatMode {
    Withheld,
    Sending {
        schedule_revision: CalibrationScheduleRevision,
        next_sequence: u32,
    },
}

enum PlaybackState {
    Dormant,
    Armed {
        playback: crate::collect::audio::Playback,
    },
    Playing(crate::collect::audio::Playback),
}

#[derive(Clone, Copy)]
enum EvidenceState {
    Fresh,
    Retained,
}

#[derive(Clone, Copy)]
enum CandidateState {
    Absent,
    Present(protocol::CalibrationCandidateValidity),
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

    fn with_validity(self, validity: protocol::CalibrationCandidateValidity) -> Self {
        match self {
            Self::Absent => Self::Absent,
            Self::Present(_) => Self::Present(validity),
        }
    }

    fn present(self) -> bool {
        matches!(self, Self::Present(_))
    }

    fn validity(self) -> protocol::CalibrationCandidateValidity {
        match self {
            Self::Absent => protocol::CalibrationCandidateValidity {
                model_numerically_valid: false,
                record_crc_valid: false,
            },
            Self::Present(validity) => validity,
        }
    }

    fn permits_activation(self) -> bool {
        self.present() && self.validity().permits_activation()
    }
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
        let playback_track_id = track.id.clone();
        let playback_entries = track.entries.clone();
        let mut prepared_playback = tokio::task::spawn_blocking(move || {
            playback_collection.open_calibration_playback(&playback_track_id, &playback_entries)
        });
        let binding = lease.binding().clone();
        let mut heartbeat = tokio::time::interval(std::time::Duration::from_millis(u64::from(
            protocol::CALIBRATION_HEARTBEAT_INTERVAL_MILLISECONDS,
        )));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut commit_phase = ScheduleCommitPhase::AwaitingDeviceReadiness;
        let mut device_ready_for_commit = false;
        let mut heartbeat_mode = HeartbeatMode::Sending {
            schedule_revision,
            next_sequence: 0,
        };
        let mut song_counts = Vec::new();
        let mut evidence = EvidenceState::Fresh;
        let mut candidate = CandidateState::Absent;
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
        let mut chunk_ack_retry = tokio::time::interval(CHUNK_ACK_RETRY_AFTER);
        chunk_ack_retry.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Tokio intervals tick immediately. Consume that first tick so a
        // newly acknowledged Begin cannot instantly duplicate Chunk 0.
        chunk_ack_retry.tick().await;
        let mut chunk_ack_retries = 0u8;
        let outcome = loop {
            tokio::select! {
                biased;
                changed = gate.changed() => match changed {
                    Ok(()) => match *gate.borrow_and_update() {
                        RunGate::Interrupted => heartbeat_mode = HeartbeatMode::Withheld,
                        RunGate::Running => {
                            if matches!(heartbeat_mode, HeartbeatMode::Withheld)
                                && matches!(playback, PlaybackState::Playing(_) | PlaybackState::Armed { .. })
                            {
                                heartbeat_mode = HeartbeatMode::Sending {
                                    schedule_revision,
                                    next_sequence: 0,
                                };
                            }
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
                                let position = playback.timeline().position.get() as u64;
                                let _ = self.coordinator.update_calibration(
                                    &binding,
                                    playing_snapshot(&track, position),
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
                    Ok(Frame::CalibrationResidentActivated { activation })
                        if activation.run == run => {
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
                        if !matches!(commit_phase, ScheduleCommitPhase::Sent) {
                            break SessionExit::DependencyFailed(
                                "the device accepted a schedule that this host did not commit".into(),
                            );
                        }
                        commit_phase = ScheduleCommitPhase::Accepted;
                        heartbeat_mode = HeartbeatMode::Sending {
                            schedule_revision,
                            next_sequence: 0,
                        };
                        let host_anchor = self
                            .timing
                            .lock()
                            .unwrap()
                            .as_ref()
                            .and_then(|timing| timing.corrected_host_milliseconds(
                                &device.device_id,
                                accepted.anchor_device_monotonic_microseconds,
                            ));
                        let deadline = match playback_deadline(host_anchor, unix_milliseconds(), tokio::time::Instant::now()) {
                            Ok(deadline) => deadline,
                            Err(detail) => break SessionExit::DependencyFailed(detail.into()),
                        };
                        let opened = match (&mut prepared_playback).await {
                            Ok(Ok(opened)) => opened,
                            Ok(Err(error)) => break SessionExit::TaskFailed(format!("calibration audio failed: {error:#}")),
                            Err(error) => break SessionExit::TaskFailed(format!("calibration audio task failed: {error}")),
                        };
                        playback = PlaybackState::Armed { playback: opened };
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
                        chunk_ack_retry.reset();
                        if matches!(upload, UploadPhase::Complete)
                            && device_ready_for_commit
                            && matches!(commit_phase, ScheduleCommitPhase::AwaitingDeviceReadiness)
                        {
                            if let Err(error) = send_schedule_commit(
                                &self.registry, &device, run, schedule_revision, &track,
                            ) {
                                break delivery_failure(error);
                            }
                            commit_phase = ScheduleCommitPhase::Sent;
                        }
                    }
                    Ok(Frame::CalibrationPreparationStatus { status })
                        if status.run == run
                            && status.schedule_revision == schedule_revision
                            && matches!(evidence, EvidenceState::Fresh) => {
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
                        device_ready_for_commit = ready_for_schedule;
                        if ready_for_schedule
                            && matches!(upload, UploadPhase::Complete)
                            && matches!(commit_phase, ScheduleCommitPhase::AwaitingDeviceReadiness)
                        {
                            if let Err(error) = send_schedule_commit(
                                &self.registry, &device, run, schedule_revision, &track,
                            ) {
                                break delivery_failure(error);
                            }
                            commit_phase = ScheduleCommitPhase::Sent;
                        }
                    }
                    Ok(Frame::CalibrationSongInterrupted { interruption })
                        if interruption.run == run => {
                        evidence = EvidenceState::Retained;
                        heartbeat_mode = HeartbeatMode::Withheld;
                        playback = PlaybackState::Dormant;
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            between_songs_snapshot(
                                &track,
                                &song_counts,
                                evidence,
                                candidate,
                                &self.tracks,
                            ),
                        );
                    }
                    Ok(Frame::CalibrationSongResult { result })
                        if result.run == run => {
                        song_counts = result.counts;
                        evidence = EvidenceState::Retained;
                        candidate = candidate.with_validity(result.validity);
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            between_songs_snapshot(
                                &track,
                                &song_counts,
                                evidence,
                                candidate,
                                &self.tracks,
                            ),
                        );
                    }
                    Ok(Frame::CalibrationCandidateStatus { candidate: status })
                        if status.run == run
                            && status.schedule_revision == schedule_revision
                            && candidate_presence_matches_track(&status.presence, &track) => {
                        candidate = CandidateState::from_presence(status.presence);
                        if matches!(heartbeat_mode, HeartbeatMode::Withheld)
                            && matches!(evidence, EvidenceState::Retained) {
                            let _ = self.coordinator.update_calibration(
                                &binding,
                                between_songs_snapshot(
                                    &track,
                                    &song_counts,
                                    evidence,
                                    candidate,
                                    &self.tracks,
                                ),
                            );
                            }
                        }
                    Ok(Frame::CalibrationScheduleCommitDeferred { deferred })
                        if deferred.run == run
                            && deferred.schedule_revision == schedule_revision
                            && matches!(commit_phase, ScheduleCommitPhase::Sent) => {
                        // The preparation status is the retry authority.  A
                        // retryable refusal returns to its explicit ready
                        // phase; no backend stopwatch recreates that phase.
                        commit_phase = ScheduleCommitPhase::AwaitingDeviceReadiness;
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
                _ = chunk_ack_retry.tick(), if matches!(upload, UploadPhase::AwaitingChunkAcknowledgement { .. }) => {
                    if chunk_ack_retries >= MAX_CHUNK_ACK_RETRIES {
                        break SessionExit::DependencyFailed(
                            "calibration Chunk acknowledgement timed out after exact idempotent retries".into(),
                        );
                    }
                    // Chunks alone are safe to retry: firmware records an
                    // exact duplicate as `UploadEffect::Duplicate` and ACKs
                    // it. Begin and Commit are intentionally never retried:
                    // their duplicate semantics are refusal/terminal rather
                    // than an idempotent acknowledgement.
                    if let Err(error) = retry_upload_chunk(
                        &self.registry, &device, run, schedule_revision, &track, &upload,
                    ) {
                        break delivery_failure(error);
                    }
                    chunk_ack_retries = chunk_ack_retries.saturating_add(1);
                }
                command = actions.recv() => {
                    let Some(command) = command else {
                        break SessionExit::TaskFailed("calibration actor action channel closed".into());
                    };
                    let terminal_discard = matches!(&command, RunAction::Discard);
                    if matches!(&command, RunAction::Save) && !candidate.permits_activation()
                    {
                        continue;
                    }
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
                            commit_phase = ScheduleCommitPhase::AwaitingDeviceReadiness;
                            device_ready_for_commit = false;
                            evidence = EvidenceState::Fresh;
                            heartbeat_mode = HeartbeatMode::Sending {
                                schedule_revision,
                                next_sequence: 0,
                            };
                            playback = PlaybackState::Dormant;
                            continue;
                        }
                        RunAction::SelectNextTrack(next_track) => {
                            // A live song owns its authored identity until its
                            // result/interruption is projected. Retargeting is
                            // deliberately restricted to that boundary.
                            if !matches!(evidence, EvidenceState::Retained)
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
                                    evidence,
                                    candidate,
                                    &self.tracks,
                                ),
                            );
                            continue;
                        }
                        RunAction::Save => Frame::CalibrationSave { run },
                        RunAction::Discard => Frame::CalibrationDiscard { run },
                    };
                    if let Err(error) = self.registry.send_bound_control(&device, frame) {
                        break delivery_failure(error);
                    }
                    if terminal_discard {
                        break SessionExit::Completed;
                    }
                }
                _ = &mut playback_alarm, if matches!(playback, PlaybackState::Armed { .. }) => {
                    let PlaybackState::Armed { playback: opened, .. } = core::mem::replace(&mut playback, PlaybackState::Dormant) else {
                        break SessionExit::TaskFailed("calibration playback deadline had no armed output".into());
                    };
                    opened.play();
                    playback = PlaybackState::Playing(opened);
                    let _ = self.coordinator.update_calibration(
                        &binding,
                        playing_snapshot(&track, 0),
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
        enqueue_run_action(actions, action)
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

    fn select_track(
        &self,
        track_id: String,
        request: &GuidedIntentRequest,
    ) -> Result<(), CoordinatorError> {
        if !self.tracks.iter().any(|track| track.id.0 == track_id) {
            return Err(CoordinatorError::UnknownCalibrationTrack);
        }
        *self.selected_track_id.lock().unwrap() = Some(track_id.clone());
        self.coordinator.publish_calibration(
            request.expected_revision,
            request.expected_run_revision,
            request.expected_session_id,
            setup_snapshot(&self.tracks, Some(track_id)),
        )?;
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
        let snapshot = self.coordinator.snapshot();
        if snapshot.active() != Some(binding)
            || !matches!(
                snapshot.calibration(),
                Some(GuidedCalibrationSnapshot::BetweenSongs { .. })
            )
        {
            return Ok(());
        }
        self.send_action(binding, RunAction::SelectNextTrack(track))
    }
}

impl GuidedModeAdapter for CalibrationModeAdapter {
    fn pause_for_no_visible_views(&self, session: &GuidedSessionBinding) {
        let active = self.active.lock().unwrap();
        if let Some(active) = active.as_ref().filter(|active| &active.binding == session) {
            let _ = active.gate.send(RunGate::Interrupted);
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
            GuidedSessionAction::DiscardCalibration => self.send_action(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunAction::Discard,
            ),
            GuidedSessionAction::SaveCalibration => self.send_action(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunAction::Save,
            ),
            GuidedSessionAction::ContinueCalibration => self.send_action(
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
    now_wall_milliseconds: i64,
    now: tokio::time::Instant,
) -> Result<tokio::time::Instant, &'static str> {
    let host_anchor_milliseconds = host_anchor_milliseconds.ok_or(
        "calibration playback is withheld until production clock probes map the device anchor",
    )?;
    let delay_milliseconds = host_anchor_milliseconds
        .checked_sub(now_wall_milliseconds)
        .ok_or("the accepted calibration anchor is already in the past")?;
    let delay_milliseconds = u64::try_from(delay_milliseconds)
        .map_err(|_| "the accepted calibration anchor is already in the past")?;
    if delay_milliseconds == 0 {
        return Err("the accepted calibration anchor is already in the past");
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

fn playing_snapshot(
    track: &crate::collect::beatmap::CalibrationTrack,
    position_ms: u64,
) -> GuidedCalibrationSnapshot {
    let lanes = protocol::CalibrationGesture::ALL
        .into_iter()
        .map(|gesture| protocol::GuidedCalibrationLane {
            visual_lane: gesture.index(),
            id: format!("{gesture:?}"),
            label: format!("{gesture:?}"),
            color_name: "brand".into(),
            motion: None,
        })
        .collect();
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
        position_ms,
        valid_reps: 0,
        invalid_reps: 0,
        paused_reason: None,
        counts: Vec::new(),
    }
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
    evidence: EvidenceState,
    candidate: CandidateState,
    tracks: &[crate::collect::beatmap::CalibrationTrack],
) -> GuidedCalibrationSnapshot {
    let valid_reps = counts.iter().map(|count| count.accepted_count).sum();
    let invalid_reps = counts.iter().map(|count| count.rejected_count).sum();
    let deficits = counts
        .iter()
        .filter(|count| count.deficit_count > 0)
        .map(|count| {
            format!(
                "{:?} {:?}: {} short",
                count.gesture, count.modifier, count.deficit_count
            )
        })
        .collect();
    GuidedCalibrationSnapshot::BetweenSongs {
        track_title: track.title.clone(),
        tracks: tracks.iter().map(guided_track).collect(),
        selected_track_id: Some(track.id.0.clone()),
        candidate_available: candidate.permits_activation(),
        continue_available: matches!(evidence, EvidenceState::Retained),
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
    fn run_action_queue_is_bounded_and_distinguishes_full_from_closed() {
        let (actions, mut receiver) = mpsc::channel(2);
        enqueue_run_action(&actions, RunAction::Continue).unwrap();
        enqueue_run_action(&actions, RunAction::Save).unwrap();
        assert_eq!(
            enqueue_run_action(&actions, RunAction::Discard),
            Err(CoordinatorError::AdapterTaskBusy)
        );

        assert!(matches!(receiver.try_recv(), Ok(RunAction::Continue)));
        drop(receiver);
        assert_eq!(
            enqueue_run_action(&actions, RunAction::Discard),
            Err(CoordinatorError::AdapterTaskUnavailable)
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
                    expected_revision: snapshot.revision,
                    expected_run_revision: snapshot.run_revision,
                    expected_session_id: None,
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
                    expected_revision: snapshot.revision,
                    expected_run_revision: snapshot.run_revision,
                    expected_session_id: None,
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
        assert_eq!(
            playback_deadline(None, 1_000, now),
            Err("calibration playback is withheld until production clock probes map the device anchor")
        );
        assert_eq!(
            playback_deadline(Some(1_000), 1_000, now),
            Err("the accepted calibration anchor is already in the past")
        );
        assert_eq!(
            playback_deadline(Some(1_050), 1_000, now).unwrap(),
            now + std::time::Duration::from_millis(50)
        );
    }

    #[test]
    fn accepted_schedule_is_a_one_way_boundary_for_preparation_projection() {
        assert!(ScheduleCommitPhase::AwaitingDeviceReadiness.projects_preparation());
        assert!(ScheduleCommitPhase::Sent.projects_preparation());
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
}
