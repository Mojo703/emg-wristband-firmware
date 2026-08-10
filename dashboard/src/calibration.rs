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
use tokio::sync::{broadcast, mpsc};

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
    commands: mpsc::UnboundedSender<RunCommand>,
}

enum RunCommand {
    /// Stop host heartbeats when the guided view disappears.  The device's
    /// two-second watchdog then emits the interruption while retaining rows.
    Interrupt,
    Resume,
    /// Continue retains the completed evidence on the device and asks it to
    /// wait for a newly authored schedule.  The adapter owns the revision
    /// minting; the browser never talks to the device directly.
    Continue,
    /// Request candidate activation.  Firmware remains the authority for
    /// numerical validity and record CRC checks.
    Save,
    /// Drop the candidate while retaining the previous resident model.
    Discard,
}

impl CalibrationModeAdapter {
    pub fn new(
        registry: Arc<Registry>,
        coordinator: GuidedSessionCoordinator,
        tracks: Vec<crate::collect::beatmap::CalibrationTrack>,
    ) -> Arc<Self> {
        let wire_tracks = tracks.iter().map(guided_track).collect::<Vec<_>>();
        coordinator
            .update_idle_calibration(GuidedCalibrationSnapshot::Setup {
                tracks: wire_tracks,
                selected_track_id: None,
            })
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
        if let Err(error) = upload_schedule(&self.registry, &device, run, schedule_revision, &track)
        {
            lease.finish(SessionExit::DependencyFailed(
                error.calibration_message().into(),
            ));
            return Err(CoordinatorError::AdapterTaskUnavailable);
        }

        let (commands, command_rx) = mpsc::unbounded_channel();
        *active = Some(ActiveRun {
            binding: binding.clone(),
            commands,
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
                    lease,
                    device,
                    bound,
                    run,
                    schedule_revision,
                    track,
                    command_rx,
                )
                .await;
        });
        Ok(())
    }

    async fn run(
        self: Arc<Self>,
        lease: SessionLease,
        device: DeviceConnectionIdentity,
        mut bound: BoundDeviceHandle,
        run: CalibrationRunKey,
        mut schedule_revision: CalibrationScheduleRevision,
        track: crate::collect::beatmap::CalibrationTrack,
        mut commands: mpsc::UnboundedReceiver<RunCommand>,
    ) {
        let binding = lease.binding().clone();
        let mut heartbeat = tokio::time::interval(std::time::Duration::from_millis(u64::from(
            protocol::CALIBRATION_HEARTBEAT_INTERVAL_MILLISECONDS,
        )));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut heartbeat_sequence = 0u32;
        let run_started = tokio::time::Instant::now();
        let mut commit_sent = false;
        let mut committed_revision = None;
        let mut heartbeat_enabled = true;
        let mut song_counts = Vec::new();
        let mut retained_evidence = false;
        let mut candidate_present = false;
        let mut candidate_validity = protocol::CalibrationCandidateValidity {
            model_numerically_valid: false,
            record_crc_valid: false,
        };
        let mut playback: Option<crate::collect::audio::Playback> = None;
        let mut playback_start_at: Option<tokio::time::Instant> = None;
        let outcome = loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    if !heartbeat_enabled {
                        continue;
                    }
                    if let Some(committed_revision) = committed_revision {
                        let frame = Frame::CalibrationHeartbeat {
                            heartbeat: protocol::CalibrationHeartbeat {
                                run,
                                schedule_revision: committed_revision,
                                sequence: heartbeat_sequence,
                            },
                        };
                        heartbeat_sequence = heartbeat_sequence.wrapping_add(1);
                            if let Err(error) = self.registry.send_bound_control(&device, frame) {
                                break delivery_failure(error);
                            }
                            if let Some(playback) = playback.as_ref() {
                                let position = playback.timeline().position.get() as u64;
                                let _ = self.coordinator.update_calibration(
                                    &binding,
                                    playing_snapshot(&track, position),
                                );
                            }
                            if playback.is_none()
                                && playback_start_at.is_some_and(|start| tokio::time::Instant::now() >= start)
                            {
                                let Some(collection) = self.collection.lock().unwrap().clone() else {
                                    break SessionExit::TaskFailed("calibration playback manager is unavailable".into());
                                };
                                match collection.open_calibration_playback(&track.id, &track.entries) {
                                    Ok(opened) => {
                                        opened.play();
                                        playback = Some(opened);
                                        playback_start_at = None;
                                        let _ = self.coordinator.update_calibration(
                                            &binding,
                                            playing_snapshot(&track, 0),
                                        );
                                    }
                                    Err(error) => break SessionExit::TaskFailed(format!("calibration audio failed: {error:#}")),
                                }
                            }
                    } else if !commit_sent && run_started.elapsed() >= std::time::Duration::from_secs(30) {
                        let frame = Frame::CalibrationScheduleCommit {
                            run,
                            schedule_revision,
                            content_identity: track.content_identity.clone(),
                            total_count: track.entries.len() as u32,
                        };
                        if let Err(error) = self.registry.send_bound_control(&device, frame) {
                            break delivery_failure(error);
                        }
                        commit_sent = true;
                    }
                }
                received = bound.frames.recv() => match received {
                    Ok(Frame::CalibrationResidentActivated { activation })
                        if activation.run == run => {
                        break SessionExit::Completed;
                    }
                    Ok(Frame::CalibrationScheduleAccepted { accepted })
                        if accepted.run == run && accepted.schedule_revision == schedule_revision => {
                        committed_revision = Some(schedule_revision);
                        let timing = { self.timing.lock().unwrap().clone() };
                        let delay_milliseconds = if let Some(timing) = timing {
                            if let Some(host_anchor) = timing.corrected_host_milliseconds(
                                &device.device_id,
                                accepted.anchor_device_monotonic_microseconds,
                            ) {
                                let now = unix_milliseconds();
                                (host_anchor - now).max(0) as u64
                            } else {
                                0
                            }
                        } else {
                            0
                        };
                        playback_start_at = Some(tokio::time::Instant::now()
                            + std::time::Duration::from_millis(delay_milliseconds));
                    }
                    Ok(Frame::CalibrationSongInterrupted { interruption })
                        if interruption.run == run => {
                        retained_evidence = true;
                        heartbeat_enabled = false;
                        committed_revision = None;
                        playback = None;
                        playback_start_at = None;
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            between_songs_snapshot(
                                &track,
                                &song_counts,
                                retained_evidence,
                                candidate_present,
                                candidate_validity,
                            ),
                        );
                    }
                    Ok(Frame::CalibrationSongResult { result })
                        if result.run == run => {
                        song_counts = result.counts;
                        retained_evidence = true;
                        candidate_validity = result.validity;
                        let _ = self.coordinator.update_calibration(
                            &binding,
                            between_songs_snapshot(
                                &track,
                                &song_counts,
                                retained_evidence,
                                candidate_present,
                                candidate_validity,
                            ),
                        );
                    }
                    Ok(Frame::CalibrationCandidateStatus { candidate })
                        if candidate.run == run => {
                        candidate_present = candidate.candidate_present;
                        candidate_validity = candidate.validity;
                        if committed_revision.is_none() && retained_evidence {
                            let _ = self.coordinator.update_calibration(
                                &binding,
                                between_songs_snapshot(
                                    &track,
                                    &song_counts,
                                    retained_evidence,
                                    candidate_present,
                                    candidate_validity,
                                ),
                            );
                        }
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
                command = commands.recv() => {
                    let Some(command) = command else {
                        break SessionExit::TaskFailed("calibration actor command channel closed".into());
                    };
                    let terminal_discard = matches!(&command, RunCommand::Discard);
                    if matches!(&command, RunCommand::Save)
                        && (!candidate_present || !candidate_validity.permits_activation())
                    {
                        continue;
                    }
                    let frame = match command {
                        RunCommand::Interrupt => {
                            heartbeat_enabled = false;
                            continue;
                        }
                        RunCommand::Resume => {
                            heartbeat_enabled = true;
                            continue;
                        }
                        RunCommand::Continue => {
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
                            if let Err(error) = upload_schedule(&self.registry, &device, run, schedule_revision, &track) {
                                break delivery_failure(error);
                            }
                            commit_sent = false;
                            committed_revision = None;
                            retained_evidence = false;
                            heartbeat_enabled = true;
                            playback = None;
                            playback_start_at = None;
                            continue;
                        }
                        RunCommand::Save => Frame::CalibrationSave { run },
                        RunCommand::Discard => Frame::CalibrationDiscard { run },
                    };
                    if let Err(error) = self.registry.send_bound_control(&device, frame) {
                        break delivery_failure(error);
                    }
                    if terminal_discard {
                        break SessionExit::Completed;
                    }
                }
            }
        };

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

    fn send(
        &self,
        binding: &GuidedSessionBinding,
        command: RunCommand,
    ) -> Result<(), CoordinatorError> {
        let active = self.active.lock().unwrap();
        active
            .as_ref()
            .filter(|active| &active.binding == binding)
            .ok_or(CoordinatorError::LeaseMismatch)?
            .commands
            .send(command)
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
            GuidedCalibrationSnapshot::Setup {
                tracks: self.tracks.iter().map(guided_track).collect(),
                selected_track_id: Some(track_id),
            },
        )?;
        Ok(())
    }
}

impl GuidedModeAdapter for CalibrationModeAdapter {
    fn pause_for_no_visible_views(&self, session: &GuidedSessionBinding) {
        let active = self.active.lock().unwrap();
        if let Some(active) = active.as_ref().filter(|active| &active.binding == session) {
            let _ = active.commands.send(RunCommand::Interrupt);
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
            GuidedSessionAction::StartCalibration if session.is_none() => {
                self.start(device.ok_or(CoordinatorError::DeviceRequired)?, &request)
            }
            GuidedSessionAction::PauseCalibration => self.send(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunCommand::Interrupt,
            ),
            GuidedSessionAction::ResumeCalibration => self.send(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunCommand::Resume,
            ),
            GuidedSessionAction::DiscardCalibration => self.send(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunCommand::Discard,
            ),
            GuidedSessionAction::SaveCalibration => self.send(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunCommand::Save,
            ),
            GuidedSessionAction::ContinueCalibration => self.send(
                session.ok_or(CoordinatorError::LeaseMismatch)?,
                RunCommand::Continue,
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

fn upload_schedule(
    registry: &Registry,
    device: &DeviceConnectionIdentity,
    run: CalibrationRunKey,
    revision: CalibrationScheduleRevision,
    track: &crate::collect::beatmap::CalibrationTrack,
) -> Result<(), ControlDeliveryError> {
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
    for (first_entry, entries) in track
        .entries
        .chunks(protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES)
        .enumerate()
    {
        registry.send_bound_control(
            device,
            Frame::CalibrationScheduleChunk {
                run,
                schedule_revision: revision,
                content_identity: track.content_identity.clone(),
                total_count,
                first_entry: (first_entry * protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES)
                    as u32,
                entries: entries.to_vec(),
            },
        )?;
    }
    Ok(())
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

fn between_songs_snapshot(
    track: &crate::collect::beatmap::CalibrationTrack,
    counts: &[protocol::CalibrationClassCounts],
    retained_evidence: bool,
    candidate_present: bool,
    validity: protocol::CalibrationCandidateValidity,
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
        candidate_available: candidate_present && validity.permits_activation(),
        continue_available: retained_evidence,
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
            cue_count: 100,
            content_identity: "test-content".into(),
            cue_shortfall: 0,
            entries: vec![protocol::CalibrationScheduleEntry {
                cue_id: protocol::CalibrationCueId::new(1).unwrap(),
                gesture: protocol::CalibrationGesture::WristPronation,
                modifier: protocol::CalibrationModifier::ThumbUp,
                track_offset: protocol::TrackMilliseconds::new(0),
                hold: protocol::DurationMilliseconds::new(1_500),
            }],
        }
    }

    #[tokio::test]
    async fn song_result_survives_candidate_not_yet_ready_for_continue() {
        let registry = Arc::new(Registry::new());
        let coordinator = GuidedSessionCoordinator::new();
        let adapter =
            CalibrationModeAdapter::new(registry.clone(), coordinator.clone(), vec![track()]);
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
        let _ = device.control_rx.recv().await;
        let _ = device.control_rx.recv().await;
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
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
                    validity: protocol::CalibrationCandidateValidity {
                        model_numerically_valid: false,
                        record_crc_valid: true,
                    },
                    candidate_present: false,
                },
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(
            coordinator.snapshot().calibration,
            Some(GuidedCalibrationSnapshot::BetweenSongs {
                continue_available: true,
                valid_reps: 3,
                invalid_reps: 1,
                ..
            })
        ));
    }

    #[test]
    fn complete_schedule_upload_uses_five_bounded_chunks_for_130_entries() {
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
        upload_schedule(
            &registry,
            &identity,
            CalibrationRunKey {
                session_id: CalibrationSessionId::new(1).unwrap(),
                run_id: CalibrationRunId::new(1).unwrap(),
            },
            CalibrationScheduleRevision::new(1).unwrap(),
            &track,
        )
        .unwrap();
        let mut sizes = Vec::new();
        while let Ok(frame) = device.control_rx.try_recv() {
            if let Frame::CalibrationScheduleChunk { entries, .. } = frame {
                sizes.push(entries.len());
            }
        }
        assert_eq!(sizes, vec![32, 32, 32, 32, 2]);
    }
}
