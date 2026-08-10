//! Reliable, bounded delivery for calibration control-plane events.
//!
//! Calibration advances durable/runtime state before its browser-visible event
//! is written.  The ordinary live stream is deliberately lossy, so these
//! events need their own ownership: retain one exact frame until CDC accepts
//! it, while replacing periodic progress with its newest value.  A new Probe
//! starts a new dashboard connection epoch and retires frames owned by the old
//! actor; replaying an old acknowledgement into a new actor is less safe than
//! dropping it.

use protocol::{
    CalibrationRunKey, CalibrationScheduleRevision,
    CalibrationScheduleUploadOperationAcknowledgement, Frame,
};
use std::collections::VecDeque;

const RELIABLE_CAPACITY: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UploadOperationIdentity {
    Begin(u32),
    Chunk { first_entry: u32, fingerprint: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReliableIdentity {
    Upload {
        run: CalibrationRunKey,
        revision: CalibrationScheduleRevision,
        operation: UploadOperationIdentity,
    },
    CommitDeferred(CalibrationRunKey, CalibrationScheduleRevision),
    ScheduleAccepted(CalibrationRunKey, CalibrationScheduleRevision),
    SongInterrupted(CalibrationRunKey, CalibrationScheduleRevision),
    SongResult(CalibrationRunKey, CalibrationScheduleRevision),
    CandidateStatus(CalibrationRunKey, CalibrationScheduleRevision),
    ResidentActivated(CalibrationRunKey, CalibrationScheduleRevision),
    /// Refusals have no run identity. Only the newest undelivered diagnostic
    /// is useful, and coalescing them prevents a malformed host from growing a
    /// second log queue through the calibration path.
    Diagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeriodicIdentity {
    Preparation(CalibrationRunKey, CalibrationScheduleRevision),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameClass {
    Reliable(ReliableIdentity),
    Periodic(PeriodicIdentity),
}

#[derive(Debug)]
pub(crate) struct Full {
    frame: Box<Frame>,
}

impl Full {
    pub(crate) fn into_frame(self) -> Frame {
        *self.frame
    }
}

/// Frames are bound to the connection generation current when they are first
/// queued. A write stall does not change that generation, so recovery can
/// retry. A fresh Probe does, so stale run authority is discarded atomically.
pub(crate) struct CalibrationOutbox {
    connection_generation: Option<u32>,
    reliable: VecDeque<(ReliableIdentity, Frame)>,
    latest_preparation: Option<(PeriodicIdentity, Frame)>,
}

impl CalibrationOutbox {
    pub(crate) fn new() -> Self {
        Self {
            connection_generation: None,
            reliable: VecDeque::with_capacity(RELIABLE_CAPACITY),
            latest_preparation: None,
        }
    }

    /// Reconcile before queueing or attempting delivery. Disconnection alone
    /// retains work for a same-epoch write recovery; only a new claimed
    /// generation proves the receiving actor changed.
    fn observe_connection(&mut self, generation: u32, connected: bool) {
        if !connected {
            return;
        }
        match self.connection_generation {
            None => self.connection_generation = Some(generation),
            Some(current) if current != generation => {
                self.reliable.clear();
                self.latest_preparation = None;
                self.connection_generation = Some(generation);
            }
            Some(_) => {}
        }
    }

    pub(crate) fn push(
        &mut self,
        generation: u32,
        connected: bool,
        frame: Frame,
    ) -> Result<(), Full> {
        // Epoch reconciliation is part of queueing, rather than a convention
        // imposed on the caller. In particular, a Probe and Begin can arrive
        // in the same serve pass: old frames must be retired *before* the new
        // Begin acknowledgement is installed.
        self.observe_connection(generation, connected);
        match classify(&frame) {
            Some(FrameClass::Periodic(identity)) => {
                self.latest_preparation = Some((identity, frame));
                Ok(())
            }
            Some(FrameClass::Reliable(identity)) => {
                if let Some((_, queued)) = self
                    .reliable
                    .iter_mut()
                    .find(|(queued_identity, _)| *queued_identity == identity)
                {
                    *queued = frame;
                    return Ok(());
                }
                if self.reliable.len() == RELIABLE_CAPACITY {
                    return Err(Full {
                        frame: Box::new(frame),
                    });
                }
                self.reliable.push_back((identity, frame));
                Ok(())
            }
            None => Err(Full {
                frame: Box::new(frame),
            }),
        }
    }

    /// Attempt exactly one frame. This makes a successful return an exact
    /// acknowledgement of the frame removed; a multi-frame send could write a
    /// prefix and report only a single aggregate failure.
    pub(crate) fn try_send_one(
        &mut self,
        generation: u32,
        connected: bool,
        mut send: impl FnMut(&Frame) -> bool,
    ) -> bool {
        self.observe_connection(generation, connected);
        if !connected {
            return false;
        }
        if let Some((_, frame)) = self.reliable.front() {
            if send(frame) {
                self.reliable.pop_front();
                return true;
            }
            return false;
        }
        if let Some((_, frame)) = self.latest_preparation.as_ref() {
            if send(frame) {
                self.latest_preparation = None;
                return true;
            }
        }
        false
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.reliable.len() + usize::from(self.latest_preparation.is_some())
    }
}

fn classify(frame: &Frame) -> Option<FrameClass> {
    let class = match frame {
        Frame::CalibrationScheduleUploadAcknowledged { acknowledgement } => {
            let operation = match acknowledgement.operation {
                CalibrationScheduleUploadOperationAcknowledgement::Begin {
                    operation_fingerprint,
                } => UploadOperationIdentity::Begin(operation_fingerprint),
                CalibrationScheduleUploadOperationAcknowledgement::Chunk {
                    first_entry,
                    operation_fingerprint,
                } => UploadOperationIdentity::Chunk {
                    first_entry,
                    fingerprint: operation_fingerprint,
                },
            };
            FrameClass::Reliable(ReliableIdentity::Upload {
                run: acknowledgement.run,
                revision: acknowledgement.schedule_revision,
                operation,
            })
        }
        Frame::CalibrationScheduleCommitDeferred { deferred } => FrameClass::Reliable(
            ReliableIdentity::CommitDeferred(deferred.run, deferred.schedule_revision),
        ),
        Frame::CalibrationPreparationStatus { status } => FrameClass::Periodic(
            PeriodicIdentity::Preparation(status.run, status.schedule_revision),
        ),
        Frame::CalibrationScheduleAccepted { accepted } => FrameClass::Reliable(
            ReliableIdentity::ScheduleAccepted(accepted.run, accepted.schedule_revision),
        ),
        Frame::CalibrationSongInterrupted { interruption } => FrameClass::Reliable(
            ReliableIdentity::SongInterrupted(interruption.run, interruption.schedule_revision),
        ),
        Frame::CalibrationSongResult { result } => FrameClass::Reliable(
            ReliableIdentity::SongResult(result.run, result.schedule_revision),
        ),
        Frame::CalibrationCandidateStatus { candidate } => FrameClass::Reliable(
            ReliableIdentity::CandidateStatus(candidate.run, candidate.schedule_revision),
        ),
        Frame::CalibrationResidentActivated { activation } => FrameClass::Reliable(
            ReliableIdentity::ResidentActivated(activation.run, activation.schedule_revision),
        ),
        Frame::BenchError {
            source: protocol::BenchErrorSource::Calibration,
            ..
        } => FrameClass::Reliable(ReliableIdentity::Diagnostic),
        _ => return None,
    };
    Some(class)
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{
        CalibrationPreparationPhase, CalibrationPreparationStatus, CalibrationRunId,
        CalibrationScheduleAccepted, CalibrationSessionId,
    };

    fn run(session: u64, run: u32) -> CalibrationRunKey {
        CalibrationRunKey {
            session_id: CalibrationSessionId::new(session).unwrap(),
            run_id: CalibrationRunId::new(run).unwrap(),
        }
    }

    fn revision(value: u32) -> CalibrationScheduleRevision {
        CalibrationScheduleRevision::new(value).unwrap()
    }

    fn accepted(run: CalibrationRunKey, revision: CalibrationScheduleRevision) -> Frame {
        Frame::CalibrationScheduleAccepted {
            accepted: CalibrationScheduleAccepted {
                run,
                schedule_revision: revision,
                content_identity: "song".into(),
                acknowledged_device_monotonic_microseconds: 10,
                anchor_device_monotonic_microseconds: 3_000_010,
                acquisition_sample: 20,
            },
        }
    }

    fn preparation(
        run: CalibrationRunKey,
        revision: CalibrationScheduleRevision,
        elapsed_milliseconds: u32,
    ) -> Frame {
        Frame::CalibrationPreparationStatus {
            status: CalibrationPreparationStatus {
                run,
                schedule_revision: revision,
                phase: CalibrationPreparationPhase::Settling {
                    elapsed_milliseconds,
                    remaining_milliseconds: 10_000 - elapsed_milliseconds,
                },
            },
        }
    }

    fn is_accepted_for(
        frame: &Frame,
        expected_run: CalibrationRunKey,
        expected_revision: CalibrationScheduleRevision,
    ) -> bool {
        matches!(
            frame,
            Frame::CalibrationScheduleAccepted { accepted }
                if accepted.run == expected_run
                    && accepted.schedule_revision == expected_revision
        )
    }

    fn preparation_elapsed(frame: &Frame) -> Option<u32> {
        match frame {
            Frame::CalibrationPreparationStatus {
                status:
                    CalibrationPreparationStatus {
                        phase:
                            CalibrationPreparationPhase::Settling {
                                elapsed_milliseconds,
                                ..
                            },
                        ..
                    },
            } => Some(*elapsed_milliseconds),
            _ => None,
        }
    }

    #[test]
    fn failed_send_retains_exact_terminal_frame_until_recovery() {
        let key = run(1, 2);
        let rev = revision(3);
        let mut outbox = CalibrationOutbox::new();
        outbox.push(7, true, accepted(key, rev)).unwrap();

        assert!(!outbox.try_send_one(7, true, |frame| {
            assert!(is_accepted_for(frame, key, rev));
            false
        }));
        assert_eq!(outbox.len(), 1);
        assert!(outbox.try_send_one(7, true, |frame| {
            assert!(is_accepted_for(frame, key, rev));
            true
        }));
        assert_eq!(outbox.len(), 0);
    }

    #[test]
    fn fresh_connection_epoch_drops_old_actor_authority() {
        let old = accepted(run(1, 1), revision(1));
        let new_run = run(2, 1);
        let new_revision = revision(1);
        let mut outbox = CalibrationOutbox::new();
        outbox.push(9, true, old).unwrap();
        assert!(!outbox.try_send_one(9, true, |_| false));

        outbox
            .push(10, true, accepted(new_run, new_revision))
            .unwrap();
        assert_eq!(outbox.len(), 1);
        assert!(outbox.try_send_one(10, true, |frame| is_accepted_for(
            frame,
            new_run,
            new_revision
        )));
    }

    #[test]
    fn disconnection_without_new_epoch_preserves_pending_delivery() {
        let key = run(1, 1);
        let rev = revision(1);
        let mut outbox = CalibrationOutbox::new();
        outbox.push(4, true, accepted(key, rev)).unwrap();

        assert!(!outbox.try_send_one(4, false, |_| true));
        assert!(outbox.try_send_one(4, true, |frame| is_accepted_for(frame, key, rev)));
    }

    #[test]
    fn periodic_preparation_is_coalesced_to_newest_observation() {
        let key = run(1, 1);
        let rev = revision(1);
        let first = preparation(key, rev, 100);
        let mut outbox = CalibrationOutbox::new();
        outbox.push(1, true, first).unwrap();
        outbox.push(1, true, preparation(key, rev, 600)).unwrap();

        assert_eq!(outbox.len(), 1);
        assert!(outbox.try_send_one(1, true, |frame| preparation_elapsed(frame) == Some(600)));
    }

    #[test]
    fn reliable_terminal_precedes_periodic_progress() {
        let key = run(1, 1);
        let rev = revision(1);
        let progress = preparation(key, rev, 100);
        let mut outbox = CalibrationOutbox::new();
        outbox.push(1, true, progress).unwrap();
        outbox.push(1, true, accepted(key, rev)).unwrap();

        assert!(outbox.try_send_one(1, true, |frame| is_accepted_for(frame, key, rev)));
        assert!(outbox.try_send_one(1, true, |frame| preparation_elapsed(frame) == Some(100)));
    }
}
