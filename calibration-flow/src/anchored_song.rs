//! Device-owned execution state for one committed calibration song.
//!
//! This deliberately models the schedule boundary rather than a transport
//! frame.  The protocol contract is still changing, but firmware needs one
//! deterministic place that proves an uploaded song is complete before it can
//! become runnable.  The adapter supplies identities and device-clock values;
//! this module never reads a clock or sends a frame.

use alloc::{string::String, vec::Vec};
use core::fmt;
use protocol::{
    CalibrationRunKey, CalibrationScheduleEntry, CalibrationScheduleRevision, RepRejection,
};
pub use protocol::{
    CALIBRATION_CUE_HOLD_MILLISECONDS as REQUIRED_CUE_HOLD_MILLISECONDS,
    CALIBRATION_CUE_RECOVERY_MILLISECONDS as REQUIRED_CUE_RECOVERY_MILLISECONDS,
};

use crate::RepEvidence;

/// The confirmed maximum at the upload boundary.  This is intentionally local
/// while the protocol crate's cancelled partial implementation still says 16.
pub const MAX_ANCHORED_SONG_CHUNK_CUES: usize = 32;
/// A complete upload is bounded before either of its two exact-reserve calls.
/// The shipped recipe needs 130 cues; 256 leaves room for alternate authored
/// songs while keeping malformed input from consuming the device heap or
/// flooding the 24-frame reliable acknowledgement outbox.
pub const MAX_ANCHORED_SONG_CUES: u32 = 256;
pub const HEARTBEAT_INTERVAL_MICROSECONDS: u64 = 500_000;
pub const HEARTBEAT_TIMEOUT_MICROSECONDS: u64 = 2_000_000;
pub const SONG_ANCHOR_LEAD_MICROSECONDS: u64 = 3_000_000;

/// Identity which must agree on every upload and heartbeat operation.
///
/// `content_identity` is a domain value rather than a second wire type.  The
/// replacement protocol will carry its existing track content identity here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredSongIdentity {
    pub run: CalibrationRunKey,
    pub revision: CalibrationScheduleRevision,
    pub content_identity: String,
    pub total_count: u32,
}

impl AnchoredSongIdentity {
    pub fn new(
        run: CalibrationRunKey,
        revision: CalibrationScheduleRevision,
        content_identity: String,
        total_count: u32,
    ) -> Result<Self, AnchoredSongError> {
        let identity = Self {
            run,
            revision,
            content_identity,
            total_count,
        };
        identity.validate()?;
        Ok(identity)
    }

    fn validate(&self) -> Result<(), AnchoredSongError> {
        if self.content_identity.is_empty() {
            return Err(AnchoredSongError::EmptyContentIdentity);
        }
        if self.total_count == 0 {
            return Err(AnchoredSongError::EmptySchedule);
        }
        if self.total_count > MAX_ANCHORED_SONG_CUES {
            return Err(AnchoredSongError::TooManyCues {
                limit: MAX_ANCHORED_SONG_CUES,
                received: self.total_count,
            });
        }
        Ok(())
    }
}

/// The device coordinate system the dashboard maps onto its corrected clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SongAnchor {
    pub acknowledged_device_monotonic_microseconds: u64,
    pub device_monotonic_microseconds: u64,
    pub acquisition_sample: u64,
}

impl SongAnchor {
    pub fn three_seconds_after(
        acknowledged_device_monotonic_microseconds: u64,
        acquisition_sample: u64,
    ) -> Result<Self, AnchoredSongError> {
        let device_monotonic_microseconds = acknowledged_device_monotonic_microseconds
            .checked_add(SONG_ANCHOR_LEAD_MICROSECONDS)
            .ok_or(AnchoredSongError::AnchorOverflow)?;
        Ok(Self {
            acknowledged_device_monotonic_microseconds,
            device_monotonic_microseconds,
            acquisition_sample,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadEffect {
    Applied,
    Duplicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SongInterruption {
    HeartbeatTimedOut,
    OperatorStopped,
    DeviceLinkLost,
}

/// The only action an adapter needs to carry into firmware feedback, label
/// collection, and fitting.  A cue's authored device instant is always carried
/// with the action; a delayed transport cannot choose a new label window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchoredSongAction {
    OpenCue {
        entry: CalibrationScheduleEntry,
        device_monotonic_microseconds: u64,
    },
    CloseCue {
        entry: CalibrationScheduleEntry,
        device_monotonic_microseconds: u64,
    },
    Interrupted {
        reason: SongInterruption,
        rejected_open_cue: Option<CalibrationScheduleEntry>,
    },
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SongState {
    AwaitingUpload,
    ReceivingUpload,
    Anchored,
    CueOpen,
    AwaitingCueEvidence,
    Interrupted,
    Completed,
}

/// Evidence and bounded-fit work survive an interrupted song and are never
/// reset by the next upload.  The storage/fitter owns the actual rows and
/// checkpoints; this state makes their retention observable to the adapter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetainedSongProgress {
    pub accepted_cues: u32,
    pub rejected_cues: u32,
    pub accepted_rows: u32,
    pub completed_fit_checkpoints: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnchoredSongError {
    EmptyContentIdentity,
    EmptySchedule,
    TooManyCues {
        limit: u32,
        received: u32,
    },
    /// The device could not reserve its bounded upload transaction before
    /// accepting it.  This is a protocol refusal, never an allocator panic or
    /// watchdog-reset halfway through a calibration.
    UploadAllocationFailed,
    WrongRun {
        expected: CalibrationRunKey,
        received: CalibrationRunKey,
    },
    WrongRevision {
        expected: CalibrationScheduleRevision,
        received: CalibrationScheduleRevision,
    },
    WrongContentIdentity,
    WrongTotalCount {
        expected: u32,
        received: u32,
    },
    RevisionDidNotAdvance {
        previous: CalibrationScheduleRevision,
        received: CalibrationScheduleRevision,
    },
    UploadAlreadyInProgress,
    NoUploadInProgress,
    ChunkTooLarge {
        limit: usize,
        received: usize,
    },
    EmptyChunk,
    ChunkOutOfOrder {
        expected: u32,
        received: u32,
    },
    ConflictingDuplicateChunk {
        first_entry: u32,
    },
    TotalCountExceeded {
        total_count: u32,
    },
    IncompleteSchedule {
        expected: u32,
        received: u32,
    },
    CueIdentifiersNotStrictlyIncreasing,
    CueOffsetsNotStrictlyIncreasing,
    WrongCueHold {
        expected: u32,
        received: u32,
    },
    RecoveryTooShort {
        required: u32,
        actual: u32,
    },
    TimestampOverflow,
    AnchorOverflow,
    ScheduleNotAnchored,
    HeartbeatExpired,
    NoOpenCue,
    CueStillOpen,
    NoClosedCue,
    SongNotRunnable,
    SongMustBeReuploaded,
}

impl fmt::Display for AnchoredSongError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl core::error::Error for AnchoredSongError {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReceivedChunk {
    first_entry: u32,
    entries: Vec<CalibrationScheduleEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingUpload {
    identity: AnchoredSongIdentity,
    entries: Vec<CalibrationScheduleEntry>,
    received_chunks: Vec<ReceivedChunk>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenCue {
    entry: CalibrationScheduleEntry,
    opens_at_microseconds: u64,
    closes_at_microseconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RunnableSong {
    identity: AnchoredSongIdentity,
    entries: Vec<CalibrationScheduleEntry>,
    anchor: SongAnchor,
    committed_at_microseconds: u64,
    last_heartbeat_microseconds: Option<u64>,
    cue: CueLifecycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CueLifecycle {
    Waiting { next_entry: usize },
    Open { next_entry: usize, cue: OpenCue },
    AwaitingEvidence { next_entry: usize, cue: OpenCue },
}

/// A song has exactly one lifecycle owner.  Upload and runnable storage cannot
/// coexist, and a public `SongState` is derived from this value instead of
/// being a second mutable account of the same transition.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SongLifecycle {
    AwaitingUpload,
    ReceivingUpload(PendingUpload),
    Running(RunnableSong),
    Interrupted(RunnableSong),
    Completed(RunnableSong),
}

/// Pure state machine for complete device-anchored songs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredSong {
    run: CalibrationRunKey,
    previous_revision: Option<CalibrationScheduleRevision>,
    lifecycle: SongLifecycle,
    retained: RetainedSongProgress,
}

impl AnchoredSong {
    pub const fn new(run: CalibrationRunKey) -> Self {
        Self {
            run,
            previous_revision: None,
            lifecycle: SongLifecycle::AwaitingUpload,
            retained: RetainedSongProgress {
                accepted_cues: 0,
                rejected_cues: 0,
                accepted_rows: 0,
                completed_fit_checkpoints: 0,
            },
        }
    }

    pub const fn state(&self) -> SongState {
        match &self.lifecycle {
            SongLifecycle::AwaitingUpload => SongState::AwaitingUpload,
            SongLifecycle::ReceivingUpload(_) => SongState::ReceivingUpload,
            SongLifecycle::Running(song) => match song.cue {
                CueLifecycle::Waiting { .. } => SongState::Anchored,
                CueLifecycle::Open { .. } => SongState::CueOpen,
                CueLifecycle::AwaitingEvidence { .. } => SongState::AwaitingCueEvidence,
            },
            SongLifecycle::Interrupted(_) => SongState::Interrupted,
            SongLifecycle::Completed(_) => SongState::Completed,
        }
    }

    pub const fn retained_progress(&self) -> RetainedSongProgress {
        self.retained
    }

    pub fn identity(&self) -> Option<&AnchoredSongIdentity> {
        match &self.lifecycle {
            SongLifecycle::AwaitingUpload => None,
            SongLifecycle::ReceivingUpload(upload) => Some(&upload.identity),
            SongLifecycle::Running(song)
            | SongLifecycle::Interrupted(song)
            | SongLifecycle::Completed(song) => Some(&song.identity),
        }
    }

    /// The immutable device/acquisition coordinate of a committed song.
    /// Upload phases have no anchor; interruption and completion retain it so
    /// evidence and diagnostics keep the exact authored coordinate.
    pub const fn anchor(&self) -> Option<SongAnchor> {
        match &self.lifecycle {
            SongLifecycle::Running(song)
            | SongLifecycle::Interrupted(song)
            | SongLifecycle::Completed(song) => Some(song.anchor),
            SongLifecycle::AwaitingUpload | SongLifecycle::ReceivingUpload(_) => None,
        }
    }

    /// Starts an isolated upload.  The next song in the same run must use a
    /// later revision, which makes Continue require a fresh upload and anchor.
    pub fn begin_upload(
        &mut self,
        identity: AnchoredSongIdentity,
    ) -> Result<(), AnchoredSongError> {
        identity.validate()?;
        self.check_run(identity.run)?;
        match self.lifecycle {
            SongLifecycle::ReceivingUpload(_) => {
                return Err(AnchoredSongError::UploadAlreadyInProgress);
            }
            SongLifecycle::Running(_) => return Err(AnchoredSongError::SongNotRunnable),
            SongLifecycle::AwaitingUpload
            | SongLifecycle::Interrupted(_)
            | SongLifecycle::Completed(_) => {}
        }
        if let Some(previous) = self.previous_revision {
            if identity.revision <= previous {
                return Err(AnchoredSongError::RevisionDidNotAdvance {
                    previous,
                    received: identity.revision,
                });
            }
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(identity.total_count as usize)
            .map_err(|_| AnchoredSongError::UploadAllocationFailed)?;
        let mut received_chunks = Vec::new();
        let chunk_count = (identity.total_count as usize)
            .saturating_add(MAX_ANCHORED_SONG_CHUNK_CUES - 1)
            / MAX_ANCHORED_SONG_CHUNK_CUES;
        received_chunks
            .try_reserve_exact(chunk_count)
            .map_err(|_| AnchoredSongError::UploadAllocationFailed)?;
        self.lifecycle = SongLifecycle::ReceivingUpload(PendingUpload {
            identity,
            entries,
            received_chunks,
        });
        Ok(())
    }

    /// Accepts one complete bounded chunk.  Duplicate transmission of a
    /// previously accepted whole chunk is idempotent; overlap is not silently
    /// merged because that could change a label without moving an index.
    pub fn upload_chunk(
        &mut self,
        identity: &AnchoredSongIdentity,
        first_entry: u32,
        entries: &[CalibrationScheduleEntry],
    ) -> Result<UploadEffect, AnchoredSongError> {
        let SongLifecycle::ReceivingUpload(pending) = &mut self.lifecycle else {
            return Err(AnchoredSongError::NoUploadInProgress);
        };
        check_identity(&pending.identity, identity)?;
        if entries.is_empty() {
            return Err(AnchoredSongError::EmptyChunk);
        }
        if entries.len() > MAX_ANCHORED_SONG_CHUNK_CUES {
            return Err(AnchoredSongError::ChunkTooLarge {
                limit: MAX_ANCHORED_SONG_CHUNK_CUES,
                received: entries.len(),
            });
        }
        let expected = pending.entries.len() as u32;
        if first_entry < expected {
            let duplicate = pending
                .received_chunks
                .iter()
                .find(|chunk| chunk.first_entry == first_entry);
            return match duplicate {
                Some(chunk) if chunk.entries.as_slice() == entries => Ok(UploadEffect::Duplicate),
                _ => Err(AnchoredSongError::ConflictingDuplicateChunk { first_entry }),
            };
        }
        if first_entry != expected {
            return Err(AnchoredSongError::ChunkOutOfOrder {
                expected,
                received: first_entry,
            });
        }
        let new_count = expected.checked_add(entries.len() as u32).ok_or(
            AnchoredSongError::TotalCountExceeded {
                total_count: pending.identity.total_count,
            },
        )?;
        if new_count > pending.identity.total_count {
            return Err(AnchoredSongError::TotalCountExceeded {
                total_count: pending.identity.total_count,
            });
        }
        validate_appended_entries(pending.entries.last().copied(), entries)?;
        let mut duplicate_entries = Vec::new();
        duplicate_entries
            .try_reserve_exact(entries.len())
            .map_err(|_| AnchoredSongError::UploadAllocationFailed)?;
        duplicate_entries.extend_from_slice(entries);
        pending.entries.extend_from_slice(entries);
        pending.received_chunks.push(ReceivedChunk {
            first_entry,
            entries: duplicate_entries,
        });
        Ok(UploadEffect::Applied)
    }

    /// Atomically makes the complete song runnable and returns its immutable
    /// device/acquisition anchor.  A partial upload remains non-runnable.
    pub fn commit(
        &mut self,
        identity: &AnchoredSongIdentity,
        acknowledged_device_monotonic_microseconds: u64,
        acquisition_sample: u64,
    ) -> Result<SongAnchor, AnchoredSongError> {
        let SongLifecycle::ReceivingUpload(pending) = &self.lifecycle else {
            return Err(AnchoredSongError::NoUploadInProgress);
        };
        check_identity(&pending.identity, identity)?;
        if pending.entries.len() as u32 != pending.identity.total_count {
            return Err(AnchoredSongError::IncompleteSchedule {
                expected: pending.identity.total_count,
                received: pending.entries.len() as u32,
            });
        }
        let anchor = SongAnchor::three_seconds_after(
            acknowledged_device_monotonic_microseconds,
            acquisition_sample,
        )?;
        let previous = core::mem::replace(&mut self.lifecycle, SongLifecycle::AwaitingUpload);
        let SongLifecycle::ReceivingUpload(pending) = previous else {
            self.lifecycle = previous;
            return Err(AnchoredSongError::NoUploadInProgress);
        };
        self.previous_revision = Some(pending.identity.revision);
        self.lifecycle = SongLifecycle::Running(RunnableSong {
            identity: pending.identity,
            entries: pending.entries,
            anchor,
            committed_at_microseconds: acknowledged_device_monotonic_microseconds,
            last_heartbeat_microseconds: None,
            cue: CueLifecycle::Waiting { next_entry: 0 },
        });
        Ok(anchor)
    }

    /// Records an inbound host heartbeat.  The adapter supplies the device
    /// clock instant at which it arrived; protocol sequencing remains outside
    /// this domain engine.
    pub fn heartbeat(
        &mut self,
        identity: &AnchoredSongIdentity,
        received_device_monotonic_microseconds: u64,
    ) -> Result<(), AnchoredSongError> {
        let song = match &mut self.lifecycle {
            SongLifecycle::Running(song) => song,
            SongLifecycle::Interrupted(_) | SongLifecycle::Completed(_) => {
                return Err(AnchoredSongError::SongMustBeReuploaded);
            }
            SongLifecycle::AwaitingUpload | SongLifecycle::ReceivingUpload(_) => {
                return Err(AnchoredSongError::ScheduleNotAnchored);
            }
        };
        check_identity(&song.identity, identity)?;
        song.last_heartbeat_microseconds = Some(received_device_monotonic_microseconds);
        Ok(())
    }

    /// Advances using the device monotonic clock.  Calling it after a heartbeat
    /// deadline interrupts exactly the currently open cue, if any.
    pub fn poll(
        &mut self,
        now_device_monotonic_microseconds: u64,
    ) -> Result<Option<AnchoredSongAction>, AnchoredSongError> {
        let song = match &mut self.lifecycle {
            SongLifecycle::Running(song) => song,
            SongLifecycle::Interrupted(_) | SongLifecycle::Completed(_) => return Ok(None),
            SongLifecycle::AwaitingUpload | SongLifecycle::ReceivingUpload(_) => {
                return Err(AnchoredSongError::ScheduleNotAnchored);
            }
        };
        let heartbeat_from = song
            .last_heartbeat_microseconds
            .unwrap_or(song.committed_at_microseconds);
        let heartbeat_deadline = heartbeat_from.saturating_add(HEARTBEAT_TIMEOUT_MICROSECONDS);
        // A close that was authored before the liveness deadline remains an
        // ordinary completed cue even when this poll arrives late.  Otherwise
        // a delayed service loop could incorrectly reject a cue that the
        // device-owned clock had already closed.
        if let CueLifecycle::Open {
            next_entry,
            cue: open,
        } = song.cue
        {
            if now_device_monotonic_microseconds >= open.closes_at_microseconds
                && open.closes_at_microseconds <= heartbeat_deadline
            {
                song.cue = CueLifecycle::AwaitingEvidence {
                    next_entry,
                    cue: open,
                };
                return Ok(Some(AnchoredSongAction::CloseCue {
                    entry: open.entry,
                    device_monotonic_microseconds: open.closes_at_microseconds,
                }));
            }
        }
        if now_device_monotonic_microseconds >= heartbeat_deadline {
            let rejected_open_cue = match song.cue {
                CueLifecycle::Open { cue, .. } => Some(cue.entry),
                CueLifecycle::Waiting { .. } | CueLifecycle::AwaitingEvidence { .. } => None,
            };
            if rejected_open_cue.is_some() {
                self.retained.rejected_cues += 1;
            }
            let previous = core::mem::replace(&mut self.lifecycle, SongLifecycle::AwaitingUpload);
            let SongLifecycle::Running(song) = previous else {
                self.lifecycle = previous;
                return Err(AnchoredSongError::ScheduleNotAnchored);
            };
            self.lifecycle = SongLifecycle::Interrupted(song);
            return Ok(Some(AnchoredSongAction::Interrupted {
                reason: SongInterruption::HeartbeatTimedOut,
                rejected_open_cue,
            }));
        }
        if let CueLifecycle::Open {
            next_entry,
            cue: open,
        } = song.cue
        {
            if now_device_monotonic_microseconds >= open.closes_at_microseconds {
                song.cue = CueLifecycle::AwaitingEvidence {
                    next_entry,
                    cue: open,
                };
                return Ok(Some(AnchoredSongAction::CloseCue {
                    entry: open.entry,
                    device_monotonic_microseconds: open.closes_at_microseconds,
                }));
            }
            return Ok(None);
        }
        let CueLifecycle::Waiting { next_entry } = song.cue else {
            return Ok(None);
        };
        let Some(entry) = song.entries.get(next_entry).copied() else {
            let previous = core::mem::replace(&mut self.lifecycle, SongLifecycle::AwaitingUpload);
            let SongLifecycle::Running(song) = previous else {
                self.lifecycle = previous;
                return Err(AnchoredSongError::ScheduleNotAnchored);
            };
            self.lifecycle = SongLifecycle::Completed(song);
            return Ok(Some(AnchoredSongAction::Completed));
        };
        let opens_at_microseconds = cue_instant(song.anchor, entry)?;
        if now_device_monotonic_microseconds < opens_at_microseconds {
            return Ok(None);
        }
        let closes_at_microseconds = opens_at_microseconds
            .checked_add(u64::from(entry.hold.get()) * 1_000)
            .ok_or(AnchoredSongError::TimestampOverflow)?;
        song.cue = CueLifecycle::Open {
            next_entry,
            cue: OpenCue {
                entry,
                opens_at_microseconds,
                closes_at_microseconds,
            },
        };
        Ok(Some(AnchoredSongAction::OpenCue {
            entry,
            device_monotonic_microseconds: opens_at_microseconds,
        }))
    }

    /// Resolves evidence only after its cue closed.  This does not modify the
    /// authored schedule; rejected evidence simply advances to the next
    /// authored cue, and Continue uploads another complete song for deficits.
    pub fn record_closed_evidence(
        &mut self,
        evidence: RepEvidence,
        accepted_rows: u32,
    ) -> Result<Result<(), RepRejection>, AnchoredSongError> {
        let song = match &mut self.lifecycle {
            SongLifecycle::Running(song) | SongLifecycle::Interrupted(song) => song,
            SongLifecycle::Completed(_) => {
                return Err(AnchoredSongError::SongMustBeReuploaded);
            }
            SongLifecycle::AwaitingUpload | SongLifecycle::ReceivingUpload(_) => {
                return Err(AnchoredSongError::ScheduleNotAnchored);
            }
        };
        let (next_entry, _closed) = match song.cue {
            CueLifecycle::AwaitingEvidence { next_entry, cue } => (next_entry, cue),
            CueLifecycle::Open { .. } => return Err(AnchoredSongError::CueStillOpen),
            CueLifecycle::Waiting { .. } => return Err(AnchoredSongError::NoClosedCue),
        };
        song.cue = CueLifecycle::Waiting {
            next_entry: next_entry + 1,
        };
        match evidence.rejection() {
            Some(reason) => {
                self.retained.rejected_cues += 1;
                Ok(Err(reason))
            }
            None => {
                self.retained.accepted_cues += 1;
                self.retained.accepted_rows += accepted_rows;
                Ok(Ok(()))
            }
        }
    }

    /// Notes a completed bounded-fit checkpoint.  It deliberately remains
    /// valid after interruption and through the next Continue upload.
    pub fn fit_checkpoint_completed(&mut self) {
        self.retained.completed_fit_checkpoints += 1;
    }

    pub fn interrupt(
        &mut self,
        reason: SongInterruption,
    ) -> Result<AnchoredSongAction, AnchoredSongError> {
        let rejected_open_cue = match &self.lifecycle {
            SongLifecycle::Running(song) => match song.cue {
                CueLifecycle::Open { cue, .. } => Some(cue.entry),
                CueLifecycle::Waiting { .. } | CueLifecycle::AwaitingEvidence { .. } => None,
            },
            SongLifecycle::Interrupted(_) | SongLifecycle::Completed(_) => {
                return Err(AnchoredSongError::SongMustBeReuploaded);
            }
            SongLifecycle::AwaitingUpload | SongLifecycle::ReceivingUpload(_) => {
                return Err(AnchoredSongError::ScheduleNotAnchored);
            }
        };
        if rejected_open_cue.is_some() {
            self.retained.rejected_cues += 1;
        }
        let previous = core::mem::replace(&mut self.lifecycle, SongLifecycle::AwaitingUpload);
        let SongLifecycle::Running(song) = previous else {
            self.lifecycle = previous;
            return Err(AnchoredSongError::SongMustBeReuploaded);
        };
        self.lifecycle = SongLifecycle::Interrupted(song);
        Ok(AnchoredSongAction::Interrupted {
            reason,
            rejected_open_cue,
        })
    }

    fn check_run(&self, received: CalibrationRunKey) -> Result<(), AnchoredSongError> {
        if self.run == received {
            Ok(())
        } else {
            Err(AnchoredSongError::WrongRun {
                expected: self.run,
                received,
            })
        }
    }
}

fn check_identity(
    expected: &AnchoredSongIdentity,
    received: &AnchoredSongIdentity,
) -> Result<(), AnchoredSongError> {
    if expected.run != received.run {
        return Err(AnchoredSongError::WrongRun {
            expected: expected.run,
            received: received.run,
        });
    }
    if expected.revision != received.revision {
        return Err(AnchoredSongError::WrongRevision {
            expected: expected.revision,
            received: received.revision,
        });
    }
    if expected.content_identity != received.content_identity {
        return Err(AnchoredSongError::WrongContentIdentity);
    }
    if expected.total_count != received.total_count {
        return Err(AnchoredSongError::WrongTotalCount {
            expected: expected.total_count,
            received: received.total_count,
        });
    }
    Ok(())
}

fn validate_appended_entries(
    previous: Option<CalibrationScheduleEntry>,
    entries: &[CalibrationScheduleEntry],
) -> Result<(), AnchoredSongError> {
    let mut previous = previous;
    for entry in entries {
        if entry.hold.get() != REQUIRED_CUE_HOLD_MILLISECONDS {
            return Err(AnchoredSongError::WrongCueHold {
                expected: REQUIRED_CUE_HOLD_MILLISECONDS,
                received: entry.hold.get(),
            });
        }
        if let Some(before) = previous {
            if entry.cue_id <= before.cue_id {
                return Err(AnchoredSongError::CueIdentifiersNotStrictlyIncreasing);
            }
            if entry.track_offset <= before.track_offset {
                return Err(AnchoredSongError::CueOffsetsNotStrictlyIncreasing);
            }
            let required = before
                .track_offset
                .get()
                .checked_add(before.hold.get())
                .and_then(|value| value.checked_add(REQUIRED_CUE_RECOVERY_MILLISECONDS))
                .ok_or(AnchoredSongError::TimestampOverflow)?;
            if entry.track_offset.get() < required {
                return Err(AnchoredSongError::RecoveryTooShort {
                    required: REQUIRED_CUE_RECOVERY_MILLISECONDS,
                    actual: entry.track_offset.get().saturating_sub(
                        before.track_offset.get().saturating_add(before.hold.get()),
                    ),
                });
            }
        }
        previous = Some(*entry);
    }
    Ok(())
}

fn cue_instant(
    anchor: SongAnchor,
    entry: CalibrationScheduleEntry,
) -> Result<u64, AnchoredSongError> {
    anchor
        .device_monotonic_microseconds
        .checked_add(u64::from(entry.track_offset.get()) * 1_000)
        .ok_or(AnchoredSongError::TimestampOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use protocol::{
        CalibrationCueId, CalibrationGesture, CalibrationModifier, CalibrationRunId,
        CalibrationSessionId, DurationMilliseconds, TrackMilliseconds,
    };

    fn run() -> CalibrationRunKey {
        CalibrationRunKey {
            session_id: CalibrationSessionId::new(8).unwrap(),
            run_id: CalibrationRunId::new(3).unwrap(),
        }
    }

    fn identity(revision: u32, count: u32) -> AnchoredSongIdentity {
        AnchoredSongIdentity::new(
            run(),
            CalibrationScheduleRevision::new(revision).unwrap(),
            "track-sha256".to_string(),
            count,
        )
        .unwrap()
    }

    fn cue(id: u32, at: u32) -> CalibrationScheduleEntry {
        CalibrationScheduleEntry {
            cue_id: CalibrationCueId::new(id).unwrap(),
            gesture: CalibrationGesture::WristPronation,
            modifier: CalibrationModifier::ThumbUp,
            track_offset: TrackMilliseconds::new(at),
            hold: DurationMilliseconds::new(REQUIRED_CUE_HOLD_MILLISECONDS),
        }
    }

    fn song_with_two_cues() -> (AnchoredSong, AnchoredSongIdentity, SongAnchor) {
        let identity = identity(1, 2);
        let mut song = AnchoredSong::new(run());
        song.begin_upload(identity.clone()).unwrap();
        song.upload_chunk(&identity, 0, &[cue(1, 0), cue(2, 2_000)])
            .unwrap();
        let anchor = song.commit(&identity, 10_000, 77).unwrap();
        (song, identity, anchor)
    }

    #[test]
    fn complete_song_allocation_is_bounded_before_upload_begins() {
        let error = AnchoredSongIdentity::new(
            run(),
            CalibrationScheduleRevision::new(1).unwrap(),
            "oversized".to_string(),
            MAX_ANCHORED_SONG_CUES + 1,
        )
        .unwrap_err();
        assert_eq!(
            error,
            AnchoredSongError::TooManyCues {
                limit: MAX_ANCHORED_SONG_CUES,
                received: MAX_ANCHORED_SONG_CUES + 1,
            }
        );
    }

    #[test]
    fn chunks_are_bounded_ordered_and_exactly_idempotent() {
        let identity = identity(1, 33);
        let mut song = AnchoredSong::new(run());
        song.begin_upload(identity.clone()).unwrap();
        let first: Vec<_> = (1..=32).map(|id| cue(id, (id - 1) * 2_000)).collect();
        assert_eq!(
            song.upload_chunk(&identity, 0, &first),
            Ok(UploadEffect::Applied)
        );
        assert_eq!(
            song.upload_chunk(&identity, 0, &first),
            Ok(UploadEffect::Duplicate)
        );
        let mut changed = first.clone();
        changed[0].gesture = CalibrationGesture::ThumbExtension;
        assert_eq!(
            song.upload_chunk(&identity, 0, &changed),
            Err(AnchoredSongError::ConflictingDuplicateChunk { first_entry: 0 })
        );
        assert_eq!(
            song.upload_chunk(&identity, 33, &[cue(33, 64_000)]),
            Err(AnchoredSongError::ChunkOutOfOrder {
                expected: 32,
                received: 33
            })
        );
        assert_eq!(
            song.upload_chunk(&identity, 32, &[cue(33, 64_000)]),
            Ok(UploadEffect::Applied)
        );
    }

    #[test]
    fn schedule_validation_rejects_hold_order_and_recovery_faults_without_publishing() {
        let identity = identity(1, 2);
        let mut song = AnchoredSong::new(run());
        song.begin_upload(identity.clone()).unwrap();
        let mut wrong_hold = cue(1, 0);
        wrong_hold.hold = DurationMilliseconds::new(1_499);
        assert!(matches!(
            song.upload_chunk(&identity, 0, &[wrong_hold]),
            Err(AnchoredSongError::WrongCueHold { .. })
        ));
        assert!(matches!(
            song.upload_chunk(&identity, 0, &[cue(2, 0), cue(1, 2_000)]),
            Err(AnchoredSongError::CueIdentifiersNotStrictlyIncreasing)
        ));
        assert!(matches!(
            song.upload_chunk(&identity, 0, &[cue(1, 0), cue(2, 1_999)]),
            Err(AnchoredSongError::RecoveryTooShort { actual: 499, .. })
        ));
        assert_eq!(song.state(), SongState::ReceivingUpload);
        assert!(matches!(
            song.commit(&identity, 1, 1),
            Err(AnchoredSongError::IncompleteSchedule { received: 0, .. })
        ));
    }

    #[test]
    fn anchor_is_atomic_three_seconds_ahead_and_carries_acquisition_coordinate() {
        let (song, identity, anchor) = song_with_two_cues();
        assert_eq!(song.state(), SongState::Anchored);
        assert_eq!(anchor.device_monotonic_microseconds, 3_010_000);
        assert_eq!(anchor.acquisition_sample, 77);
        assert_eq!(song.identity(), Some(&identity));
        assert_eq!(song.anchor(), Some(anchor));
    }

    #[test]
    fn an_arbitrary_nonempty_short_song_completes_after_a_rejected_cue() {
        let identity = identity(1, 1);
        let mut song = AnchoredSong::new(run());
        song.begin_upload(identity.clone()).unwrap();
        song.upload_chunk(&identity, 0, &[cue(1, 0)]).unwrap();
        let anchor = song.commit(&identity, 10_000, 77).unwrap();

        song.heartbeat(&identity, anchor.device_monotonic_microseconds)
            .unwrap();
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds),
            Ok(Some(AnchoredSongAction::OpenCue { entry, .. })) if entry == cue(1, 0)
        ));
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_000),
            Ok(Some(AnchoredSongAction::CloseCue { entry, .. })) if entry == cue(1, 0)
        ));
        assert_eq!(
            song.record_closed_evidence(
                RepEvidence {
                    windows_present: 8,
                    windows_expected: 9,
                    ..RepEvidence::default()
                },
                0,
            ),
            Ok(Err(RepRejection::MissingSamples))
        );
        assert_eq!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_000),
            Ok(Some(AnchoredSongAction::Completed))
        );
        assert_eq!(song.retained_progress().rejected_cues, 1);
    }

    #[test]
    fn interruption_rejects_only_the_open_cue_and_retains_finished_work() {
        let (mut song, first_identity, anchor) = song_with_two_cues();
        song.heartbeat(&first_identity, anchor.device_monotonic_microseconds)
            .unwrap();
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds),
            Ok(Some(AnchoredSongAction::OpenCue { entry, .. })) if entry == cue(1, 0)
        ));
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_000),
            Ok(Some(AnchoredSongAction::CloseCue { entry, .. })) if entry == cue(1, 0)
        ));
        assert_eq!(
            song.record_closed_evidence(
                RepEvidence {
                    windows_present: 9,
                    windows_expected: 9,
                    ..RepEvidence::default()
                },
                9,
            ),
            Ok(Ok(()))
        );
        song.fit_checkpoint_completed();
        song.heartbeat(
            &first_identity,
            anchor.device_monotonic_microseconds + 1_600_000,
        )
        .unwrap();
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds + 2_000_000),
            Ok(Some(AnchoredSongAction::OpenCue { entry, .. })) if entry == cue(2, 2_000)
        ));
        assert_eq!(
            song.interrupt(SongInterruption::OperatorStopped),
            Ok(AnchoredSongAction::Interrupted {
                reason: SongInterruption::OperatorStopped,
                rejected_open_cue: Some(cue(2, 2_000)),
            })
        );
        assert_eq!(
            song.retained_progress(),
            RetainedSongProgress {
                accepted_cues: 1,
                rejected_cues: 1,
                accepted_rows: 9,
                completed_fit_checkpoints: 1
            }
        );
    }

    #[test]
    fn interruption_without_an_open_cue_keeps_closed_or_future_evidence_intact() {
        let (mut song, identity, _anchor) = song_with_two_cues();
        assert_eq!(
            song.interrupt(SongInterruption::DeviceLinkLost),
            Ok(AnchoredSongAction::Interrupted {
                reason: SongInterruption::DeviceLinkLost,
                rejected_open_cue: None,
            })
        );
        assert_eq!(song.retained_progress().rejected_cues, 0);
        assert!(matches!(
            song.heartbeat(&identity, 20),
            Err(AnchoredSongError::SongMustBeReuploaded)
        ));
    }

    #[test]
    fn closed_evidence_can_finish_after_interruption_without_resuming_the_song() {
        let (mut song, identity, anchor) = song_with_two_cues();
        song.heartbeat(&identity, anchor.device_monotonic_microseconds)
            .unwrap();
        song.poll(anchor.device_monotonic_microseconds).unwrap();
        song.poll(anchor.device_monotonic_microseconds + 1_500_000)
            .unwrap();
        assert_eq!(song.state(), SongState::AwaitingCueEvidence);

        song.interrupt(SongInterruption::OperatorStopped).unwrap();
        assert_eq!(song.state(), SongState::Interrupted);
        song.record_closed_evidence(
            RepEvidence {
                windows_present: 9,
                windows_expected: 9,
                ..RepEvidence::default()
            },
            9,
        )
        .unwrap()
        .unwrap();
        assert_eq!(song.state(), SongState::Interrupted);
        assert_eq!(song.retained_progress().accepted_cues, 1);
        assert_eq!(
            song.heartbeat(&identity, anchor.device_monotonic_microseconds + 1_500_001),
            Err(AnchoredSongError::SongMustBeReuploaded)
        );
    }

    #[test]
    fn heartbeat_timeout_counts_an_open_cue_once_but_never_rejects_an_already_closed_cue() {
        let (mut song, identity, anchor) = song_with_two_cues();
        song.heartbeat(&identity, anchor.device_monotonic_microseconds)
            .unwrap();
        song.poll(anchor.device_monotonic_microseconds).unwrap();
        // The host is still alive during the first cue, then disappears.
        song.heartbeat(&identity, anchor.device_monotonic_microseconds + 1_400_000)
            .unwrap();
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_000),
            Ok(Some(AnchoredSongAction::CloseCue { entry, .. })) if entry == cue(1, 0)
        ));
        assert_eq!(
            song.record_closed_evidence(
                RepEvidence {
                    windows_present: 9,
                    windows_expected: 9,
                    ..RepEvidence::default()
                },
                9,
            ),
            Ok(Ok(()))
        );
        song.poll(anchor.device_monotonic_microseconds + 2_000_000)
            .unwrap();
        assert_eq!(
            song.poll(anchor.device_monotonic_microseconds + 3_400_000),
            Ok(Some(AnchoredSongAction::Interrupted {
                reason: SongInterruption::HeartbeatTimedOut,
                rejected_open_cue: Some(cue(2, 2_000)),
            }))
        );
        assert_eq!(song.retained_progress().accepted_cues, 1);
        assert_eq!(song.retained_progress().rejected_cues, 1);
    }

    #[test]
    fn continue_requires_new_revision_upload_and_new_anchor_but_keeps_progress() {
        let (mut song, first_identity, anchor) = song_with_two_cues();
        song.heartbeat(&first_identity, anchor.device_monotonic_microseconds)
            .unwrap();
        song.fit_checkpoint_completed();
        song.interrupt(SongInterruption::OperatorStopped).unwrap();
        assert!(matches!(
            song.begin_upload(first_identity.clone()),
            Err(AnchoredSongError::RevisionDidNotAdvance { .. })
        ));
        let continued = identity(2, 1);
        song.begin_upload(continued.clone()).unwrap();
        song.upload_chunk(&continued, 0, &[cue(3, 0)]).unwrap();
        let next_anchor = song.commit(&continued, 20_000, 99).unwrap();
        assert_ne!(next_anchor, anchor);
        assert_eq!(song.retained_progress().completed_fit_checkpoints, 1);
    }

    #[test]
    fn identity_is_checked_for_every_boundary() {
        let identity = identity(1, 1);
        let mut song = AnchoredSong::new(run());
        song.begin_upload(identity.clone()).unwrap();
        let wrong = AnchoredSongIdentity::new(
            run(),
            CalibrationScheduleRevision::new(1).unwrap(),
            "other-track".to_string(),
            1,
        )
        .unwrap();
        assert_eq!(
            song.upload_chunk(&wrong, 0, &[cue(1, 0)]),
            Err(AnchoredSongError::WrongContentIdentity)
        );
        song.upload_chunk(&identity, 0, &[cue(1, 0)]).unwrap();
        song.commit(&identity, 10, 4).unwrap();
        assert_eq!(
            song.heartbeat(&wrong, 11),
            Err(AnchoredSongError::WrongContentIdentity)
        );
    }

    #[test]
    fn every_public_phase_has_one_exhaustive_transition_path() {
        let first = identity(1, 1);
        let mut song = AnchoredSong::new(run());
        assert_eq!(song.state(), SongState::AwaitingUpload);
        assert_eq!(song.identity(), None);
        assert_eq!(song.poll(0), Err(AnchoredSongError::ScheduleNotAnchored));

        song.begin_upload(first.clone()).unwrap();
        assert_eq!(song.state(), SongState::ReceivingUpload);
        assert_eq!(song.identity(), Some(&first));
        assert_eq!(
            song.begin_upload(first.clone()),
            Err(AnchoredSongError::UploadAlreadyInProgress)
        );
        assert!(matches!(
            song.commit(&first, 10, 4),
            Err(AnchoredSongError::IncompleteSchedule { .. })
        ));
        assert_eq!(song.state(), SongState::ReceivingUpload);

        song.upload_chunk(&first, 0, &[cue(1, 0)]).unwrap();
        let anchor = song.commit(&first, 10, 4).unwrap();
        assert_eq!(song.state(), SongState::Anchored);
        assert_eq!(
            song.begin_upload(identity(2, 1)),
            Err(AnchoredSongError::SongNotRunnable)
        );

        song.heartbeat(&first, anchor.device_monotonic_microseconds)
            .unwrap();
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds),
            Ok(Some(AnchoredSongAction::OpenCue { .. }))
        ));
        assert_eq!(song.state(), SongState::CueOpen);
        assert_eq!(
            song.record_closed_evidence(RepEvidence::default(), 0),
            Err(AnchoredSongError::CueStillOpen)
        );
        assert_eq!(song.state(), SongState::CueOpen);

        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_000),
            Ok(Some(AnchoredSongAction::CloseCue { .. }))
        ));
        assert_eq!(song.state(), SongState::AwaitingCueEvidence);
        assert_eq!(
            song.begin_upload(identity(2, 1)),
            Err(AnchoredSongError::SongNotRunnable)
        );
        song.record_closed_evidence(
            RepEvidence {
                windows_present: 9,
                windows_expected: 9,
                ..RepEvidence::default()
            },
            9,
        )
        .unwrap()
        .unwrap();
        assert_eq!(song.state(), SongState::Anchored);

        assert_eq!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_000),
            Ok(Some(AnchoredSongAction::Completed))
        );
        assert_eq!(song.state(), SongState::Completed);
        assert_eq!(song.identity(), Some(&first));
        assert_eq!(
            song.heartbeat(&first, anchor.device_monotonic_microseconds + 1_500_001),
            Err(AnchoredSongError::SongMustBeReuploaded)
        );

        let second = identity(2, 1);
        song.begin_upload(second.clone()).unwrap();
        assert_eq!(song.state(), SongState::ReceivingUpload);
        assert_eq!(song.identity(), Some(&second));
    }
}
