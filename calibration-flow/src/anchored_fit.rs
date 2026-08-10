//! Pure work planning for bounded fitting between anchored songs.
//!
//! The executor owns the fitter and its in-flight pass. This type owns the
//! ordering invariant: every accepted-row checkpoint requested before Save is
//! completed before the one final polish. Keeping the plan in the host-runnable
//! core makes Continue/Save races testable without waiting for a wristband.

/// One bounded fit operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchoredFitStage {
    Checkpoint,
    FinalPolish,
}

/// Work which remains after the currently running fit, or is ready to start.
///
/// The queue is deliberately bounded. Checkpoints coalesce because each pass
/// sees every row flushed so far; at most one extra checkpoint is needed after
/// a checkpoint already in flight. Final polish is terminal and must follow
/// that queued checkpoint when both are requested.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnchoredFitPlan {
    #[default]
    Idle,
    Checkpoint,
    Finalize,
    CheckpointThenFinalize,
}

impl AnchoredFitPlan {
    /// Add a checkpoint for newly retained rows. Repeated requests coalesce.
    /// Once finalization is requested, no later collection is legal, so the
    /// terminal plans remain terminal rather than accepting impossible work.
    pub const fn request_checkpoint(self) -> Self {
        match self {
            Self::Idle | Self::Checkpoint => Self::Checkpoint,
            Self::Finalize | Self::CheckpointThenFinalize => self,
        }
    }

    /// Request Save's final polish after all checkpoint work already owed.
    pub const fn request_finalization(self) -> Self {
        match self {
            Self::Idle | Self::Finalize => Self::Finalize,
            Self::Checkpoint | Self::CheckpointThenFinalize => Self::CheckpointThenFinalize,
        }
    }

    /// Take the first operation and return the exact remaining plan.
    pub const fn take_first(self) -> Option<(AnchoredFitStage, Self)> {
        match self {
            Self::Idle => None,
            Self::Checkpoint => Some((AnchoredFitStage::Checkpoint, Self::Idle)),
            Self::Finalize => Some((AnchoredFitStage::FinalPolish, Self::Idle)),
            Self::CheckpointThenFinalize => Some((AnchoredFitStage::Checkpoint, Self::Finalize)),
        }
    }

    pub const fn is_finalizing(self) -> bool {
        matches!(self, Self::Finalize | Self::CheckpointThenFinalize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_checkpoints_coalesce_before_one_final_polish() {
        let plan = AnchoredFitPlan::Idle
            .request_checkpoint()
            .request_checkpoint()
            .request_finalization();
        assert_eq!(plan, AnchoredFitPlan::CheckpointThenFinalize);
        let (first, remainder) = plan.take_first().unwrap();
        assert_eq!(first, AnchoredFitStage::Checkpoint);
        let (second, remainder) = remainder.take_first().unwrap();
        assert_eq!(second, AnchoredFitStage::FinalPolish);
        assert_eq!(remainder, AnchoredFitPlan::Idle);
    }

    #[test]
    fn finalization_is_a_terminal_collection_boundary() {
        assert_eq!(
            AnchoredFitPlan::Finalize.request_checkpoint(),
            AnchoredFitPlan::Finalize
        );
        assert_eq!(
            AnchoredFitPlan::CheckpointThenFinalize.request_checkpoint(),
            AnchoredFitPlan::CheckpointThenFinalize
        );
    }
}
