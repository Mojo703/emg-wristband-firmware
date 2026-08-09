//! Whether a labeled span was worth keeping.
//!
//! This is validity, not segmentation. The span is already fixed by the grid
//! arithmetic and nothing here may move it: a rep either counts or is thrown
//! away and asked for again. Rule 6 — labels come from the cue clock, and
//! nothing ever relabels one.
//!
//! Every reason here is a hardware fact. "No gesture was performed" is not
//! among them and has no detector, which is a recorded gap rather than an
//! oversight: V's experiment 11 found no statistic and no threshold that
//! separates a dead rep from a real one. The shipped floor rejected 83.1% of
//! genuine fixture reps, including every thumb extension — that gesture sits at
//! rest-level band energy at these electrodes, and real reps span 716 to 1944
//! permille of baseline against idle windows spanning 390 to 1606. The floor
//! that accepts every real rep rejects barely half the idle.
//!
//! Dead reps do cost: one per class moves the four numbers out of the
//! acceptance region. Nothing on the device catches that. The per-class
//! self-test reports a weak class to the panel afterwards, which is a report
//! and not a gate — log 0022 leaves its pass-fail threshold unresolved.

use protocol::RepRejection;

/// What the device observed over one labeled span. Everything the rules below
/// need, gathered by the caller because every field of it needs hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RepEvidence {
    /// Any channel the front end flagged lead-off during the span.
    pub lead_off_channels: bool,
    /// The span overlapped an ADC recovery settle.
    pub adc_recovery_settle: bool,
    /// The span overlapped a flash write or erase. The flush schedule is
    /// supposed to make this impossible, which is exactly why it is checked.
    pub flash_operation: bool,
    /// Windows the acquisition path actually produced inside the span, against
    /// the W it was supposed to.
    pub windows_present: u32,
    pub windows_expected: u32,
}

impl RepEvidence {
    /// Why this rep cannot be used, or `None` if it can.
    ///
    /// Ordered by how much the answer tells whoever is reading the telemetry.
    /// Missing windows first: with a hole in the span nothing else measured
    /// over it means anything. Then the flash overlap, because it is an
    /// invariant failing rather than a wearer doing something — it should never
    /// appear, and if it does it must not be reported as a wearer's fault. Then
    /// the front end's own two flags.
    pub fn rejection(&self) -> Option<RepRejection> {
        if self.windows_present < self.windows_expected {
            return Some(RepRejection::MissingSamples);
        }
        if self.flash_operation {
            return Some(RepRejection::FlashOperationOverlap);
        }
        if self.lead_off_channels {
            return Some(RepRejection::LeadOffChannels);
        }
        if self.adc_recovery_settle {
            return Some(RepRejection::AdcRecoverySettle);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> RepEvidence {
        RepEvidence {
            windows_present: 9,
            windows_expected: 9,
            ..RepEvidence::default()
        }
    }

    #[test]
    fn a_clean_rep_is_kept() {
        assert_eq!(good().rejection(), None);
    }

    #[test]
    fn a_rep_is_never_rejected_for_being_quiet() {
        // The check that would have is gone, and nothing replaced it. Thumb
        // extension sits at rest-level band energy at these electrodes, so a
        // rep indistinguishable from idle is exactly what a correct rep can
        // look like.
        assert_eq!(good().rejection(), None);
    }

    #[test]
    fn a_hole_in_the_span_outranks_everything_else() {
        // With a window missing, whatever else was measured is over the wrong
        // samples, so reporting one of those as the reason would be a guess
        // dressed as a diagnosis.
        let torn = RepEvidence {
            windows_present: 8,
            lead_off_channels: true,
            ..good()
        };
        assert_eq!(torn.rejection(), Some(RepRejection::MissingSamples));
    }

    #[test]
    fn a_flash_overlap_is_never_reported_as_the_wearers_fault() {
        // Flushes are scheduled strictly between rounds, so this reason should
        // never appear in a run report. When it does it is the schedule that
        // broke, and it has to outrank the flags a write would itself disturb.
        let during_a_write = RepEvidence {
            flash_operation: true,
            lead_off_channels: true,
            adc_recovery_settle: true,
            ..good()
        };
        assert_eq!(
            during_a_write.rejection(),
            Some(RepRejection::FlashOperationOverlap)
        );
    }

    #[test]
    fn the_front_ends_own_flags_are_each_reported() {
        let lead_off = RepEvidence {
            lead_off_channels: true,
            ..good()
        };
        assert_eq!(lead_off.rejection(), Some(RepRejection::LeadOffChannels));
        let settling = RepEvidence {
            adc_recovery_settle: true,
            ..good()
        };
        assert_eq!(settling.rejection(), Some(RepRejection::AdcRecoverySettle));
    }

    #[test]
    fn every_reason_that_survives_is_a_hardware_fact() {
        // The list is closed on purpose. Anything about what the wearer did
        // needs a detector, and V measured that none exists.
        for evidence in [
            RepEvidence {
                windows_present: 0,
                ..good()
            },
            RepEvidence {
                flash_operation: true,
                ..good()
            },
            RepEvidence {
                lead_off_channels: true,
                ..good()
            },
            RepEvidence {
                adc_recovery_settle: true,
                ..good()
            },
        ] {
            assert_ne!(
                evidence.rejection(),
                Some(RepRejection::AtRestBaseline),
                "the device emitted a reason it no longer detects"
            );
            assert!(evidence.rejection().is_some());
        }
    }
}
