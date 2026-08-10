//! Recovery of one logical resident and one candidate/scratch over two slots.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PhysicalSlot {
    First,
    Second,
}

impl PhysicalSlot {
    pub(crate) const ALL: [Self; 2] = [Self::First, Self::Second];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
        }
    }

    pub(crate) const fn other(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StoredIdentity {
    pub physical: PhysicalSlot,
    pub generation: u32,
    pub crc: u32,
    pub role: StoredRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoredRole {
    Resident,
    ExportableCandidate,
    Inactive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentIdentity {
    pub physical: PhysicalSlot,
    pub generation: u32,
    pub crc: u32,
}

impl ResidentIdentity {
    pub(crate) const fn from_stored(stored: StoredIdentity) -> Self {
        Self {
            physical: stored.physical,
            generation: stored.generation,
            crc: stored.crc,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectorPersistenceCapability {
    SlotSequenceAndCrc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StoreSelector {
    resident: Option<ResidentIdentity>,
    exportable: Option<StoredIdentity>,
    scratch: Option<PhysicalSlot>,
}

impl StoreSelector {
    pub(crate) fn recover(slots: [Option<StoredIdentity>; 2]) -> Self {
        let newest_state = slots
            .into_iter()
            .flatten()
            .filter(|stored| matches!(stored.role, StoredRole::Resident | StoredRole::Inactive))
            .max_by_key(|stored| (stored.generation, stored.physical.index()));
        let resident = newest_state
            .filter(|stored| stored.role == StoredRole::Resident)
            .map(ResidentIdentity::from_stored);
        let state_generation = newest_state.map_or(0, |stored| stored.generation);
        let exportable = slots
            .into_iter()
            .flatten()
            .filter(|stored| {
                stored.role == StoredRole::ExportableCandidate
                    && stored.generation > state_generation
            })
            .max_by_key(|stored| (stored.generation, stored.physical.index()));
        let occupied = exportable
            .map(|stored| stored.physical)
            .or_else(|| newest_state.map(|stored| stored.physical));
        let scratch = if exportable.is_some() {
            None
        } else {
            Some(occupied.map_or(PhysicalSlot::First, PhysicalSlot::other))
        };
        Self {
            resident,
            exportable,
            scratch,
        }
    }

    pub(crate) const fn resident(&self) -> Option<ResidentIdentity> {
        self.resident
    }

    pub(crate) const fn exportable(&self) -> Option<StoredIdentity> {
        self.exportable
    }

    pub(crate) const fn scratch(&self) -> Option<PhysicalSlot> {
        self.scratch
    }

    pub(crate) const fn classification_enabled(&self) -> bool {
        self.resident.is_some()
    }

    pub(crate) const fn persistence_capability(&self) -> SelectorPersistenceCapability {
        SelectorPersistenceCapability::SlotSequenceAndCrc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(physical: PhysicalSlot, generation: u32, role: StoredRole) -> StoredIdentity {
        StoredIdentity {
            physical,
            generation,
            crc: generation.wrapping_mul(0x0101_0101),
            role,
        }
    }

    #[test]
    fn promoted_candidate_is_the_resident_after_reboot_recovery() {
        let old_resident = stored(PhysicalSlot::First, 7, StoredRole::Resident);
        let candidate = stored(PhysicalSlot::Second, 8, StoredRole::ExportableCandidate);
        let before_save = StoreSelector::recover([Some(old_resident), Some(candidate)]);
        assert_eq!(
            before_save.resident(),
            Some(ResidentIdentity::from_stored(old_resident))
        );
        assert_eq!(before_save.exportable(), Some(candidate));

        // Save rewrites only the candidate's metadata role and CRC last. A
        // reboot reconstructs solely from that validated metadata, so no RAM
        // state is needed to restore the promoted model and its gains.
        let promoted = stored(PhysicalSlot::Second, 8, StoredRole::Resident);
        let after_reboot = StoreSelector::recover([Some(old_resident), Some(promoted)]);
        assert_eq!(
            after_reboot.resident(),
            Some(ResidentIdentity::from_stored(promoted))
        );
        assert!(after_reboot.classification_enabled());
        assert_eq!(after_reboot.exportable(), None);
    }

    #[test]
    fn interrupted_or_discarded_candidate_leaves_prior_resident_usable() {
        let resident = stored(PhysicalSlot::First, 7, StoredRole::Resident);
        let interrupted = StoreSelector::recover([Some(resident), None]);
        assert_eq!(
            interrupted.resident(),
            Some(ResidentIdentity::from_stored(resident))
        );
        assert!(interrupted.classification_enabled());

        let candidate = stored(PhysicalSlot::Second, 8, StoredRole::ExportableCandidate);
        let before_discard = StoreSelector::recover([Some(resident), Some(candidate)]);
        assert_eq!(
            before_discard.resident(),
            Some(ResidentIdentity::from_stored(resident))
        );
        let after_discard = StoreSelector::recover([Some(resident), None]);
        assert_eq!(
            after_discard.resident(),
            Some(ResidentIdentity::from_stored(resident))
        );
        assert!(after_discard.classification_enabled());
    }
}
