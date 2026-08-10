#[path = "../../opal-firmware/src/calibration/resident_selector.rs"]
mod resident_selector;

use resident_selector::{
    PhysicalSlot, ResidentIdentity, SelectorPersistenceCapability, StoreSelector, StoredIdentity,
    StoredRole,
};

fn stored(physical: PhysicalSlot, generation: u32, crc: u32, role: StoredRole) -> StoredIdentity {
    StoredIdentity {
        physical,
        generation,
        crc,
        role,
    }
}

#[test]
fn selector_persists_exactly_two_sequence_and_crc_slots() {
    assert_eq!(
        PhysicalSlot::ALL,
        [PhysicalSlot::First, PhysicalSlot::Second]
    );

    let selector = StoreSelector::recover([None, None]);
    assert_eq!(
        selector.persistence_capability(),
        SelectorPersistenceCapability::SlotSequenceAndCrc
    );
}

#[test]
fn existing_two_residents_migrate_to_newest_plus_erased_scratch() {
    let older = stored(PhysicalSlot::First, 4, 0x1111, StoredRole::Resident);
    let newer = stored(PhysicalSlot::Second, 9, 0x2222, StoredRole::Resident);
    let selector = StoreSelector::recover([Some(older), Some(newer)]);

    assert_eq!(
        selector.resident(),
        Some(ResidentIdentity::from_stored(newer))
    );
    assert_eq!(selector.scratch(), Some(PhysicalSlot::First));
    assert_eq!(selector.exportable(), None);
}

#[test]
fn saving_candidate_promotes_it_and_demotes_old_resident_to_scratch() {
    let old = stored(PhysicalSlot::First, 4, 0x1111, StoredRole::Resident);
    let candidate = stored(PhysicalSlot::Second, 5, 0x2222, StoredRole::Resident);
    let before = StoreSelector::recover([Some(old), None]);
    let after = StoreSelector::recover([Some(old), Some(candidate)]);

    assert_eq!(before.resident(), Some(ResidentIdentity::from_stored(old)));
    assert_eq!(before.scratch(), Some(PhysicalSlot::Second));
    assert_eq!(
        after.resident(),
        Some(ResidentIdentity::from_stored(candidate))
    );
    assert_eq!(after.scratch(), Some(PhysicalSlot::First));
}

#[test]
fn discard_preserves_resident_and_returns_candidate_region_to_scratch() {
    let resident = stored(PhysicalSlot::First, 4, 0x1111, StoredRole::Resident);
    let selector = StoreSelector::recover([Some(resident), None]);

    assert_eq!(
        selector.resident(),
        Some(ResidentIdentity::from_stored(resident))
    );
    assert_eq!(selector.scratch(), Some(PhysicalSlot::Second));
}

#[test]
fn host_export_stays_available_without_displacing_resident() {
    let resident = stored(PhysicalSlot::First, 4, 0x1111, StoredRole::Resident);
    let export = stored(
        PhysicalSlot::Second,
        5,
        0x2222,
        StoredRole::ExportableCandidate,
    );
    let pending = StoreSelector::recover([Some(resident), Some(export)]);
    let transferred = StoreSelector::recover([Some(resident), None]);

    assert_eq!(
        pending.resident(),
        Some(ResidentIdentity::from_stored(resident))
    );
    assert_eq!(pending.exportable(), Some(export));
    assert_eq!(pending.scratch(), None);
    assert_eq!(transferred.exportable(), None);
    assert_eq!(transferred.scratch(), Some(PhysicalSlot::Second));
}

#[test]
fn newer_inactive_record_clears_classification_and_frees_old_resident_region() {
    let resident = stored(PhysicalSlot::First, 4, 0x1111, StoredRole::Resident);
    let inactive = stored(PhysicalSlot::Second, 5, 0x3333, StoredRole::Inactive);
    let selector = StoreSelector::recover([Some(resident), Some(inactive)]);

    assert_eq!(selector.resident(), None);
    assert!(!selector.classification_enabled());
    assert_eq!(selector.scratch(), Some(PhysicalSlot::First));
}

#[test]
fn restoring_archive_through_scratch_promotes_the_restored_record() {
    let inactive = stored(PhysicalSlot::Second, 5, 0x3333, StoredRole::Inactive);
    let restored = stored(PhysicalSlot::First, 6, 0xABCD, StoredRole::Resident);
    let selector = StoreSelector::recover([Some(restored), Some(inactive)]);

    assert_eq!(
        selector.resident(),
        Some(ResidentIdentity {
            physical: PhysicalSlot::First,
            generation: 6,
            crc: 0xABCD,
        })
    );
    assert_eq!(selector.scratch(), Some(PhysicalSlot::Second));
}
