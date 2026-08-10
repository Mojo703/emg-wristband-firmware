//! Pure domain state for calibration placement in the dashboard.
//!
//! This module models successful state transitions and recoverable conflicts.
//! It does not claim that a later transport or persistence adapter can commit a
//! transition atomically across host and device storage.

use core::fmt;

pub const RESIDENT_SLOT_COUNT: usize = 2;
pub const ARCHIVE_SLOT_COUNT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CalibrationIdentity(u64);

impl CalibrationIdentity {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Revision(u64);

impl Revision {
    pub const fn initial() -> Self {
        Self(0)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(next) => Some(Self(next)),
            None => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionIndexError {
    Resident(usize),
    Archive(usize),
}

impl fmt::Display for PositionIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resident(index) => write!(
                formatter,
                "resident index {index} is outside 0..{RESIDENT_SLOT_COUNT}"
            ),
            Self::Archive(index) => write!(
                formatter,
                "archive index {index} is outside 0..{ARCHIVE_SLOT_COUNT}"
            ),
        }
    }
}

impl std::error::Error for PositionIndexError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentIndex(u8);

impl ResidentIndex {
    pub fn new(index: usize) -> Result<Self, PositionIndexError> {
        if index < RESIDENT_SLOT_COUNT {
            Ok(Self(index as u8))
        } else {
            Err(PositionIndexError::Resident(index))
        }
    }

    pub const fn get(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArchiveIndex(u8);

impl ArchiveIndex {
    pub fn new(index: usize) -> Result<Self, PositionIndexError> {
        if index < ARCHIVE_SLOT_COUNT {
            Ok(Self(index as u8))
        } else {
            Err(PositionIndexError::Archive(index))
        }
    }

    pub const fn get(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Position {
    Candidate,
    Resident(ResidentIndex),
    Archive(ArchiveIndex),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedPosition {
    pub position: Position,
    pub identity: Option<CalibrationIdentity>,
}

impl ExpectedPosition {
    pub const fn new(position: Position, identity: Option<CalibrationIdentity>) -> Self {
        Self { position, identity }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calibration<T> {
    identity: CalibrationIdentity,
    value: T,
}

impl<T> Calibration<T> {
    pub const fn new(identity: CalibrationIdentity, value: T) -> Self {
        Self { identity, value }
    }

    pub const fn identity(&self) -> CalibrationIdentity {
        self.identity
    }

    pub const fn value(&self) -> &T {
        &self.value
    }

    pub fn into_value(self) -> T {
        self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibraryError {
    StaleRevision {
        expected: Revision,
        actual: Revision,
    },
    StaleContent {
        position: Position,
        expected: Option<CalibrationIdentity>,
        actual: Option<CalibrationIdentity>,
    },
    EmptyPosition(Position),
    SamePosition(Position),
    RevisionExhausted,
}

impl fmt::Display for LibraryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleRevision { expected, actual } => write!(
                formatter,
                "library revision changed from {} to {}",
                expected.get(),
                actual.get()
            ),
            Self::StaleContent {
                position,
                expected,
                actual,
            } => write!(
                formatter,
                "content at {position:?} changed from {expected:?} to {actual:?}"
            ),
            Self::EmptyPosition(position) => write!(formatter, "{position:?} is empty"),
            Self::SamePosition(position) => {
                write!(
                    formatter,
                    "{position:?} cannot be both ends of an operation"
                )
            }
            Self::RevisionExhausted => formatter.write_str("library revision is exhausted"),
        }
    }
}

impl std::error::Error for LibraryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationLibrary<T> {
    revision: Revision,
    candidate: Option<Calibration<T>>,
    residents: [Option<Calibration<T>>; RESIDENT_SLOT_COUNT],
    archives: [Option<Calibration<T>>; ARCHIVE_SLOT_COUNT],
    active: Option<ResidentIndex>,
}

impl<T> Default for CalibrationLibrary<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CalibrationLibrary<T> {
    pub fn new() -> Self {
        Self {
            revision: Revision::initial(),
            candidate: None,
            residents: core::array::from_fn(|_| None),
            archives: core::array::from_fn(|_| None),
            active: None,
        }
    }

    pub const fn revision(&self) -> Revision {
        self.revision
    }

    pub const fn active(&self) -> Option<ResidentIndex> {
        self.active
    }

    pub const fn candidate(&self) -> Option<&Calibration<T>> {
        self.candidate.as_ref()
    }

    pub fn resident(&self, index: ResidentIndex) -> Option<&Calibration<T>> {
        self.residents[index.get()].as_ref()
    }

    pub fn archive(&self, index: ArchiveIndex) -> Option<&Calibration<T>> {
        self.archives[index.get()].as_ref()
    }

    pub fn at(&self, position: Position) -> Option<&Calibration<T>> {
        match position {
            Position::Candidate => self.candidate(),
            Position::Resident(index) => self.resident(index),
            Position::Archive(index) => self.archive(index),
        }
    }

    pub fn replace_candidate(
        &mut self,
        expected_revision: Revision,
        expected_candidate: Option<CalibrationIdentity>,
        candidate: Calibration<T>,
    ) -> Result<Revision, LibraryError> {
        self.check_revision(expected_revision)?;
        self.check_content(Position::Candidate, expected_candidate)?;
        self.candidate = Some(candidate);
        self.advance_revision()
    }

    pub fn start_run(
        &mut self,
        expected_revision: Revision,
        expected_candidate: Option<CalibrationIdentity>,
    ) -> Result<Revision, LibraryError> {
        self.clear_candidate(expected_revision, expected_candidate)
    }

    pub fn discard_candidate(
        &mut self,
        expected_revision: Revision,
        expected_candidate: Option<CalibrationIdentity>,
    ) -> Result<Revision, LibraryError> {
        self.clear_candidate(expected_revision, expected_candidate)
    }

    pub fn move_or_swap(
        &mut self,
        expected_revision: Revision,
        source: ExpectedPosition,
        target: ExpectedPosition,
    ) -> Result<Revision, LibraryError> {
        self.check_revision(expected_revision)?;
        self.check_distinct(source.position, target.position)?;
        self.check_expectations(&[source, target])?;
        if source.identity.is_none() {
            return Err(LibraryError::EmptyPosition(source.position));
        }

        let source_value = self.take(source.position);
        let target_value = self.take(target.position);
        self.put(source.position, target_value);
        self.put(target.position, source_value);
        self.normalize_active();
        self.advance_revision()
    }

    pub fn duplicate(
        &mut self,
        expected_revision: Revision,
        source: ExpectedPosition,
        target: ExpectedPosition,
    ) -> Result<Revision, LibraryError>
    where
        T: Clone,
    {
        self.check_revision(expected_revision)?;
        self.check_distinct(source.position, target.position)?;
        self.check_expectations(&[source, target])?;
        let value = self
            .at(source.position)
            .cloned()
            .ok_or(LibraryError::EmptyPosition(source.position))?;
        self.put(target.position, Some(value));
        self.advance_revision()
    }

    pub fn delete(
        &mut self,
        expected_revision: Revision,
        expected: ExpectedPosition,
    ) -> Result<Revision, LibraryError> {
        self.check_revision(expected_revision)?;
        self.check_content(expected.position, expected.identity)?;
        if self.take(expected.position).is_none() {
            return Err(LibraryError::EmptyPosition(expected.position));
        }
        self.normalize_active();
        self.advance_revision()
    }

    pub fn activate(
        &mut self,
        expected_revision: Revision,
        resident: ResidentIndex,
        expected_identity: Option<CalibrationIdentity>,
    ) -> Result<Revision, LibraryError> {
        let position = Position::Resident(resident);
        self.check_revision(expected_revision)?;
        self.check_content(position, expected_identity)?;
        if self.resident(resident).is_none() {
            return Err(LibraryError::EmptyPosition(position));
        }
        self.active = Some(resident);
        self.advance_revision()
    }

    pub fn save_candidate_to_resident(
        &mut self,
        expected_revision: Revision,
        expected_candidate: Option<CalibrationIdentity>,
        destination: ResidentIndex,
        expected_destination: Option<CalibrationIdentity>,
    ) -> Result<Revision, LibraryError> {
        let destination_position = Position::Resident(destination);
        self.check_revision(expected_revision)?;
        self.check_expectations(&[
            ExpectedPosition::new(Position::Candidate, expected_candidate),
            ExpectedPosition::new(destination_position, expected_destination),
        ])?;
        let candidate = self
            .candidate
            .take()
            .ok_or(LibraryError::EmptyPosition(Position::Candidate))?;
        self.residents[destination.get()] = Some(candidate);
        self.active = Some(destination);
        self.advance_revision()
    }

    pub fn save_candidate_to_archive(
        &mut self,
        expected_revision: Revision,
        expected_candidate: Option<CalibrationIdentity>,
        destination: ArchiveIndex,
        expected_destination: Option<CalibrationIdentity>,
    ) -> Result<Revision, LibraryError> {
        let destination_position = Position::Archive(destination);
        self.check_revision(expected_revision)?;
        self.check_expectations(&[
            ExpectedPosition::new(Position::Candidate, expected_candidate),
            ExpectedPosition::new(destination_position, expected_destination),
        ])?;
        let candidate = self
            .candidate
            .take()
            .ok_or(LibraryError::EmptyPosition(Position::Candidate))?;
        self.archives[destination.get()] = Some(candidate);
        self.advance_revision()
    }

    fn clear_candidate(
        &mut self,
        expected_revision: Revision,
        expected_candidate: Option<CalibrationIdentity>,
    ) -> Result<Revision, LibraryError> {
        self.check_revision(expected_revision)?;
        self.check_content(Position::Candidate, expected_candidate)?;
        self.candidate = None;
        self.advance_revision()
    }

    fn check_revision(&self, expected: Revision) -> Result<(), LibraryError> {
        if expected != self.revision {
            return Err(LibraryError::StaleRevision {
                expected,
                actual: self.revision,
            });
        }
        self.revision
            .next()
            .map(|_| ())
            .ok_or(LibraryError::RevisionExhausted)
    }

    fn check_expectations(&self, expected: &[ExpectedPosition]) -> Result<(), LibraryError> {
        expected
            .iter()
            .try_for_each(|value| self.check_content(value.position, value.identity))
    }

    fn check_content(
        &self,
        position: Position,
        expected: Option<CalibrationIdentity>,
    ) -> Result<(), LibraryError> {
        let actual = self.at(position).map(Calibration::identity);
        if expected == actual {
            Ok(())
        } else {
            Err(LibraryError::StaleContent {
                position,
                expected,
                actual,
            })
        }
    }

    fn check_distinct(&self, source: Position, target: Position) -> Result<(), LibraryError> {
        if source == target {
            Err(LibraryError::SamePosition(source))
        } else {
            Ok(())
        }
    }

    fn take(&mut self, position: Position) -> Option<Calibration<T>> {
        match position {
            Position::Candidate => self.candidate.take(),
            Position::Resident(index) => self.residents[index.get()].take(),
            Position::Archive(index) => self.archives[index.get()].take(),
        }
    }

    fn put(&mut self, position: Position, value: Option<Calibration<T>>) {
        match position {
            Position::Candidate => self.candidate = value,
            Position::Resident(index) => self.residents[index.get()] = value,
            Position::Archive(index) => self.archives[index.get()] = value,
        }
    }

    fn normalize_active(&mut self) {
        if self
            .active
            .is_some_and(|index| self.resident(index).is_none())
        {
            self.active = None;
        }
    }

    fn advance_revision(&mut self) -> Result<Revision, LibraryError> {
        self.revision = self
            .revision
            .next()
            .expect("revision capacity checked before mutation");
        Ok(self.revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration(identity: u64) -> Calibration<&'static str> {
        Calibration::new(CalibrationIdentity::new(identity), "snapshot")
    }

    fn resident(index: usize) -> ResidentIndex {
        ResidentIndex::new(index).unwrap()
    }

    fn archive(index: usize) -> ArchiveIndex {
        ArchiveIndex::new(index).unwrap()
    }

    fn expected(position: Position, identity: Option<u64>) -> ExpectedPosition {
        ExpectedPosition::new(position, identity.map(CalibrationIdentity::new))
    }

    fn install_candidate(library: &mut CalibrationLibrary<&'static str>, identity: u64) {
        library
            .replace_candidate(library.revision(), None, calibration(identity))
            .unwrap();
    }

    #[test]
    fn typed_indices_enforce_position_bounds() {
        assert_eq!(ResidentIndex::new(0).unwrap().get(), 0);
        assert_eq!(ResidentIndex::new(1).unwrap().get(), 1);
        assert_eq!(ResidentIndex::new(2), Err(PositionIndexError::Resident(2)));
        assert_eq!(ArchiveIndex::new(7).unwrap().get(), 7);
        assert_eq!(ArchiveIndex::new(8), Err(PositionIndexError::Archive(8)));
    }

    #[test]
    fn saving_candidate_to_resident_clears_candidate_and_activates_destination() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 10);
        let revision = library.revision();

        library
            .save_candidate_to_resident(
                revision,
                Some(CalibrationIdentity::new(10)),
                resident(1),
                None,
            )
            .unwrap();

        assert!(library.candidate().is_none());
        assert_eq!(
            library.resident(resident(1)).unwrap().identity(),
            CalibrationIdentity::new(10)
        );
        assert_eq!(library.active(), Some(resident(1)));
        assert_eq!(library.revision(), revision.next().unwrap());
    }

    #[test]
    fn saving_candidate_to_archive_preserves_active_resident() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 1);
        library
            .save_candidate_to_resident(
                library.revision(),
                Some(CalibrationIdentity::new(1)),
                resident(0),
                None,
            )
            .unwrap();
        install_candidate(&mut library, 2);
        let active = library.active();

        library
            .save_candidate_to_archive(
                library.revision(),
                Some(CalibrationIdentity::new(2)),
                archive(3),
                None,
            )
            .unwrap();

        assert!(library.candidate().is_none());
        assert_eq!(
            library.archive(archive(3)).unwrap().identity(),
            CalibrationIdentity::new(2)
        );
        assert_eq!(library.active(), active);
    }

    #[test]
    fn final_save_overwrites_destination_instead_of_swapping_it_into_candidate() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 1);
        library
            .save_candidate_to_archive(
                library.revision(),
                Some(CalibrationIdentity::new(1)),
                archive(0),
                None,
            )
            .unwrap();
        install_candidate(&mut library, 2);

        library
            .save_candidate_to_archive(
                library.revision(),
                Some(CalibrationIdentity::new(2)),
                archive(0),
                Some(CalibrationIdentity::new(1)),
            )
            .unwrap();

        assert!(library.candidate().is_none());
        assert_eq!(
            library.archive(archive(0)).unwrap().identity(),
            CalibrationIdentity::new(2)
        );
    }

    #[test]
    fn normal_placement_moves_to_empty_and_swaps_with_occupied() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 1);
        library
            .move_or_swap(
                library.revision(),
                expected(Position::Candidate, Some(1)),
                expected(Position::Archive(archive(0)), None),
            )
            .unwrap();
        assert!(library.candidate().is_none());
        assert_eq!(
            library.archive(archive(0)).unwrap().identity(),
            CalibrationIdentity::new(1)
        );

        install_candidate(&mut library, 2);
        library
            .move_or_swap(
                library.revision(),
                expected(Position::Candidate, Some(2)),
                expected(Position::Archive(archive(0)), Some(1)),
            )
            .unwrap();
        assert_eq!(
            library.candidate().unwrap().identity(),
            CalibrationIdentity::new(1)
        );
        assert_eq!(
            library.archive(archive(0)).unwrap().identity(),
            CalibrationIdentity::new(2)
        );
    }

    #[test]
    fn swapping_into_active_position_changes_content_without_moving_active_marker() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 1);
        library
            .save_candidate_to_resident(
                library.revision(),
                Some(CalibrationIdentity::new(1)),
                resident(0),
                None,
            )
            .unwrap();
        install_candidate(&mut library, 2);

        library
            .move_or_swap(
                library.revision(),
                expected(Position::Candidate, Some(2)),
                expected(Position::Resident(resident(0)), Some(1)),
            )
            .unwrap();

        assert_eq!(library.active(), Some(resident(0)));
        assert_eq!(
            library.resident(resident(0)).unwrap().identity(),
            CalibrationIdentity::new(2)
        );
        assert_eq!(
            library.candidate().unwrap().identity(),
            CalibrationIdentity::new(1)
        );
    }

    #[test]
    fn moving_active_content_to_empty_position_deactivates_library() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 1);
        library
            .save_candidate_to_resident(
                library.revision(),
                Some(CalibrationIdentity::new(1)),
                resident(0),
                None,
            )
            .unwrap();

        library
            .move_or_swap(
                library.revision(),
                expected(Position::Resident(resident(0)), Some(1)),
                expected(Position::Archive(archive(0)), None),
            )
            .unwrap();

        assert_eq!(library.active(), None);
        assert!(library.resident(resident(0)).is_none());
    }

    #[test]
    fn duplicate_copies_content_and_keeps_source() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 8);

        library
            .duplicate(
                library.revision(),
                expected(Position::Candidate, Some(8)),
                expected(Position::Archive(archive(7)), None),
            )
            .unwrap();

        assert_eq!(
            library.candidate().unwrap().identity(),
            CalibrationIdentity::new(8)
        );
        assert_eq!(
            library.archive(archive(7)).unwrap().identity(),
            CalibrationIdentity::new(8)
        );
    }

    #[test]
    fn deleting_active_resident_turns_classification_off() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 4);
        library
            .save_candidate_to_resident(
                library.revision(),
                Some(CalibrationIdentity::new(4)),
                resident(1),
                None,
            )
            .unwrap();

        library
            .delete(
                library.revision(),
                expected(Position::Resident(resident(1)), Some(4)),
            )
            .unwrap();

        assert_eq!(library.active(), None);
        assert!(library.resident(resident(1)).is_none());
    }

    #[test]
    fn activation_requires_the_expected_occupied_resident() {
        let mut library = CalibrationLibrary::new();
        assert_eq!(
            library.activate(library.revision(), resident(0), None),
            Err(LibraryError::EmptyPosition(Position::Resident(resident(0))))
        );

        install_candidate(&mut library, 9);
        library
            .move_or_swap(
                library.revision(),
                expected(Position::Candidate, Some(9)),
                expected(Position::Resident(resident(0)), None),
            )
            .unwrap();
        library
            .activate(
                library.revision(),
                resident(0),
                Some(CalibrationIdentity::new(9)),
            )
            .unwrap();
        assert_eq!(library.active(), Some(resident(0)));
    }

    #[test]
    fn discard_and_starting_run_clear_candidate() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 3);
        library
            .discard_candidate(library.revision(), Some(CalibrationIdentity::new(3)))
            .unwrap();
        assert!(library.candidate().is_none());

        install_candidate(&mut library, 5);
        library
            .start_run(library.revision(), Some(CalibrationIdentity::new(5)))
            .unwrap();
        assert!(library.candidate().is_none());
    }

    #[test]
    fn stale_revision_and_content_reject_without_mutation() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 6);
        let unchanged = library.clone();

        assert_eq!(
            library.discard_candidate(Revision::initial(), Some(CalibrationIdentity::new(6))),
            Err(LibraryError::StaleRevision {
                expected: Revision::initial(),
                actual: unchanged.revision(),
            })
        );
        assert_eq!(library, unchanged);

        assert_eq!(
            library.discard_candidate(library.revision(), Some(CalibrationIdentity::new(99)),),
            Err(LibraryError::StaleContent {
                position: Position::Candidate,
                expected: Some(CalibrationIdentity::new(99)),
                actual: Some(CalibrationIdentity::new(6)),
            })
        );
        assert_eq!(library, unchanged);
    }

    #[test]
    fn stale_destination_identity_rejects_before_source_is_consumed() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 6);
        let unchanged = library.clone();

        assert_eq!(
            library.save_candidate_to_archive(
                library.revision(),
                Some(CalibrationIdentity::new(6)),
                archive(0),
                Some(CalibrationIdentity::new(90)),
            ),
            Err(LibraryError::StaleContent {
                position: Position::Archive(archive(0)),
                expected: Some(CalibrationIdentity::new(90)),
                actual: None,
            })
        );
        assert_eq!(library, unchanged);
    }

    #[test]
    fn exhausted_revision_rejects_before_mutating_content() {
        let mut library = CalibrationLibrary::new();
        library.revision = Revision(u64::MAX);
        let unchanged = library.clone();

        assert_eq!(
            library.replace_candidate(Revision(u64::MAX), None, calibration(1),),
            Err(LibraryError::RevisionExhausted)
        );
        assert_eq!(library, unchanged);
    }

    #[test]
    fn all_successful_transitions_preserve_active_slot_invariant() {
        for active_index in 0..RESIDENT_SLOT_COUNT {
            for archive_index in 0..ARCHIVE_SLOT_COUNT {
                let mut library = CalibrationLibrary::new();
                install_candidate(&mut library, 1);
                let active = resident(active_index);
                library
                    .save_candidate_to_resident(
                        library.revision(),
                        Some(CalibrationIdentity::new(1)),
                        active,
                        None,
                    )
                    .unwrap();
                install_candidate(&mut library, 2);
                library
                    .move_or_swap(
                        library.revision(),
                        expected(Position::Candidate, Some(2)),
                        expected(Position::Archive(archive(archive_index)), None),
                    )
                    .unwrap();
                assert!(library
                    .active()
                    .is_none_or(|index| library.resident(index).is_some()));

                library
                    .move_or_swap(
                        library.revision(),
                        expected(Position::Archive(archive(archive_index)), Some(2)),
                        expected(Position::Resident(active), Some(1)),
                    )
                    .unwrap();
                assert!(library
                    .active()
                    .is_none_or(|index| library.resident(index).is_some()));
            }
        }
    }

    #[test]
    fn confirmed_resident_save_discards_the_old_value_and_clears_candidate() {
        let mut library = CalibrationLibrary::new();
        install_candidate(&mut library, 1);
        library
            .save_candidate_to_resident(
                library.revision(),
                Some(CalibrationIdentity::new(1)),
                resident(0),
                None,
            )
            .unwrap();
        install_candidate(&mut library, 2);

        library
            .save_candidate_to_resident(
                library.revision(),
                Some(CalibrationIdentity::new(2)),
                resident(0),
                Some(CalibrationIdentity::new(1)),
            )
            .unwrap();

        assert!(library.candidate().is_none());
        assert_eq!(
            library.resident(resident(0)).unwrap().identity(),
            CalibrationIdentity::new(2)
        );
        assert_eq!(library.active(), Some(resident(0)));
    }
}
