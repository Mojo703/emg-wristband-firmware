//! What the backend has to remember between sessions, because neither the
//! device nor the setup form can supply it.
//!
//! The board and harness a device is soldered to are invisible to the firmware,
//! so the operator says once per device id and every later session reuses it.
//! The don count is not in any single session either: it is how many times the
//! band has gone on this subject's arm, which only something outliving the
//! session can count. The audio settings are here for the same reason: an
//! operator sets the level and the output device once and expects the next
//! session to sound the same.
//!
//! All of it lives in one CBOR file under `config/`, alongside `collection.json` and
//! ignored by git like the rest of the runtime state there. A file that cannot
//! be read is reported and replaced rather than failing the backend's start: the
//! cost of losing it is re-entering a board revision, and refusing to run would
//! be worse.

use anyhow::Context;
use protocol::{Arm, BoardRevision, SubjectId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The highest don number recorded for one subject's arm. A struct rather than a
/// bare number so a stored entry can gain a field without the whole file becoming
/// unreadable.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct DonHistory {
    count: u32,
}

/// How loud the music sits under the cue clicks when nobody has said. Well
/// below the clicks, which are the point of the exercise.
const DEFAULT_VOLUME_PERMILLE: u32 = 600;

/// How the game sounded last time. `output` is a device name, or `None` for the
/// host's default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredAudio {
    pub output: Option<String>,
    pub volume_permille: u32,
}

impl Default for StoredAudio {
    fn default() -> Self {
        Self {
            output: None,
            volume_permille: DEFAULT_VOLUME_PERMILLE,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct StoredProvenance {
    /// Board and harness per device id.
    boards: BTreeMap<String, BoardRevision>,
    /// The highest don number per subject and arm, keyed by [`don_key`].
    dons: BTreeMap<String, DonHistory>,
    /// Absent in a store written before the audio controls existed, which is
    /// why it is an `Option` rather than a defaulted struct: an operator who
    /// has never touched the controls gets the default, not a stored zero.
    #[serde(default)]
    audio: Option<StoredAudio>,
}

/// The map key for one subject's arm. A pair key would encode as a CBOR array
/// key, which nothing that reads this file by eye handles gracefully.
fn don_key(subject: &SubjectId, arm: Arm) -> String {
    let arm = match arm {
        Arm::Left => "left",
        Arm::Right => "right",
    };
    format!("{subject}/{arm}")
}

pub struct ProvenanceStore {
    path: PathBuf,
    state: Mutex<StoredProvenance>,
}

impl ProvenanceStore {
    /// Load the store, or start empty if the file is absent or unreadable.
    pub fn load(path: PathBuf) -> ProvenanceStore {
        let state = match std::fs::read(&path) {
            Ok(bytes) => match ciborium::from_reader(bytes.as_slice()) {
                Ok(state) => state,
                Err(error) => {
                    tracing::warn!(
                        "{} is not a provenance store ({error}); starting empty",
                        path.display()
                    );
                    StoredProvenance::default()
                }
            },
            Err(_) => StoredProvenance::default(),
        };
        ProvenanceStore {
            path,
            state: Mutex::new(state),
        }
    }

    /// What the operator last said this device is soldered to.
    pub fn board_revision(&self, device_id: &str) -> Option<BoardRevision> {
        self.state.lock().unwrap().boards.get(device_id).cloned()
    }

    /// Record a device's board and harness, replacing whatever was remembered.
    pub fn set_board_revision(&self, device_id: &str, revision: BoardRevision) {
        self.state
            .lock()
            .unwrap()
            .boards
            .insert(device_id.to_string(), revision);
        self.write();
    }

    /// The don number for a session about to be recorded: the highest this
    /// subject's arm has been given, plus one. Every session is its own don, so a
    /// second take counts again — nothing here inspects the `donned` stamp.
    pub fn next_don_count(&self, subject: &SubjectId, arm: Arm) -> u32 {
        let count = {
            let mut state = self.state.lock().unwrap();
            let history = state.dons.entry(don_key(subject, arm)).or_default();
            history.count += 1;
            history.count
        };
        self.write();
        count
    }

    /// How the game sounded last time, or the defaults if it has never been set.
    pub fn audio(&self) -> StoredAudio {
        self.state.lock().unwrap().audio.clone().unwrap_or_default()
    }

    /// Remember how the game sounds.
    pub fn set_audio(&self, audio: StoredAudio) {
        self.state.lock().unwrap().audio = Some(audio);
        self.write();
    }

    /// Persist the whole store. Staged and renamed, so a crash mid-write leaves
    /// the previous file rather than a truncated one.
    fn write(&self) {
        if let Err(error) = self.write_or_fail() {
            tracing::warn!("provenance store not saved: {error:#}");
        }
    }

    fn write_or_fail(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut encoded = Vec::new();
        ciborium::into_writer(&*self.state.lock().unwrap(), &mut encoded)
            .context("encoding the provenance store")?;
        let temporary = self.path.with_extension("cbor.tmp");
        std::fs::write(&temporary, &encoded)
            .with_context(|| format!("writing {}", temporary.display()))?;
        std::fs::rename(&temporary, &self.path)
            .with_context(|| format!("renaming into {}", self.path.display()))?;
        Ok(())
    }
}

/// The store's default location, beside the collection config.
pub fn default_path() -> PathBuf {
    Path::new("config").join("provenance.cbor")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_path(label: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("dashboard-provenance-{label}-{stamp}.cbor"))
    }

    #[test]
    fn a_board_revision_outlives_the_process_that_entered_it() {
        let path = temporary_path("boards");
        let store = ProvenanceStore::load(path.clone());
        assert_eq!(store.board_revision("opal-1a2b3c"), None);
        store.set_board_revision(
            "opal-1a2b3c",
            BoardRevision {
                board: "rev A bodged".to_string(),
                harness: "ribbon 2".to_string(),
            },
        );

        let reopened = ProvenanceStore::load(path.clone());
        let revision = reopened.board_revision("opal-1a2b3c").unwrap();
        assert_eq!(revision.board, "rev A bodged");
        assert_eq!(revision.harness, "ribbon 2");
        assert_eq!(reopened.board_revision("opal-other"), None);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn the_don_count_is_the_highest_stored_plus_one_per_arm() {
        let path = temporary_path("dons");
        let store = ProvenanceStore::load(path.clone());
        let subject = SubjectId("matthew".to_string());

        assert_eq!(store.next_don_count(&subject, Arm::Right), 1);
        assert_eq!(store.next_don_count(&subject, Arm::Right), 2);
        // The other arm is its own band placement, so its own count.
        assert_eq!(store.next_don_count(&subject, Arm::Left), 1);
        // As is another subject's.
        assert_eq!(
            store.next_don_count(&SubjectId("alex".to_string()), Arm::Right),
            1
        );

        // A restart continues from what is stored rather than from zero.
        let reopened = ProvenanceStore::load(path.clone());
        assert_eq!(reopened.next_don_count(&subject, Arm::Right), 3);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn an_unreadable_store_starts_empty_instead_of_failing() {
        let path = temporary_path("garbage");
        std::fs::write(&path, b"not cbor at all").unwrap();
        let store = ProvenanceStore::load(path.clone());
        assert_eq!(store.board_revision("opal-1a2b3c"), None);
        std::fs::remove_file(&path).unwrap();
    }
}
