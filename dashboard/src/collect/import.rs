//! Adding and removing tracks: a Beat Saber map zip in, a track directory out.
//!
//! Both intake paths — a zip uploaded from the browser and a BeatSaver key the
//! backend downloads — produce the same thing, the bytes of a map archive, so
//! everything after [`import_archive`] is shared. From the archive exactly three
//! files are taken: `Info.dat`, the one difficulty [`beatsaber::read_info`]
//! settled on, and the audio the info file names. Entries are matched by file
//! name alone and nothing is ever written to a path the archive chose, so a
//! crafted entry name cannot escape the track directory.
//!
//! The audio is copied verbatim; the only thing read out of it here is the
//! duration, which an Ogg container states in its last page's granule position.
//! Decoding happens later, in the mixer that plays the track.
//!
//! A track is built in a temporary directory beside the library and renamed into
//! place, so an import that fails partway leaves nothing for
//! [`TrackCatalog::load`](super::beatmap::TrackCatalog::load) to trip over.

use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context};
use serde::Serialize;

use super::beatmap::{LevelEntry, TrackEntry, TRACK_AUDIO_NAME, TRACK_FILE_NAME};
use super::beatsaber::{self, LevelSummary};
use super::calibration_level::{CalibrationLevelProduct, CALIBRATION_SOURCE_LEVEL};

/// The largest map archive accepted. Beat Saber maps run a few megabytes of Ogg
/// plus a cover image; ten times the largest map in the library is room enough
/// to be generous without letting an upload fill memory.
pub const MAXIMUM_ARCHIVE_BYTES: usize = 128 * 1024 * 1024;

/// The most entries a map archive may hold. A map is a handful of files; a
/// listing this long is not one.
const MAXIMUM_ARCHIVE_ENTRIES: usize = 512;

/// The largest single file taken out of an archive, which bounds what a
/// deliberately over-compressed entry can cost.
const MAXIMUM_ENTRY_BYTES: u64 = 128 * 1024 * 1024;

/// Where the map a track was converted from is kept, so a conversion-rule change
/// can be replayed without the original archive.
const SOURCE_DIRECTORY_NAME: &str = "source";

/// What an import produces, and what the picker needs to show the result: the
/// track's identity plus, per level, whether the song is worth collecting on.
#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub id: String,
    pub title: String,
    pub beats_per_minute: f64,
    pub duration_ms: u32,
    /// The Beat Saber difficulty file the map's own preference order chose.
    pub difficulty_file: String,
    pub levels: Vec<LevelSummary>,
    pub calibration: CalibrationImportReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationImportAvailability {
    Available,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationImportReport {
    pub availability: CalibrationImportAvailability,
    pub cue_count: usize,
    pub content_identity: String,
    pub cue_shortfall: usize,
}

/// Convert one map archive into a track directory under `tracks_root`.
///
/// Blocking: decompression, conversion, and the filesystem work all happen on
/// the calling thread.
pub fn import_archive(archive_bytes: &[u8], tracks_root: &Path) -> anyhow::Result<ImportReport> {
    if archive_bytes.len() > MAXIMUM_ARCHIVE_BYTES {
        anyhow::bail!(
            "the map archive is {} MB; the importer accepts at most {} MB",
            archive_bytes.len() / (1024 * 1024),
            MAXIMUM_ARCHIVE_BYTES / (1024 * 1024)
        );
    }
    let mut archive = zip::ZipArchive::new(Cursor::new(archive_bytes))
        .context("the upload is not a readable zip archive")?;
    if archive.len() > MAXIMUM_ARCHIVE_ENTRIES {
        anyhow::bail!(
            "the archive holds {} entries; a Beat Saber map holds a handful",
            archive.len()
        );
    }

    let (info_name, info_bytes) =
        take_entry(&mut archive, |name| name.eq_ignore_ascii_case("info.dat"))?
            .ok_or_else(|| anyhow!("the archive holds no Info.dat"))?;
    let info_text = String::from_utf8(info_bytes).context("Info.dat is not UTF-8")?;
    let info = beatsaber::read_info(&info_text)?;

    let (difficulty_name, difficulty_bytes) =
        take_entry(&mut archive, |name| name == info.difficulty_file_name)?.ok_or_else(|| {
            anyhow!(
                "Info.dat names the difficulty file {} but the archive does not hold it",
                info.difficulty_file_name
            )
        })?;
    let difficulty_text =
        String::from_utf8(difficulty_bytes).context("the difficulty file is not UTF-8")?;
    let difficulty = beatsaber::read_difficulty(&difficulty_text)?;

    let (_, audio_bytes) = take_entry(&mut archive, |name| name == info.audio_file_name)?
        .ok_or_else(|| {
            anyhow!(
                "Info.dat names the audio file {} but the archive does not hold it",
                info.audio_file_name
            )
        })?;
    let duration_ms = ogg_duration_milliseconds(&audio_bytes)
        .with_context(|| format!("reading the duration of {}", info.audio_file_name))?;

    // v4 keeps its tempo map in a file of its own, which the source copy needs
    // too so a conversion-rule change can be replayed from the track directory.
    let audio_data = match &info.audio_data_file_name {
        Some(name) => take_entry(&mut archive, |entry| entry == name)?
            .and_then(|(name, bytes)| Some((name, String::from_utf8(bytes).ok()?))),
        None => None,
    };
    let audio_clock = audio_data
        .as_ref()
        .and_then(|(_, text)| beatsaber::read_audio_data_clock(text));

    let (levels, summaries) =
        beatsaber::convert(&difficulty, info.beats_per_minute, audio_clock, duration_ms)?;
    let track_id = track_id_of(&info.title)?;
    let entry = build_imported_track_entry(
        protocol::TrackId(track_id.clone()),
        info.title.clone(),
        info.beats_per_minute,
        duration_ms,
        levels,
    );

    let mut source_files = vec![
        (info_name.as_str(), info_text.as_bytes()),
        (difficulty_name.as_str(), difficulty_text.as_bytes()),
    ];
    if let Some((name, text)) = &audio_data {
        source_files.push((name.as_str(), text.as_bytes()));
    }
    write_track_directory(
        tracks_root,
        &directory_name_of(&track_id),
        &entry,
        &audio_bytes,
        &source_files,
    )?;

    Ok(ImportReport {
        id: track_id,
        title: info.title,
        beats_per_minute: entry.beats_per_minute,
        duration_ms,
        difficulty_file: difficulty_name,
        levels: summaries,
        calibration: calibration_import_report(&entry),
    })
}

fn calibration_import_report(entry: &TrackEntry) -> CalibrationImportReport {
    let product = entry
        .calibration
        .as_ref()
        .expect("new imports always contain a calibration product");
    CalibrationImportReport {
        availability: CalibrationImportAvailability::Available,
        cue_count: product.cue_count(),
        content_identity: product.content_identity.clone(),
        cue_shortfall: crate::collect::calibration_level::MAXIMUM_CUES
            .saturating_sub(product.cue_count()),
    }
}

fn build_imported_track_entry(
    id: protocol::TrackId,
    title: String,
    beats_per_minute: f64,
    duration_ms: u32,
    levels: std::collections::BTreeMap<String, LevelEntry>,
) -> TrackEntry {
    let calibration_source = levels
        .get(CALIBRATION_SOURCE_LEVEL)
        .expect("the converter always produces the hard source level");
    let calibration = Some(CalibrationLevelProduct::generate(
        &calibration_source.map_notes,
        duration_ms,
    ));
    TrackEntry {
        id,
        title,
        beats_per_minute: (beats_per_minute * 100.0).round() / 100.0,
        duration_ms,
        levels,
        calibration,
        rest: None,
    }
}

/// Remove one track's directory from the library.
pub fn delete_track(track_id: &str, tracks_root: &Path) -> anyhow::Result<()> {
    let directory = tracks_root.join(track_directory_name(track_id)?);
    if !directory.join(TRACK_FILE_NAME).is_file() {
        anyhow::bail!("no track {track_id} in the library");
    }
    std::fs::remove_dir_all(&directory).with_context(|| format!("removing {}", directory.display()))
}

/// The directory name a track id maps to, refusing anything that is not one.
///
/// Ids come from the browser, so this is the only thing standing between a
/// request and an arbitrary path: a track id is lowercase alphanumerics with
/// separators, which cannot spell `..` or a path separator.
fn track_directory_name(track_id: &str) -> anyhow::Result<String> {
    let acceptable = !track_id.is_empty()
        && track_id.chars().all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || character == '_'
                || character == '-'
        });
    if !acceptable {
        anyhow::bail!("'{track_id}' is not a track id");
    }
    Ok(directory_name_of(track_id))
}

/// A track id as a directory name.
fn directory_name_of(track_id: &str) -> String {
    track_id.replace('_', "-")
}

/// The catalog id derived from a title: lowercase, runs of anything else
/// collapsed to one underscore.
fn track_id_of(title: &str) -> anyhow::Result<String> {
    let mut id = String::new();
    for character in title.to_lowercase().chars() {
        if character.is_ascii_lowercase() || character.is_ascii_digit() {
            id.push(character);
        } else if !id.ends_with('_') {
            id.push('_');
        }
    }
    let id = id.trim_matches('_').to_string();
    if id.is_empty() {
        anyhow::bail!("cannot derive a track id from the title '{title}'");
    }
    Ok(id)
}

/// Build the track in a temporary directory beside the library, then rename it
/// into place — the whole track appears at once or not at all.
fn write_track_directory(
    tracks_root: &Path,
    directory_name: &str,
    entry: &TrackEntry,
    audio_bytes: &[u8],
    source_files: &[(&str, &[u8])],
) -> anyhow::Result<()> {
    let destination = tracks_root.join(directory_name);
    if destination.exists() {
        anyhow::bail!(
            "a track named {directory_name} is already in the library; delete it first to \
             re-import it"
        );
    }
    std::fs::create_dir_all(tracks_root)
        .with_context(|| format!("creating the track library {}", tracks_root.display()))?;

    // Beside the library rather than in the system temporary directory, so the
    // rename cannot cross a filesystem boundary.
    let staging = tracks_root.join(format!(".importing-{directory_name}"));
    let _ = std::fs::remove_dir_all(&staging);
    let staged = || -> anyhow::Result<()> {
        std::fs::create_dir_all(staging.join(SOURCE_DIRECTORY_NAME))?;
        std::fs::write(staging.join(TRACK_AUDIO_NAME), audio_bytes)?;
        std::fs::write(
            staging.join(TRACK_FILE_NAME),
            serde_json::to_string_pretty(entry)? + "\n",
        )?;
        for (name, bytes) in source_files {
            let name = Path::new(name)
                .file_name()
                .ok_or_else(|| anyhow!("the archive entry {name} has no file name"))?;
            std::fs::write(staging.join(SOURCE_DIRECTORY_NAME).join(name), bytes)?;
        }
        Ok(())
    }();
    if let Err(error) = staged {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error).context("writing the track directory");
    }
    std::fs::rename(&staging, &destination)
        .with_context(|| format!("moving the imported track into {}", destination.display()))
}

/// Atomically replace only a track's derived manifest. The retained source and
/// personal audio are outside the destination path and are never opened here.
/// HTTP regeneration wiring can call this after rebuilding a [`TrackEntry`]
/// from the retained source directory.
pub fn replace_derived_metadata(track_directory: &Path, entry: &TrackEntry) -> anyhow::Result<()> {
    if let Some(calibration) = &entry.calibration {
        let source = entry
            .levels
            .get(CALIBRATION_SOURCE_LEVEL)
            .ok_or_else(|| anyhow!("regenerated metadata has no hard source level"))?;
        calibration
            .validate_against_source(&source.map_notes, entry.duration_ms)
            .context("validating regenerated calibration metadata")?;
    }
    let bytes =
        serde_json::to_vec_pretty(entry).context("serializing regenerated track metadata")?;
    let destination = track_directory.join(TRACK_FILE_NAME);
    replace_file_atomically(&destination, |file| {
        file.write_all(&bytes)?;
        file.write_all(b"\n")
    })
    .with_context(|| format!("replacing {}", destination.display()))
}

static ATOMIC_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn replace_file_atomically(
    destination: &Path,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let parent = destination.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "atomic destination has no parent directory",
        )
    })?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "atomic destination has no UTF-8 file name",
            )
        })?;
    let sequence = ATOMIC_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));

    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        write(&mut file)?;
        file.sync_all()?;
        std::fs::rename(&temporary, destination)?;
        std::fs::File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// Read one archive entry by file name, ignoring whatever directory path the
/// entry claims to sit in — a map's files sit either at the archive root or
/// inside one directory, and neither placement changes what is taken.
fn take_entry(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    matches: impl Fn(&str) -> bool,
) -> anyhow::Result<Option<(String, Vec<u8>)>> {
    let found = (0..archive.len()).find_map(|index| {
        let entry = archive.by_index_raw(index).ok()?;
        if !entry.is_file() {
            return None;
        }
        let file_name = file_name_of(entry.name());
        matches(&file_name).then_some((index, file_name))
    });
    let Some((index, file_name)) = found else {
        return Ok(None);
    };

    let entry = archive.by_index(index)?;
    if entry.size() > MAXIMUM_ENTRY_BYTES {
        anyhow::bail!(
            "the archive entry {file_name} unpacks to {} MB",
            entry.size() / (1024 * 1024)
        );
    }
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry
        .take(MAXIMUM_ENTRY_BYTES)
        .read_to_end(&mut bytes)
        .with_context(|| format!("unpacking {file_name}"))?;
    Ok(Some((file_name, bytes)))
}

/// The last component of an archive entry's name, taking both separators a zip
/// writer might have used.
fn file_name_of(entry_name: &str) -> String {
    entry_name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(entry_name)
        .to_string()
}

/// Length of an Ogg Vorbis stream in milliseconds: the sample rate from the
/// Vorbis identification header, and the total sample count from the granule
/// position of the stream's last page.
pub fn ogg_duration_milliseconds(bytes: &[u8]) -> anyhow::Result<u32> {
    const CAPTURE_PATTERN: &[u8; 4] = b"OggS";
    const VORBIS_IDENTIFICATION: &[u8; 7] = b"\x01vorbis";

    if bytes.len() < 4 || &bytes[..4] != CAPTURE_PATTERN {
        anyhow::bail!("the audio is not an Ogg container; Beat Saber maps ship Ogg Vorbis");
    }

    let mut sample_rate = None;
    let mut vorbis_stream = None;
    let mut last_granule = 0u64;
    let mut offset = 0usize;
    while offset + 27 <= bytes.len() {
        if &bytes[offset..offset + 4] != CAPTURE_PATTERN {
            anyhow::bail!("the Ogg stream is truncated or corrupt at byte {offset}");
        }
        let granule = u64::from_le_bytes(bytes[offset + 6..offset + 14].try_into().unwrap());
        let stream = u32::from_le_bytes(bytes[offset + 14..offset + 18].try_into().unwrap());
        let segment_count = usize::from(bytes[offset + 26]);
        let table_end = offset + 27 + segment_count;
        if table_end > bytes.len() {
            break;
        }
        let payload_length: usize = bytes[offset + 27..table_end]
            .iter()
            .map(|&length| usize::from(length))
            .sum();
        let payload_end = table_end + payload_length;
        if payload_end > bytes.len() {
            break;
        }

        let payload = &bytes[table_end..payload_end];
        if sample_rate.is_none()
            && payload.len() >= 16
            && payload.starts_with(VORBIS_IDENTIFICATION)
        {
            sample_rate = Some(u32::from_le_bytes(payload[12..16].try_into().unwrap()));
            vorbis_stream = Some(stream);
        }
        if vorbis_stream == Some(stream) && granule != u64::MAX {
            last_granule = last_granule.max(granule);
        }
        offset = payload_end;
    }

    let sample_rate = sample_rate
        .filter(|&rate| rate > 0)
        .ok_or_else(|| anyhow!("the Ogg container holds no Vorbis stream"))?;
    if last_granule == 0 {
        anyhow::bail!("the Vorbis stream states no length");
    }
    Ok((last_granule as f64 / f64::from(sample_rate) * 1_000.0).round() as u32)
}

/// A BeatSaver map key or a link to one, as the browser typed it.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatSaverKey(String);

impl std::fmt::Display for BeatSaverKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl BeatSaverKey {
    /// Read a key out of what the browser sent: a bare key (`858`) or a map
    /// page's address (`https://beatsaver.com/maps/858`).
    pub fn parse(reference: &str) -> anyhow::Result<BeatSaverKey> {
        let trimmed = reference.trim().trim_end_matches('/');
        let candidate = match trimmed.rsplit_once("/maps/") {
            Some((_, tail)) => tail.split('/').next().unwrap_or_default(),
            None => trimmed,
        };
        let key = candidate.to_lowercase();
        if key.is_empty()
            || !key
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
        {
            anyhow::bail!(
                "'{reference}' is neither a BeatSaver key nor a beatsaver.com map address"
            );
        }
        Ok(BeatSaverKey(key))
    }
}

/// BeatSaver refuses requests without one, so the importer names itself.
const USER_AGENT: &str = concat!("emg-dashboard/", env!("CARGO_PKG_VERSION"));

/// Fetch one map's archive from BeatSaver.
///
/// The three ways this fails are told apart in the message, because the fix
/// differs: the network, the key, and the map itself.
pub async fn download_map(key: &BeatSaverKey) -> anyhow::Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("building the BeatSaver client")?;

    let metadata = client
        .get(format!("https://api.beatsaver.com/maps/id/{key}"))
        .send()
        .await
        .map_err(|error| anyhow!("cannot reach BeatSaver: {error}"))?;
    if metadata.status() == reqwest::StatusCode::NOT_FOUND {
        anyhow::bail!("BeatSaver has no map with the key {key}");
    }
    let metadata: serde_json::Value = metadata
        .error_for_status()
        .map_err(|error| anyhow!("BeatSaver refused the lookup of {key}: {error}"))?
        .json()
        .await
        .map_err(|error| anyhow!("BeatSaver's answer for {key} was unreadable: {error}"))?;

    let download_url = metadata
        .get("versions")
        .and_then(serde_json::Value::as_array)
        .and_then(|versions| versions.first())
        .and_then(|version| version.get("downloadURL"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("BeatSaver lists no downloadable version of {key}"))?;

    let archive = client
        .get(download_url)
        .send()
        .await
        .map_err(|error| anyhow!("cannot reach BeatSaver's download host: {error}"))?
        .error_for_status()
        .map_err(|error| anyhow!("BeatSaver would not serve the map {key}: {error}"))?
        .bytes()
        .await
        .map_err(|error| anyhow!("the download of {key} was cut short: {error}"))?;
    if archive.len() > MAXIMUM_ARCHIVE_BYTES {
        anyhow::bail!(
            "the map {key} is {} MB; the importer accepts at most {} MB",
            archive.len() / (1024 * 1024),
            MAXIMUM_ARCHIVE_BYTES / (1024 * 1024)
        );
    }
    Ok(archive.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collect::beatmap::{assign_columns, LevelEntry, MapNote, MAXIMUM_COLUMNS};

    fn unique_test_directory(name: &str) -> std::path::PathBuf {
        let sequence = ATOMIC_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "dashboard-import-{name}-{}-{sequence}",
            std::process::id()
        ))
    }

    fn test_track_entry(id: &str) -> TrackEntry {
        let map_notes = (0..12)
            .map(|index| MapNote {
                time_ms: 1_000 + index * 2_500,
                cell: (index % 12) as u8,
                hold_ms: 1_250,
            })
            .collect::<Vec<_>>();
        let level = LevelEntry {
            column_assignments: (1..=MAXIMUM_COLUMNS)
                .map(|count| (count.to_string(), assign_columns(&map_notes, count)))
                .collect(),
            map_notes,
        };
        let levels = ["easy", "medium", "hard"]
            .into_iter()
            .map(|name| (name.to_string(), level.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let calibration = Some(CalibrationLevelProduct::generate(
            &levels[CALIBRATION_SOURCE_LEVEL].map_notes,
            60_000,
        ));
        TrackEntry {
            id: protocol::TrackId(id.to_string()),
            title: id.to_string(),
            beats_per_minute: 120.0,
            duration_ms: 60_000,
            levels,
            calibration,
            rest: None,
        }
    }

    #[test]
    fn a_title_becomes_a_track_id_and_a_directory_name() {
        assert_eq!(
            track_id_of("LiSa (Sword Art Online) - Crossing Field").unwrap(),
            "lisa_sword_art_online_crossing_field"
        );
        assert_eq!(
            track_id_of("K-forest - A W A K E").unwrap(),
            "k_forest_a_w_a_k_e"
        );
        assert_eq!(
            directory_name_of("toby_fox_the_world_revolving"),
            "toby-fox-the-world-revolving"
        );
        assert!(track_id_of("???").is_err());
    }

    #[test]
    fn a_track_id_that_could_name_another_directory_is_refused() {
        assert!(track_directory_name("../../etc").is_err());
        assert!(track_directory_name("a/b").is_err());
        assert!(track_directory_name("..").is_err());
        assert!(track_directory_name("").is_err());
        assert!(track_directory_name("Uppercase").is_err());
        assert_eq!(
            track_directory_name("usao_perfect_army").unwrap(),
            "usao-perfect-army"
        );
    }

    #[test]
    fn a_beatsaver_reference_reduces_to_its_key() {
        assert_eq!(BeatSaverKey::parse("858").unwrap().to_string(), "858");
        assert_eq!(
            BeatSaverKey::parse("https://beatsaver.com/maps/2Ff1")
                .unwrap()
                .to_string(),
            "2ff1"
        );
        assert_eq!(
            BeatSaverKey::parse(" https://beatsaver.com/maps/858/ ")
                .unwrap()
                .to_string(),
            "858"
        );
        assert!(BeatSaverKey::parse("").is_err());
        assert!(BeatSaverKey::parse("https://beatsaver.com/maps/").is_err());
        assert!(BeatSaverKey::parse("not a key").is_err());
    }

    #[test]
    fn an_entry_name_reduces_to_its_file_name() {
        assert_eq!(file_name_of("Info.dat"), "Info.dat");
        assert_eq!(file_name_of("Some Map/Info.dat"), "Info.dat");
        assert_eq!(file_name_of("../../../Info.dat"), "Info.dat");
        assert_eq!(file_name_of("windows\\Info.dat"), "Info.dat");
    }

    #[test]
    fn audio_that_is_not_an_ogg_container_is_refused() {
        let error = ogg_duration_milliseconds(b"ID3\x04not an ogg at all")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not an Ogg container"), "{error}");
        assert!(ogg_duration_milliseconds(b"").is_err());
    }

    /// One Ogg page: the header, its segment table, and the payload.
    fn ogg_page(granule: u64, stream: u32, payload: &[u8]) -> Vec<u8> {
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS");
        page.extend_from_slice(&[0, 0]);
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&stream.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes());
        let segments: Vec<u8> = payload.chunks(255).map(|chunk| chunk.len() as u8).collect();
        page.push(segments.len() as u8);
        page.extend_from_slice(&segments);
        page.extend_from_slice(payload);
        page
    }

    fn identification_header(sample_rate: u32) -> Vec<u8> {
        let mut packet = b"\x01vorbis".to_vec();
        packet.extend_from_slice(&0u32.to_le_bytes());
        packet.push(2);
        packet.extend_from_slice(&sample_rate.to_le_bytes());
        packet.extend_from_slice(&[0; 12]);
        packet
    }

    #[test]
    fn the_duration_is_the_last_granule_over_the_sample_rate() {
        let mut stream = ogg_page(0, 7, &identification_header(44_100));
        stream.extend(ogg_page(44_100, 7, b"audio"));
        // 2.5 seconds of samples at 44.1 kHz.
        stream.extend(ogg_page(110_250, 7, b"audio"));
        assert_eq!(ogg_duration_milliseconds(&stream).unwrap(), 2_500);
    }

    #[test]
    fn a_second_logical_stream_does_not_set_the_length() {
        let mut stream = ogg_page(0, 7, &identification_header(48_000));
        stream.extend(ogg_page(48_000, 7, b"audio"));
        // A cover-art or skeleton stream with a granule on its own timeline.
        stream.extend(ogg_page(9_999_999, 9, b"other"));
        assert_eq!(ogg_duration_milliseconds(&stream).unwrap(), 1_000);
    }

    #[test]
    fn an_ogg_without_a_vorbis_header_is_refused() {
        let stream = ogg_page(48_000, 7, b"\x80theora placeholder payload");
        let error = ogg_duration_milliseconds(&stream).unwrap_err().to_string();
        assert!(error.contains("no Vorbis stream"), "{error}");
    }

    #[test]
    fn atomic_metadata_replacement_never_touches_audio() {
        let directory = unique_test_directory("atomic-metadata");
        std::fs::create_dir_all(&directory).unwrap();
        let audio_path = directory.join(TRACK_AUDIO_NAME);
        std::fs::write(&audio_path, b"personal audio").unwrap();
        std::fs::write(directory.join(TRACK_FILE_NAME), b"old metadata").unwrap();
        let entry = test_track_entry("replacement");

        replace_derived_metadata(&directory, &entry).unwrap();

        assert_eq!(std::fs::read(&audio_path).unwrap(), b"personal audio");
        let loaded: TrackEntry =
            serde_json::from_slice(&std::fs::read(directory.join(TRACK_FILE_NAME)).unwrap())
                .unwrap();
        assert_eq!(loaded.id.0, "replacement");
        assert!(loaded.calibration.is_some());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn imported_entry_adds_calibration_from_hard_without_changing_ordinary_levels() {
        let original = test_track_entry("source");
        let levels = original.levels.clone();
        let entry = build_imported_track_entry(
            protocol::TrackId("imported".into()),
            "Imported".into(),
            120.0,
            60_000,
            levels.clone(),
        );
        let calibration = entry.calibration.as_ref().unwrap();

        assert_eq!(
            serde_json::to_vec(&entry.levels).unwrap(),
            serde_json::to_vec(&levels).unwrap()
        );
        assert_eq!(calibration.source_level, "hard");
        assert!(calibration.notes.iter().all(|cue| {
            let source = &entry.levels["hard"].map_notes[cue.source_index];
            cue.map_note.time_ms == source.time_ms && cue.map_note.cell == source.cell
        }));
        calibration.validate().unwrap();
    }

    #[test]
    fn import_report_exposes_calibration_availability_count_and_identity() {
        let entry = test_track_entry("reported");
        let report = calibration_import_report(&entry);
        let product = entry.calibration.as_ref().unwrap();

        assert_eq!(
            report.availability,
            CalibrationImportAvailability::Available
        );
        assert_eq!(report.cue_count, product.cue_count());
        assert_eq!(report.content_identity, product.content_identity);
        assert_eq!(
            report.cue_shortfall,
            crate::collect::calibration_level::MAXIMUM_CUES - product.cue_count()
        );
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["availability"], "available");
        assert!(json.get("available").is_none());
    }

    #[test]
    fn failed_atomic_write_keeps_previous_metadata_and_audio() {
        let directory = unique_test_directory("failed-atomic-metadata");
        std::fs::create_dir_all(&directory).unwrap();
        let metadata_path = directory.join(TRACK_FILE_NAME);
        let audio_path = directory.join(TRACK_AUDIO_NAME);
        std::fs::write(&metadata_path, b"old metadata").unwrap();
        std::fs::write(&audio_path, b"personal audio").unwrap();

        let result = replace_file_atomically(&metadata_path, |file| {
            file.write_all(b"partial replacement")?;
            Err(std::io::Error::other("injected failure"))
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read(&metadata_path).unwrap(), b"old metadata");
        assert_eq!(std::fs::read(&audio_path).unwrap(), b"personal audio");
        let _ = std::fs::remove_dir_all(directory);
    }
}
