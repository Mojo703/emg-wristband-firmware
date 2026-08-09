//! Which of a model's training rows go where.
//!
//! The live-plus-flash split is the architecture the full training matrix
//! needs, so a bench case has to say which rows are streamed into RAM and which
//! are baked into the flash partition — and the two selections must be exact
//! complements, or the fit trains on a row twice or not at all.
//!
//! `model.json` lists `training_sources` in the order the rows appear in
//! `training_rows.npy`, each with a session, a role, and a row count summing to
//! the whole set. That makes a selection a set of contiguous ranges, which is
//! what this module resolves. The complement check is the point: a filter that
//! silently kept nothing would produce an empty partition and a fit that looked
//! like it worked.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// One entry of `model.json`'s `training_sources`, with the row range it
/// occupies resolved.
#[derive(Clone, Debug)]
pub struct Source {
    pub session: String,
    pub role: String,
    pub start: usize,
    pub rows: usize,
}

impl Source {
    pub fn end(&self) -> usize {
        self.start + self.rows
    }
}

/// Read the source breakdown and check it against the row matrix it describes.
pub fn read(directory: &Path, total_rows: usize) -> Result<Vec<Source>> {
    let path = directory.join("model.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).context("parse model.json")?;
    let entries = json["training_sources"]
        .as_array()
        .with_context(|| format!("{} has no training_sources", path.display()))?;

    let mut sources = Vec::with_capacity(entries.len());
    let mut start = 0usize;
    for entry in entries {
        let rows = entry["rows"].as_u64().context("source has no rows")? as usize;
        sources.push(Source {
            session: entry["session"]
                .as_str()
                .context("source has no session")?
                .to_string(),
            role: entry["role"]
                .as_str()
                .context("source has no role")?
                .to_string(),
            start,
            rows,
        });
        start += rows;
    }
    if start != total_rows {
        bail!(
            "training_sources account for {start} rows, training_rows.npy holds {total_rows}; \
             the breakdown cannot be used to split them"
        );
    }
    Ok(sources)
}

/// A selection over the sources, by session and by role. Empty lists match
/// everything, so a case that names nothing gets the whole set.
pub struct Selection {
    pub sessions: Vec<String>,
    pub roles: Vec<String>,
    /// Whether the named sources are the ones kept or the ones dropped.
    pub exclude: bool,
}

impl Selection {
    /// Whether the filter names this source.
    ///
    /// With both lists non-empty a source is named only if it matches **both** —
    /// `--session A --role command` means "session A's command rows", not
    /// "everything from A plus every command row". That reading is worth
    /// stating because it is the surprising one for an exclusion, where
    /// "exclude A and exclude commands" sounds like a union.
    ///
    /// It must not be changed to a union on the exclusion side alone.
    /// [`Selection::resolve`] partitions by `names(source) != exclude`, so the
    /// include and exclude halves of the *same* filter are exact complements
    /// whatever this returns — which is what keeps a row from landing in both
    /// the flash image and the live stream. Making exclusion a union while
    /// inclusion stayed an intersection would break that, and break it in the
    /// direction of double-training a row. The test below covers the combined
    /// case for exactly this reason.
    fn names(&self, source: &Source) -> bool {
        let session_matches = self.sessions.is_empty() || self.sessions.contains(&source.session);
        let role_matches = self.roles.is_empty() || self.roles.contains(&source.role);
        // Naming nothing at all cannot mean "every source is named": an exclude
        // filter with no names must drop nothing, not everything.
        if self.sessions.is_empty() && self.roles.is_empty() {
            return !self.exclude;
        }
        session_matches && role_matches
    }

    /// The selected sources, and the row indices they cover.
    pub fn resolve<'a>(&self, sources: &'a [Source]) -> Result<(Vec<&'a Source>, Vec<usize>)> {
        for name in &self.sessions {
            if !sources.iter().any(|source| source.session == *name) {
                bail!("no training source from session {name}");
            }
        }
        for name in &self.roles {
            if !sources.iter().any(|source| source.role == *name) {
                bail!("no training source with role {name}");
            }
        }
        let kept: Vec<&Source> = sources
            .iter()
            .filter(|source| self.names(source) != self.exclude)
            .collect();
        if kept.is_empty() {
            bail!("the filter selected no rows");
        }
        let mut indices = Vec::new();
        for source in &kept {
            indices.extend(source.start..source.end());
        }
        Ok((kept, indices))
    }
}

/// Gather the selected rows out of a row-major feature matrix.
pub fn gather(values: &[f32], stride: usize, indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * stride);
    for index in indices {
        out.extend_from_slice(&values[index * stride..(index + 1) * stride]);
    }
    out
}

/// One line per selected source, so a case's split is visible in its log.
pub fn describe(kept: &[&Source]) -> String {
    kept.iter()
        .map(|source| format!("{} {} ({} rows)", source.session, source.role, source.rows))
        .collect::<Vec<_>>()
        .join("\n  ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources() -> Vec<Source> {
        let counts = [
            ("modifier", "command", 750),
            ("rest_a", "rest", 720),
            ("noop_a", "no_op", 1200),
        ];
        let mut start = 0;
        counts
            .iter()
            .map(|(session, role, rows)| {
                let source = Source {
                    session: (*session).into(),
                    role: (*role).into(),
                    start,
                    rows: *rows,
                };
                start += rows;
                source
            })
            .collect()
    }

    #[test]
    fn including_a_session_takes_exactly_its_rows() {
        let selection = Selection {
            sessions: vec!["modifier".into()],
            roles: Vec::new(),
            exclude: false,
        };
        let sources = sources();
        let (kept, indices) = selection.resolve(&sources).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(indices.len(), 750);
        assert_eq!(indices[0], 0);
        assert_eq!(*indices.last().unwrap(), 749);
    }

    /// The property the split depends on: what goes to flash and what is
    /// streamed must partition the row set, with no row in both and none in
    /// neither.
    #[test]
    fn include_and_exclude_stay_complements_when_both_filters_are_set() {
        let sources = sources();
        let total: usize = sources.iter().map(|source| source.rows).sum();
        // Every shape a case can name, including the two-flag combination the
        // single-flag tests never reach.
        let filters = [
            (vec!["modifier"], vec![]),
            (vec![], vec!["command"]),
            (vec!["modifier"], vec!["command"]),
            (vec!["modifier", "rest_static"], vec!["command", "rest"]),
        ];
        for (sessions, roles) in filters {
            let build = |exclude: bool| Selection {
                sessions: sessions.iter().map(|name| name.to_string()).collect(),
                roles: roles.iter().map(|name| name.to_string()).collect(),
                exclude,
            };
            let live = build(false).resolve(&sources);
            let flash = build(true).resolve(&sources);
            // One half may legitimately be empty and refuse; the pair is only
            // meaningful when both resolve.
            let (Ok((_, live_rows)), Ok((_, flash_rows))) = (live, flash) else {
                continue;
            };
            let mut all = [live_rows, flash_rows].concat();
            all.sort_unstable();
            assert_eq!(
                all,
                (0..total).collect::<Vec<_>>(),
                "sessions {sessions:?} roles {roles:?}: the two halves are not a partition, \
                 so a row is either trained twice or not at all"
            );
        }
    }

    #[test]
    fn include_and_exclude_of_the_same_filter_are_complements() {
        let sources = sources();
        let live = Selection {
            sessions: vec!["modifier".into()],
            roles: Vec::new(),
            exclude: false,
        };
        let flash = Selection {
            sessions: vec!["modifier".into()],
            roles: Vec::new(),
            exclude: true,
        };
        let (_, live_rows) = live.resolve(&sources).unwrap();
        let (_, flash_rows) = flash.resolve(&sources).unwrap();
        let total: usize = sources.iter().map(|source| source.rows).sum();
        assert_eq!(live_rows.len() + flash_rows.len(), total);
        let mut all = [live_rows, flash_rows].concat();
        all.sort_unstable();
        assert!(all.windows(2).all(|pair| pair[0] != pair[1]), "row in both");
        assert_eq!(all, (0..total).collect::<Vec<_>>());
    }

    #[test]
    fn an_empty_filter_keeps_everything_and_excludes_nothing() {
        let sources = sources();
        let total: usize = sources.iter().map(|source| source.rows).sum();
        let keep = Selection {
            sessions: Vec::new(),
            roles: Vec::new(),
            exclude: false,
        };
        assert_eq!(keep.resolve(&sources).unwrap().1.len(), total);
        let drop_nothing = Selection {
            sessions: Vec::new(),
            roles: Vec::new(),
            exclude: true,
        };
        assert_eq!(drop_nothing.resolve(&sources).unwrap().1.len(), total);
    }

    #[test]
    fn a_name_that_matches_no_source_is_an_error_rather_than_an_empty_set() {
        // A typo'd session name silently selecting nothing is how a partition
        // ends up empty and a fit looks like it worked.
        let selection = Selection {
            sessions: vec!["typo".into()],
            roles: Vec::new(),
            exclude: true,
        };
        assert!(selection.resolve(&sources()).is_err());
    }

    #[test]
    fn a_filter_that_selects_every_source_is_refused() {
        let selection = Selection {
            sessions: Vec::new(),
            roles: vec!["command".into(), "rest".into(), "no_op".into()],
            exclude: true,
        };
        assert!(selection.resolve(&sources()).is_err());
    }

    #[test]
    fn gather_takes_whole_rows_in_index_order() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let gathered = gather(&values, 3, &[2, 0]);
        assert_eq!(gathered, vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0]);
    }
}
