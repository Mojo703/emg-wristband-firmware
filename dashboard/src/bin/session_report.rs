//! `session_report <session directory> [--tracks <directory>]`
//!
//! Reads one recorded collection session and prints whether it is usable for
//! training. Everything it measures is in `dashboard::session_report`.

use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    let mut arguments = std::env::args().skip(1);
    let mut session_directory: Option<PathBuf> = None;
    let mut tracks_root: Option<PathBuf> = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--tracks" => {
                tracks_root = arguments.next().map(PathBuf::from);
                if tracks_root.is_none() {
                    anyhow::bail!("--tracks needs a directory");
                }
            }
            "-h" | "--help" => {
                println!("usage: session_report <session directory> [--tracks <directory>]");
                return Ok(());
            }
            _ if session_directory.is_none() => session_directory = Some(PathBuf::from(argument)),
            _ => anyhow::bail!("unexpected argument {argument}"),
        }
    }
    let Some(session_directory) = session_directory else {
        anyhow::bail!("usage: session_report <session directory> [--tracks <directory>]");
    };

    let tracks_root = tracks_root.unwrap_or_else(|| default_tracks_root(&session_directory));
    let report = dashboard::session_report::analyze(&session_directory, &tracks_root)?;
    print!("{}", dashboard::session_report::render(&report));
    Ok(())
}

/// `EMG_TRACKS_DIR` if it is set, else the library beside the sessions root the
/// session was read from, else the working directory's own.
fn default_tracks_root(session_directory: &Path) -> PathBuf {
    if let Ok(configured) = std::env::var("EMG_TRACKS_DIR") {
        return PathBuf::from(configured);
    }
    let beside = session_directory
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("tracks"));
    match beside {
        Some(path) if path.is_dir() => path,
        _ => PathBuf::from("tracks"),
    }
}
