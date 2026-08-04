use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    embuild::espidf::sysenv::output();
    emit_build_identity();
}

/// Stamp the build's git commit, working-tree state, and time into the binary as
/// environment variables `src/provenance.rs` reads. A device reports these on
/// connect, so a recorded session names the build that produced it.
fn emit_build_identity() {
    let repository = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .expect("the firmware crate has a parent directory")
        .to_path_buf();

    // A commit or a checkout has to re-run this, or the stamp names whatever was
    // current the last time the crate happened to rebuild.
    let git_directory = repository.join(".git");
    for path in ["HEAD", "index"] {
        println!(
            "cargo:rerun-if-changed={}",
            git_directory.join(path).display()
        );
    }

    let commit = git(&repository, &["rev-parse", "--short", "HEAD"]).unwrap_or_default();
    // Only the sources that end up in this binary count as dirty: a change under
    // `dashboard/` does not alter the firmware, and calling the build modified
    // because of one would make the flag mean nothing.
    let modified = git(
        &repository,
        &[
            "status",
            "--porcelain",
            "--",
            "opal-firmware",
            "protocol",
            "emg-runtime",
        ],
    )
    .is_some_and(|changes| !changes.is_empty());
    let built_at = Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_default();

    println!("cargo:rustc-env=FIRMWARE_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=FIRMWARE_WORKING_TREE_MODIFIED={modified}");
    println!("cargo:rustc-env=FIRMWARE_BUILT_AT={built_at}");
}

/// One git invocation's trimmed output, or `None` if git is unavailable or the
/// command failed — a build outside a checkout still has to succeed.
fn git(repository: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
