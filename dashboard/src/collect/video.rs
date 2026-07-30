//! Unit 2: the webcam, as an ffmpeg child process.
//!
//! One [`FfmpegVideoCapture`] owns the camera for the dashboard's lifetime. A
//! v4l2 device cannot be opened twice, so at most one ffmpeg child exists at a
//! time: either a session recording or a short-lived placement-photo run, never
//! both. Nothing is probed at construction — a missing camera or a missing
//! ffmpeg surfaces as an `Err` from [`VideoCapture::start_recording`] or
//! [`VideoCapture::capture_photo`], where a caller can report it.
//!
//! Liveness is measured at the disk, not from ffmpeg's chatter: every
//! [`VideoCapture::health`] call stats the output file and compares its size
//! with the previous call. That same lazy check is what dates the recording —
//! the first call that sees the file grow records the wall-clock moment, and the
//! distance from the session's requested start becomes
//! [`VideoReport::start_offset`]. A crashed ffmpeg therefore reports
//! `advancing: false` forever, which is exactly the tripwire the manager wants;
//! the dead child is also noticed directly and logged once.

use super::interfaces::{RecordingHandle, VideoCapture, VideoReport};
use anyhow::{anyhow, Context};
use protocol::{OffsetMilliseconds, StreamProgress, UnixMilliseconds};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Capture geometry. Fixed rather than configurable: the recording is a
/// reference view of the hand, and the numbers appear verbatim in
/// [`VideoReport::detail`].
const FRAMES_PER_SECOND: u32 = 30;
const CAPTURE_WIDTH: u32 = 1280;
const CAPTURE_HEIGHT: u32 = 720;

/// How long a graceful stop is given before the child is killed outright.
const GRACEFUL_STOP_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a single still is given before the child is killed outright.
const PHOTO_TIMEOUT: Duration = Duration::from_secs(5);
/// Poll granularity while waiting on a child.
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// The camera, driven by ffmpeg child processes.
pub struct FfmpegVideoCapture {
    camera_device: PathBuf,
    /// The running session recording, if any. Also the "camera is busy" flag
    /// that blocks a mid-recording photo.
    live: Option<LiveRecording>,
}

/// State of the one in-flight session recording.
struct LiveRecording {
    child: Child,
    output: PathBuf,
    requested_start: UnixMilliseconds,
    progress: OutputProgress,
    /// The ffmpeg child was already found dead and logged; don't log again.
    death_reported: bool,
}

/// Size tracking for one output file: the previous observation, and the moment
/// the file was first seen to have grown. Separated from the child process so it
/// can be exercised against any file that something else grows.
#[derive(Debug, Default)]
struct OutputProgress {
    last_bytes: u64,
    first_growth: Option<UnixMilliseconds>,
    /// Any observation has been made at all — the first stat of a file that
    /// already has bytes in it counts as growth.
    observed: bool,
}

impl OutputProgress {
    /// Fold one observation of the output's size in, taken at `now`, and report
    /// the disk-level progress it implies.
    fn observe(&mut self, bytes_on_disk: u64, now: UnixMilliseconds) -> StreamProgress {
        let advancing = if self.observed {
            bytes_on_disk > self.last_bytes
        } else {
            bytes_on_disk > 0
        };
        self.observed = true;
        self.last_bytes = bytes_on_disk;
        if advancing && self.first_growth.is_none() {
            self.first_growth = Some(now);
        }
        StreamProgress {
            bytes_on_disk,
            advancing,
        }
    }
}

impl FfmpegVideoCapture {
    /// The camera to record from, e.g. `/dev/video0`. Nothing is opened or
    /// probed here.
    pub fn new(camera_device: PathBuf) -> Self {
        Self {
            camera_device,
            live: None,
        }
    }

    /// The video input arguments both the recording and the still share.
    fn camera_input_arguments(&self) -> Vec<String> {
        vec![
            "-nostdin".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
            "-f".to_string(),
            "v4l2".to_string(),
            "-framerate".to_string(),
            FRAMES_PER_SECOND.to_string(),
            "-video_size".to_string(),
            format!("{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}"),
            "-i".to_string(),
            self.camera_device.display().to_string(),
        ]
    }
}

impl VideoCapture for FfmpegVideoCapture {
    fn start_recording(
        &mut self,
        output: &Path,
        requested_start: UnixMilliseconds,
    ) -> anyhow::Result<RecordingHandle> {
        if self.live.is_some() {
            return Err(anyhow!(
                "a recording is already running; the camera takes one recording at a time"
            ));
        }

        // stdin is piped so a graceful stop can ask ffmpeg to quit by writing
        // "q"; -nostdin keeps ffmpeg from otherwise treating the pipe as a
        // terminal it can read commands from.
        let child = Command::new("ffmpeg")
            .args(self.camera_input_arguments())
            .args([
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "28",
                "-y",
            ])
            .arg(output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!(
                    "spawning ffmpeg to record {} to {}",
                    self.camera_device.display(),
                    output.display()
                )
            })?;

        tracing::info!(
            camera = %self.camera_device.display(),
            output = %output.display(),
            "video recording started"
        );

        // Deliberately no wait for readiness: ffmpeg takes a moment to open the
        // camera and the session must not stall on it. health() finds out when
        // frames actually landed.
        self.live = Some(LiveRecording {
            child,
            output: output.to_path_buf(),
            requested_start,
            progress: OutputProgress::default(),
            death_reported: false,
        });
        Ok(RecordingHandle(()))
    }

    fn capture_photo(&mut self, output: &Path) -> anyhow::Result<()> {
        // The trait allows this call at any time, but a v4l2 device cannot be
        // opened twice: the second ffmpeg would fail with EBUSY and, worse,
        // could look like a camera fault. The collection flow only takes
        // placement photos between sessions, so refuse clearly instead.
        if self.live.is_some() {
            return Err(anyhow!(
                "cannot take a photo while recording: {} is a v4l2 device and \
                 opens once at a time, so take placement photos before the \
                 session starts",
                self.camera_device.display()
            ));
        }

        let mut child = Command::new("ffmpeg")
            .args(self.camera_input_arguments())
            .args(["-frames:v", "1", "-y"])
            .arg(output)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!(
                    "spawning ffmpeg to photograph {} into {}",
                    self.camera_device.display(),
                    output.display()
                )
            })?;

        let status = match wait_with_timeout(&mut child, PHOTO_TIMEOUT)? {
            Some(status) => status,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!(
                    "ffmpeg did not produce a still from {} within {} seconds",
                    self.camera_device.display(),
                    PHOTO_TIMEOUT.as_secs()
                ));
            }
        };

        if !status.success() {
            return Err(anyhow!(
                "ffmpeg failed to photograph {} ({status})",
                self.camera_device.display()
            ));
        }
        tracing::info!(output = %output.display(), "placement photo captured");
        Ok(())
    }

    fn health(&mut self, _recording: &RecordingHandle) -> StreamProgress {
        let Some(live) = self.live.as_mut() else {
            // A handle exists only while a recording does, so this is
            // unreachable through the trait; report honestly rather than panic.
            return StreamProgress {
                bytes_on_disk: 0,
                advancing: false,
            };
        };

        // A dead child cannot grow the file again, so say so once and let the
        // size comparison carry the signal from here on.
        if !live.death_reported {
            if let Ok(Some(status)) = live.child.try_wait() {
                live.death_reported = true;
                tracing::warn!(
                    output = %live.output.display(),
                    "ffmpeg exited while recording ({status}); video will not advance"
                );
            }
        }

        let bytes = bytes_on_disk(&live.output);
        live.progress.observe(bytes, now_unix_milliseconds())
    }

    fn stop_recording(&mut self, _recording: RecordingHandle) -> anyhow::Result<VideoReport> {
        let mut live = self
            .live
            .take()
            .ok_or_else(|| anyhow!("stop_recording called with no recording running"))?;

        // "q" on stdin is ffmpeg's own request to finish: it stops reading the
        // camera and writes the container's trailer, which is what makes the
        // .mkv seekable rather than merely playable.
        if let Some(mut standard_input) = live.child.stdin.take() {
            if let Err(error) = standard_input
                .write_all(b"q\n")
                .and_then(|()| standard_input.flush())
            {
                tracing::warn!("could not ask ffmpeg to quit: {error}");
            }
            // Dropping the pipe also gives ffmpeg end-of-file.
        }

        match wait_with_timeout(&mut live.child, GRACEFUL_STOP_TIMEOUT)? {
            Some(status) if status.success() => {}
            Some(status) => {
                tracing::warn!(
                    output = %live.output.display(),
                    "ffmpeg exited unsuccessfully ({status}); keeping what it wrote"
                );
            }
            None => {
                tracing::warn!(
                    output = %live.output.display(),
                    "ffmpeg ignored the quit request for {} seconds; killing it",
                    GRACEFUL_STOP_TIMEOUT.as_secs()
                );
                let _ = live.child.kill();
                let _ = live.child.wait();
            }
        }

        // One last look, so a recording that was never health-checked still
        // dates itself and reports its true size.
        let bytes = bytes_on_disk(&live.output);
        live.progress.observe(bytes, now_unix_milliseconds());

        let start_offset = match live.progress.first_growth {
            Some(first_growth) => first_growth.since(live.requested_start),
            None => OffsetMilliseconds::new(0),
        };
        let mut detail = format!("{FRAMES_PER_SECOND} fps · {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}");
        if live.progress.first_growth.is_none() {
            detail.push_str(" · start not observed");
        }

        tracing::info!(
            output = %live.output.display(),
            bytes,
            start_offset = start_offset.get(),
            "video recording stopped"
        );
        Ok(VideoReport {
            bytes,
            start_offset,
            detail,
        })
    }
}

impl Drop for FfmpegVideoCapture {
    fn drop(&mut self) {
        // Backend shutdown with a session still running: an orphaned ffmpeg
        // would hold the camera against the next process to start.
        if let Some(live) = self.live.as_mut() {
            tracing::warn!(
                output = %live.output.display(),
                "video capture dropped with a recording live; killing ffmpeg"
            );
            let _ = live.child.kill();
            let _ = live.child.wait();
        }
    }
}

/// Size of a path, or zero if it cannot be stated (ffmpeg has not created it
/// yet, most often). Absence and emptiness mean the same thing here: no frames.
fn bytes_on_disk(path: &Path) -> u64 {
    std::fs::metadata(path).map(|data| data.len()).unwrap_or(0)
}

fn now_unix_milliseconds() -> UnixMilliseconds {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    UnixMilliseconds::new(since_epoch.as_millis() as u64)
}

/// Wait for `child` for at most `limit`, returning `None` on overrun. Polls
/// rather than blocking so the caller keeps the choice of what to do next.
fn wait_with_timeout(child: &mut Child, limit: Duration) -> anyhow::Result<Option<ExitStatus>> {
    let deadline = std::time::Instant::now() + limit;
    loop {
        if let Some(status) = child.try_wait().context("waiting on the ffmpeg child")? {
            return Ok(Some(status));
        }
        if std::time::Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(WAIT_POLL_INTERVAL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique scratch path per test, since no temp-file crate is available.
    fn scratch_path(name: &str, extension: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "dashboard_video_test_{name}_{}_{unique}.{extension}",
            std::process::id()
        ))
    }

    fn append_bytes(path: &Path, count: usize) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("opening the scratch file");
        file.write_all(&vec![0_u8; count]).expect("appending bytes");
        file.flush().expect("flushing");
    }

    /// Is a usable ffmpeg on PATH? Tests needing one skip rather than fail.
    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[test]
    fn progress_follows_a_file_that_grows_then_stalls() {
        let path = scratch_path("growth", "mkv");
        let _ = std::fs::remove_file(&path);
        let mut progress = OutputProgress::default();

        // Nothing written yet: the file does not even exist.
        let before = progress.observe(bytes_on_disk(&path), UnixMilliseconds::new(1_000));
        assert_eq!(before.bytes_on_disk, 0);
        assert!(!before.advancing);
        assert_eq!(progress.first_growth, None);

        // First frames land; this observation dates the recording.
        append_bytes(&path, 64);
        let growing = progress.observe(bytes_on_disk(&path), UnixMilliseconds::new(1_500));
        assert_eq!(growing.bytes_on_disk, 64);
        assert!(growing.advancing);
        assert_eq!(progress.first_growth, Some(UnixMilliseconds::new(1_500)));

        // Still growing: the first-growth moment does not move.
        append_bytes(&path, 32);
        let still_growing = progress.observe(bytes_on_disk(&path), UnixMilliseconds::new(2_000));
        assert_eq!(still_growing.bytes_on_disk, 96);
        assert!(still_growing.advancing);
        assert_eq!(progress.first_growth, Some(UnixMilliseconds::new(1_500)));

        // The writer stops: same size, so not advancing — the tripwire.
        let stalled = progress.observe(bytes_on_disk(&path), UnixMilliseconds::new(2_500));
        assert_eq!(stalled.bytes_on_disk, 96);
        assert!(!stalled.advancing);

        // And it stays stalled for as long as nobody writes.
        let still_stalled = progress.observe(bytes_on_disk(&path), UnixMilliseconds::new(3_000));
        assert!(!still_stalled.advancing);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_file_with_bytes_at_the_first_look_counts_as_growth() {
        let path = scratch_path("prefilled", "mkv");
        let _ = std::fs::remove_file(&path);
        append_bytes(&path, 8);

        let mut progress = OutputProgress::default();
        let first = progress.observe(bytes_on_disk(&path), UnixMilliseconds::new(500));
        assert!(first.advancing);
        assert_eq!(progress.first_growth, Some(UnixMilliseconds::new(500)));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn start_offset_is_the_first_growth_measured_against_the_request() {
        let requested = UnixMilliseconds::new(10_000);
        let first_growth = UnixMilliseconds::new(10_420);
        assert_eq!(first_growth.since(requested), OffsetMilliseconds::new(420));
    }

    #[test]
    fn a_photo_is_refused_while_a_recording_is_live() {
        if !ffmpeg_available() {
            eprintln!("skipping: no ffmpeg on PATH");
            return;
        }
        // A camera is not needed: start_recording only spawns the child, so the
        // capture is "live" from this side even where ffmpeg will fail to open
        // the device.
        let recording_path = scratch_path("busy", "mkv");
        let photo_path = scratch_path("busy", "jpg");
        let mut capture = FfmpegVideoCapture::new(PathBuf::from("/dev/video-does-not-exist"));
        let handle = capture
            .start_recording(&recording_path, now_unix_milliseconds())
            .expect("spawning ffmpeg should succeed even for a bogus camera");

        let refusal = capture
            .capture_photo(&photo_path)
            .expect_err("a photo mid-recording must be refused");
        let message = refusal.to_string();
        assert!(
            message.contains("cannot take a photo while recording"),
            "unhelpful refusal: {message}"
        );

        // And a second recording is refused too.
        let second = capture
            .start_recording(&recording_path, now_unix_milliseconds())
            .expect_err("a second recording must be refused");
        assert!(second.to_string().contains("already running"));

        let report = capture.stop_recording(handle).expect("stopping");
        // The bogus camera means no frames, so the offset falls back to zero
        // and the detail says why.
        assert_eq!(report.start_offset, OffsetMilliseconds::new(0));
        assert!(report.detail.contains("30 fps · 1280x720"));
        assert!(report.detail.contains("start not observed"));

        // Once stopped, the photo path is open again (it will fail on the
        // missing device, not on the busy check).
        let photo = capture.capture_photo(&photo_path);
        if let Err(error) = photo {
            assert!(!error
                .to_string()
                .contains("cannot take a photo while recording"));
        }

        let _ = std::fs::remove_file(&recording_path);
        let _ = std::fs::remove_file(&photo_path);
    }

    #[test]
    fn health_without_a_recording_reports_no_progress() {
        let mut capture = FfmpegVideoCapture::new(PathBuf::from("/dev/video0"));
        let progress = capture.health(&RecordingHandle(()));
        assert_eq!(progress.bytes_on_disk, 0);
        assert!(!progress.advancing);
    }
}
