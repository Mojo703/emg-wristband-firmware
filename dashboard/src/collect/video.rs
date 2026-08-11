//! Unit 2: the webcam, as an ffmpeg child process.
//!
//! One [`FfmpegVideoCapture`] owns the camera for the dashboard's lifetime. A
//! v4l2 device cannot be opened twice, so at most one ffmpeg child exists at a
//! time and [`CameraUse`] names which of the three jobs holds it: the setup
//! preview, a session recording, or a short-lived placement photo. Recording
//! and photographing evict a running preview and wait for its child to exit
//! before opening the device themselves; starting a preview while a recording
//! runs is refused. The camera therefore cannot end up held by a preview with
//! no recording running.
//!
//! A capture that produces nothing is an error at the moment it is asked for,
//! not a discovery made afterwards. [`VideoCapture::start_recording`] watches
//! the child for [`READINESS_TIMEOUT`] and only returns once the camera has
//! delivered a frame; ffmpeg dying, or capturing nothing in that window, comes
//! back as an `Err` carrying ffmpeg's own stderr.
//!
//! Readiness and liveness both read ffmpeg's `-progress` stream rather than the
//! output file's size. The matroska muxer writes whole clusters, so a healthy
//! 30 fps recording leaves the file untouched for seconds at a time and a size
//! comparison at the manager's ~500 ms cadence would call it stalled far more
//! often than not; the frame counter tracks the camera, which is what the
//! tripwire is about. The first frame observed dates the recording, and its
//! distance from the session's requested start becomes
//! [`VideoReport::start_offset`]. A child that dies mid-recording stops
//! advancing the counter, and is noticed directly and logged once. The size
//! reported alongside is still stat'd from the file.

use super::interfaces::{RecordingHandle, VideoCapture, VideoReport};
use anyhow::{anyhow, Context};
use protocol::{OffsetMilliseconds, StreamProgress, UnixMilliseconds};
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The preview streams the capture stream itself, scaled down only if the
/// capture is larger than these — so what it frames is the field of view the
/// recording gets, not a different mode's crop.
const PREVIEW_MAXIMUM_WIDTH: u32 = 640;
const PREVIEW_MAXIMUM_FRAMES_PER_SECOND: u32 = 30;
/// mpjpeg part separator, pinned here so the HTTP content type can name it.
pub const PREVIEW_BOUNDARY: &str = "emgpreview";

/// How long a recording is given to put its first bytes on disk before the
/// attempt is called a failure.
const READINESS_TIMEOUT: Duration = Duration::from_secs(3);
/// How long a preview is given to emit its first part.
const PREVIEW_READINESS_TIMEOUT: Duration = Duration::from_secs(3);
/// How long a graceful stop is given before the child is killed outright.
const GRACEFUL_STOP_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a single still is given before the child is killed outright.
const PHOTO_TIMEOUT: Duration = Duration::from_secs(5);
/// Poll granularity while waiting on a child.
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How often ffmpeg reports its captured-frame count. Below the manager's
/// ~500 ms health cadence, so every health check has a fresh number.
const PROGRESS_PERIOD_SECONDS: &str = "0.25";

/// Preview parts buffered toward the browser. Small on purpose: a browser that
/// falls behind should stall the pump, and through it ffmpeg, rather than build
/// a backlog of frames that are stale by the time anyone sees them.
const PREVIEW_QUEUE_DEPTH: usize = 4;
/// Cap on retained ffmpeg stderr. Enough for the diagnosis, bounded so a
/// chatty camera cannot grow it without limit.
const STDERR_RETAINED_BYTES: usize = 4096;

/// What to ask the camera for, and what the recording is therefore made of.
/// The numbers appear verbatim in [`VideoReport::detail`].
///
/// Configurable because the modes a given camera offers cannot be known from
/// here: a device that does not offer the default should be an environment
/// change rather than a rebuild. The default is 240p30, which is what checking
/// a hand against a label needs and no more — a session's video should not
/// outweigh its EMG.
///
/// Bandwidth is set by what the camera is asked to send, not by what is written
/// to disk: capturing 720p and scaling down costs the bus exactly as much as
/// recording 720p. At 240p that cost is small enough not to matter. Above it,
/// name `mjpeg` as the input format — it is compressed on the wire where the
/// usual `yuyv422` is not, and raw 720p30 needs more bandwidth than USB 2.0 has
/// to give.
#[derive(Debug, Clone)]
pub struct CameraSettings {
    pub width: u32,
    pub height: u32,
    pub frames_per_second: u32,
    /// v4l2 input pixel format, or `None` to let the driver choose.
    pub input_format: Option<String>,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            width: 320,
            height: 240,
            frames_per_second: 30,
            input_format: None,
        }
    }
}

impl CameraSettings {
    /// How the geometry reads on the summary screen's files card.
    fn describe(&self) -> String {
        let format = match &self.input_format {
            Some(format) => format!(", {format}"),
            None => String::new(),
        };
        format!(
            "{} fps, {}x{}{format}",
            self.frames_per_second, self.width, self.height
        )
    }
}

/// The camera, driven by ffmpeg child processes.
pub struct FfmpegVideoCapture {
    camera_device: PathBuf,
    settings: CameraSettings,
    /// Who holds the camera, if anyone. The one place the exclusivity lives.
    use_of_camera: Option<CameraUse>,
    /// Handed out with each preview so a later stop can tell whether it is
    /// stopping its own preview or one that has since replaced it.
    next_preview_token: u64,
}

/// The job the single ffmpeg child is doing. A placement photo runs entirely
/// inside its own call and so has no variant here.
enum CameraUse {
    Preview(LivePreview),
    Recording(LiveRecording),
}

/// Identifies one preview, so stopping is idempotent and cannot reach across
/// to a preview that replaced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewToken(u64);

/// A started preview: the token that owns it, and the JPEG parts to write to
/// the browser.
#[derive(Debug)]
pub struct PreviewSession {
    pub token: PreviewToken,
    pub parts: tokio::sync::mpsc::Receiver<Vec<u8>>,
}

/// State of the running preview.
struct LivePreview {
    child: Child,
    token: PreviewToken,
}

/// State of the one in-flight session recording.
struct LiveRecording {
    child: Child,
    output: PathBuf,
    requested_start: UnixMilliseconds,
    captured: CapturedFrames,
    progress: CaptureProgress,
    /// The ffmpeg child was already found dead and logged; don't log again.
    death_reported: bool,
}

/// Frames ffmpeg says it has taken from the camera, followed on a thread that
/// reads its `-progress` stream.
///
/// Liveness has to come from here rather than from the output file's size. The
/// matroska muxer writes whole clusters, so a healthy 30 fps recording leaves
/// the file untouched for seconds at a time and a size comparison taken every
/// 500 ms would call it stalled far more often than not. The frame counter
/// moves with the camera, which is the thing the tripwire is actually about.
#[derive(Clone, Default)]
struct CapturedFrames(Arc<Mutex<u64>>);

impl CapturedFrames {
    /// Follow `progress`, ffmpeg's `-progress pipe:1` stream, to end of file.
    fn following(progress: std::process::ChildStdout) -> Self {
        let count = Arc::new(Mutex::new(0));
        let sink = Arc::clone(&count);
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(progress)
                .lines()
                .map_while(Result::ok)
            {
                if let Some(frames) = line.strip_prefix("frame=") {
                    if let Ok(frames) = frames.trim().parse::<u64>() {
                        *sink.lock().unwrap() = frames;
                    }
                }
            }
        });
        Self(count)
    }

    fn count(&self) -> u64 {
        *self.0.lock().unwrap()
    }
}

/// One recording's progress: the frames seen so far, and the moment the first
/// one arrived — which is what dates the recording against the session's
/// requested start.
#[derive(Debug, Default)]
struct CaptureProgress {
    last_frames: u64,
    first_frame: Option<UnixMilliseconds>,
}

impl CaptureProgress {
    /// Fold one observation in, taken at `now`, and report what it implies.
    fn observe(
        &mut self,
        frames: u64,
        bytes_on_disk: u64,
        now: UnixMilliseconds,
    ) -> StreamProgress {
        let advancing = frames > self.last_frames;
        self.last_frames = frames;
        if frames > 0 && self.first_frame.is_none() {
            self.first_frame = Some(now);
        }
        StreamProgress {
            bytes_on_disk,
            advancing,
        }
    }
}

/// ffmpeg's stderr, drained by a thread so a full pipe can never wedge the
/// child, and readable at any moment for the tail of what it complained about.
struct RetainedStderr(Arc<Mutex<String>>);

impl RetainedStderr {
    /// Take `stderr` off the child and read it until end of file on its own
    /// thread, keeping the last [`STDERR_RETAINED_BYTES`].
    fn draining(mut stderr: ChildStderr) -> Self {
        let text = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&text);
        std::thread::spawn(move || {
            let mut buffer = [0_u8; 1024];
            while let Ok(count) = stderr.read(&mut buffer) {
                if count == 0 {
                    break;
                }
                let mut held = sink.lock().unwrap();
                held.push_str(&String::from_utf8_lossy(&buffer[..count]));
                while held.len() > STDERR_RETAINED_BYTES {
                    let trimmed = held[held.len() - STDERR_RETAINED_BYTES..].to_string();
                    *held = trimmed;
                }
            }
        });
        Self(text)
    }

    /// What ffmpeg has said so far, collapsed to one line for an error message,
    /// or a stand-in when it said nothing.
    fn message(&self) -> String {
        let text = self.0.lock().unwrap();
        let joined = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("; ");
        if joined.is_empty() {
            "ffmpeg reported nothing on stderr".to_string()
        } else {
            joined
        }
    }
}

impl FfmpegVideoCapture {
    /// The camera to record from, e.g. `/dev/video0`, and what to ask it for.
    /// Nothing is opened or probed here.
    pub fn new(camera_device: PathBuf, settings: CameraSettings) -> Self {
        Self {
            camera_device,
            settings,
            use_of_camera: None,
            next_preview_token: 0,
        }
    }

    /// The video input arguments the recording, the preview and the still all
    /// share. Every consumer asks the camera for the same mode, so switching
    /// between them never renegotiates the sensor.
    fn camera_input_arguments(&self) -> Vec<String> {
        let mut arguments = vec![
            "-nostdin".to_string(),
            "-loglevel".to_string(),
            "error".to_string(),
            "-f".to_string(),
            "v4l2".to_string(),
        ];
        if let Some(format) = &self.settings.input_format {
            arguments.push("-input_format".to_string());
            arguments.push(format.clone());
        }
        arguments.extend([
            "-framerate".to_string(),
            self.settings.frames_per_second.to_string(),
            "-video_size".to_string(),
            format!("{}x{}", self.settings.width, self.settings.height),
            "-i".to_string(),
            self.camera_device.display().to_string(),
        ]);
        arguments
    }

    /// A recording running right now, if there is one.
    fn recording(&mut self) -> Option<&mut LiveRecording> {
        match self.use_of_camera.as_mut() {
            Some(CameraUse::Recording(recording)) => Some(recording),
            Some(CameraUse::Preview(_)) | None => None,
        }
    }

    /// Free the camera for a recording or a photo, waiting for the preview's
    /// child to actually exit — the device stays open until it does, and the
    /// next ffmpeg would meet EBUSY.
    fn evict_preview(&mut self) {
        let Some(CameraUse::Preview(mut preview)) = self.use_of_camera.take() else {
            return;
        };
        let _ = preview.child.kill();
        let _ = preview.child.wait();
        tracing::debug!("camera preview stopped; the device is free");
    }

    /// What the preview does to the capture stream on its way to the browser.
    /// Both caps are ceilings, not targets: a 240p capture is already smaller
    /// than the wire needs, so it passes through untouched and the operator
    /// frames the shot against the exact pixels the recording keeps.
    fn preview_filter(&self) -> String {
        let mut stages = Vec::new();
        if self.settings.width > PREVIEW_MAXIMUM_WIDTH {
            stages.push(format!("scale={PREVIEW_MAXIMUM_WIDTH}:-2"));
        }
        if self.settings.frames_per_second > PREVIEW_MAXIMUM_FRAMES_PER_SECOND {
            stages.push(format!("fps={PREVIEW_MAXIMUM_FRAMES_PER_SECOND}"));
        }
        if stages.is_empty() {
            // A filtergraph cannot be empty, and copying is the honest no-op.
            stages.push("null".to_string());
        }
        stages.join(",")
    }

    /// Begin streaming JPEG parts from the camera for the setup preview,
    /// replacing any preview already running. Refused while a recording holds
    /// the camera, which is what keeps a stray browser reconnect from taking
    /// the device out from under a session.
    pub fn start_preview(&mut self) -> anyhow::Result<PreviewSession> {
        if matches!(self.use_of_camera, Some(CameraUse::Recording(_))) {
            return Err(anyhow!(
                "the camera is recording a session. The preview is a setup-time view"
            ));
        }
        self.evict_preview();

        self.next_preview_token += 1;
        let token = PreviewToken(self.next_preview_token);

        let mut child = Command::new("ffmpeg")
            .args(self.camera_input_arguments())
            .args(["-vf", &self.preview_filter()])
            .args([
                "-q:v",
                "7",
                "-f",
                "mpjpeg",
                "-boundary_tag",
                PREVIEW_BOUNDARY,
                "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "spawning ffmpeg to preview {}",
                    self.camera_device.display()
                )
            })?;

        let stderr = RetainedStderr::draining(child.stderr.take().expect("stderr was piped"));
        let mut standard_output = child.stdout.take().expect("stdout was piped");
        let (sender, parts) = tokio::sync::mpsc::channel(PREVIEW_QUEUE_DEPTH);

        // One thread pumps the child's stdout into the channel for as long as
        // the browser reads, and reports on `ready` whether the very first read
        // produced anything — which is what proves the camera opened.
        let (ready_sender, ready) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut buffer = vec![0_u8; 32 * 1024];
            loop {
                let count = match standard_output.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => count,
                };
                // Only the first read answers the readiness question; later
                // sends fail harmlessly once the receiver is gone.
                let _ = ready_sender.send(());
                if sender.blocking_send(buffer[..count].to_vec()).is_err() {
                    break;
                }
            }
        });

        // A failure has to be this call's error rather than an empty stream the
        // browser is left to interpret.
        if ready.recv_timeout(PREVIEW_READINESS_TIMEOUT).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(anyhow!(
                "no preview frame from {} within {} s: {}",
                self.camera_device.display(),
                PREVIEW_READINESS_TIMEOUT.as_secs(),
                stderr.message()
            ));
        }

        tracing::info!(camera = %self.camera_device.display(), "camera preview started");
        self.use_of_camera = Some(CameraUse::Preview(LivePreview { child, token }));
        Ok(PreviewSession { token, parts })
    }

    /// Stop the preview `token` names. Does nothing if that preview has already
    /// been replaced or evicted, so a browser disconnecting late cannot cut off
    /// the preview that succeeded it.
    pub fn stop_preview(&mut self, token: PreviewToken) {
        if matches!(
            self.use_of_camera.as_ref(),
            Some(CameraUse::Preview(preview)) if preview.token == token
        ) {
            self.evict_preview();
        }
    }
}

impl VideoCapture for FfmpegVideoCapture {
    fn start_recording(
        &mut self,
        output: &Path,
        requested_start: UnixMilliseconds,
    ) -> anyhow::Result<RecordingHandle> {
        if self.recording().is_some() {
            return Err(anyhow!(
                "a recording is already running. The camera takes one recording at a time"
            ));
        }
        self.evict_preview();

        let mut child = Command::new("ffmpeg")
            .args(self.camera_input_arguments())
            .args([
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "28",
                // So the size the summary reports tracks what has been written
                // rather than lagging a 32 KB buffer behind it.
                "-flush_packets",
                "1",
                "-stats_period",
                PROGRESS_PERIOD_SECONDS,
                "-progress",
                "pipe:1",
                "-y",
            ])
            .arg(output)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "spawning ffmpeg to record {} to {}",
                    self.camera_device.display(),
                    output.display()
                )
            })?;
        let stderr = RetainedStderr::draining(child.stderr.take().expect("stderr was piped"));
        let captured = CapturedFrames::following(child.stdout.take().expect("stdout was piped"));

        // A frame off the camera is the only proof it opened, so wait for one
        // here: a recording that never starts must fail the session start, not
        // leave an empty file to be discovered afterwards.
        let mut progress = CaptureProgress::default();
        let readiness = wait_for_first_frame(&mut child, output, &captured, &mut progress);
        if let Err(reason) = readiness {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(output);
            return Err(anyhow!(
                "recording {} to {} produced no video within {} s ({reason}): {}",
                self.camera_device.display(),
                output.display(),
                READINESS_TIMEOUT.as_secs(),
                stderr.message()
            ));
        }

        tracing::info!(
            camera = %self.camera_device.display(),
            output = %output.display(),
            "video recording started"
        );

        self.use_of_camera = Some(CameraUse::Recording(LiveRecording {
            child,
            output: output.to_path_buf(),
            requested_start,
            captured,
            progress,
            death_reported: false,
        }));
        Ok(RecordingHandle(()))
    }

    fn capture_photo(&mut self, output: &Path) -> anyhow::Result<()> {
        // The trait allows this call at any time, but a v4l2 device cannot be
        // opened twice: the second ffmpeg would fail with EBUSY and, worse,
        // could look like a camera fault. The collection flow only takes
        // placement photos between sessions, so refuse clearly instead.
        if self.recording().is_some() {
            return Err(anyhow!(
                "cannot take a photo while recording: {} is a v4l2 device and \
                 opens once at a time, so take placement photos before the \
                 session starts",
                self.camera_device.display()
            ));
        }
        self.evict_preview();

        let mut child = Command::new("ffmpeg")
            .args(self.camera_input_arguments())
            .args(["-frames:v", "1", "-y"])
            .arg(output)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "spawning ffmpeg to photograph {} into {}",
                    self.camera_device.display(),
                    output.display()
                )
            })?;
        let stderr = RetainedStderr::draining(child.stderr.take().expect("stderr was piped"));

        let status = match wait_with_timeout(&mut child, PHOTO_TIMEOUT)? {
            Some(status) => status,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!(
                    "ffmpeg did not produce a still from {} within {} seconds: {}",
                    self.camera_device.display(),
                    PHOTO_TIMEOUT.as_secs(),
                    stderr.message()
                ));
            }
        };

        if !status.success() {
            return Err(anyhow!(
                "ffmpeg failed to photograph {} ({status}): {}",
                self.camera_device.display(),
                stderr.message()
            ));
        }
        tracing::info!(output = %output.display(), "placement photo captured");
        Ok(())
    }

    fn health(&mut self, _recording: &RecordingHandle) -> StreamProgress {
        let Some(live) = self.recording() else {
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

        let frames = live.captured.count();
        let bytes = bytes_on_disk(&live.output);
        live.progress
            .observe(frames, bytes, now_unix_milliseconds())
    }

    fn stop_recording(&mut self, _recording: RecordingHandle) -> anyhow::Result<VideoReport> {
        let Some(CameraUse::Recording(mut live)) = self.use_of_camera.take() else {
            return Err(anyhow!("stop_recording called with no recording running"));
        };

        // SIGINT is ffmpeg's request to finish: it stops reading the camera and
        // writes the container's trailer, which is what gives the .mkv a
        // duration and makes it seekable rather than merely playable. It has to
        // be a signal — the child runs with -nostdin, so it never reads the "q"
        // an interactive ffmpeg would take. Interrupting is a non-zero exit by
        // definition, so only a child that ignores it is worth reporting.
        interrupt(&live.child);
        if wait_with_timeout(&mut live.child, GRACEFUL_STOP_TIMEOUT)?.is_none() {
            tracing::warn!(
                output = %live.output.display(),
                "ffmpeg ignored the interrupt for {} seconds; killing it, so the \
                 recording keeps what was flushed but has no trailer",
                GRACEFUL_STOP_TIMEOUT.as_secs()
            );
            let _ = live.child.kill();
            let _ = live.child.wait();
        }

        // One last look, so a recording that was never health-checked still
        // reports its true size.
        let bytes = bytes_on_disk(&live.output);
        live.progress
            .observe(live.captured.count(), bytes, now_unix_milliseconds());

        let start_offset = match live.progress.first_frame {
            Some(first_frame) => first_frame.since(live.requested_start),
            None => OffsetMilliseconds::new(0),
        };
        let detail = self.settings.describe();

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
        // Backend shutdown with the camera in use: an orphaned ffmpeg would
        // hold the device against the next process to start.
        match self.use_of_camera.as_mut() {
            Some(CameraUse::Recording(live)) => {
                tracing::warn!(
                    output = %live.output.display(),
                    "video capture dropped with a recording live; killing ffmpeg"
                );
                let _ = live.child.kill();
                let _ = live.child.wait();
            }
            Some(CameraUse::Preview(preview)) => {
                let _ = preview.child.kill();
                let _ = preview.child.wait();
            }
            None => {}
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

/// Watch a freshly spawned recording until ffmpeg reports its first captured
/// frame, folding that observation into `progress` so the recording dates
/// itself from the moment it truly started. The `Err` is why no video appeared.
fn wait_for_first_frame(
    child: &mut Child,
    output: &Path,
    captured: &CapturedFrames,
    progress: &mut CaptureProgress,
) -> Result<(), String> {
    let deadline = std::time::Instant::now() + READINESS_TIMEOUT;
    loop {
        progress.observe(
            captured.count(),
            bytes_on_disk(output),
            now_unix_milliseconds(),
        );
        if progress.first_frame.is_some() {
            return Ok(());
        }
        // Checked after the counter, so frames from a child that has already
        // finished still count.
        match child.try_wait() {
            Ok(Some(status)) => return Err(format!("ffmpeg exited with {status}")),
            Ok(None) => {}
            Err(error) => return Err(format!("could not wait on ffmpeg: {error}")),
        }
        if std::time::Instant::now() >= deadline {
            return Err("ffmpeg is running but captured no frames".to_string());
        }
        std::thread::sleep(WAIT_POLL_INTERVAL);
    }
}

/// Ask `child` to finish the way a terminal's Ctrl-C would, so ffmpeg
/// finalizes the container instead of dying mid-write.
fn interrupt(child: &Child) {
    // Safety: `kill` on a live child's own pid, with a signal number that is
    // valid on every platform this runs on.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }
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
    fn progress_follows_a_camera_that_delivers_then_stops() {
        let mut progress = CaptureProgress::default();

        // Nothing captured yet: no frames, so nothing advances and the
        // recording has no start moment.
        let before = progress.observe(0, 0, UnixMilliseconds::new(1_000));
        assert!(!before.advancing);
        assert_eq!(progress.first_frame, None);

        // First frames land; this observation dates the recording.
        let starting = progress.observe(7, 589, UnixMilliseconds::new(1_500));
        assert_eq!(starting.bytes_on_disk, 589);
        assert!(starting.advancing);
        assert_eq!(progress.first_frame, Some(UnixMilliseconds::new(1_500)));

        // Still capturing, with the file untouched since: the muxer is holding
        // a cluster, which must not read as a stalled camera.
        let buffered = progress.observe(22, 589, UnixMilliseconds::new(2_000));
        assert!(buffered.advancing);
        assert_eq!(progress.first_frame, Some(UnixMilliseconds::new(1_500)));

        // The camera stops: the frame count sits still, so the tripwire trips
        // even as the muxer flushes what it had.
        let stalled = progress.observe(22, 183_214, UnixMilliseconds::new(2_500));
        assert!(!stalled.advancing);
        assert_eq!(stalled.bytes_on_disk, 183_214);

        // And it stays stalled for as long as no frame arrives.
        assert!(
            !progress
                .observe(22, 183_214, UnixMilliseconds::new(3_000))
                .advancing
        );
    }

    #[test]
    fn start_offset_is_the_first_frame_measured_against_the_request() {
        let requested = UnixMilliseconds::new(10_000);
        let first_frame = UnixMilliseconds::new(10_420);
        assert_eq!(first_frame.since(requested), OffsetMilliseconds::new(420));
    }

    /// The bug this module exists to prevent: a camera that cannot be opened
    /// must fail the call that asked for the recording, carrying ffmpeg's own
    /// reason, and must leave no file behind to be mistaken for a take.
    #[test]
    fn a_camera_that_cannot_be_opened_fails_the_start() {
        if !ffmpeg_available() {
            eprintln!("skipping: no ffmpeg on PATH");
            return;
        }
        let recording_path = scratch_path("missing_camera", "mkv");
        let _ = std::fs::remove_file(&recording_path);
        let mut capture = FfmpegVideoCapture::new(
            PathBuf::from("/dev/video-does-not-exist"),
            CameraSettings::default(),
        );

        let failure = capture
            .start_recording(&recording_path, now_unix_milliseconds())
            .expect_err("a missing camera must fail the start");
        let message = format!("{failure:#}");
        assert!(
            message.contains("produced no video"),
            "unhelpful failure: {message}"
        );
        assert!(
            message.contains("/dev/video-does-not-exist"),
            "the failure must name the camera: {message}"
        );
        assert!(
            !std::fs::metadata(&recording_path).is_ok_and(|data| data.len() > 0),
            "a failed start must not leave a file to be mistaken for a take"
        );

        // Nothing holds the camera afterwards, so the next attempt is free to
        // try rather than being refused as busy.
        let again = capture
            .start_recording(&recording_path, now_unix_milliseconds())
            .expect_err("still no camera");
        assert!(!format!("{again:#}").contains("already running"));

        let _ = std::fs::remove_file(&recording_path);
    }

    /// Likewise for the preview: a camera that will not open is this call's
    /// error, not an empty stream the browser has to puzzle over.
    #[test]
    fn a_preview_of_a_missing_camera_fails_the_start() {
        if !ffmpeg_available() {
            eprintln!("skipping: no ffmpeg on PATH");
            return;
        }
        let mut capture = FfmpegVideoCapture::new(
            PathBuf::from("/dev/video-does-not-exist"),
            CameraSettings::default(),
        );
        let failure = capture
            .start_preview()
            .expect_err("a missing camera must fail the preview");
        assert!(
            format!("{failure:#}").contains("/dev/video-does-not-exist"),
            "the failure must name the camera: {failure:#}"
        );
        assert!(capture.use_of_camera.is_none(), "the camera must be free");
    }

    /// Stopping a preview that has already been replaced must not reach across
    /// and kill its successor.
    #[test]
    fn stopping_a_replaced_preview_leaves_the_current_one_alone() {
        let mut capture =
            FfmpegVideoCapture::new(PathBuf::from("/dev/video0"), CameraSettings::default());
        let stale = PreviewToken(1);
        capture.next_preview_token = 2;
        capture.use_of_camera = Some(CameraUse::Preview(LivePreview {
            child: Command::new("sleep")
                .arg("30")
                .spawn()
                .expect("spawning a stand-in child"),
            token: PreviewToken(2),
        }));

        capture.stop_preview(stale);
        assert!(
            matches!(capture.use_of_camera, Some(CameraUse::Preview(_))),
            "the live preview must survive its predecessor's stop"
        );
        capture.stop_preview(PreviewToken(2));
        assert!(capture.use_of_camera.is_none());
    }

    /// The default has to be the small one: a session's video is a reference
    /// view, and the bus it arrives over is shared with the device link.
    #[test]
    fn capture_defaults_to_240p30() {
        let settings = CameraSettings::default();
        assert_eq!((settings.width, settings.height), (320, 240));
        assert_eq!(settings.frames_per_second, 30);
        assert_eq!(settings.input_format, None);
        assert_eq!(settings.describe(), "30 fps, 320x240");
    }

    /// The preview must not resample a capture that is already small enough,
    /// or the operator frames the shot against pixels the recording never saw.
    #[test]
    fn the_preview_only_shrinks_a_capture_bigger_than_the_wire_needs() {
        let small =
            FfmpegVideoCapture::new(PathBuf::from("/dev/video0"), CameraSettings::default());
        assert_eq!(small.preview_filter(), "null");

        let large = FfmpegVideoCapture::new(
            PathBuf::from("/dev/video0"),
            CameraSettings {
                width: 1280,
                height: 720,
                frames_per_second: 60,
                input_format: Some("mjpeg".to_string()),
            },
        );
        assert_eq!(large.preview_filter(), "scale=640:-2,fps=30");
    }

    /// A named input format has to reach ffmpeg, since it is the difference
    /// between a compressed stream and a raw one on a shared USB bus.
    #[test]
    fn a_named_input_format_reaches_ffmpeg_and_the_report() {
        let settings = CameraSettings {
            width: 640,
            height: 360,
            frames_per_second: 30,
            input_format: Some("mjpeg".to_string()),
        };
        assert_eq!(settings.describe(), "30 fps, 640x360, mjpeg");

        let capture = FfmpegVideoCapture::new(PathBuf::from("/dev/video0"), settings);
        let arguments = capture.camera_input_arguments();
        let format_position = arguments
            .iter()
            .position(|argument| argument == "-input_format")
            .expect("the input format must be passed");
        assert_eq!(arguments[format_position + 1], "mjpeg");
        assert!(arguments.contains(&"640x360".to_string()));
        // Before -i, or ffmpeg reads it as an output option.
        let input_position = arguments.iter().position(|a| a == "-i").expect("input");
        assert!(format_position < input_position);
    }

    #[test]
    fn health_without_a_recording_reports_no_progress() {
        let mut capture =
            FfmpegVideoCapture::new(PathBuf::from("/dev/video0"), CameraSettings::default());
        let progress = capture.health(&RecordingHandle(()));
        assert_eq!(progress.bytes_on_disk, 0);
        assert!(!progress.advancing);
    }
}
