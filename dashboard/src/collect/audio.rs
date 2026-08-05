//! Backend audio playback for the collection game: the track, the cue clicks and
//! the debug metronome, mixed into one output stream the backend can read an
//! exact position out of.
//!
//! The whole track is decoded into memory up front. That buys two things the
//! session manager depends on. Seeking is exact, because a position is an index.
//! And the playhead is exact, because it is the count of frames the mixer has
//! submitted rather than an estimate read back out of a player — [`Timeline`]
//! pairs a track position with the instant the subject will *hear* it, which is
//! the anchor every cue event is logged against.
//!
//! The gap between submitting a sample and hearing it is the output device's
//! latency, measured from the stream's own timestamps rather than asked of the
//! driver (a driver that answers zero, as this host's does, would put every cue
//! instant out by the length of the device's buffer). It is already inside
//! [`Timeline::heard_at`], so a cue logged against the timeline is in heard
//! time and needs no correction later. [`Playback::output_latency`] exposes the
//! figure for the log line at session start and nothing else records it: a
//! recorded copy is something a reader could subtract a second time.
//!
//! `EMG_AUDIO_OUTPUT` picks the sink this backend starts on. Unset means the
//! default output device; a name (or part of one) picks a specific device;
//! `silent` runs the same mixer against no device at all, paced by the wall
//! clock, which is what automated runs use so a test does not play music at
//! whoever is sitting there. The operator can change device and level while a
//! track plays; `silent` is the one setting a browser cannot undo.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use protocol::{DurationMilliseconds, TrackMilliseconds, UnixMilliseconds};

/// How many frames the silent sink renders per iteration, which is also the
/// latency it reports. Close enough to a real ALSA period that the timeline
/// behaves the same way under test as it does on the bench.
const SILENT_SINK_FRAMES: u64 = 1024;

/// Buffers the stream must have run before its reported delay is taken as the
/// output latency, and how long opening will wait for that.
const SETTLING_BUFFERS: u64 = 16;
const MEASUREMENT_TIMEOUT: Duration = Duration::from_secs(1);

/// The cue click: loud and high, on every note onset.
const CUE_CLICK_HERTZ: f32 = 1100.0;
const CUE_CLICK_SECONDS: f32 = 0.04;
const CUE_CLICK_GAIN: f32 = 0.25;

/// The debug metronome: soft and low, on every measured beat, so the grid the
/// notes were scheduled on is audible under the cue clicks.
const METRONOME_HERTZ: f32 = 700.0;
const METRONOME_SECONDS: f32 = 0.025;
const METRONOME_GAIN: f32 = 0.1;

/// Simultaneous clicks the mixer will voice. A cue onset landing on a beat is
/// two; the headroom is for a burst of sixteenth notes.
const VOICE_CAPACITY: usize = 8;

fn now() -> UnixMilliseconds {
    let since_epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch");
    UnixMilliseconds::new(since_epoch.as_millis() as u64)
}

/// One track's audio, fully decoded, interleaved by channel.
pub struct DecodedTrack {
    sample_rate: u32,
    channels: usize,
    samples: Vec<f32>,
}

impl DecodedTrack {
    fn frames(&self) -> u64 {
        (self.samples.len() / self.channels.max(1)) as u64
    }

    /// One channel's sample at a fractional frame index, linearly interpolated.
    /// Reading past the end gives silence, so the mixer can run the cursor on
    /// past the audio without a bounds check of its own.
    fn sample_at(&self, frame: f64, channel: usize) -> f32 {
        let frames = self.frames();
        if frame < 0.0 || frames == 0 {
            return 0.0;
        }
        let lower = frame.floor();
        let index = lower as u64;
        if index >= frames {
            return 0.0;
        }
        let fraction = (frame - lower) as f32;
        let channel = channel.min(self.channels - 1);
        let at = |frame_index: u64| -> f32 {
            let offset = frame_index as usize * self.channels + channel;
            self.samples.get(offset).copied().unwrap_or(0.0)
        };
        let first = at(index);
        let second = if index + 1 < frames {
            at(index + 1)
        } else {
            0.0
        };
        first + (second - first) * fraction
    }
}

/// Decode a whole audio file into memory.
pub fn decode(path: &Path) -> anyhow::Result<DecodedTrack> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .with_context(|| format!("opening track audio {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("reading {}", path.display()))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("{} holds no decodable audio track", path.display()))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| format!("no decoder for {}", path.display()))?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels = track
        .codec_params
        .channels
        .map_or(0, |channels| channels.count());
    let mut samples: Vec<f32> = Vec::new();
    let mut buffer: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(Error::ResetRequired) => break,
            Err(error) => return Err(anyhow::Error::new(error).context("reading an audio packet")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A damaged packet is a hole in the music, not a reason to refuse
            // the take; the position and the schedule are unaffected.
            Err(Error::DecodeError(_)) => continue,
            Err(error) => return Err(anyhow::Error::new(error).context("decoding audio")),
        };
        let spec = *decoded.spec();
        sample_rate = spec.rate;
        channels = spec.channels.count();
        let target =
            buffer.get_or_insert_with(|| SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
        target.copy_interleaved_ref(decoded);
        samples.extend_from_slice(target.samples());
    }
    if sample_rate == 0 || channels == 0 || samples.is_empty() {
        anyhow::bail!("{} decoded to no audio", path.display());
    }
    Ok(DecodedTrack {
        sample_rate,
        channels,
        samples,
    })
}

/// Which sink the game plays through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioOutput {
    /// The host's default output device.
    Default,
    /// The first output device whose name contains this text.
    Named(String),
    /// No device: the mixer still runs and the timeline still advances in real
    /// time, but nothing is audible.
    Silent,
}

/// Every output device this host offers, for the operator's picker. Enumerating
/// is a driver call that can fail or hang on a wedged sound server; a failure
/// reads as no choices rather than as a reason not to run.
pub fn output_devices() -> Vec<String> {
    let host = cpal::default_host();
    let Ok(devices) = host.output_devices() else {
        return Vec::new();
    };
    devices.filter_map(|device| device.name().ok()).collect()
}

/// Read the sink out of `EMG_AUDIO_OUTPUT`.
pub fn output_from_environment() -> AudioOutput {
    match std::env::var("EMG_AUDIO_OUTPUT") {
        Err(_) => AudioOutput::Default,
        Ok(value) => match value.trim() {
            "" => AudioOutput::Default,
            "silent" => AudioOutput::Silent,
            name => AudioOutput::Named(name.to_string()),
        },
    }
}

/// Where playback stands and when that will reach the subject's ears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeline {
    pub position: TrackMilliseconds,
    /// The instant `position` is heard, which is a little after the read
    /// because it includes the output device's latency.
    pub heard_at: UnixMilliseconds,
    pub playing: bool,
    /// The cursor has run past the end of the decoded audio.
    pub finished: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClickKind {
    Cue,
    Beat,
}

struct Voice {
    kind: ClickKind,
    offset: usize,
}

struct MixerState {
    /// Output frames rendered so far, which is the playhead. Everything else
    /// about position is derived from this one number.
    cursor: u64,
    playing: bool,
    voices: Vec<Voice>,
    /// Index of the next click in the schedule; re-derived after a seek.
    next_click: usize,
}

struct Shared {
    /// Shared rather than owned so switching the output device rebuilds the
    /// mixer around the same decoded audio instead of reading the file again.
    track: Arc<DecodedTrack>,
    /// Click onsets in output frames, time-ordered, with the tone each wants.
    clicks: Vec<(u64, ClickKind)>,
    cue_waveform: Vec<f32>,
    beat_waveform: Vec<f32>,
    output_sample_rate: u32,
    output_channels: usize,
    /// Source frames per output frame, for the resampling read.
    source_step: f64,
    /// Output frames the whole track occupies.
    track_output_frames: u64,
    state: Mutex<MixerState>,
    timeline: Mutex<Timeline>,
    pacing: Mutex<Pacing>,
    /// The music's level, as `f32` bits. The clicks keep their own gains: they
    /// are the cue, not the entertainment, and turning the music down is how an
    /// operator makes them clearer.
    music_gain: AtomicU32,
    /// The smoothed queue depth, in microseconds, for readers.
    latency_microseconds: AtomicU64,
    /// Buffers the sink has rendered. Opening waits for a few, because the
    /// output settles into real-time pacing rather than starting there.
    buffers: AtomicU64,
    stop: AtomicBool,
}

/// How the sink is running ahead of real time — the whole of the output
/// latency, measured against nothing but the wall clock.
struct Pacing {
    /// When the first buffer was rendered, which is when the sink began
    /// consuming in real time.
    started: Instant,
    /// Frames handed to the sink so far.
    submitted: u64,
    /// Smoothed queue depth in frames, which is the reported output latency.
    smoothed_queue: f64,
}

impl Shared {
    fn frames_to_position(&self, frames: u64) -> TrackMilliseconds {
        let milliseconds = frames.saturating_mul(1000) / u64::from(self.output_sample_rate);
        TrackMilliseconds::new(milliseconds.min(u64::from(u32::MAX)) as u32)
    }

    /// Place the cursor, and put the click schedule wherever that lands.
    /// Without the second half, rebuilding the mixer at a position mid-track
    /// would fire every click the track has already played.
    fn seek(&self, frames: u64, playing: bool) {
        let mut state = self.state.lock().unwrap();
        state.cursor = frames;
        state.voices.clear();
        state.next_click = self
            .clicks
            .partition_point(|(click_frame, _)| *click_frame <= frames);
        state.playing = playing;
    }

    /// Render one buffer and publish where its first frame lands, and when the
    /// subject will hear it.
    ///
    /// The output latency is the sink's queue depth: frames submitted, less the
    /// frames real time has had room for since the sink started consuming. It
    /// is measured here rather than asked of the driver, because a driver that
    /// answers zero (this host's does) would silently put every cue instant out
    /// by the length of the device's buffer.
    fn render(&self, output: &mut [f32]) {
        let frames = (output.len() / self.output_channels) as u64;
        let queue_frames = {
            let mut pacing = self.pacing.lock().unwrap();
            let rate = u64::from(self.output_sample_rate);
            let first = self.buffers.load(Ordering::Acquire) == 0;
            if first {
                // Real time starts when the sink starts consuming, not when the
                // stream was built; opening a device takes tens of milliseconds.
                pacing.started = Instant::now();
            }
            let elapsed_frames = (pacing.started.elapsed().as_micros() as u64) * rate / 1_000_000;
            let queue = pacing.submitted.saturating_sub(elapsed_frames) as f64;
            pacing.smoothed_queue = if first {
                queue
            } else {
                pacing.smoothed_queue * 0.75 + queue * 0.25
            };
            self.latency_microseconds.store(
                (pacing.smoothed_queue * 1_000_000.0 / rate as f64) as u64,
                Ordering::Release,
            );
            pacing.submitted += frames;
            queue
        };
        let heard_at = UnixMilliseconds::new(
            now().get() + (queue_frames * 1000.0 / f64::from(self.output_sample_rate)) as u64,
        );
        self.buffers.fetch_add(1, Ordering::Release);

        let mut state = self.state.lock().unwrap();
        let start = state.cursor;
        let playing = state.playing;
        *self.timeline.lock().unwrap() = Timeline {
            position: self.frames_to_position(start),
            heard_at,
            playing,
            finished: start >= self.track_output_frames,
        };

        let frames = frames as usize;
        if !playing {
            output.fill(0.0);
            return;
        }
        for frame in 0..frames {
            let cursor = state.cursor;
            while let Some((_, kind)) = self
                .clicks
                .get(state.next_click)
                .copied()
                .filter(|(click_frame, _)| *click_frame <= cursor)
            {
                if state.voices.len() < VOICE_CAPACITY {
                    state.voices.push(Voice { kind, offset: 0 });
                }
                state.next_click += 1;
            }

            let mut click = 0.0f32;
            for voice in &mut state.voices {
                let waveform = match voice.kind {
                    ClickKind::Cue => &self.cue_waveform,
                    ClickKind::Beat => &self.beat_waveform,
                };
                click += waveform.get(voice.offset).copied().unwrap_or(0.0);
                voice.offset += 1;
            }
            state.voices.retain(|voice| {
                let length = match voice.kind {
                    ClickKind::Cue => self.cue_waveform.len(),
                    ClickKind::Beat => self.beat_waveform.len(),
                };
                voice.offset < length
            });

            let source_frame = cursor as f64 * self.source_step;
            let gain = f32::from_bits(self.music_gain.load(Ordering::Relaxed));
            for channel in 0..self.output_channels {
                let music = self.track.sample_at(source_frame, channel) * gain;
                output[frame * self.output_channels + channel] = (music + click).clamp(-1.0, 1.0);
            }
            state.cursor = cursor + 1;
        }
    }
}

/// Everything the mixer needs, with the click schedule resolved onto the output
/// stream's own frame grid.
fn build_mixer(
    track: Arc<DecodedTrack>,
    note_onsets: &[TrackMilliseconds],
    beat_times: &[TrackMilliseconds],
    output_sample_rate: u32,
    output_channels: usize,
    music_gain: f32,
) -> Arc<Shared> {
    let source_step = f64::from(track.sample_rate) / f64::from(output_sample_rate);
    let track_output_frames = (track.frames() as f64 / source_step).ceil() as u64;
    let mut clicks: Vec<(u64, ClickKind)> = note_onsets
        .iter()
        .map(|onset| (onset, ClickKind::Cue))
        .chain(beat_times.iter().map(|beat| (beat, ClickKind::Beat)))
        .map(|(position, kind)| {
            (
                u64::from(position.get()) * u64::from(output_sample_rate) / 1000,
                kind,
            )
        })
        .collect();
    clicks.sort_by_key(|(frame, _)| *frame);

    Arc::new(Shared {
        track,
        clicks,
        cue_waveform: click_waveform(
            output_sample_rate,
            CUE_CLICK_HERTZ,
            CUE_CLICK_SECONDS,
            CUE_CLICK_GAIN,
        ),
        beat_waveform: click_waveform(
            output_sample_rate,
            METRONOME_HERTZ,
            METRONOME_SECONDS,
            METRONOME_GAIN,
        ),
        output_sample_rate,
        output_channels,
        source_step,
        track_output_frames,
        state: Mutex::new(MixerState {
            cursor: 0,
            playing: false,
            voices: Vec::with_capacity(VOICE_CAPACITY),
            next_click: 0,
        }),
        timeline: Mutex::new(Timeline {
            position: TrackMilliseconds::new(0),
            heard_at: now(),
            playing: false,
            finished: false,
        }),
        pacing: Mutex::new(Pacing {
            started: Instant::now(),
            submitted: 0,
            smoothed_queue: 0.0,
        }),
        music_gain: AtomicU32::new(music_gain.to_bits()),
        latency_microseconds: AtomicU64::new(0),
        buffers: AtomicU64::new(0),
        stop: AtomicBool::new(false),
    })
}

/// A square-wave burst with an exponential decay, the same shape the browser's
/// oscillator produced.
fn click_waveform(sample_rate: u32, hertz: f32, seconds: f32, gain: f32) -> Vec<f32> {
    let length = (sample_rate as f32 * seconds) as usize;
    (0..length)
        .map(|index| {
            let time = index as f32 / sample_rate as f32;
            let phase = time * hertz;
            let square = if phase - phase.floor() < 0.5 {
                1.0
            } else {
                -1.0
            };
            let envelope = gain * (0.001f32 / gain.max(0.001)).powf(time / seconds.max(0.001));
            square * envelope
        })
        .collect()
}

/// The running mixer, plus everything a rebuild of it needs: switching the
/// output device builds a second mixer around the same decoded audio and the
/// same schedule, at whatever rate the new device runs.
pub struct Playback {
    shared: Arc<Shared>,
    sink: Option<std::thread::JoinHandle<()>>,
    output_name: String,
    output: AudioOutput,
    track: Arc<DecodedTrack>,
    note_onsets: Vec<TrackMilliseconds>,
    beat_times: Vec<TrackMilliseconds>,
}

impl Drop for Playback {
    fn drop(&mut self) {
        self.stop_sink();
    }
}

impl Playback {
    /// Decode a track, build the click schedule, and open the sink. Playback is
    /// silent until [`Playback::play`]: the stream runs from here so the output
    /// latency is measured before the session's manifest is written.
    pub fn open(
        audio_path: &Path,
        note_onsets: &[TrackMilliseconds],
        beat_times: &[TrackMilliseconds],
        output: &AudioOutput,
        music_gain: f32,
    ) -> anyhow::Result<Playback> {
        let track = Arc::new(decode(audio_path)?);
        let mut playback = Playback {
            shared: build_mixer(
                Arc::clone(&track),
                note_onsets,
                beat_times,
                1,
                1,
                music_gain,
            ),
            sink: None,
            output_name: String::new(),
            output: output.clone(),
            track,
            note_onsets: note_onsets.to_vec(),
            beat_times: beat_times.to_vec(),
        };
        // The placeholder mixer above is never played; `open_sink` replaces it
        // with one built for the rate the device actually reports.
        playback.open_sink(output, TrackMilliseconds::new(0), false)?;
        Ok(playback)
    }

    /// Build a mixer for `output`'s own rate, start its sink, and leave the
    /// cursor at `position`.
    ///
    /// The wait at the end is what makes [`Playback::output_latency`] a
    /// measurement rather than a guess: a device reports no delay until it has
    /// samples queued, so the reading only becomes real a few buffers in, and
    /// the session's first cue must not be placed before then.
    fn open_sink(
        &mut self,
        output: &AudioOutput,
        position: TrackMilliseconds,
        playing: bool,
    ) -> anyhow::Result<()> {
        // The outgoing sink goes quiet before the incoming one is built, so no
        // moment has both playing, and it comes back if the new one refuses:
        // an operator picking the wrong device gets an error, not a dead
        // session in the middle of a take.
        let outgoing_was_playing = self.shared.state.lock().unwrap().playing;
        self.shared.state.lock().unwrap().playing = false;
        let opened = self.build_sink(output, position, playing);
        if opened.is_err() {
            self.shared.state.lock().unwrap().playing = outgoing_was_playing;
        }
        opened
    }

    fn build_sink(
        &mut self,
        output: &AudioOutput,
        position: TrackMilliseconds,
        playing: bool,
    ) -> anyhow::Result<()> {
        let (sink_kind, output_sample_rate, output_channels, output_name) = match output {
            AudioOutput::Silent => (
                SinkKind::Silent,
                self.track.sample_rate,
                self.track.channels.min(2),
                "silent".to_string(),
            ),
            AudioOutput::Default | AudioOutput::Named(_) => {
                let device = open_device(output)?;
                let name = device
                    .name()
                    .unwrap_or_else(|_| "unnamed output device".to_string());
                let config = device
                    .default_output_config()
                    .context("the output device offers no default configuration")?;
                (
                    SinkKind::Device(device),
                    config.sample_rate().0,
                    config.channels() as usize,
                    name,
                )
            }
        };

        let shared = build_mixer(
            Arc::clone(&self.track),
            &self.note_onsets,
            &self.beat_times,
            output_sample_rate,
            output_channels,
            self.music_gain(),
        );
        let frames = u64::from(position.get()) * u64::from(output_sample_rate) / 1000;
        shared.seek(frames, playing);
        let sink = spawn_sink(sink_kind, Arc::clone(&shared))?;

        self.stop_sink();
        self.shared = shared;
        self.sink = Some(sink);
        self.output_name = output_name;
        self.output = output.clone();

        let deadline = Instant::now() + MEASUREMENT_TIMEOUT;
        loop {
            let measured = self.shared.latency_microseconds.load(Ordering::Acquire) > 0;
            let settled = self.shared.buffers.load(Ordering::Acquire) >= SETTLING_BUFFERS;
            if measured && settled {
                break;
            }
            if Instant::now() >= deadline {
                if self.shared.buffers.load(Ordering::Acquire) == 0 {
                    anyhow::bail!("the audio output produced no buffers within a second");
                }
                tracing::warn!("audio output reported no latency; treating it as none");
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    }

    fn stop_sink(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(sink) = self.sink.take() {
            let _ = sink.join();
        }
    }

    /// Move playback to another device without losing the track's place.
    ///
    /// The new sink has its own rate and its own latency, so the caller has to
    /// re-anchor: [`Timeline::heard_at`] afterwards describes the new device,
    /// and cues placed against the old one would be out by the difference.
    /// A failure to open leaves the old sink running rather than the session
    /// silent, so the operator can pick something else.
    pub fn switch_output(&mut self, output: &AudioOutput) -> anyhow::Result<()> {
        let timeline = self.timeline();
        self.open_sink(output, timeline.position, timeline.playing)
    }

    /// Start, or continue from where a pause left the cursor.
    pub fn play(&self) {
        self.shared.state.lock().unwrap().playing = true;
    }

    /// Freeze the cursor. The sink keeps running and keeps publishing, so a
    /// frozen timeline is still readable.
    pub fn pause(&self) {
        self.shared.state.lock().unwrap().playing = false;
    }

    pub fn timeline(&self) -> Timeline {
        *self.shared.timeline.lock().unwrap()
    }

    /// How long it takes a submitted sample to become audible, measured from
    /// the stream's own timestamps.
    pub fn output_latency(&self) -> DurationMilliseconds {
        let microseconds = self.shared.latency_microseconds.load(Ordering::Acquire);
        DurationMilliseconds::new((microseconds / 1000).min(u64::from(u32::MAX)) as u32)
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn music_gain(&self) -> f32 {
        f32::from_bits(self.shared.music_gain.load(Ordering::Acquire))
    }

    /// Set the music's level. Takes effect on the next buffer, so it is safe to
    /// call mid-song; the cue clicks are unaffected.
    pub fn set_music_gain(&self, gain: f32) {
        self.shared
            .music_gain
            .store(gain.clamp(0.0, 1.0).to_bits(), Ordering::Release);
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.shared.output_sample_rate
    }
}

enum SinkKind {
    Device(cpal::Device),
    Silent,
}

fn open_device(output: &AudioOutput) -> anyhow::Result<cpal::Device> {
    let host = cpal::default_host();
    match output {
        AudioOutput::Named(wanted) => {
            let wanted_lowercase = wanted.to_lowercase();
            host.output_devices()
                .context("listing audio output devices")?
                .find(|device| {
                    device
                        .name()
                        .is_ok_and(|name| name.to_lowercase().contains(&wanted_lowercase))
                })
                .ok_or_else(|| anyhow::anyhow!("no audio output device matching '{wanted}'"))
        }
        AudioOutput::Default | AudioOutput::Silent => host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("this host has no default audio output device")),
    }
}

/// Run the sink on its own thread. `cpal::Stream` is not `Send`, so the stream
/// is built, played and dropped entirely inside the thread that owns it; the
/// session talks to the mixer through `Shared` instead.
fn spawn_sink(kind: SinkKind, shared: Arc<Shared>) -> anyhow::Result<std::thread::JoinHandle<()>> {
    let (ready, opened) = mpsc::channel::<anyhow::Result<()>>();
    let thread = std::thread::Builder::new()
        .name("collection-audio".into())
        .spawn(move || match kind {
            SinkKind::Device(device) => run_device_sink(device, shared, ready),
            SinkKind::Silent => run_silent_sink(shared, ready),
        })
        .context("spawning the audio sink thread")?;
    match opened.recv() {
        Ok(Ok(())) => Ok(thread),
        Ok(Err(error)) => {
            let _ = thread.join();
            Err(error)
        }
        Err(_) => {
            let _ = thread.join();
            Err(anyhow::anyhow!("the audio sink thread died before opening"))
        }
    }
}

fn run_device_sink(
    device: cpal::Device,
    shared: Arc<Shared>,
    ready: mpsc::Sender<anyhow::Result<()>>,
) {
    let config = match device.default_output_config() {
        Ok(config) => config,
        Err(error) => {
            let _ = ready.send(Err(
                anyhow::Error::new(error).context("reading the output config")
            ));
            return;
        }
    };
    let mixer = Arc::clone(&shared);
    let stream = device.build_output_stream(
        &config.config(),
        move |output: &mut [f32], _: &cpal::OutputCallbackInfo| mixer.render(output),
        |error| tracing::warn!("collection audio output error: {error}"),
        None,
    );
    let stream = match stream {
        Ok(stream) => stream,
        Err(error) => {
            let _ = ready.send(Err(
                anyhow::Error::new(error).context("opening the output stream")
            ));
            return;
        }
    };
    if let Err(error) = stream.play() {
        let _ = ready.send(Err(
            anyhow::Error::new(error).context("starting the output stream")
        ));
        return;
    }
    let _ = ready.send(Ok(()));
    while !shared.stop.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The same mixer, paced by the wall clock and thrown away. A run against this
/// sink advances the timeline exactly as a device does, so timing can be checked
/// without anything being audible.
fn run_silent_sink(shared: Arc<Shared>, ready: mpsc::Sender<anyhow::Result<()>>) {
    let rate = u64::from(shared.output_sample_rate);
    let period = Duration::from_micros(SILENT_SINK_FRAMES * 1_000_000 / rate);
    let _ = ready.send(Ok(()));

    let mut buffer = vec![0.0f32; SILENT_SINK_FRAMES as usize * shared.output_channels];
    let start = Instant::now();
    let mut submitted: u64 = 0;
    while !shared.stop.load(Ordering::Acquire) {
        // Stay one buffer ahead of real time, which is what a device period
        // does, so the measured queue depth means the same thing here.
        let elapsed_frames = (start.elapsed().as_micros() as u64) * rate / 1_000_000;
        while submitted < elapsed_frames + SILENT_SINK_FRAMES {
            shared.render(&mut buffer);
            submitted += SILENT_SINK_FRAMES;
        }
        std::thread::sleep(period / 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent_playback(track: &Path) -> Playback {
        Playback::open(
            track,
            &[TrackMilliseconds::new(1_000)],
            &[TrackMilliseconds::new(500)],
            &AudioOutput::Silent,
            1.0,
        )
        .expect("the silent sink always opens")
    }

    fn sample_track() -> Option<std::path::PathBuf> {
        let root = std::path::Path::new("tracks");
        let entry = std::fs::read_dir(root).ok()?.flatten().find_map(|entry| {
            let audio = entry.path().join("audio.ogg");
            audio.exists().then_some(audio)
        })?;
        Some(entry)
    }

    /// Play until the sink publishes a reading taken after this call.
    fn playing_timeline(playback: &Playback) -> Timeline {
        let commanded = now();
        let deadline = Instant::now() + Duration::from_millis(500);
        loop {
            let timeline = playback.timeline();
            if timeline.playing && timeline.heard_at >= commanded {
                return timeline;
            }
            assert!(Instant::now() < deadline, "the sink published nothing");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn a_paused_cursor_does_not_move() {
        let Some(track) = sample_track() else { return };
        let playback = silent_playback(&track);
        std::thread::sleep(Duration::from_millis(60));
        let frozen = playback.timeline();
        assert!(!frozen.playing);
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(playback.timeline().position, frozen.position);
    }

    /// The cursor is a sample count, and the whole design rests on that count
    /// tracking real time rather than drifting away from it.
    #[test]
    fn the_cursor_advances_with_the_wall_clock() {
        let Some(track) = sample_track() else { return };
        let playback = silent_playback(&track);
        playback.play();
        let first = playing_timeline(&playback);
        std::thread::sleep(Duration::from_millis(400));
        let second = playback.timeline();
        let advanced = second.position.get() as i64 - first.position.get() as i64;
        let elapsed = second.heard_at.since(first.heard_at).get();
        assert!(
            (advanced - elapsed).abs() < 60,
            "cursor advanced {advanced} ms while {elapsed} ms of wall clock passed"
        );
    }

    /// Pausing and resuming picks up where it froze; nothing is skipped and
    /// nothing replays.
    #[test]
    fn a_resume_continues_from_where_the_pause_froze_the_cursor() {
        let Some(track) = sample_track() else { return };
        let playback = silent_playback(&track);
        playback.play();
        playing_timeline(&playback);
        std::thread::sleep(Duration::from_millis(200));
        playback.pause();
        std::thread::sleep(Duration::from_millis(50));
        let frozen = playback.timeline().position;
        std::thread::sleep(Duration::from_millis(300));

        playback.play();
        let resumed = playing_timeline(&playback).position;
        let skipped = resumed.get() as i64 - frozen.get() as i64;
        assert!(
            (0..60).contains(&skipped),
            "resumed at {resumed:?} after freezing at {frozen:?}"
        );
    }

    /// Turning the music down leaves the cue clicks where they were: the point
    /// of the control is to make the cues clearer, so a mixer that scaled both
    /// would be useless for it.
    #[test]
    fn the_volume_moves_the_music_and_not_the_clicks() {
        let Some(track) = sample_track() else { return };
        let rate = decode(&track)
            .expect("the library's audio decodes")
            .sample_rate;
        let onset = TrackMilliseconds::new(0);

        let render = |gain: f32, cues: &[TrackMilliseconds]| {
            let decoded = decode(&track).expect("the library's audio decodes");
            let channels = decoded.channels.min(2);
            let mixer = build_mixer(Arc::new(decoded), cues, &[], rate, channels, gain);
            mixer.state.lock().unwrap().playing = true;
            let mut buffer = vec![0.0f32; rate as usize * channels];
            mixer.render(&mut buffer);
            buffer
        };
        let peak = |buffer: &[f32]| buffer.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));

        let loud = render(1.0, &[]);
        let quiet = render(0.25, &[]);
        assert!(
            (peak(&quiet) - peak(&loud) * 0.25).abs() < 0.01,
            "the music did not scale with the gain"
        );

        // The click's own contribution, measured against the music underneath
        // it at each level, is unchanged.
        let loud_cued = render(1.0, &[onset]);
        let quiet_cued = render(0.25, &[onset]);
        let click_of = |cued: &[f32], plain: &[f32]| {
            cued.iter()
                .zip(plain)
                .fold(0.0f32, |peak, (a, b)| peak.max((a - b).abs()))
        };
        assert!(
            (click_of(&quiet_cued, &quiet) - click_of(&loud_cued, &loud)).abs() < 0.01,
            "the cue click moved with the music volume"
        );
    }

    /// Switching output keeps the track's place. The silent sink is the only
    /// device a test can rely on existing, so it switches to itself — which
    /// still exercises the rebuild, the seek and the click re-derivation.
    #[test]
    fn switching_output_keeps_the_place_and_does_not_replay_past_clicks() {
        let Some(track) = sample_track() else { return };
        let mut playback = silent_playback(&track);
        playback.play();
        playing_timeline(&playback);
        // Far enough in that clicks are behind the playhead when the sink is
        // rebuilt. Rebuilding waits for the new sink to settle, so the position
        // afterwards is not something the test can predict — it is read.
        std::thread::sleep(Duration::from_millis(300));

        let before = playback.timeline().position;
        playback
            .switch_output(&AudioOutput::Silent)
            .expect("the silent sink always opens");
        let after = playing_timeline(&playback).position;
        let drift = after.get() as i64 - before.get() as i64;
        assert!(
            (0..400).contains(&drift),
            "switching output moved the track from {before:?} to {after:?}"
        );
        assert!(
            after.get() >= 500,
            "the test needs a playhead past the first click, not {after:?}"
        );
        let already_played = playback
            .shared
            .clicks
            .iter()
            .filter(|(frame, _)| {
                *frame * 1000 / u64::from(playback.shared.output_sample_rate)
                    <= u64::from(after.get())
            })
            .count();
        assert_eq!(
            playback.shared.state.lock().unwrap().next_click,
            already_played,
            "the new sink would replay the clicks the old one already played"
        );
    }

    /// The click envelope has to reach zero, or a voice would keep sounding for
    /// the whole buffer it was retired in.
    #[test]
    fn a_click_decays_to_silence() {
        let waveform = click_waveform(48_000, CUE_CLICK_HERTZ, CUE_CLICK_SECONDS, CUE_CLICK_GAIN);
        assert_eq!(waveform.len(), (48_000.0 * CUE_CLICK_SECONDS) as usize);
        assert!(waveform[0].abs() > 0.2);
        assert!(waveform[waveform.len() - 1].abs() < 0.01);
    }

    /// What the sink is handed: the track's own audio, with a cue click on top
    /// at the scheduled onset. Checked here rather than by ear, because by ear
    /// means playing music at whoever is sitting at this machine.
    #[test]
    fn the_buffer_carries_the_track_with_a_click_at_the_cue() {
        let Some(track) = sample_track() else { return };
        let rate = decode(&track)
            .expect("the library's audio decodes")
            .sample_rate;
        let onset = TrackMilliseconds::new(500);
        let onset_frame = 500 * rate as usize / 1000;

        let render_one_second = |cues: &[TrackMilliseconds]| {
            let decoded = decode(&track).expect("the library's audio decodes");
            let channels = decoded.channels.min(2);
            let mixer = build_mixer(Arc::new(decoded), cues, &[], rate, channels, 1.0);
            mixer.state.lock().unwrap().playing = true;
            let mut buffer = vec![0.0f32; rate as usize * channels];
            mixer.render(&mut buffer);
            (buffer, channels)
        };
        let (music_only, channels) = render_one_second(&[]);
        let (with_cue, _) = render_one_second(&[onset]);

        let music_peak = music_only.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        assert!(music_peak > 0.001, "the track rendered as silence");

        // Everything the cue added, and where it landed.
        let (loudest_frame, added) = with_cue
            .iter()
            .zip(&music_only)
            .enumerate()
            .map(|(index, (mixed, music))| (index / channels, (mixed - music).abs()))
            .fold((0, 0.0f32), |worst, current| {
                if current.1 > worst.1 {
                    current
                } else {
                    worst
                }
            });
        assert!(
            added > CUE_CLICK_GAIN * 0.9,
            "no cue click on top of the track (added {added})"
        );
        assert!(
            loudest_frame.abs_diff(onset_frame) < rate as usize / 100,
            "the click landed at frame {loudest_frame}, not the cue's {onset_frame}"
        );
    }
}
