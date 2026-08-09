//! The firmware validation bench's playback engine: replay a recorded EMG
//! session streamed in from a host, run the calibrated gesture pipeline on it,
//! and report what came out.
//!
//! This exists so the gesture pipeline can be validated on the real silicon
//! before an analog front end is attached to it. A bare ESP32-S3 brings up no
//! ADC, so the serve loop starts this instead of an acquisition source and the
//! samples arrive over the same USB-serial CBOR link everything else uses.
//! `firmware-bench/PROTOCOL.md` documents the frames and the orchestration
//! sequences; `firmware-bench/ARITHMETIC.md` pins the arithmetic, which lives in
//! `emg-runtime` — nothing here does math on a sample beyond widening it.
//!
//! Two threads, because the two jobs have incompatible deadlines. The serve loop
//! must poll the link and feed the task watchdog every few milliseconds; a
//! calibration fit occupies a core for seconds and a chunk of samples for
//! milliseconds. So the loop only ever moves bytes — it drains the link, hands
//! whole `Control` values to this engine, and sends back whatever the engine has
//! produced — and a worker thread owns every piece of pipeline state and does
//! all the work. Neither ever waits on the other: the command queue is bounded
//! and refuses rather than blocks, and the flow control below is what keeps it
//! from ever having to.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use emg_runtime::band_features::{
    BandFeaturePipeline, CHANNEL_COUNT, FEATURE_COUNT, WINDOW_SAMPLES,
};
use emg_runtime::calibration::{
    fit_calibration, CalibrationModel, FeaturePrecision, FeatureStore, Int8Quantization,
    StaticFeatureRows,
};
use emg_runtime::pipeline::RejectPipeline;
use log::{info, warn};
use protocol::{BenchDecision, Frame, PLAYBACK_MAX_CHUNK_SAMPLES};

use crate::telemetry::{heap_free_bytes, largest_free_block_bytes};
use crate::training_rows::TrainingRows;
use crate::transport::Control;

/// Command slots between the serve loop and the worker. Deeper than the credit
/// window below so a status request or a fit command never finds the queue full
/// of samples: the host is allowed to interleave them, and a refused command
/// would strand an orchestration sequence.
const COMMAND_QUEUE_DEPTH: usize = 12;

/// Sample bytes the host may have in flight. A byte budget rather than a chunk
/// count, because the chunk size is the host's to choose and a count that was
/// safe at 4 KB chunks would be four times the receive ring at the largest
/// chunk the protocol allows.
///
/// It has to stay inside the ring the serve loop drains into
/// (`SERIAL_RX_BUFFER_BYTES` in `main`) with room to spare: bytes the ring
/// cannot hold are dropped by the driver rather than delayed, and they resurface
/// as a sequence gap and an abandoned run. The margin covers each chunk's
/// framing and any control frame the host interleaves.
const CREDIT_BUDGET_BYTES: usize = 12 * 1024;

/// Chunks the host may have in flight regardless of how small they are. Bounds
/// the command queue's occupancy, and past a few chunks the round trip stops
/// being what limits throughput anyway.
const MAX_CREDIT_CHUNKS: u32 = 8;

/// Windows batched into one `BenchFeatures` frame. Each window is 256 bytes of
/// features, so this is a 4 KB payload — comfortably inside the serve loop's
/// 18 KB encode buffer, and few enough frames that the per-frame CBOR overhead
/// stays negligible against the feature bytes.
const FEATURE_BATCH_WINDOWS: usize = 16;

/// Decisions batched into one `BenchCommits` frame.
const DECISION_BATCH_WINDOWS: usize = 64;

/// Windows between unsolicited status reports while a session streams. About
/// eight seconds of recorded time, which is often enough to watch a long
/// session's throughput without adding meaningful traffic to it.
const STATUS_INTERVAL_WINDOWS: u32 = 64;

/// Command classes the reject pipeline scores over, per `ARITHMETIC.md`. The
/// class count a model carries is larger — it includes no-op and rest, whose
/// probability mass is never eligible to commit.
const COMMAND_CLASSES: usize = 5;

/// The reject threshold, per `ARITHMETIC.md`. Fixed here rather than taken from
/// the device's sensitivity setting: the bench compares against a host
/// simulation that uses this number, and a run that silently used the stored
/// preset would disagree for a reason no one would look for.
const REJECT_TAU: f32 = 0.5;

/// The worker's stack. It holds a window of features, the probability vector,
/// and whatever `fit_calibration` puts on the stack; 16 KB was not obviously
/// enough for the last of those, and the thread is the only one this feature
/// adds.
const WORKER_STACK_BYTES: usize = 24 * 1024;

/// The serve loop's handle on the worker. Owns no pipeline state: everything
/// crosses as a message, so there is no lock for a link write to contend with.
pub struct PlaybackEngine {
    commands: mpsc::SyncSender<Control>,
    outbound: mpsc::Receiver<Frame>,
    /// Completed windows, for a calibration running off this stream instead of
    /// off a front end. The same values the features frame carries, handed to
    /// the flow so the scripted wearer is fed by the recording rather than by
    /// a second copy of the pipeline.
    windows: mpsc::Receiver<crate::calibration::CalibrationWindow>,
    /// The reference gains the session declared, so a scripted calibration
    /// adopts the recording's own rather than estimating them from it.
    gains: Arc<Mutex<Option<[f32; CHANNEL_COUNT]>>>,
    /// Commands the queue refused. Counted here rather than in the worker
    /// because the worker is exactly what was too busy to hear about them.
    refused: Arc<AtomicU32>,
}

impl PlaybackEngine {
    /// Spawn the worker. Runs on core 1, which a playback build leaves idle —
    /// there is no front end, so the acquisition threads and the GPIO
    /// dispatcher that own that core on real hardware do not exist.
    pub fn start() -> anyhow::Result<Self> {
        // A calibration fit occupies core 1 for minutes without yielding, which
        // starves that core's idle task past the 5 s task watchdog and reboots
        // the chip mid-measurement. Stop watching the idle tasks; the serve
        // loop keeps its own subscription and feeds it every iteration, so a
        // hung link still reboots.
        let watchdog = esp_idf_svc::sys::esp_task_wdt_config_t {
            timeout_ms: 5000,
            idle_core_mask: 0,
            trigger_panic: true,
        };
        let reconfigured = unsafe { esp_idf_svc::sys::esp_task_wdt_reconfigure(&watchdog) };
        if reconfigured != esp_idf_svc::sys::ESP_OK {
            warn!("task watchdog reconfigure failed ({reconfigured}); long fits will reboot");
        }
        let (commands, command_queue) = mpsc::sync_channel(COMMAND_QUEUE_DEPTH);
        let (produced, outbound) = mpsc::channel();
        let (produced_windows, windows) = mpsc::channel();
        let gains: Arc<Mutex<Option<[f32; CHANNEL_COUNT]>>> = Arc::new(Mutex::new(None));
        let worker_gains = Arc::clone(&gains);
        let refused = Arc::new(AtomicU32::new(0));
        let worker_refused = Arc::clone(&refused);
        crate::cores::spawn_pinned(crate::cores::PLAYBACK_WORKER_CORE, || {
            std::thread::Builder::new()
                .stack_size(WORKER_STACK_BYTES)
                .spawn(move || {
                    crate::cores::log_thread_priority("playback worker");
                    let mut bench =
                        Bench::new(produced, worker_refused, produced_windows, worker_gains);
                    while let Ok(control) = command_queue.recv() {
                        bench.handle(control);
                    }
                })
        })??;
        info!("playback engine started; streaming sessions over the serial link");
        Ok(Self {
            commands,
            outbound,
            windows,
            gains,
            refused,
        })
    }

    /// Hand the worker a control frame if it is one of the bench frames, and
    /// return it untouched otherwise so the serve loop can apply it as config.
    ///
    /// Never blocks. A full queue means the host outran its credit window, which
    /// is a protocol violation on its side; the refusal is counted and surfaces
    /// in the next status frame as `dropped_chunks`, because a run that lost
    /// samples must be visibly wrong rather than quietly plausible.
    pub fn accept(&self, control: Control) -> Option<Control> {
        if !control.is_bench() {
            return Some(control);
        }
        if self.commands.try_send(control).is_err() {
            self.refused.fetch_add(1, Ordering::Relaxed);
        }
        None
    }

    /// Everything the worker has produced since the last call, in order.
    pub fn drain_outbound(&self) -> Vec<Frame> {
        self.outbound.try_iter().collect()
    }

    /// Completed windows since the last call, oldest first. A calibration
    /// running scripted takes these where a wearer's would take the ADC's.
    pub fn drain_windows(&self) -> Vec<crate::calibration::CalibrationWindow> {
        self.windows.try_iter().collect()
    }

    /// The reference gains of the session being replayed, once one has begun.
    pub fn session_gains(&self) -> Option<[f32; CHANNEL_COUNT]> {
        self.gains.lock().ok().and_then(|gains| *gains)
    }
}

/// Per-window feature compute cost over one session.
#[derive(Default)]
struct ComputeCost {
    minimum_microseconds: u32,
    maximum_microseconds: u32,
    sum_microseconds: u64,
    windows: u32,
    /// Cost accumulated by chunks whose window has not closed yet.
    pending_microseconds: u32,
}

impl ComputeCost {
    fn add_chunk(&mut self, microseconds: u32) {
        self.pending_microseconds = self.pending_microseconds.saturating_add(microseconds);
    }

    fn close_window(&mut self) {
        let cost = self.pending_microseconds;
        self.pending_microseconds = 0;
        self.minimum_microseconds = if self.windows == 0 {
            cost
        } else {
            self.minimum_microseconds.min(cost)
        };
        self.maximum_microseconds = self.maximum_microseconds.max(cost);
        self.sum_microseconds += cost as u64;
        self.windows += 1;
    }

    fn mean_microseconds(&self) -> u32 {
        if self.windows == 0 {
            0
        } else {
            (self.sum_microseconds / self.windows as u64) as u32
        }
    }
}

/// One session being replayed.
struct Session {
    identifier: String,
    features: BandFeaturePipeline,
    /// The chunk sequence number the next `PlaybackSamples` must carry.
    expected_sequence: u32,
    samples_received: u64,
    windows: u32,
    sequence_gaps: u32,
    /// Abandoned after a gap: the filter state is continuous across the whole
    /// session, so every later feature is wrong once samples are missing, and
    /// continuing would produce numbers that look like measurements.
    abandoned: bool,
    /// Little-endian `f32` feature bits for the windows not yet sent.
    batch: Vec<u8>,
    batch_first_window: u32,
    batch_windows: u32,
    compute: ComputeCost,
    windows_at_last_status: u32,
    /// Chunks granted at a time, from [`CREDIT_BUDGET_BYTES`] and this
    /// session's chunk size.
    credit_chunks: u32,
}

/// The worker's whole world.
struct Bench {
    outbound: mpsc::Sender<Frame>,
    refused: Arc<AtomicU32>,
    /// Completed windows for a scripted calibration. Send-only and unbounded:
    /// the serve loop drains every iteration, and a calibration that is not
    /// running simply drops them.
    windows: mpsc::Sender<crate::calibration::CalibrationWindow>,
    gains: Arc<Mutex<Option<[f32; CHANNEL_COUNT]>>>,
    session: Option<Session>,
    model: Option<CalibrationModel>,
    reject: RejectPipeline,
    store: Option<FeatureStore>,
    fit_class_count: usize,
    stored_rows: u32,
    /// The flash training partition, mapped once at start-up. `None` when the
    /// partition is absent or unwritten, which is what a device that has never
    /// been given an image looks like.
    flash_rows: Option<TrainingRows>,
    decisions: Vec<BenchDecision>,
    probabilities: Vec<f32>,
    /// Sample instants published to a calibration since the last reset, across
    /// however many sessions that took.
    ///
    /// Separate from the session's own window counter on purpose. A session's
    /// windows number from zero because that is what the feature and commit
    /// frames index by, and the parity comparison reads those; a calibration
    /// needs one monotonic sample space, because a scripted run splices the
    /// thumb-up session and the thumb-down one into a single protocol and its
    /// labeling arithmetic cannot have time run backwards in the middle. So
    /// the two counters exist and mean different things.
    published_samples: u64,
    /// Where the current session's own sample space starts inside the run's.
    /// Set at each `playback_begin` from what the previous sessions
    /// contributed, which is what splices them.
    published_base: u64,
    mode: &'static str,
}

impl Bench {
    fn new(
        outbound: mpsc::Sender<Frame>,
        refused: Arc<AtomicU32>,
        windows: mpsc::Sender<crate::calibration::CalibrationWindow>,
        gains: Arc<Mutex<Option<[f32; CHANNEL_COUNT]>>>,
    ) -> Self {
        Self {
            outbound,
            refused,
            windows,
            gains,
            session: None,
            model: None,
            reject: RejectPipeline::new(COMMAND_CLASSES, REJECT_TAU),
            store: None,
            fit_class_count: 0,
            stored_rows: 0,
            flash_rows: crate::training_rows::map_or_warn(),
            decisions: Vec::with_capacity(DECISION_BATCH_WINDOWS),
            probabilities: Vec::new(),
            published_samples: 0,
            published_base: 0,
            mode: "idle",
        }
    }

    fn handle(&mut self, control: Control) {
        match control {
            Control::PlaybackBegin {
                session,
                sample_count,
                chunk_samples,
                constants,
            } => self.handle_begin(session, sample_count, chunk_samples, &constants),
            Control::PlaybackSamples { sequence, samples } => {
                self.handle_samples(sequence, &samples)
            }
            Control::PlaybackEnd {} => self.handle_end(),
            Control::BenchModelLoad { class_count, model } => {
                self.handle_model_load(class_count as usize, &model)
            }
            Control::BenchReplayRows { first_window, rows } => {
                self.handle_replay_rows(first_window, &rows)
            }
            Control::BenchFitBegin {
                row_capacity,
                precision,
                class_count,
                quantization,
            } => self.handle_fit_begin(row_capacity, precision, class_count as usize, quantization),
            Control::BenchFitRows {
                labels,
                row_weights,
                rows,
            } => self.handle_fit_rows(&labels, &row_weights, &rows),
            Control::BenchFitRun { use_static_rows } => self.handle_fit_run(use_static_rows),
            Control::BenchStatusRequest {} => self.send_status(),
            Control::BenchReset {} => self.handle_reset(),
            // Config and link management belong to the serve loop; `accept`
            // never forwards them.
            other => warn!("playback worker ignored a non-bench control: {other:?}"),
        }
    }

    fn handle_begin(
        &mut self,
        identifier: String,
        sample_count: u32,
        chunk_samples: u32,
        constants: &[u8],
    ) {
        const EXPECTED_CONSTANTS: usize = (1 + CHANNEL_COUNT) * 4;
        if constants.len() != EXPECTED_CONSTANTS {
            self.fail(
                "playback_begin",
                format!(
                    "constants blob is {} bytes, want {EXPECTED_CONSTANTS}",
                    constants.len()
                ),
            );
            return;
        }
        if chunk_samples == 0 || chunk_samples as usize > PLAYBACK_MAX_CHUNK_SAMPLES {
            self.fail(
                "playback_begin",
                format!("chunk_samples {chunk_samples} outside 1..={PLAYBACK_MAX_CHUNK_SAMPLES}"),
            );
            return;
        }
        let microvolts_per_count = float_at(constants, 0);
        // Published for a scripted calibration to adopt: the recording's own
        // gains are the right ones, and re-estimating them from the recording
        // would measure the recording rather than a wearer.
        let mut reference_gains = [0.0f32; CHANNEL_COUNT];
        for (slot, gain) in reference_gains.iter_mut().enumerate() {
            *gain = float_at(constants, 1 + slot);
        }
        if let Ok(mut published) = self.gains.lock() {
            *published = Some(reference_gains);
        }

        // A fresh pipeline rather than a reset one: the filters carry state
        // across a whole session and must start from zero for the next, and
        // `BandFeaturePipeline` allocates only in `new`.
        self.session = Some(Session {
            identifier,
            features: BandFeaturePipeline::new(microvolts_per_count, reference_gains),
            expected_sequence: 0,
            samples_received: 0,
            windows: 0,
            sequence_gaps: 0,
            abandoned: false,
            batch: Vec::with_capacity(FEATURE_BATCH_WINDOWS * FEATURE_COUNT * 4),
            batch_first_window: 0,
            batch_windows: 0,
            compute: ComputeCost::default(),
            windows_at_last_status: 0,
            credit_chunks: credit_chunks(chunk_samples as usize),
        });
        self.reject = RejectPipeline::new(COMMAND_CLASSES, REJECT_TAU);
        self.decisions.clear();
        // A new session's windows number from zero again, so the run's own
        // space carries on from where the last one stopped.
        self.published_base = self.published_samples;
        self.mode = "streaming";
        info!("playback session begins: {sample_count} samples in chunks of {chunk_samples}");
        self.grant_credit(0);
    }

    fn handle_samples(&mut self, sequence: u32, samples: &[u8]) {
        if self.session.is_none() {
            self.fail(
                "playback_samples",
                "no session; send playback_begin first".into(),
            );
            return;
        }
        // Whether the chunk was consumed, and the features of the window it
        // closed. A chunk is at most one window long, so it can close at most
        // one window; the pipeline is asked for the rest anyway and a second
        // would be a broken invariant rather than a dropped result.
        let mut closed: Option<[f32; FEATURE_COUNT]> = None;
        // Sliding windows the chunk produced, for a calibration running off
        // this stream. Up to four per chunk at the quarter stride, against the
        // one aligned window `closed` carries.
        let mut sliding: Vec<(u64, [f32; FEATURE_COUNT])> = Vec::new();
        let mut refusal = None;

        {
            let session = self.session.as_mut().expect("checked above");
            if session.abandoned {
                // Credit is still granted, so the host's send loop drains and
                // reaches `playback_end` rather than stalling on a grant that
                // never comes.
                let next = session.expected_sequence;
                self.grant_credit(next);
                return;
            }
            if sequence != session.expected_sequence {
                session.sequence_gaps += 1;
                session.abandoned = true;
                let expected = session.expected_sequence;
                refusal = Some(format!(
                    "sequence {sequence}, expected {expected}; session abandoned"
                ));
            } else if samples.len() % (CHANNEL_COUNT * 2) != 0 {
                session.abandoned = true;
                refusal = Some(format!(
                    "{} bytes is not whole sample instants",
                    samples.len()
                ));
            } else {
                session.expected_sequence = sequence.wrapping_add(1);

                // The chunk's whole push loop is timed once, not each sample: a
                // per-sample clock read would be several percent of the cost it
                // is measuring. The host tool sends chunk sizes that divide the
                // 500-sample window, so a chunk lies inside one window and the
                // attribution below is exact.
                let started = Instant::now();
                let mut windows_closed = 0usize;
                for instant in samples.chunks_exact(CHANNEL_COUNT * 2) {
                    let mut counts = [0i16; CHANNEL_COUNT];
                    for (slot, raw) in instant.chunks_exact(2).enumerate() {
                        counts[slot] = i16::from_le_bytes([raw[0], raw[1]]);
                    }
                    // Sliding rather than aligned, so a scripted calibration
                    // sees the same nine-window reps a wearer's would. Every
                    // fourth is the aligned window bit for bit, and only those
                    // reach the feature and commit frames — the parity
                    // comparison indexes by 500-aligned windows and reads
                    // exactly what it always did.
                    if let Some(window) = session.features.push_sliding(&counts) {
                        sliding.push((window.end_sample, window.features));
                        if window.aligned {
                            closed = Some(window.features);
                            windows_closed += 1;
                        }
                    }
                }
                let elapsed_microseconds = started.elapsed().as_micros() as u32;
                session.samples_received += (samples.len() / (CHANNEL_COUNT * 2)) as u64;
                session.compute.add_chunk(elapsed_microseconds);
                if windows_closed > 1 {
                    session.abandoned = true;
                    refusal = Some(format!(
                        "chunk closed {windows_closed} windows; chunk_samples must not exceed {WINDOW_SAMPLES}"
                    ));
                    closed = None;
                } else if windows_closed == 1 {
                    session.compute.close_window();
                }
            }
        }

        if let Some(detail) = refusal {
            self.fail("playback_samples", detail);
            return;
        }
        for (end_sample, features) in sliding {
            self.publish_window(end_sample, &features);
        }
        if let Some(features) = closed {
            self.record_window(&features);
        }

        let (next, windows, since_status) = {
            let session = self.session.as_ref().expect("checked above");
            (
                session.expected_sequence,
                session.windows,
                session.windows - session.windows_at_last_status,
            )
        };
        self.grant_credit(next);
        if since_status >= STATUS_INTERVAL_WINDOWS {
            self.session
                .as_mut()
                .expect("checked above")
                .windows_at_last_status = windows;
            self.send_status();
        }
    }

    /// One completed window: batch its features for the host, and score it when
    /// a model is loaded so a streamed session produces its commit sequence in
    /// the same pass that produced its features.
    fn record_window(&mut self, features: &[f32; FEATURE_COUNT]) {
        let window = {
            let Some(session) = self.session.as_mut() else {
                return;
            };
            if session.batch_windows == 0 {
                session.batch_first_window = session.windows;
            }
            for value in features {
                session
                    .batch
                    .extend_from_slice(&value.to_bits().to_le_bytes());
            }
            session.batch_windows += 1;
            let window = session.windows;
            session.windows += 1;
            window
        };
        if self
            .session
            .as_ref()
            .is_some_and(|session| session.batch_windows as usize >= FEATURE_BATCH_WINDOWS)
        {
            self.flush_features();
        }
        if self.model.is_some() {
            self.score_row(window, features);
        }
    }

    /// Hand the window to a calibration running off this stream.
    ///
    /// Sample indices, not the device clock: the flow is paced by the recording
    /// so a scripted run reproduces regardless of how fast the board chewed
    /// through the bytes. The lead-off and recovery flags are false because a
    /// recording has neither — the validity checks that read them are exercised
    /// on a wearer, and a replay tests the rest of the rule. The gains come
    /// with the session rather than from an estimate over it, so no instants
    /// travel with the window either.
    fn publish_window(&mut self, end_sample: u64, features: &[f32; FEATURE_COUNT]) {
        // The session's own position plus everything the sessions before it
        // contributed, so a spliced run is one monotonic space.
        let end_sample = self.published_base + end_sample;
        self.published_samples = end_sample;
        let _ = self.windows.send(crate::calibration::CalibrationWindow {
            end_sample,
            features: *features,
            lead_off: false,
            adc_recovery: false,
        });
    }

    fn score_row(&mut self, window: u32, features: &[f32; FEATURE_COUNT]) {
        let Some(model) = self.model.as_ref() else {
            return;
        };
        self.probabilities.clear();
        self.probabilities.resize(model.class_count, 0.0);
        model.probabilities(features, &mut self.probabilities);
        let decision = self.reject.step(&self.probabilities);
        self.decisions.push(BenchDecision {
            window,
            command: decision.argmax,
            accepted: decision.accepted,
            reject_score_bits: decision.reject_score.to_bits(),
        });
        if self.decisions.len() >= DECISION_BATCH_WINDOWS {
            self.flush_decisions();
        }
    }

    fn handle_end(&mut self) {
        self.flush_features();
        self.flush_decisions();
        self.mode = "idle";
        self.send_status();
        if let Some(session) = self.session.as_ref() {
            info!(
                "playback session ends: {} windows, {} samples, {} gaps",
                session.windows, session.samples_received, session.sequence_gaps
            );
        }
    }

    fn handle_model_load(&mut self, class_count: usize, bits: &[u8]) {
        match CalibrationModel::from_bits(class_count, bits) {
            Some(model) => {
                self.probabilities = vec![0.0; model.class_count];
                self.model = Some(model);
                self.reject = RejectPipeline::new(COMMAND_CLASSES, REJECT_TAU);
                self.decisions.clear();
                info!(
                    "calibration model loaded: {class_count} classes, {} bytes",
                    bits.len()
                );
            }
            None => self.fail(
                "bench_model_load",
                format!("{} bytes do not describe {class_count} classes", bits.len()),
            ),
        }
    }

    fn handle_replay_rows(&mut self, first_window: u32, rows: &[u8]) {
        if self.model.is_none() {
            self.fail(
                "bench_replay_rows",
                "no model; send bench_model_load first".into(),
            );
            return;
        }
        const ROW_BYTES: usize = FEATURE_COUNT * 4;
        if rows.len() % ROW_BYTES != 0 {
            self.fail(
                "bench_replay_rows",
                format!(
                    "{} bytes is not whole {FEATURE_COUNT}-feature rows",
                    rows.len()
                ),
            );
            return;
        }
        self.mode = "replaying";
        for (index, row) in rows.chunks_exact(ROW_BYTES).enumerate() {
            let features: [f32; FEATURE_COUNT] =
                std::array::from_fn(|feature| float_at(row, feature));
            self.score_row(first_window + index as u32, &features);
        }
        self.flush_decisions();
        self.mode = "idle";
    }

    fn handle_fit_begin(
        &mut self,
        row_capacity: u32,
        precision: u8,
        class_count: usize,
        quantization: Vec<u8>,
    ) {
        let precision = match precision {
            0 => FeaturePrecision::Float32,
            1 => FeaturePrecision::Float16,
            2 => FeaturePrecision::Int8,
            other => {
                self.fail("bench_fit_begin", format!("precision selector {other}"));
                return;
            }
        };
        if precision == FeaturePrecision::Int8 && quantization.len() != FEATURE_COUNT * 2 * 4 {
            self.fail(
                "bench_fit_begin",
                format!(
                    "i8 precision wants {} bytes of offset/scale, got {}",
                    FEATURE_COUNT * 2 * 4,
                    quantization.len()
                ),
            );
            return;
        }
        // At `i8` the store quantizes with the host's constants rather than its
        // identity default: the comparison is against a host that used these,
        // and a store left on the default would disagree by a whole
        // quantization step at every feature.
        self.store = Some(match precision {
            FeaturePrecision::Int8 => {
                let Some(constants) = Int8Quantization::from_bits(&quantization) else {
                    self.fail("bench_fit_begin", "i8 constants did not decode".into());
                    return;
                };
                FeatureStore::with_int8_capacity(row_capacity as usize, constants)
            }
            _ => FeatureStore::with_capacity(row_capacity as usize, precision),
        });
        self.fit_class_count = class_count;
        self.stored_rows = 0;
        info!("calibration store: {row_capacity} rows, {class_count} classes");
    }

    fn handle_fit_rows(&mut self, labels: &[u8], row_weights: &[u8], rows: &[u8]) {
        const ROW_BYTES: usize = FEATURE_COUNT * 4;
        if self.store.is_none() {
            self.fail(
                "bench_fit_rows",
                "no store; send bench_fit_begin first".into(),
            );
            return;
        }
        if row_weights.len() != labels.len() * 4 || rows.len() != labels.len() * ROW_BYTES {
            let count = labels.len();
            self.fail(
                "bench_fit_rows",
                format!(
                    "{count} labels want {} weight bytes and {} row bytes, got {} and {}",
                    count * 4,
                    count * ROW_BYTES,
                    row_weights.len(),
                    rows.len()
                ),
            );
            return;
        }
        let mut stored = 0u32;
        let mut full = false;
        {
            let store = self.store.as_mut().expect("checked above");
            for (index, row) in rows.chunks_exact(ROW_BYTES).enumerate() {
                let features: [f32; FEATURE_COUNT] =
                    std::array::from_fn(|feature| float_at(row, feature));
                if !store.push(&features, labels[index], float_at(row_weights, index)) {
                    full = true;
                    break;
                }
                stored += 1;
            }
        }
        self.stored_rows += stored;
        if full {
            self.fail(
                "bench_fit_rows",
                format!("store full after {} rows", self.stored_rows),
            );
        }
    }

    fn handle_fit_run(&mut self, use_static_rows: bool) {
        match self.store.as_ref() {
            None => {
                self.fail(
                    "bench_fit_run",
                    "no store; send bench_fit_begin first".into(),
                );
                return;
            }
            Some(store) if store.is_empty() => {
                self.fail("bench_fit_run", "no rows stored".into());
                return;
            }
            Some(_) => {}
        }
        let store = self.store.as_ref().expect("checked above");
        self.mode = "fitting";

        // Which experiment this is. Live-only measures how many rows RAM can
        // hold; live-plus-flash measures the split the full training matrix
        // actually needs. A run that asked for flash and did not get it would
        // report the other experiment's numbers under this one's name, so the
        // two ways of not getting it are refusals rather than warnings.
        let flash = match (use_static_rows, self.flash_rows.as_ref()) {
            (false, _) => None,
            (true, None) => {
                self.fail(
                    "bench_fit_run",
                    "asked for flash rows, none are mapped".into(),
                );
                self.mode = "idle";
                return;
            }
            (true, Some(flash)) => Some(flash),
        };
        // One pass over the flash bytes before the fit, so the wall time below
        // can be split between arithmetic and the ~250 passes the fit makes
        // through the flash cache.
        let flash_walk_microseconds = flash.map(TrainingRows::walk_microseconds).unwrap_or(0);
        // Each source decodes through its own layout, so the live store and
        // the flash image may hold different precisions — the validated
        // arrangement is a float live pool over int8 flash rows.
        let static_rows = flash.and_then(|flash| match flash.precision() {
            FeaturePrecision::Int8 => {
                StaticFeatureRows::with_int8(flash.rows(), flash.quantization())
            }
            precision => StaticFeatureRows::new(flash.rows(), precision),
        });
        if use_static_rows && static_rows.is_none() {
            // Unreachable: the mapping already checked the image's byte count
            // against its own stride. Refused rather than assumed, because the
            // alternative is running the live-only experiment under the
            // live-plus-flash name, which is the one outcome no reader of the
            // results could detect.
            self.fail(
                "bench_fit_run",
                "flash rows are not a whole number of rows at their own precision".into(),
            );
            self.mode = "idle";
            return;
        }
        let flash_row_count = static_rows
            .as_ref()
            .map(StaticFeatureRows::len)
            .unwrap_or(0) as u32;

        let heap_free_before_bytes = heap_free_bytes();
        let largest_free_block_before_bytes = largest_free_block_bytes();
        let started = Instant::now();
        let model = fit_calibration(store, static_rows.as_ref(), self.fit_class_count);
        let wall_milliseconds = started.elapsed().as_millis() as u32;
        let heap_free_after_bytes = heap_free_bytes();
        let largest_free_block_after_bytes = largest_free_block_bytes();
        let bits = model.to_bits();
        info!(
            "calibration fit: {} live rows + {flash_row_count} flash rows, {wall_milliseconds} ms \
             ({flash_walk_microseconds} us per flash pass), heap {heap_free_before_bytes} -> {heap_free_after_bytes}",
            self.stored_rows
        );
        self.probabilities = vec![0.0; model.class_count];
        let class_count = model.class_count as u32;
        self.model = Some(model);
        self.reject = RejectPipeline::new(COMMAND_CLASSES, REJECT_TAU);
        self.mode = "idle";
        self.send(Frame::BenchFitResult {
            wall_milliseconds,
            rows: self.stored_rows,
            flash_rows: flash_row_count,
            flash_walk_microseconds,
            class_count,
            heap_free_before_bytes,
            heap_free_after_bytes,
            largest_free_block_before_bytes,
            largest_free_block_after_bytes,
            model: bits,
        });
    }

    fn handle_reset(&mut self) {
        self.session = None;
        self.model = None;
        self.store = None;
        self.fit_class_count = 0;
        self.stored_rows = 0;
        self.decisions.clear();
        self.probabilities = Vec::new();
        self.published_samples = 0;
        self.published_base = 0;
        self.reject = RejectPipeline::new(COMMAND_CLASSES, REJECT_TAU);
        self.refused.store(0, Ordering::Relaxed);
        self.mode = "idle";
        self.send_status();
    }

    fn flush_features(&mut self) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if session.batch_windows == 0 {
            return;
        }
        let frame = Frame::BenchFeatures {
            first_window: session.batch_first_window,
            window_count: session.batch_windows,
            // Taken rather than cloned, and the emptied buffer keeps its
            // capacity for the next batch.
            features: std::mem::take(&mut session.batch),
        };
        session.batch = Vec::with_capacity(FEATURE_BATCH_WINDOWS * FEATURE_COUNT * 4);
        session.batch_windows = 0;
        self.send(frame);
    }

    fn flush_decisions(&mut self) {
        if self.decisions.is_empty() {
            return;
        }
        let decisions = std::mem::take(&mut self.decisions);
        self.decisions = Vec::with_capacity(DECISION_BATCH_WINDOWS);
        self.send(Frame::BenchCommits { decisions });
    }

    fn grant_credit(&self, next_sequence: u32) {
        let free_chunks = self
            .session
            .as_ref()
            .map(|session| session.credit_chunks)
            .unwrap_or(1);
        self.send(Frame::PlaybackCredit {
            next_sequence,
            free_chunks,
        });
    }

    fn send_status(&self) {
        let (identifier, samples_received, windows, gaps, compute) = match self.session.as_ref() {
            Some(session) => (
                session.identifier.clone(),
                session.samples_received,
                session.windows,
                session.sequence_gaps,
                (
                    session.compute.minimum_microseconds,
                    session.compute.mean_microseconds(),
                    session.compute.maximum_microseconds,
                ),
            ),
            None => (String::new(), 0, 0, 0, (0, 0, 0)),
        };
        self.send(Frame::BenchStatus {
            mode: self.mode.into(),
            session: identifier,
            samples_received,
            windows_processed: windows,
            feature_minimum_microseconds: compute.0,
            feature_mean_microseconds: compute.1,
            feature_maximum_microseconds: compute.2,
            heap_free_bytes: heap_free_bytes(),
            largest_free_block_bytes: largest_free_block_bytes(),
            dropped_chunks: self.refused.load(Ordering::Relaxed),
            sequence_gaps: gaps,
            stored_rows: self.stored_rows,
            flash_rows: self
                .flash_rows
                .as_ref()
                .map(TrainingRows::row_count)
                .unwrap_or(0) as u32,
        });
    }

    fn fail(&self, stage: &str, detail: String) {
        warn!("bench {stage}: {detail}");
        self.send(Frame::BenchError {
            stage: stage.into(),
            detail,
        });
    }

    /// A closed outbound channel means the serve loop is gone, which means the
    /// device is on its way down; there is nowhere to report that to.
    fn send(&self, frame: Frame) {
        let _ = self.outbound.send(frame);
    }
}

/// How many chunks of `chunk_samples` instants fit the in-flight byte budget.
/// Always at least one, so a host that picked a chunk larger than the budget
/// still makes progress — one chunk at a time, throughput-limited rather than
/// deadlocked.
fn credit_chunks(chunk_samples: usize) -> u32 {
    let chunk_bytes = protocol::playback_chunk_bytes(chunk_samples).max(1);
    ((CREDIT_BUDGET_BYTES / chunk_bytes) as u32).clamp(1, MAX_CREDIT_CHUNKS)
}

/// The `index`-th little-endian `f32` in a bit blob, or zero past the end.
///
/// Every caller has already checked the blob's length against what it expects,
/// so the bounds case is unreachable; it returns zero rather than panicking
/// because a panic here reboots a device whose only console is the link the
/// panic would have travelled on.
fn float_at(bits: &[u8], index: usize) -> f32 {
    let start = index * 4;
    match bits.get(start..start + 4) {
        Some(bytes) => f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The credit window must not let the host put more bytes on the wire than
    /// the USB receive ring can hold: the driver drops the overflow instead of
    /// delaying it, and the loss reads as a sequence gap several seconds into a
    /// run. Checked at every chunk size the protocol allows, because the host
    /// picks that number and the grant is computed from it.
    #[test]
    fn the_credit_window_fits_the_receive_ring_at_every_chunk_size() {
        for chunk_samples in 1..=PLAYBACK_MAX_CHUNK_SAMPLES {
            let outstanding = credit_chunks(chunk_samples) as usize
                * protocol::playback_chunk_bytes(chunk_samples);
            assert!(
                outstanding <= crate::SERIAL_RX_BUFFER_BYTES,
                "chunks of {chunk_samples} leave {outstanding} bytes outstanding, ring holds {}",
                crate::SERIAL_RX_BUFFER_BYTES
            );
        }
    }

    #[test]
    fn a_chunk_larger_than_the_budget_still_gets_one_credit() {
        assert_eq!(credit_chunks(PLAYBACK_MAX_CHUNK_SAMPLES), 1);
        assert!(credit_chunks(125) > 1);
    }

    /// One `BenchFeatures` payload has to fit the serve loop's single encode
    /// buffer, which is reserved at boot and sized for the EMG path.
    #[test]
    fn a_feature_batch_fits_the_encode_buffer() {
        let payload = FEATURE_BATCH_WINDOWS * FEATURE_COUNT * 4;
        assert!(payload < 16 * 1024, "{payload} bytes of features per frame");
    }

    #[test]
    fn chunk_timing_attribution_needs_chunks_inside_one_window() {
        // The per-window cost is exact only when a chunk cannot straddle a
        // window boundary, which is what the host tool's default guarantees.
        assert_eq!(WINDOW_SAMPLES % 125, 0);
    }

    #[test]
    fn float_at_reads_little_endian_bits() {
        let values = [1.0f32, -0.5, 1e-12];
        let bits: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        for (index, expected) in values.iter().enumerate() {
            assert_eq!(float_at(&bits, index), *expected);
        }
        assert_eq!(float_at(&bits, values.len()), 0.0);
    }

    #[test]
    fn compute_cost_reports_the_first_window_as_the_minimum() {
        let mut cost = ComputeCost::default();
        cost.add_chunk(4_000);
        cost.close_window();
        cost.add_chunk(6_000);
        cost.close_window();
        assert_eq!(cost.minimum_microseconds, 4_000);
        assert_eq!(cost.maximum_microseconds, 6_000);
        assert_eq!(cost.mean_microseconds(), 5_000);
    }
}
