//! The combining stage: two per-chip sample streams onto one grid, through
//! conditioning, into windows the main loop consumes.
//!
//! Acquisition is three threads and two interrupts. Each chip's frames are clocked
//! out inside its DRDY interrupt ([`super::frame_reader`]); each chip then has a
//! [`pipeline`] thread that drains that ring, owns the chip's health,
//! and emits batches of timestamped 8-channel frames. This module's combiner drains both
//! pipelines into an [`emg_runtime::alignment::GridAligner`], which places the two
//! independently clocked streams onto one 2 kHz grid on the device clock, with the
//! pairing policy and its accounting in one explicit, host-tested place. Each
//! emitted grid step then runs through the conditioning stage
//! into the model's int8 window and, in lockstep, into the raw i16 wire window.
//!
//! The aligner is where the dual-clock reality lives, measured instead of assumed:
//! its per-source counters say exactly how many frames each chip's oscillator
//! surplus (dropped), lagged (duplicated), or failed to deliver (missing), and the
//! combiner logs them periodically. The output timeline is grid-regular —
//! consecutive windows sit exactly `window / 2000 Hz` apart on the device clock —
//! so the wire's claimed sample rate is finally the truth rather than a nominal
//! figure the oscillators miss by percents.
//!
//! Presence is the contract between the pipelines and the grid. A pipeline brackets
//! every untrustworthy stretch (death, warm recovery, reference settling) with
//! `Absent`/`Present` events on the same ordered channel as its frames. On
//! `Absent`, the combiner marks the aligner source absent (its slots read zero
//! downstream, which [`super::preprocess`] documents as the honest value), resets
//! that chip's conditioning, and discards the partial window — a window silently
//! spanning a recovery would feed the model 250 ms that never happened on half its
//! channels. The other chip streams through its peer's whole outage untouched.
//!
//! The window channel is the handoff to the main loop, and the consumer drains it
//! rather than keeping the newest, because the stream is recorded: a window that
//! arrives late is still data, and a window dropped for being 250 ms stale is a
//! hole in the training set. Latency semantics are unchanged — the main loop still
//! classifies only the newest of a drained batch. What the queue cannot do is grow:
//! [`WINDOW_QUEUE_DEPTH`] documents the heap that stops it. Overflow past it is
//! counted, not absorbed.

use anyhow::Result;
use emg_runtime::alignment::GridAligner;
use emg_runtime::model::INPUT_CH;
use log::{info, warn};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

mod pipeline;

use super::ads1298::{self, SAMPLE_RATE_HZ};
use super::channel::{Board, DEVICE_COUNT};
use super::decode::Sample;
use super::preprocess::{wire_time_step, InputStage};
use super::FrontEnds;
use pipeline::ChipEvent;

/// Windows the channel will hold.
///
/// The depth is not about the consumer's speed — inference is far inside the window
/// period — but about how long a link write may stall before recorded data starts
/// going missing. It wants to be deeper than this; heap is what stops it. Every unit
/// of depth is ~24 KB of standing allocation (a window's model-input buffer plus its
/// packed payload, and the recycle pool holds a buffer set per slot to match),
/// against a steady state that must leave room for serial transport and BLE
/// buffers on top. Depth one still covers a main-loop absence of a full window
/// period — the loop polls every 5 ms — and overflow past it is counted
/// ([`HealthCounters::dropped`]), not absorbed.
const WINDOW_QUEUE_DEPTH: usize = 1;

/// Pipeline events the combiner may fall behind by before a pipeline's send blocks.
/// Each `Frames` event is ~8 ms of one chip, so this is over two seconds of slack —
/// combining a batch costs microseconds, so depth this deep is only ever touched if
/// the combiner itself is wedged.
const EVENT_QUEUE_DEPTH: usize = 64;

/// Spent window buffers waiting to be reused (see [`AdcSource::recycle`]). One is
/// enough for the steady cycle — the main loop returns a set as each window ships,
/// and the combiner takes it back for the next — so in steady state every window
/// reuses pooled buffers and the window path performs no recurring multi-kilobyte
/// allocation. Each pooled set parks ~24 KB of capacity, and heap headroom (the
/// link threads need two 8 KB contiguous stacks at dial time) outranks covering the
/// rare cold-start pool miss, which falls back to a fresh allocation. A window
/// rejected by the output queue is reclaimed directly by the combiner instead
/// of being freed and reallocated; calibration makes that overflow path common
/// enough for large-buffer churn to fragment the heap.
const RECYCLE_POOL_DEPTH: usize = 1;

/// Empty chip-frame batches returned from the combiner to each chip pipeline.
/// One returned batch plus the pipeline's active batch forms a steady double
/// buffer, removing the two pipelines' ~256 allocations per second.
const BATCH_RECYCLE_DEPTH: usize = 1;

/// Stack for the combiner thread. The window buffers live on the heap; esp-idf's
/// default stack is tight and an overflow presents as an unexplained reboot.
const THREAD_STACK_BYTES: usize = 8192;

/// Below the chip pipelines (frame reads win under contention), above the main
/// loop's default, below wifi.
const THREAD_PRIORITY: u8 = 9;

/// Grid ticks between aligner telemetry reports: ~4 s at 2 kHz. Telemetry never
/// touches log retention, so the rate is set by trend resolution alone.
const ALIGNER_REPORT_TICKS: u64 = 8_192;

/// An emitted-step gap larger than this means the grid skipped ticks (every present
/// source gapped, or the whole front end was out): the partial window spans a hole
/// in time and is discarded rather than silently stitched. Three half-periods:
/// tolerant of nothing — one period is the only legal spacing — but not tripped by
/// integer rounding of the grid arithmetic.
const STEP_DISCONTINUITY_US: u64 = 3 * 1_000_000 / (2 * SAMPLE_RATE_HZ as u64);

/// One window handed to the main loop, stamped with its first grid tick.
///
/// The wire stream travels already packed: the raw i16 counts live in a buffer the
/// combiner owns permanently, and what leaves it is the delta+varint payload
/// ([`protocol::pack_sample_stream`]) that goes onto the wire verbatim — sized
/// exactly, allocated once, and returned for reuse after the frame has been sent.
/// The conditioned model input rides alongside because the conditioning is stateful
/// and not invertible; its buffer returns through the recycle pool rather than being
/// reallocated per window. Both choices serve the same constraint: the
/// device heap's out-of-memory margin is spent on multi-kilobyte buffers, so both
/// allocations cycle through the bounded queue and recycle paths rather than churn.
pub(crate) struct AcquiredWindow {
    /// Device-clock microseconds at the first time step of this window. This is the
    /// window's place on the real timeline — *not* a count of windows times a
    /// nominal duration: recoveries and outages remove real time that a synthetic
    /// `seq × window_us` timeline would silently paper over. Consumers anchoring
    /// this timeline to their own clock see data stay put instead of drifting.
    pub(crate) started_us: u64,
    /// One past the last emitted acquisition-grid sample in this window.
    /// This counter advances in the combiner whether optional feature work runs.
    pub(crate) end_sample: u64,
    /// Conditioned model input, time-major `[t, c]`. Never leaves the device; the
    /// buffer goes back via [`AdcSource::recycle`].
    pub(crate) samples: Vec<i8>,
    /// The window's raw counts at [`super::MICROVOLTS_PER_WIRE_COUNT`],
    /// channel-major, delta+varint packed — the `Frame::Emg` payload byte-for-byte.
    pub(crate) packed_wire: Vec<u8>,
    /// Which steps of which source are aligner gaps — the `Frame::Emg::missing`
    /// bit planes byte-for-byte, so a consumer can tell the placeholder zeros in
    /// the wire stream from measurements. The model input needs no equivalent:
    /// zero already means "no signal" there.
    pub(crate) missing: Vec<u8>,
}

/// All large acquisition buffers, reserved while the boot heap is still whole.
/// The live path may move these buffers between threads but never creates a
/// replacement: if backpressure withholds the spare set, that window is dropped.
pub(crate) struct AcquisitionBuffers {
    building: Vec<i8>,
    next_building: Vec<i8>,
    building_wire: Vec<i16>,
    packed_wire: Vec<u8>,
    building_missing: Vec<u8>,
    next_missing: Vec<u8>,
}

impl AcquisitionBuffers {
    pub(crate) fn reserve(window_length: usize) -> Self {
        let samples_per_window = window_length * INPUT_CH;
        let missing_mask_len = DEVICE_COUNT * protocol::missing_plane_stride(window_length);
        Self {
            building: Vec::with_capacity(samples_per_window),
            next_building: Vec::with_capacity(samples_per_window),
            building_wire: Vec::with_capacity(samples_per_window),
            packed_wire: Vec::with_capacity(protocol::max_packed_sample_bytes(samples_per_window)),
            building_missing: vec![0; missing_mask_len],
            next_missing: vec![0; missing_mask_len],
        }
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.building.capacity()
            + self.next_building.capacity()
            + self.building_wire.capacity() * core::mem::size_of::<i16>()
            + self.packed_wire.capacity()
            + self.building_missing.capacity()
            + self.next_missing.capacity()
    }
}

/// Health totals across the front end. These outlive any single window, so they sit
/// beside the channel rather than in it; the pipelines' log lines carry the per-chip
/// attribution, and the aligner's counters (logged by the combiner) carry the
/// clock-alignment accounting.
#[derive(Default)]
pub(super) struct HealthCounters {
    /// Windows the channel had no room for.
    pub(super) dropped: AtomicU32,
    /// Frames the interrupt read path could not clock out, both chips: the SPI
    /// host never reported the transfer complete.
    pub(super) read_errors: AtomicU32,
    /// Frames whose status word lost its fixed marker bits, summed over both
    /// chips. Each pipeline keeps its own count for its telemetry, since a
    /// device-wide sum cannot say which chip is failing.
    pub(super) bad_status: AtomicU32,
    /// Times a chip was warm-recovered after its conversions died. The bring-up
    /// campaign (documentation/ads1298-bringup-2026-07-31/TEST-LOG.md) established
    /// the front end dies stochastically under multi-channel conversion; recovery
    /// restores it in a few milliseconds at the cost of a gap in that chip's stream.
    pub(super) recoveries: AtomicU32,
    /// Frames on which either chip flagged any channel lead-off, summed over
    /// both. A count rather than a mask: a calibration asks only whether an
    /// electrode was off the skin anywhere across a labeled span, and which
    /// one it was belongs in telemetry, where the per-chip bit patterns
    /// already go.
    pub(super) lead_off_frames: AtomicU32,
    /// Which channels the front end currently says are not in contact, one bit
    /// per channel across both chips. The newest reading rather than a count:
    /// a wearer seating a band needs to know which electrode is lifted *now*,
    /// and the count beside it answers a different question — whether anything
    /// lifted during a rep.
    pub(super) lead_off_channels: AtomicU32,
}

/// The consumer side of acquisition.
pub(crate) struct AdcSource {
    windows: Receiver<AcquiredWindow>,
    /// Returns spent window buffers to the combiner. See [`Self::recycle`].
    recycled: SyncSender<(Vec<i8>, Vec<u8>, Vec<u8>)>,
    counters: Arc<HealthCounters>,
    window_length: usize,
    acquisition_sample: Arc<MonotonicCounter>,
}

/// A single-writer u64 snapshot on targets that only provide 32-bit atomics.
/// The odd/even revision prevents a reader from combining different writes.
#[derive(Default)]
struct MonotonicCounter {
    revision: AtomicU32,
    low: AtomicU32,
    high: AtomicU32,
}

impl MonotonicCounter {
    fn store(&self, value: u64) {
        self.revision.fetch_add(1, Ordering::SeqCst);
        self.low.store(value as u32, Ordering::Relaxed);
        self.high.store((value >> 32) as u32, Ordering::Relaxed);
        self.revision.fetch_add(1, Ordering::SeqCst);
    }

    fn load(&self) -> u64 {
        loop {
            let before = self.revision.load(Ordering::Acquire);
            if before & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            let low = self.low.load(Ordering::Relaxed);
            let high = self.high.load(Ordering::Relaxed);
            let after = self.revision.load(Ordering::Acquire);
            if before == after {
                return (u64::from(high) << 32) | u64::from(low);
            }
        }
    }
}

impl AdcSource {
    pub(crate) fn acquisition_sample(&self) -> u64 {
        self.acquisition_sample.load()
    }

    /// Windows lost outright, cumulative since boot: the channel was full at
    /// [`WINDOW_QUEUE_DEPTH`] when a window came ready, so it was never handed over.
    /// Since the consumer drains the whole backlog, this only moves when the main
    /// loop has been away for longer than the queue covers — a wedged link write,
    /// not a slow consumer. Every one is a hole in a recording.
    pub(crate) fn dropped_windows(&self) -> u32 {
        self.counters.dropped.load(Ordering::Relaxed)
    }

    /// Frames the driver failed to clock out, cumulative since boot, both chips.
    pub(crate) fn read_errors(&self) -> u32 {
        self.counters.read_errors.load(Ordering::Relaxed)
    }

    /// Frames whose status marker came back wrong, cumulative since boot, both chips.
    /// Unlike `read_errors`, these are transactions the SPI driver reported as
    /// successful -- the bus is running but the data is not trustworthy.
    pub(crate) fn bad_status(&self) -> u32 {
        self.counters.bad_status.load(Ordering::Relaxed)
    }

    /// Times a chip died and was warm-recovered, cumulative since boot, both chips.
    /// Each one costs the detection timeout, ~3 ms of reset-and-rewrite, and then a
    /// reference-settling discard window for that chip — during which its eight
    /// slots read zero while the other chip's stream continues.
    pub(crate) fn recoveries(&self) -> u32 {
        self.counters.recoveries.load(Ordering::Relaxed)
    }

    /// Frames on which either chip flagged any channel lead-off, cumulative.
    ///
    /// A calibration reads the difference across a labeled span rather than the
    /// value: an electrode that came off the skin at any point during a rep
    /// makes that rep unusable, and the span is short enough that a count that
    /// moved at all is the answer.
    pub(crate) fn lead_off_frames(&self) -> u32 {
        self.counters.lead_off_frames.load(Ordering::Relaxed)
    }

    /// Channels currently flagged lead-off, one bit per channel, or `None` when
    /// the front end is not watching for it.
    ///
    /// The distinction is the whole point. With the comparators unpowered the
    /// status bits read zero forever, and zero is exactly what a seated band
    /// looks like — so a bare `u16` would tell a wearer every electrode is fine
    /// on a device that cannot tell. `None` says "no answer"; nothing
    /// downstream may read it as good news. See [`ads1298::LEAD_OFF_ENABLED`],
    /// which is off until the block earns its bench experiment.
    pub(crate) fn lead_off_channels(&self) -> Option<u16> {
        ads1298::LEAD_OFF_ENABLED
            .then(|| self.counters.lead_off_channels.load(Ordering::Relaxed) as u16)
    }

    /// The next complete window without allocating a temporary backlog vector.
    pub(crate) fn poll_window(&self) -> Option<AcquiredWindow> {
        loop {
            let window = self.windows.try_recv().ok()?;
            let expected = self.window_length * INPUT_CH;
            if window.samples.len() == expected {
                return Some(window);
            }
            warn!(
                "discarding malformed window: {} model samples, expected {expected}",
                window.samples.len(),
            );
            self.recycle(window.samples, window.packed_wire, window.missing);
        }
    }

    /// Hand a spent window's buffers back for reuse: its model-input samples, its
    /// packed wire payload, and its missing-mask planes (the latter two reclaimed
    /// from the sent frame). Fire-and-forget: a full pool just lets the buffers
    /// drop, so this can never block the main loop.
    pub(crate) fn recycle(&self, samples: Vec<i8>, packed_wire: Vec<u8>, missing: Vec<u8>) {
        let _ = self.recycled.try_send((samples, packed_wire, missing));
    }
}

/// Spawns one pipeline thread per chip and the combiner, returning the consumer
/// handle.
///
/// `window_length` is the model's `input_len`; `input_scale` is its *normalised* units
/// per count, not microvolts per count — see [`super::conditioning`].
pub(crate) fn start(
    front_ends: FrontEnds,
    window_length: usize,
    input_scale: f32,
    buffers: AcquisitionBuffers,
) -> Result<AdcSource> {
    let (window_sender, windows) = sync_channel::<AcquiredWindow>(WINDOW_QUEUE_DEPTH);
    let (event_sender, events) = sync_channel::<(usize, ChipEvent)>(EVENT_QUEUE_DEPTH);
    let (chip0_batch_sender, chip0_batches) = sync_channel(BATCH_RECYCLE_DEPTH);
    let (chip1_batch_sender, chip1_batches) = sync_channel(BATCH_RECYCLE_DEPTH);
    for sender in [&chip0_batch_sender, &chip1_batch_sender] {
        sender
            .try_send(Vec::with_capacity(pipeline::FRAMES_PER_BATCH))
            .expect("a new batch recycler has room for its initial spare");
    }
    let batch_senders = [chip0_batch_sender, chip1_batch_sender];
    let mut batch_receivers = [Some(chip0_batches), Some(chip1_batches)];
    let (recycle_sender, recycled) =
        sync_channel::<(Vec<i8>, Vec<u8>, Vec<u8>)>(RECYCLE_POOL_DEPTH);
    let counters = Arc::new(HealthCounters::default());
    let acquisition_sample = Arc::new(MonotonicCounter::default());

    let FrontEnds {
        chips,
        readers,
        _power_down: power_down,
    } = front_ends;
    for (((chip, reader), power_down), board) in chips
        .into_iter()
        .zip(readers)
        .zip(power_down)
        .zip(Board::ALL)
    {
        let index = board.device_index();
        pipeline::spawn(
            index,
            crate::cores::front_end_core(board),
            chip,
            reader,
            power_down,
            event_sender.clone(),
            batch_receivers[index]
                .take()
                .expect("each chip owns one batch recycler"),
            counters.clone(),
        )?;
    }
    drop(event_sender); // the pipelines hold the only senders now

    let combiner_counters = counters.clone();
    let combiner_acquisition_sample = acquisition_sample.clone();
    crate::cores::spawn_pinned(crate::cores::COMBINER_CORE, || {
        std::thread::Builder::new()
            .name("adc-combine".into())
            .stack_size(THREAD_STACK_BYTES)
            .spawn(move || {
                crate::cores::set_current_thread_priority(THREAD_PRIORITY);
                info!("combiner running");
                combine(
                    events,
                    batch_senders,
                    window_sender,
                    recycled,
                    combiner_counters,
                    combiner_acquisition_sample,
                    window_length,
                    input_scale,
                    buffers,
                );
            })
    })??;

    Ok(AdcSource {
        windows,
        recycled: recycle_sender,
        counters,
        window_length,
        acquisition_sample,
    })
}

/// The combiner loop: pipeline events in, aligned/conditioned windows out. Returns
/// when every pipeline is gone (the receive fails) or the main loop is gone (the
/// window send disconnects).
fn combine(
    events: Receiver<(usize, ChipEvent)>,
    batch_recyclers: [SyncSender<Vec<(u64, Sample)>>; DEVICE_COUNT],
    windows: SyncSender<AcquiredWindow>,
    recycled: Receiver<(Vec<i8>, Vec<u8>, Vec<u8>)>,
    counters: Arc<HealthCounters>,
    acquisition_sample: Arc<MonotonicCounter>,
    window_length: usize,
    input_scale: f32,
    buffers: AcquisitionBuffers,
) {
    let samples_per_window = window_length * INPUT_CH;
    let mut aligner = GridAligner::<Sample, DEVICE_COUNT>::new(SAMPLE_RATE_HZ);
    // Owned here rather than shared: the conditioning carries per-channel filter
    // state across time steps, and this thread is the only writer.
    let mut input_stage = InputStage::new(input_scale, SAMPLE_RATE_HZ as f32);
    let AcquisitionBuffers {
        mut building,
        next_building,
        mut building_wire,
        packed_wire,
        mut building_missing,
        next_missing,
    } = buffers;
    // The wire stream, filled in lockstep with `building`; the two are always the
    // same length and are cleared together.
    // The window's missing-mask bit planes (`Frame::Emg::missing`), set in
    // lockstep with the buffers above and zeroed whenever they are cleared.
    let missing_mask_len = DEVICE_COUNT * protocol::missing_plane_stride(window_length);
    let mut spare_buffers = Some((next_building, packed_wire, next_missing));
    // Device-clock time of the first time step in `building`; stamped when the
    // first step lands, cleared with the buffer.
    let mut building_started_us: u64 = 0;
    // The previous emitted grid step, for spotting skipped stretches.
    let mut previous_step_us: Option<u64> = None;
    // Device-time anchor for the acquisition counter. Unlike feature-pipeline
    // state, this is established by the first aligned sample and never resets.
    let mut acquisition_anchor_us: Option<u64> = None;
    let mut ticks_at_last_log: u64 = 0;

    loop {
        let (chip, event) = match events.recv() {
            Ok(received) => received,
            Err(_) => {
                info!("every chip pipeline is gone; combining stopping");
                return;
            }
        };
        match event {
            ChipEvent::Present => aligner.set_present(chip, true),
            ChipEvent::Absent => {
                aligner.set_present(chip, false);
                // That chip's stream is about to jump: the reset pulse re-settles
                // its reference and its electrodes may come back sitting somewhere
                // else. Only its own eight slots re-seed.
                input_stage.reset_device_after_gap(chip);
                // The partial window would silently span the outage on half its
                // channels; discard it rather than stitch across the gap.
                building.clear();
                building_wire.clear();
                building_missing.fill(0);
            }
            ChipEvent::Frames(frames) => {
                for &(at_us, frame) in &frames {
                    aligner.push(chip, at_us, frame);
                }
                let mut frames = frames;
                frames.clear();
                let _ = batch_recyclers[chip].try_send(frames);
                while let Some(step) = aligner.poll() {
                    let anchor_us = *acquisition_anchor_us.get_or_insert(step.at_us);
                    let elapsed_us = step.at_us - anchor_us;
                    let seconds = elapsed_us / 1_000_000;
                    let subsecond_us = elapsed_us % 1_000_000;
                    let current_sample = seconds * SAMPLE_RATE_HZ as u64
                        + subsecond_us * SAMPLE_RATE_HZ as u64 / 1_000_000
                        + 1;
                    acquisition_sample.store(current_sample);
                    // Skipped ticks mean the emitted timeline has a hole (every
                    // present source gapped at once); a window must not span it.
                    if previous_step_us
                        .is_some_and(|previous| step.at_us - previous > STEP_DISCONTINUITY_US)
                        && !building.is_empty()
                    {
                        building.clear();
                        building_wire.clear();
                        building_missing.fill(0);
                    }
                    previous_step_us = Some(step.at_us);
                    if building.is_empty() {
                        building_started_us = step.at_us;
                    }
                    let step_index = building.len() / INPUT_CH;
                    for (source, slot) in step.slots.iter().enumerate() {
                        if slot.is_none() {
                            let plane = source * protocol::missing_plane_stride(window_length);
                            building_missing[plane + step_index / 8] |= 1 << (step_index % 8);
                        }
                    }
                    building.extend_from_slice(&input_stage.time_step(&step.slots));
                    building_wire.extend_from_slice(&wire_time_step(&step.slots));

                    if building.len() >= samples_per_window {
                        // Both outgoing buffers cycle through the pool: whatever the
                        // main loop has returned is reused. A rejected output is
                        // reclaimed locally; only cold start costs fresh allocations.
                        let available = spare_buffers.take().or_else(|| recycled.try_recv().ok());
                        let Some((mut next_building, mut packed_wire, mut next_missing)) =
                            available
                        else {
                            counters.dropped.fetch_add(1, Ordering::Relaxed);
                            building.clear();
                            building_wire.clear();
                            building_missing.fill(0);
                            continue;
                        };
                        next_building.clear();
                        next_missing.clear();
                        next_missing.resize(missing_mask_len, 0);
                        // The wire payload leaves here already packed, straight off
                        // the persistent i16 buffer (channel-major, matching the
                        // `Frame::Emg` layout), into the pooled payload buffer that
                        // is itself the bytes the link writes.
                        let wire = building_wire.as_slice();
                        protocol::pack_sample_stream_into(
                            &mut packed_wire,
                            (0..INPUT_CH).flat_map(|ch| {
                                (0..window_length).map(move |ti| wire[ti * INPUT_CH + ch])
                            }),
                        );
                        building_wire.clear();
                        // Derive the acquisition index from the aligner's device-time
                        // grid rather than from delivered-window count. Sustained gaps
                        // skip emitted ticks, but they must still advance the clock a
                        // future cue mapping is measured against.
                        let full = AcquiredWindow {
                            started_us: building_started_us,
                            end_sample: current_sample,
                            samples: std::mem::replace(&mut building, next_building),
                            packed_wire,
                            missing: std::mem::replace(&mut building_missing, next_missing),
                        };
                        match windows.try_send(full) {
                            Ok(()) => {}
                            Err(TrySendError::Full(window)) => {
                                counters.dropped.fetch_add(1, Ordering::Relaxed);
                                spare_buffers =
                                    Some((window.samples, window.packed_wire, window.missing));
                            }
                            Err(TrySendError::Disconnected(_)) => {
                                // The main loop is gone, so there is nobody left
                                // to feed.
                                info!("ADC consumer disconnected; combining stopping");
                                return;
                            }
                        }
                    }
                }
                let ticks = aligner.ticks_emitted();
                if ticks - ticks_at_last_log >= ALIGNER_REPORT_TICKS {
                    ticks_at_last_log = ticks;
                    // The alignment accounting IS the measured clock behaviour:
                    // surplus on a healthy chip is its oscillator running fast
                    // against the grid, duplicates are it running slow, missing is
                    // a genuine gap in its stream. All counters cumulative since
                    // boot, so a dropped report costs nothing.
                    let metric = crate::telemetry::metric;
                    let mut metrics = vec![
                        metric("ticks_emitted", ticks as f64),
                        metric("ticks_skipped", aligner.ticks_skipped() as f64),
                    ];
                    for source in 0..DEVICE_COUNT {
                        let counters = aligner.counters(source);
                        for (name, value) in [
                            ("surplus_dropped", counters.surplus_dropped),
                            ("duplicated", counters.duplicated),
                            ("missing", counters.missing),
                            ("rejected", counters.rejected),
                            ("overflowed", counters.overflowed),
                        ] {
                            metrics.push(metric(&format!("chip{source}_{name}"), value as f64));
                        }
                    }
                    crate::telemetry::report("aligner", metrics);
                }
            }
        }
    }
}
