//! The combining stage: two per-chip sample streams onto one grid, through
//! conditioning, into windows the main loop consumes.
//!
//! Acquisition is three threads. Each chip has a [`super::chip_pipeline`] thread
//! that owns its SPI bus, its DRDY interrupt, and its health, and emits batches of
//! timestamped 8-channel frames. This module's combiner thread drains both
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

use super::ads1298::SAMPLE_RATE_HZ;
use super::channel::{Board, DEVICE_COUNT};
use super::chip_pipeline::{self, ChipEvent};
use super::decode::Sample;
use super::preprocess::{wire_time_step, InputStage};
use super::FrontEnds;

/// Windows the channel will hold.
///
/// The depth is not about the consumer's speed — inference is far inside the window
/// period — but about how long a link write may stall before recorded data starts
/// going missing. It wants to be deeper than this; heap is what stops it. Every unit
/// of depth is ~24 KB of standing allocation (a window's model-input buffer plus its
/// packed payload, and the recycle pool holds a buffer set per slot to match),
/// against a steady state that must leave room for a live TCP session's lwIP
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
/// allocation. Each pooled set parks ~24 KB of capacity, and heap headroom (TCP's
/// link threads need two 8 KB contiguous stacks at dial time) outranks covering the
/// rare pool miss, which just falls back to a fresh allocation.
const RECYCLE_POOL_DEPTH: usize = 1;

/// Stack for the combiner thread. The window buffers live on the heap; esp-idf's
/// default stack is tight and an overflow presents as an unexplained reboot.
const THREAD_STACK_BYTES: usize = 8192;

/// Below the chip pipelines (frame reads win under contention), above the main
/// loop's default, below wifi.
const THREAD_PRIORITY: u8 = 9;

/// Grid ticks between aligner accounting log lines: ~33 s at 2 kHz, long enough
/// that the log stays single events rather than spam.
const ALIGNER_LOG_TICKS: u64 = 65_536;

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
/// exactly, allocated once, and freed when the frame has been sent. The conditioned
/// model input rides alongside because the conditioning is stateful and not
/// invertible; its buffer returns to the combiner through the recycle pool rather
/// than being reallocated per window. Both choices serve the same constraint: the
/// device heap's out-of-memory margin is spent on multi-kilobyte transients, so a
/// window contributes exactly one — its packed payload, which *is* the product.
pub(crate) struct AcquiredWindow {
    /// Device-clock microseconds at the first time step of this window. This is the
    /// window's place on the real timeline — *not* a count of windows times a
    /// nominal duration: recoveries and outages remove real time that a synthetic
    /// `seq × window_us` timeline would silently paper over. Consumers anchoring
    /// this timeline to their own clock see data stay put instead of drifting.
    pub(crate) started_us: u64,
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

/// Health totals across the front end. These outlive any single window, so they sit
/// beside the channel rather than in it; the pipelines' log lines carry the per-chip
/// attribution, and the aligner's counters (logged by the combiner) carry the
/// clock-alignment accounting.
#[derive(Default)]
pub(super) struct HealthCounters {
    /// Windows the channel had no room for.
    pub(super) dropped: AtomicU32,
    /// Frames the driver failed to read, both chips.
    pub(super) read_errors: AtomicU32,
    /// Frames whose status word lost its fixed marker bits, both chips.
    pub(super) bad_status: AtomicU32,
    /// Times a chip was warm-recovered after its conversions died. The bring-up
    /// campaign (documentation/ads1298-bringup-2026-07-31/TEST-LOG.md) established
    /// the front end dies stochastically under multi-channel conversion; recovery
    /// restores it in a few milliseconds at the cost of a gap in that chip's stream.
    pub(super) recoveries: AtomicU32,
}

/// The consumer side of acquisition.
pub(crate) struct AdcSource {
    windows: Receiver<AcquiredWindow>,
    /// Returns spent window buffers to the combiner. See [`Self::recycle`].
    recycled: SyncSender<(Vec<i8>, Vec<u8>, Vec<u8>)>,
    counters: Arc<HealthCounters>,
    window_length: usize,
}

impl AdcSource {
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

    /// Every window waiting, oldest first, or empty if none is. Never blocks.
    ///
    /// The whole backlog comes out because the stream is recorded: the caller sends
    /// each window on the wire in order and classifies only the last, so a link stall
    /// costs latency on the decision rather than a gap in the data. Nothing is
    /// discarded here — [`HealthCounters::dropped`] only counts what the producer
    /// could not hand over at all.
    pub(crate) fn drain_windows(&self) -> Vec<AcquiredWindow> {
        let mut drained = Vec::new();
        while let Ok(window) = self.windows.try_recv() {
            // The model-input copy in `main` asserts on a length mismatch, and an
            // assert there is a reboot with no console, so check it here first.
            let expected = self.window_length * INPUT_CH;
            if window.samples.len() != expected {
                warn!(
                    "discarding malformed window: {} model samples, expected {expected}",
                    window.samples.len(),
                );
                continue;
            }
            drained.push(window);
        }
        drained
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
) -> Result<AdcSource> {
    let (window_sender, windows) = sync_channel::<AcquiredWindow>(WINDOW_QUEUE_DEPTH);
    let (event_sender, events) = sync_channel::<(usize, ChipEvent)>(EVENT_QUEUE_DEPTH);
    let (recycle_sender, recycled) =
        sync_channel::<(Vec<i8>, Vec<u8>, Vec<u8>)>(RECYCLE_POOL_DEPTH);
    let counters = Arc::new(HealthCounters::default());

    let FrontEnds {
        chips,
        _power_down: power_down,
    } = front_ends;
    for ((chip, power_down), board) in chips.into_iter().zip(power_down).zip(Board::ALL) {
        chip_pipeline::spawn(
            board.device_index(),
            crate::cores::front_end_core(board),
            chip,
            power_down,
            event_sender.clone(),
            counters.clone(),
        )?;
    }
    drop(event_sender); // the pipelines hold the only senders now

    let combiner_counters = counters.clone();
    crate::cores::spawn_pinned(crate::cores::COMBINER_CORE, || {
        std::thread::Builder::new()
            .name("adc-combine".into())
            .stack_size(THREAD_STACK_BYTES)
            .spawn(move || {
                set_current_thread_priority(THREAD_PRIORITY);
                info!("combiner running");
                combine(
                    events,
                    window_sender,
                    recycled,
                    combiner_counters,
                    window_length,
                    input_scale,
                );
            })
    })??;

    Ok(AdcSource {
        windows,
        recycled: recycle_sender,
        counters,
        window_length,
    })
}

/// The combiner loop: pipeline events in, aligned/conditioned windows out. Returns
/// when every pipeline is gone (the receive fails) or the main loop is gone (the
/// window send disconnects).
fn combine(
    events: Receiver<(usize, ChipEvent)>,
    windows: SyncSender<AcquiredWindow>,
    recycled: Receiver<(Vec<i8>, Vec<u8>, Vec<u8>)>,
    counters: Arc<HealthCounters>,
    window_length: usize,
    input_scale: f32,
) {
    let samples_per_window = window_length * INPUT_CH;
    let mut aligner = GridAligner::<Sample, DEVICE_COUNT>::new(SAMPLE_RATE_HZ);
    // Owned here rather than shared: the conditioning carries per-channel filter
    // state across time steps, and this thread is the only writer.
    let mut input_stage = InputStage::new(input_scale, SAMPLE_RATE_HZ as f32);
    let mut building: Vec<i8> = Vec::with_capacity(samples_per_window);
    // The wire stream, filled in lockstep with `building`; the two are always the
    // same length and are cleared together.
    let mut building_wire: Vec<i16> = Vec::with_capacity(samples_per_window);
    // The window's missing-mask bit planes (`Frame::Emg::missing`), set in
    // lockstep with the buffers above and zeroed whenever they are cleared.
    let missing_mask_len = DEVICE_COUNT * protocol::missing_plane_stride(window_length);
    let mut building_missing: Vec<u8> = vec![0u8; missing_mask_len];
    // Device-clock time of the first time step in `building`; stamped when the
    // first step lands, cleared with the buffer.
    let mut building_started_us: u64 = 0;
    // The previous emitted grid step, for spotting skipped stretches.
    let mut previous_step_us: Option<u64> = None;
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
                for (at_us, frame) in frames {
                    aligner.push(chip, at_us, frame);
                }
                while let Some(step) = aligner.poll() {
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
                        // main loop has returned is reused, and only an empty pool
                        // (cold start, or a dropped window) costs fresh allocations.
                        let (mut next_building, mut packed_wire, mut next_missing) =
                            recycled.try_recv().unwrap_or_else(|_| {
                                (
                                    Vec::with_capacity(samples_per_window),
                                    Vec::new(),
                                    Vec::new(),
                                )
                            });
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
                        let full = AcquiredWindow {
                            started_us: building_started_us,
                            samples: std::mem::replace(&mut building, next_building),
                            packed_wire,
                            missing: std::mem::replace(&mut building_missing, next_missing),
                        };
                        match windows.try_send(full) {
                            Ok(()) => {}
                            Err(TrySendError::Full(_)) => {
                                counters.dropped.fetch_add(1, Ordering::Relaxed);
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
                if ticks - ticks_at_last_log >= ALIGNER_LOG_TICKS {
                    ticks_at_last_log = ticks;
                    // The alignment accounting IS the measured clock behaviour:
                    // surplus on a healthy chip is its oscillator running fast
                    // against the grid, duplicates are it running slow, missing is
                    // a genuine gap in its stream.
                    let mut line = format!(
                        "aligner: {ticks} ticks emitted, {} skipped",
                        aligner.ticks_skipped()
                    );
                    for source in 0..DEVICE_COUNT {
                        let c = aligner.counters(source);
                        line.push_str(&format!(
                            " || chip {source}: surplus {} dup {} missing {} rejected {} overflow {}",
                            c.surplus_dropped, c.duplicated, c.missing, c.rejected, c.overflowed
                        ));
                    }
                    info!("{line}");
                }
            }
        }
    }
}

/// Raises the calling thread's FreeRTOS priority. `std::thread` gives every thread the
/// esp-idf default, which is below the wifi task and equal to the main loop.
pub(super) fn set_current_thread_priority(priority: u8) {
    unsafe {
        esp_idf_svc::sys::vTaskPrioritySet(std::ptr::null_mut(), priority as u32);
    }
}
