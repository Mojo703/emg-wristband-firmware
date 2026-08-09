//! Aligning independently clocked sample streams onto one shared time grid.
//!
//! The problem this solves: the wristband's two ADS1298s each convert on their own
//! internal oscillator (±0.5% at room temperature, ±2% over the rated range), so
//! their samples tick at *almost* the same rate and slip continuously in phase.
//! Anything that wants a single sixteen-channel time series — the model, the wire
//! stream, a recorded session — has to decide which of chip A's samples goes with
//! which of chip B's. This module is that decision, made explicit: a fixed grid at
//! the nominal rate on the device clock, and per-source counters for every frame
//! the alignment had to drop, reuse, or go without, so the actual clock behaviour
//! is measured instead of assumed.
//!
//! # The model
//!
//! A [`GridAligner`] owns `SOURCES` independent streams of timestamped frames. Grid
//! tick `k` sits at `anchor + k / rate` (integer microsecond arithmetic, drift-free),
//! where the anchor is the first accepted frame's timestamp. For each tick, each
//! *present* source contributes the queued frame nearest in time to the tick, if one
//! sits within the acceptance window (three quarters of a period — see
//! [`GridAligner::new`] for why not half); the outcomes are:
//!
//! - a frame skipped over entirely, because a later frame was at least as near:
//!   `surplus_dropped` — the source's clock runs faster than the grid;
//! - a frame reused for a second tick: `duplicated` — the source runs slower;
//! - no frame within the acceptance window: `missing`, and the slot reads `None` — a
//!   gap in that source's stream.
//!
//! At the nominal 2 kHz grid with the ADS1298's real ~2.05 kHz oscillators, expect a
//! steady few-percent `surplus_dropped` on every healthy source; that number *is* the
//! measured oscillator error. The output stream itself is exactly grid-regular:
//! consecutive [`AlignedStep::at_us`] values differ by one period (except across
//! skipped stretches, below), which is what lets a consumer finally trust the claimed
//! sample rate.
//!
//! # Presence
//!
//! Sources start absent and must be announced with [`GridAligner::set_present`]. An
//! absent source never blocks the grid and its slot reads `None`; marking a source
//! absent also clears its queue (its next frames describe a different epoch — after a
//! reset, a re-settle — and stitching across that gap would be a lie). The contract a
//! producer must honour: a source that stops delivering frames MUST be marked absent
//! promptly, because the grid waits on every present source before it can emit a
//! tick. The firmware's per-chip death detection provides exactly that bound.
//!
//! # Empty and skipped ticks
//!
//! When every present source is in a gap at some tick, what happens depends on how
//! long the gap has run. A short one — up to [`MAX_CONSECUTIVE_EMPTY_TICKS`] — is
//! emitted with every slot `None`: a couple of missed reads landing on both sources
//! at once must not void the 250 ms window being built around them, and an all-zero
//! time step is the same honest answer a single gapped source already gives. A
//! sustained one stops being emitted: flooding the consumer with all-`None` steps
//! across a dead stretch would manufacture data where there is none, so the ticks
//! are skipped and the consumer sees time jump between one emitted step and the
//! next — a visible discontinuity to discard partial work across.
//! [`GridAligner::ticks_skipped`] counts what was passed over.

use alloc::collections::VecDeque;

/// Frames one source may hold while waiting to be paired. Steady state needs one or
/// two, plus room for a producer's batch landing at once; depth beyond that only
/// exists to ride out a slow consumer, and past it the oldest frame is dropped and
/// counted (`overflowed`) rather than growing the heap.
const MAX_QUEUED_FRAMES: usize = 32;

/// The longest run of no-contribution ticks that still emits (all slots `None`)
/// rather than being skipped. Short simultaneous gaps — a few missed reads landing
/// on both sources at once — stay inside the emitted timeline as honest zeros, so a
/// window being built across one is not thrown away; anything longer is a genuine
/// outage and becomes a timeline discontinuity. Eight ticks is 4 ms at 2 kHz:
/// generous against read jitter, two orders of magnitude under a warm recovery's
/// settling window.
pub const MAX_CONSECUTIVE_EMPTY_TICKS: u64 = 8;

/// One emitted grid tick: the tick's place on the source clock's timeline, and one
/// slot per source — `None` for a source that is absent or had no frame near enough.
pub struct AlignedStep<T, const SOURCES: usize> {
    /// Timestamp of this tick on the same clock the pushed frames were stamped with.
    pub at_us: u64,
    pub slots: [Option<T>; SOURCES],
}

/// Everything the alignment had to do to one source's stream, cumulative. These are
/// measurements, not errors: `surplus_dropped` and `duplicated` on a healthy source
/// are its real clock rate expressed against the grid.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SourceCounters {
    /// Frames discarded without ever serving a tick: the source outpaces the grid.
    pub surplus_dropped: u32,
    /// Ticks served by a frame that had already served an earlier tick: the source
    /// lags the grid.
    pub duplicated: u32,
    /// Emitted ticks at which this source was present but had no frame within the
    /// acceptance window: a gap in its stream.
    pub missing: u32,
    /// Frames refused at the door: non-monotonic timestamp, or pushed while the
    /// source was absent.
    pub rejected: u32,
    /// Frames evicted because the queue hit [`MAX_QUEUED_FRAMES`]: the consumer is
    /// not draining the aligner.
    pub overflowed: u32,
}

struct QueuedFrame<T> {
    at_us: u64,
    payload: T,
    /// Whether this frame has already served a tick, so a second use is counted as
    /// `duplicated` and an unused eviction as `surplus_dropped`.
    used: bool,
}

struct SourceState<T> {
    present: bool,
    queue: VecDeque<QueuedFrame<T>>,
    /// Newest accepted timestamp; monotonicity gate, and the proof of decidability
    /// (once it clears `tick + acceptance`, no future frame can land inside the
    /// tick's acceptance window at all).
    newest_accepted_us: Option<u64>,
    counters: SourceCounters,
}

impl<T> SourceState<T> {
    fn new() -> Self {
        Self {
            present: false,
            queue: VecDeque::with_capacity(MAX_QUEUED_FRAMES),
            newest_accepted_us: None,
            counters: SourceCounters::default(),
        }
    }

    /// Whether this source can never again produce a frame nearer to `tick_us` than
    /// what it already holds. Absent sources are trivially decided.
    fn decided(&self, tick_us: u64, acceptance_us: u64) -> bool {
        if !self.present {
            return true;
        }
        self.newest_accepted_us
            .is_some_and(|newest| newest >= tick_us + acceptance_us)
    }
}

impl<T: Clone> SourceState<T> {
    /// The queued frame nearest to `tick_us`, if it lands within the acceptance
    /// window. Frames that a later frame beats are evicted here — that eviction is
    /// the resampling.
    fn choose(&mut self, tick_us: u64, acceptance_us: u64) -> Option<T> {
        while self.queue.len() >= 2 {
            let head_distance = distance(self.queue[0].at_us, tick_us);
            let next_distance = distance(self.queue[1].at_us, tick_us);
            if next_distance <= head_distance {
                let evicted = self.queue.pop_front().expect("len checked above");
                if !evicted.used {
                    self.counters.surplus_dropped += 1;
                }
            } else {
                break;
            }
        }
        let head = self.queue.front_mut()?;
        if distance(head.at_us, tick_us) > acceptance_us {
            return None;
        }
        if head.used {
            self.counters.duplicated += 1;
        }
        head.used = true;
        Some(head.payload.clone())
    }
}

fn distance(a: u64, b: u64) -> u64 {
    a.abs_diff(b)
}

/// See the module docs. `T` is one source's frame payload (the aligner never looks
/// inside it), `SOURCES` how many streams share the grid.
pub struct GridAligner<T, const SOURCES: usize> {
    sample_rate_hz: u64,
    acceptance_us: u64,
    /// Timestamp of grid tick zero: the first accepted frame's stamp. `None` until
    /// then, and nothing can be emitted before it.
    anchor_us: Option<u64>,
    /// Index of the next tick to decide.
    next_tick: u64,
    ticks_emitted: u64,
    ticks_skipped: u64,
    /// Length of the current run of ticks no present source contributed to,
    /// including any that were skipped rather than emitted.
    consecutive_empty: u64,
    sources: [SourceState<T>; SOURCES],
}

impl<T: Clone, const SOURCES: usize> GridAligner<T, SOURCES> {
    /// A grid at `sample_rate_hz` ticks per second of the pushed frames' clock.
    ///
    /// The acceptance window — how far from a tick its serving frame may sit — is
    /// three quarters of a period, not half. Half would be the natural reading of
    /// "nearest", but a source running slightly slower than the grid (the ADS1298
    /// oscillator is allowed ±2% over temperature) spaces its frames wider than a
    /// period, and the ticks landing in the dead zone between two frames would read
    /// as gaps when the honest answer is "reuse the nearest sample". At ¾ of a
    /// period, a source may run a third slow before it ever gaps, while a genuinely
    /// lost frame still leaves its ticks at least a full source period from any
    /// survivor and is flagged `missing`. Correctness does not depend on the width:
    /// the chosen frame is always the nearest one, the window only decides how far
    /// is too far.
    ///
    /// # Panics
    ///
    /// On a zero rate, at construction — the one place a panic beats limping on.
    pub fn new(sample_rate_hz: u32) -> Self {
        assert!(sample_rate_hz > 0, "grid rate must be positive");
        Self {
            sample_rate_hz: sample_rate_hz as u64,
            acceptance_us: (3 * 1_000_000) / (4 * sample_rate_hz as u64),
            anchor_us: None,
            next_tick: 0,
            ticks_emitted: 0,
            ticks_skipped: 0,
            consecutive_empty: 0,
            sources: core::array::from_fn(|_| SourceState::new()),
        }
    }

    /// Heap bytes reserved by the bounded per-source queues.
    pub fn reserved_queue_bytes(&self) -> usize {
        self.sources
            .iter()
            .map(|source| source.queue.capacity() * core::mem::size_of::<QueuedFrame<T>>())
            .sum()
    }

    /// Announce or retract a source. Sources start absent; an absent source never
    /// blocks the grid and its slots read `None`. Retracting clears the source's
    /// queue: what it produces after coming back describes a different epoch, and
    /// pairing across the boundary would manufacture simultaneity that never
    /// happened.
    pub fn set_present(&mut self, source: usize, present: bool) {
        let state = &mut self.sources[source];
        if state.present && !present {
            state.queue.clear();
        }
        state.present = present;
    }

    /// Offer one timestamped frame from `source`. Timestamps must be strictly
    /// monotonic per source and the source must be present; violations are counted
    /// as `rejected` and ignored rather than corrupting the grid.
    pub fn push(&mut self, source: usize, at_us: u64, payload: T) {
        let state = &mut self.sources[source];
        let monotonic = state.newest_accepted_us.is_none_or(|newest| at_us > newest);
        if !state.present || !monotonic {
            state.counters.rejected += 1;
            return;
        }
        state.newest_accepted_us = Some(at_us);
        if state.queue.len() >= MAX_QUEUED_FRAMES {
            state.queue.pop_front();
            state.counters.overflowed += 1;
        }
        state.queue.push_back(QueuedFrame {
            at_us,
            payload,
            used: false,
        });
        if self.anchor_us.is_none() {
            self.anchor_us = Some(at_us);
        }
    }

    /// The next grid step, if it can already be decided. Call in a loop after each
    /// push until it returns `None`.
    ///
    /// A tick is decidable once every present source either holds a frame past the
    /// tick's far tolerance edge (so nothing nearer can still arrive) or is absent.
    /// A tick no present source can contribute to is emitted with every slot `None`
    /// while the gap is short (up to [`MAX_CONSECUTIVE_EMPTY_TICKS`]) and skipped
    /// once it has gone on longer — see the module docs.
    pub fn poll(&mut self) -> Option<AlignedStep<T, SOURCES>> {
        let anchor_us = self.anchor_us?;
        loop {
            if !self.sources.iter().any(|source| source.present) {
                return None;
            }
            let tick_us = anchor_us + (self.next_tick * 1_000_000) / self.sample_rate_hz;
            if !self
                .sources
                .iter()
                .all(|source| source.decided(tick_us, self.acceptance_us))
            {
                return None;
            }
            let mut slots: [Option<T>; SOURCES] = core::array::from_fn(|_| None);
            let mut any_contribution = false;
            for (index, source) in self.sources.iter_mut().enumerate() {
                if !source.present {
                    continue;
                }
                slots[index] = source.choose(tick_us, self.acceptance_us);
                any_contribution |= slots[index].is_some();
            }
            self.next_tick += 1;
            if !any_contribution {
                // A short all-source gap still emits, all slots `None`, so a
                // downstream window survives it as honest zeros; a sustained one
                // stops emitting and becomes a visible discontinuity instead.
                self.consecutive_empty += 1;
                if self.consecutive_empty > MAX_CONSECUTIVE_EMPTY_TICKS {
                    self.ticks_skipped += 1;
                    continue;
                }
            } else {
                self.consecutive_empty = 0;
            }
            for (index, source) in self.sources.iter_mut().enumerate() {
                if source.present && slots[index].is_none() {
                    source.counters.missing += 1;
                }
            }
            self.ticks_emitted += 1;
            return Some(AlignedStep {
                at_us: tick_us,
                slots,
            });
        }
    }

    /// Cumulative per-source accounting. See [`SourceCounters`].
    pub fn counters(&self, source: usize) -> SourceCounters {
        self.sources[source].counters
    }

    /// Grid ticks emitted since construction.
    pub fn ticks_emitted(&self) -> u64 {
        self.ticks_emitted
    }

    /// Grid ticks passed over because no present source could contribute.
    pub fn ticks_skipped(&self) -> u64 {
        self.ticks_skipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// 2 kHz: the firmware's nominal rate, so the tests measure the real geometry.
    const RATE: u32 = 2000;
    const PERIOD_US: u64 = 500;

    /// Drives `aligner` with one source at `period_us` per frame starting at
    /// `start_us`, collecting whatever the grid emits.
    fn drive_one<const SOURCES: usize>(
        aligner: &mut GridAligner<u32, SOURCES>,
        source: usize,
        start_us: u64,
        period_us: u64,
        frames: u32,
    ) -> Vec<AlignedStep<u32, SOURCES>> {
        let mut out = Vec::new();
        for index in 0..frames {
            aligner.push(source, start_us + index as u64 * period_us, index);
            while let Some(step) = aligner.poll() {
                out.push(step);
            }
        }
        out
    }

    /// Both sources on ideal clocks: every tick pairs both, nothing is dropped,
    /// reused, or missed, and the output is exactly grid-regular.
    fn assert_clean(aligner: &GridAligner<u32, 2>) {
        for source in 0..2 {
            assert_eq!(aligner.counters(source), SourceCounters::default());
        }
    }

    #[test]
    fn ideal_clocks_pair_every_tick() {
        let mut aligner = GridAligner::<u32, 2>::new(RATE);
        aligner.set_present(0, true);
        aligner.set_present(1, true);
        let mut steps = Vec::new();
        for index in 0..1000u32 {
            let t = 1_000 + index as u64 * PERIOD_US;
            aligner.push(0, t, index);
            aligner.push(1, t + 40, index); // fixed 40 µs phase offset, same rate
            while let Some(step) = aligner.poll() {
                steps.push(step);
            }
        }
        assert!(steps.len() >= 998, "{} steps", steps.len());
        for step in &steps {
            assert!(step.slots[0].is_some() && step.slots[1].is_some());
        }
        for pair in steps.windows(2) {
            assert_eq!(pair[1].at_us - pair[0].at_us, PERIOD_US);
        }
        assert_clean(&aligner);
        assert_eq!(aligner.ticks_skipped(), 0);
    }

    /// Drives two sources with the given frame periods off one shared clock,
    /// event-driven: whichever source's next frame is due first is pushed first,
    /// exactly as two chips stamped by one device clock interleave in reality. (An
    /// index-driven loop would let the faster-stamped source's *timeline* run ahead
    /// of the grid by an ever-growing margin, a pattern one shared clock cannot
    /// produce, and the queue-overflow counters rightly flag it.)
    fn drive_shared_clock(
        aligner: &mut GridAligner<u32, 2>,
        periods_us: [u64; 2],
        until_us: u64,
    ) -> Vec<AlignedStep<u32, 2>> {
        let mut steps = Vec::new();
        let mut due_us = [0u64; 2];
        let mut sent = [0u32; 2];
        loop {
            let source = if due_us[0] <= due_us[1] { 0 } else { 1 };
            if due_us[source] > until_us {
                return steps;
            }
            aligner.push(source, due_us[source], sent[source]);
            sent[source] += 1;
            due_us[source] += periods_us[source];
            while let Some(step) = aligner.poll() {
                steps.push(step);
            }
        }
    }

    #[test]
    fn a_fast_source_shows_its_rate_as_surplus_drops() {
        let mut aligner = GridAligner::<u32, 2>::new(RATE);
        aligner.set_present(0, true);
        aligner.set_present(1, true);
        // Source 1 runs 1% fast (495 µs period): over 5 s it must shed ~1% of its
        // frames as surplus, while source 0 stays clean and nothing ever overflows.
        drive_shared_clock(&mut aligner, [500, 495], 5_000_000);
        let counters = aligner.counters(1);
        assert!(
            (80..=120).contains(&counters.surplus_dropped),
            "expected ~100 surplus drops, got {:?}",
            counters
        );
        assert_eq!(counters.missing, 0, "a fast source never gaps");
        assert_eq!(counters.overflowed, 0);
        assert_eq!(aligner.counters(0).surplus_dropped, 0);
    }

    #[test]
    fn a_slow_source_shows_its_rate_as_duplicates() {
        let mut aligner = GridAligner::<u32, 2>::new(RATE);
        aligner.set_present(0, true);
        aligner.set_present(1, true);
        // Source 1 runs 1% slow (505 µs period): ~1% of ticks reuse a frame, and —
        // this is what the ¾-period acceptance window buys — none of them read as
        // gaps, because the nearest real sample is always accepted.
        drive_shared_clock(&mut aligner, [500, 505], 5_000_000);
        let counters = aligner.counters(1);
        assert!(
            (80..=120).contains(&counters.duplicated),
            "expected ~100 duplicates, got {:?}",
            counters
        );
        assert_eq!(counters.missing, 0, "a slightly slow source must not gap");
        assert_eq!(counters.overflowed, 0);
        assert_eq!(aligner.counters(0).duplicated, 0);
    }

    #[test]
    fn an_absent_source_never_blocks_and_reads_none() {
        let mut aligner = GridAligner::<u32, 2>::new(RATE);
        aligner.set_present(0, true);
        let steps = drive_one(&mut aligner, 0, 0, PERIOD_US, 100);
        assert!(steps.len() >= 98, "{} steps", steps.len());
        for step in &steps {
            assert!(step.slots[0].is_some());
            assert!(step.slots[1].is_none());
        }
        // Absent is not "missing": nothing was expected of it.
        assert_eq!(aligner.counters(1), SourceCounters::default());
    }

    #[test]
    fn a_gap_in_one_source_reads_none_and_counts_missing() {
        let mut aligner = GridAligner::<u32, 2>::new(RATE);
        aligner.set_present(0, true);
        aligner.set_present(1, true);
        let mut steps = Vec::new();
        for index in 0..200u32 {
            let t = index as u64 * PERIOD_US;
            aligner.push(0, t, index);
            // Source 1 goes dark for frames 50..80 without being marked absent —
            // upstream discarded its frames (settling) but kept pushing afterwards.
            if !(50..80).contains(&index) {
                aligner.push(1, t + 20, index);
            }
            while let Some(step) = aligner.poll() {
                steps.push(step);
            }
        }
        let gaps = steps.iter().filter(|s| s.slots[1].is_none()).count();
        assert!((28..=31).contains(&gaps), "{gaps} gap steps");
        assert_eq!(aligner.counters(1).missing as usize, gaps);
        // The other source is untouched by its peer's gap.
        assert!(steps.iter().all(|s| s.slots[0].is_some()));
    }

    #[test]
    fn an_all_source_outage_skips_ticks_instead_of_emitting_empties() {
        let mut aligner = GridAligner::<u32, 1>::new(RATE);
        aligner.set_present(0, true);
        for index in 0..100u32 {
            aligner.push(0, index as u64 * PERIOD_US, index);
            while aligner.poll().is_some() {}
        }
        // 50 ms of silence, then the stream resumes 100 periods later.
        let resume_at = 200 * PERIOD_US;
        let mut resumed = Vec::new();
        for index in 0..100u32 {
            aligner.push(0, resume_at + index as u64 * PERIOD_US, index);
            while let Some(step) = aligner.poll() {
                resumed.push(step);
            }
        }
        assert!(resumed.len() >= 2);
        // The emitted timeline jumped across the outage rather than filling it: only
        // the short all-`None` grace run at the gap's start is emitted, the rest is
        // skipped.
        assert!(aligner.ticks_skipped() >= 88, "{}", aligner.ticks_skipped());
        assert_eq!(
            aligner.counters(0).missing,
            MAX_CONSECUTIVE_EMPTY_TICKS as u32,
            "only the grace run counts as gaps; skipped ticks do not"
        );
        // Exactly one discontinuity: the tick left undecided when the stream went
        // quiet emits first, then the timeline jumps to the resumed data and is
        // grid-regular from there.
        let jumps: Vec<u64> = resumed
            .windows(2)
            .map(|pair| pair[1].at_us - pair[0].at_us)
            .filter(|&delta| delta != PERIOD_US)
            .collect();
        assert!(jumps.len() <= 1, "more than one discontinuity: {jumps:?}");
        assert!(
            jumps.iter().all(|&delta| delta > 90 * PERIOD_US),
            "jump did not span the outage: {jumps:?}"
        );
    }

    #[test]
    fn a_brief_all_source_gap_emits_empty_ticks_instead_of_breaking_the_timeline() {
        // The hardware regression this guards: the pipelines occasionally miss a
        // couple of DRDY edges on both chips at once, and skipping the resulting
        // one-or-two tick hole reads downstream as a discontinuity that voids the
        // 250 ms window being built around it. A brief hole must stay inside the
        // emitted timeline as all-`None` steps.
        let mut aligner = GridAligner::<u32, 1>::new(RATE);
        aligner.set_present(0, true);
        let mut steps = Vec::new();
        for index in 0..100u32 {
            // Frames 40 and 41 never arrive: a two-tick hole.
            if index != 40 && index != 41 {
                aligner.push(0, index as u64 * PERIOD_US, index);
            }
            while let Some(step) = aligner.poll() {
                steps.push(step);
            }
        }
        // The timeline is unbroken: every consecutive emitted step is one period
        // apart, straight across the hole.
        for pair in steps.windows(2) {
            assert_eq!(pair[1].at_us - pair[0].at_us, PERIOD_US);
        }
        let empty = steps.iter().filter(|s| s.slots[0].is_none()).count();
        assert!((1..=3).contains(&empty), "{empty} empty steps");
        assert_eq!(aligner.ticks_skipped(), 0);
        assert_eq!(aligner.counters(0).missing as usize, empty);
    }

    #[test]
    fn marking_absent_clears_the_queue_and_resumes_cleanly() {
        let mut aligner = GridAligner::<u32, 2>::new(RATE);
        aligner.set_present(0, true);
        aligner.set_present(1, true);
        for index in 0..50u32 {
            let t = index as u64 * PERIOD_US;
            aligner.push(0, t, index);
            aligner.push(1, t, index);
            while aligner.poll().is_some() {}
        }
        aligner.set_present(1, false);
        let alone = drive_one(&mut aligner, 0, 50 * PERIOD_US, PERIOD_US, 50);
        assert!(alone.iter().all(|s| s.slots[1].is_none()));
        aligner.set_present(1, true);
        // Its clock kept running while it was away.
        let mut paired_again = 0;
        for index in 100..150u32 {
            let t = index as u64 * PERIOD_US;
            aligner.push(0, t, index);
            aligner.push(1, t, index);
            while let Some(step) = aligner.poll() {
                if step.slots[1].is_some() {
                    paired_again += 1;
                }
            }
        }
        assert!(paired_again >= 48, "{paired_again} paired after return");
    }

    #[test]
    fn non_monotonic_and_absent_pushes_are_rejected() {
        let mut aligner = GridAligner::<u32, 2>::new(RATE);
        aligner.push(0, 1_000, 0); // absent: sources start absent
        assert_eq!(aligner.counters(0).rejected, 1);
        aligner.set_present(0, true);
        aligner.push(0, 1_000, 0);
        aligner.push(0, 1_000, 1); // equal timestamp: rejected
        aligner.push(0, 900, 2); // backwards: rejected
        assert_eq!(aligner.counters(0).rejected, 3);
    }

    #[test]
    fn the_first_step_lands_on_the_first_frame() {
        let mut aligner = GridAligner::<u32, 1>::new(RATE);
        aligner.set_present(0, true);
        let steps = drive_one(&mut aligner, 0, 123_456, PERIOD_US, 10);
        assert_eq!(steps.first().map(|s| s.at_us), Some(123_456));
    }

    #[test]
    fn a_stalled_consumer_overflows_the_queue_boundedly() {
        let mut aligner = GridAligner::<u32, 1>::new(RATE);
        aligner.set_present(0, true);
        for index in 0..(MAX_QUEUED_FRAMES as u32 + 100) {
            aligner.push(0, (index as u64 + 1) * PERIOD_US, index);
            // Never polled: the consumer is wedged.
        }
        assert_eq!(aligner.counters(0).overflowed, 100);
    }
}
