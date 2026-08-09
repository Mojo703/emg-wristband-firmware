//! Which windows carry a rep's label.
//!
//! Labels are by sample index on the device's own window grid, never by
//! wall-clock time and never by segmenting the signal. The rule is one
//! sentence: the span starts at the first grid boundary at least R after the
//! prompt and covers W whole windows. Everything below is that sentence in
//! integers, kept here on its own because it is the piece a host replay has to
//! reproduce exactly.

/// The device's feature window grid. Window `index` covers samples
/// `[index * hop, index * hop + window)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowGrid {
    window_samples: u32,
    hop_samples: u32,
}

impl WindowGrid {
    pub const fn new(window_samples: u32, hop_samples: u32) -> Self {
        Self {
            window_samples,
            hop_samples,
        }
    }

    pub const fn window_samples(&self) -> u32 {
        self.window_samples
    }

    pub const fn hop_samples(&self) -> u32 {
        self.hop_samples
    }

    /// Where window `index` starts.
    pub const fn window_start(&self, index: u32) -> u64 {
        index as u64 * self.hop_samples as u64
    }

    /// One past the last sample window `index` reads.
    pub const fn window_end(&self, index: u32) -> u64 {
        self.window_start(index) + self.window_samples as u64
    }

    /// The grid index of the window that *ends* at `end_sample`, if one does.
    ///
    /// The feature pipeline reports a window by where it closed, and with a
    /// sliding stride the windows overlap — so "does this window belong to the
    /// span" cannot be an overlap test, which would take in every window that
    /// merely touches it. It is an identity test on the grid, and this is how
    /// a reported window gets its identity.
    pub const fn window_ending_at(&self, end_sample: u64) -> Option<u32> {
        if end_sample < self.window_samples as u64 {
            return None;
        }
        let start = end_sample - self.window_samples as u64;
        if start % self.hop_samples as u64 != 0 {
            return None;
        }
        Some((start / self.hop_samples as u64) as u32)
    }

    /// The first window whose start is at or after `sample`.
    ///
    /// At or after, not after: a prompt whose delay lands exactly on a boundary
    /// uses that boundary. Waiting for the next one would cost a whole window
    /// of the wearer's effort for an off-by-one.
    pub const fn first_boundary_at_or_after(&self, sample: u64) -> u32 {
        let hop = self.hop_samples as u64;
        (sample.div_ceil(hop)) as u32
    }
}

/// The windows one rep's label covers, and the samples they span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabeledSpan {
    pub first_window: u32,
    pub window_count: u32,
    /// First sample any labeled window reads.
    pub first_sample: u64,
    /// One past the last sample any labeled window reads. What the validity
    /// checks and the flush schedule are compared against — a flash write
    /// anywhere inside `[first_sample, end_sample)` invalidates the rep.
    pub end_sample: u64,
}

impl LabeledSpan {
    /// The span a prompt at `prompt_sample` produces: the first grid boundary
    /// at least `delay_samples` later, then `window_count` whole windows.
    ///
    /// The row count is therefore `window_count` regardless of what phase of
    /// the grid the prompt landed on, which is the property that makes one
    /// wearer's reps comparable with another's.
    pub const fn after_prompt(
        grid: WindowGrid,
        prompt_sample: u64,
        delay_samples: u64,
        window_count: u32,
    ) -> Self {
        let first_window = grid.first_boundary_at_or_after(prompt_sample + delay_samples);
        Self {
            first_window,
            window_count,
            first_sample: grid.window_start(first_window),
            end_sample: grid.window_end(first_window + window_count - 1),
        }
    }

    /// Whether every window of the span has been produced by the time the
    /// device has seen `sample`.
    pub const fn is_complete_at(&self, sample: u64) -> bool {
        sample >= self.end_sample
    }

    /// Whether `[start, end)` touches this span at all. The flush schedule and
    /// the ADC's recovery settles are both intervals, and both are rejections
    /// if they land inside a labeled window.
    pub const fn overlaps(&self, start: u64, end: u64) -> bool {
        start < self.end_sample && end > self.first_sample
    }

    /// The grid indices the span covers.
    pub const fn windows(&self) -> core::ops::Range<u32> {
        self.first_window..self.first_window + self.window_count
    }

    /// Whether the window at grid `index` is one of this span's.
    ///
    /// Membership, not overlap. At a quarter stride the windows around a span
    /// overlap it without belonging to it, and labeling those would give one
    /// rep more rows than another depending on nothing the wearer did.
    pub const fn covers_window(&self, index: u32) -> bool {
        index >= self.first_window && index < self.first_window + self.window_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constants;

    fn span_at(prompt_sample: u64) -> LabeledSpan {
        let constants = Constants::DEFAULT;
        LabeledSpan::after_prompt(
            constants.grid(),
            prompt_sample,
            constants.prompt_delay_samples(),
            constants.labeled_windows,
        )
    }

    #[test]
    fn a_span_starts_at_the_first_boundary_past_the_delay() {
        // Prompt on a boundary: R is 500 samples, the grid steps 125, so the
        // delay lands exactly on window 4 and window 4 is what is used.
        let span = span_at(0);
        assert_eq!(span.first_window, 4);
        assert_eq!(span.first_sample, 500);

        // One sample later, the delay lands one past the boundary and the span
        // waits for the next one.
        let span = span_at(1);
        assert_eq!(span.first_window, 5);
        assert_eq!(span.first_sample, 625);
    }

    #[test]
    fn a_rep_is_nine_overlapping_windows_over_1500_samples() {
        // The shape V validated: nine windows at a quarter stride, so the last
        // one starts 1000 samples after the first and ends 1500 past it. With
        // R that is a second from the prompt to the end of the span, which is
        // the hold a wearer is actually asked for.
        let constants = Constants::DEFAULT;
        let span = span_at(0);
        assert_eq!(span.window_count, 9);
        assert_eq!(span.end_sample - span.first_sample, 1500);
        assert_eq!(
            span.end_sample,
            constants.prompt_delay_samples() + 1500,
            "a second of signal from the prompt"
        );
        // Overlapping, not disjoint: nine windows of 500 samples each drawn
        // from 1500 samples of signal is the point of the sliding stride.
        assert!(span.window_count as u64 * constants.window_samples as u64 > 1500);
    }

    #[test]
    fn every_span_is_the_same_length_whatever_phase_it_started_on() {
        // The whole point of grid alignment: two wearers prompted a quarter of
        // a window apart contribute the same number of rows, and neither
        // contributes a partial window.
        let constants = Constants::DEFAULT;
        for offset in 0..constants.hop_samples as u64 {
            let span = span_at(10_000 + offset);
            assert_eq!(span.window_count, constants.labeled_windows);
            assert_eq!(
                span.end_sample - span.first_sample,
                // Four non-overlapping 500-sample windows.
                (constants.labeled_windows as u64 - 1) * constants.hop_samples as u64
                    + constants.window_samples as u64
            );
            assert_eq!(span.first_sample % constants.hop_samples as u64, 0);
            assert_eq!(span.windows().count(), constants.labeled_windows as usize);
        }
    }

    #[test]
    fn a_span_never_starts_before_the_delay_has_elapsed() {
        // R exists so the wearer has time to move; a span that started early
        // would label the ramp into the gesture as the gesture.
        let constants = Constants::DEFAULT;
        for offset in 0..2 * constants.hop_samples as u64 {
            let prompt = 7_000 + offset;
            let span = span_at(prompt);
            assert!(span.first_sample >= prompt + constants.prompt_delay_samples());
            // And never more than one window late.
            assert!(
                span.first_sample
                    < prompt + constants.prompt_delay_samples() + constants.hop_samples as u64
            );
        }
    }

    #[test]
    fn completeness_needs_the_last_window_whole() {
        let span = span_at(0);
        assert!(!span.is_complete_at(span.end_sample - 1));
        assert!(span.is_complete_at(span.end_sample));
    }

    #[test]
    fn a_window_belongs_to_a_span_by_identity_not_by_touching_it() {
        // The distinction the sliding stride forces. Windows on either side of
        // the span overlap it — they share 375 of their 500 samples with it —
        // and must not be labeled, or two reps would contribute different row
        // counts for reasons the wearer had no part in.
        let constants = Constants::DEFAULT;
        let grid = constants.grid();
        let span = span_at(0);

        for index in span.windows() {
            let end = grid.window_end(index);
            assert_eq!(grid.window_ending_at(end), Some(index));
            assert!(span.covers_window(index));
        }
        let before = span.first_window - 1;
        let after = span.first_window + span.window_count;
        assert!(!span.covers_window(before));
        assert!(!span.covers_window(after));
        // Both of them do overlap it, which is exactly why overlap is the
        // wrong test.
        assert!(span.overlaps(grid.window_start(before), grid.window_end(before)));
        assert!(span.overlaps(grid.window_start(after), grid.window_end(after)));

        // A window that did not close on a boundary has no identity at all.
        assert_eq!(grid.window_ending_at(grid.window_end(4) + 1), None);
        assert_eq!(grid.window_ending_at(0), None);
    }

    #[test]
    fn overlap_is_half_open_at_both_ends() {
        // A flush that ends exactly where the span begins does not touch it,
        // which is what makes "flush strictly between rounds" a schedulable
        // rule rather than an approximate one.
        let span = span_at(0);
        assert!(!span.overlaps(0, span.first_sample));
        assert!(span.overlaps(span.first_sample, span.first_sample + 1));
        assert!(span.overlaps(span.end_sample - 1, span.end_sample + 100));
        assert!(!span.overlaps(span.end_sample, span.end_sample + 100));
    }

    #[test]
    fn an_overlapping_grid_still_aligns() {
        // The device's grid does not overlap, but the arithmetic is about hop
        // boundaries rather than window edges, and a swept hop must not
        // silently change which boundaries exist.
        let grid = WindowGrid::new(500, 250);
        let span = LabeledSpan::after_prompt(grid, 0, 1000, 4);
        assert_eq!(span.first_window, 4);
        assert_eq!(span.first_sample, 1000);
        // Four windows at a 250 hop reach 750 past the first start, plus the
        // window's own 500.
        assert_eq!(span.end_sample, 1000 + 750 + 500);
    }
}
