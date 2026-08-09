//! The quality gate: a self-test that may extend collection and report, and
//! may never shorten it.
//!
//! Log 0022 measured the instrument this approximates. Under leave-two-cues-out
//! folds inside a calibration set it names the weakest class pair reliably —
//! pronation appears in every session's weakest pair, paired with supination in
//! four of five — but its rank correlation with the session outcome flips
//! between six and ten cues per class on five sessions.
//!
//! So it names, and that is all it does. It used to be able to extend
//! collection for a weak class, on the reasoning that more data cannot hurt.
//! V measured otherwise: extension rounds break misclassification
//! monotonically — two extra rounds give 3/50 against golden's exact zero, and
//! it does not matter which classes are extended. The schedule ships tuned to
//! its exact floors, so a mechanism that changed the round count could only
//! move it off them. The report is the whole product.
//!
//! The device's version is leave-*recent*-cues-out rather than the host's full
//! fold sweep: the most recent reps of each class are scored against the
//! checkpoint fitted at the end of the previous round, which has not seen them.
//! That is one held-out fold instead of all of them, and it costs no refit at
//! all — the alternative on this chip is several seconds of arithmetic per
//! round to sharpen a number that only ever adds rounds.

use protocol::{CalibrationGesture, ClassPair, GateStatus};

const CLASS_COUNT: usize = CalibrationGesture::ALL.len();

/// One class's standing in the self-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassScore {
    pub gesture: CalibrationGesture,
    /// Held-out windows scored, and how many the model recovered.
    pub scored: u32,
    pub correct: u32,
    pub status: GateStatus,
}

impl ClassScore {
    /// Window accuracy in permille, or `None` before anything was scored.
    pub fn accuracy_permille(&self) -> Option<u32> {
        (self.scored > 0).then(|| self.correct * 1000 / self.scored)
    }
}

/// What the gate makes of the round that just finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateVerdict {
    pub classes: [ClassScore; CLASS_COUNT],
    /// The two classes the self-test confused most, in prompt order. What a
    /// wearer can act on; the accuracy figure on its own is not.
    pub weak_pair: Option<ClassPair>,
}

/// Held-out windows accumulated over one round.
#[derive(Debug, Clone)]
pub struct QualityGate {
    holding_accuracy_permille: u32,
    /// `confusion[truth][predicted]`. The diagonal is the correct count.
    confusion: [[u32; CLASS_COUNT]; CLASS_COUNT],
    /// Held-out windows the reject pipeline let through as nothing at all.
    /// Counted against the class but not as confusion with another one: a
    /// window that committed nothing is a false negative, not a mix-up.
    no_command: [u32; CLASS_COUNT],
}

impl QualityGate {
    pub fn new(holding_accuracy_permille: u32) -> Self {
        Self {
            holding_accuracy_permille,
            confusion: [[0; CLASS_COUNT]; CLASS_COUNT],
            no_command: [0; CLASS_COUNT],
        }
    }

    /// One held-out window's outcome. `predicted` is `None` when the window
    /// produced no command.
    pub fn record_window(
        &mut self,
        truth: CalibrationGesture,
        predicted: Option<CalibrationGesture>,
    ) {
        let truth_index = truth.index() as usize;
        match predicted {
            Some(predicted) => self.confusion[truth_index][predicted.index() as usize] += 1,
            None => self.no_command[truth_index] += 1,
        }
    }

    /// Forget everything scored so far. Called between rounds: the gate reads
    /// the round that just happened, not the whole run, because the point of it
    /// is whether the reps coming in *now* are separable.
    pub fn clear(&mut self) {
        self.confusion = [[0; CLASS_COUNT]; CLASS_COUNT];
        self.no_command = [0; CLASS_COUNT];
    }

    pub fn verdict(&self) -> GateVerdict {
        let classes = core::array::from_fn(|index| {
            let gesture = CalibrationGesture::ALL[index];
            let correct = self.confusion[index][index];
            let scored: u32 = self.confusion[index].iter().sum::<u32>() + self.no_command[index];
            let status = match scored {
                0 => GateStatus::Unknown,
                _ if correct * 1000 / scored >= self.holding_accuracy_permille => {
                    GateStatus::Holding
                }
                _ => GateStatus::Weak,
            };
            ClassScore {
                gesture,
                scored,
                correct,
                status,
            }
        });
        GateVerdict {
            classes,
            weak_pair: self.weak_pair(),
        }
    }

    /// The unordered pair with the most confusion between them, counted both
    /// ways. Unordered because the wearer's problem is that two gestures look
    /// alike, which is symmetric even when the confusion counts are not.
    fn weak_pair(&self) -> Option<ClassPair> {
        let mut worst: Option<(u32, usize, usize)> = None;
        for first in 0..CLASS_COUNT {
            for second in (first + 1)..CLASS_COUNT {
                let between = self.confusion[first][second] + self.confusion[second][first];
                if between == 0 {
                    continue;
                }
                // Strictly greater, so the earliest pair in prompt order wins a
                // tie and the reported pair does not wander between rounds that
                // scored identically.
                if worst.is_none_or(|(count, _, _)| between > count) {
                    worst = Some((between, first, second));
                }
            }
        }
        worst.map(|(_, first, second)| ClassPair {
            first: CalibrationGesture::ALL[first],
            second: CalibrationGesture::ALL[second],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constants;
    use CalibrationGesture::*;

    fn gate() -> QualityGate {
        QualityGate::new(Constants::DEFAULT.holding_accuracy_permille)
    }

    fn score(
        gate: &mut QualityGate,
        truth: CalibrationGesture,
        predicted: CalibrationGesture,
        times: u32,
    ) {
        for _ in 0..times {
            gate.record_window(truth, Some(predicted));
        }
    }

    #[test]
    fn a_class_with_nothing_scored_is_unknown_not_weak() {
        // The difference matters: unknown must not extend collection, or the
        // first round of every run would add rounds for classes nobody has
        // scored yet.
        let verdict = gate().verdict();
        assert!(verdict
            .classes
            .iter()
            .all(|class| class.status == GateStatus::Unknown));
        assert_eq!(verdict.weak_pair, None);
    }

    #[test]
    fn a_clean_round_leaves_every_class_holding() {
        let mut gate = gate();
        for gesture in CalibrationGesture::ALL {
            score(&mut gate, gesture, gesture, 8);
        }
        let verdict = gate.verdict();
        assert!(verdict
            .classes
            .iter()
            .all(|class| class.status == GateStatus::Holding));
    }

    #[test]
    fn a_class_below_the_threshold_is_named_weak() {
        let mut gate = gate();
        for gesture in CalibrationGesture::ALL {
            score(&mut gate, gesture, gesture, 8);
        }
        // Pronation recovers five of eight, which is 625 permille against a
        // 700 threshold.
        gate.clear();
        for gesture in CalibrationGesture::ALL {
            score(&mut gate, gesture, gesture, 8);
        }
        score(&mut gate, WristPronation, WristSupination, 3);
        let verdict = gate.verdict();
        let pronation = verdict.classes[WristPronation.index() as usize];
        assert_eq!(pronation.scored, 11);
        assert_eq!(pronation.accuracy_permille(), Some(727));
        assert_eq!(pronation.status, GateStatus::Holding);

        // Three more and it drops under.
        score(&mut gate, WristPronation, WristSupination, 3);
        let verdict = gate.verdict();
        let pronation = verdict.classes[WristPronation.index() as usize];
        assert_eq!(pronation.accuracy_permille(), Some(571));
        assert_eq!(pronation.status, GateStatus::Weak);
    }

    #[test]
    fn windows_that_committed_nothing_count_against_the_class() {
        // A held-out rep the pipeline rejected is a false negative, which is
        // the number this work is judged on. Silently not scoring it would let
        // a class that never fires look perfect.
        let mut gate = gate();
        score(&mut gate, ThumbExtension, ThumbExtension, 5);
        for _ in 0..5 {
            gate.record_window(ThumbExtension, None);
        }
        let thumb = gate.verdict().classes[ThumbExtension.index() as usize];
        assert_eq!(thumb.scored, 10);
        assert_eq!(thumb.correct, 5);
        assert_eq!(thumb.status, GateStatus::Weak);
        // But it is not confusion with anything, so it names no pair.
        assert_eq!(gate.verdict().weak_pair, None);
    }

    #[test]
    fn the_weak_pair_is_unordered_and_counted_both_ways() {
        let mut gate = gate();
        // Radial and ulnar trade four windows between them; pronation loses
        // three to supination one way only.
        score(&mut gate, WristRadialDeviation, WristUlnarDeviation, 2);
        score(&mut gate, WristUlnarDeviation, WristRadialDeviation, 2);
        score(&mut gate, WristPronation, WristSupination, 3);
        assert_eq!(
            gate.verdict().weak_pair,
            Some(ClassPair {
                first: WristRadialDeviation,
                second: WristUlnarDeviation
            })
        );
    }

    #[test]
    fn a_tie_names_the_earlier_pair_so_the_report_does_not_wander() {
        let mut gate = gate();
        score(&mut gate, WristPronation, WristSupination, 2);
        score(&mut gate, WristRadialDeviation, WristUlnarDeviation, 2);
        assert_eq!(
            gate.verdict().weak_pair,
            Some(ClassPair {
                first: WristPronation,
                second: WristSupination
            })
        );
    }

    #[test]
    fn clearing_forgets_the_previous_round() {
        let mut gate = gate();
        score(&mut gate, WristPronation, WristSupination, 10);
        assert_eq!(
            gate.verdict().classes[WristPronation.index() as usize].status,
            GateStatus::Weak
        );
        gate.clear();
        assert!(gate
            .verdict()
            .classes
            .iter()
            .all(|class| class.status == GateStatus::Unknown));
        assert_eq!(gate.verdict().weak_pair, None);
    }
}
