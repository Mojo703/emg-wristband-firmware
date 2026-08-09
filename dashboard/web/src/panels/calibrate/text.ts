// The calibration vocabulary as a person reads it. The wire carries named enums
// (protocol.ts), so this file is a display table and nothing else — no code
// elsewhere branches on these strings.

import {
  CalibrationGesture,
  CalibrationOutcome,
  CalibrationPhase,
  GateStatus,
  RepRejection,
} from '../../lib/protocol';

const GESTURE_LABELS: Record<CalibrationGesture, string> = {
  [CalibrationGesture.WristPronation]: 'Pronation',
  [CalibrationGesture.WristSupination]: 'Supination',
  [CalibrationGesture.WristRadialDeviation]: 'Radial deviation',
  [CalibrationGesture.WristUlnarDeviation]: 'Ulnar deviation',
  [CalibrationGesture.ThumbExtension]: 'Thumb extension',
};

export function gestureLabel(gesture: CalibrationGesture): string {
  return GESTURE_LABELS[gesture];
}

/** The phases in the order the device walks them, for the progress strip.
 * `idle` and `stopped` are not steps: one is before the walk and the other is a
 * terminal state the strip cannot place. */
export const PHASE_STEPS: readonly CalibrationPhase[] = [
  CalibrationPhase.Settling,
  CalibrationPhase.ThumbUpRounds,
  CalibrationPhase.Handover,
  CalibrationPhase.ThumbDownRounds,
  CalibrationPhase.Polish,
  CalibrationPhase.Install,
  CalibrationPhase.Complete,
];

const PHASE_LABELS: Record<CalibrationPhase, string> = {
  [CalibrationPhase.Idle]: 'Idle',
  [CalibrationPhase.Settling]: 'Settling',
  [CalibrationPhase.ThumbUpRounds]: 'Thumb up',
  [CalibrationPhase.Handover]: 'Handover',
  [CalibrationPhase.ThumbDownRounds]: 'Thumb down',
  [CalibrationPhase.Polish]: 'Polish',
  [CalibrationPhase.Install]: 'Install',
  [CalibrationPhase.Complete]: 'Done',
  [CalibrationPhase.Stopped]: 'Stopped',
};

export function phaseLabel(phase: CalibrationPhase): string {
  return PHASE_LABELS[phase];
}

/** What the wearer is doing in each phase, shown under the prompt. */
const PHASE_INSTRUCTIONS: Record<CalibrationPhase, string> = {
  [CalibrationPhase.Idle]: 'No run in progress.',
  [CalibrationPhase.Settling]: 'Hold still. The device is erasing its slot and measuring gains.',
  [CalibrationPhase.ThumbUpRounds]: 'Pole in hand, thumb extended clear of the grip.',
  [CalibrationPhase.Handover]: 'Change the pole to the other hand.',
  [CalibrationPhase.ThumbDownRounds]: 'Pole in hand, thumb gripping.',
  [CalibrationPhase.Polish]: 'Collection is done. The device is finishing the fit.',
  [CalibrationPhase.Install]: 'Writing the model.',
  [CalibrationPhase.Complete]: 'Calibration installed.',
  [CalibrationPhase.Stopped]: 'The run ended early.',
};

export function phaseInstruction(phase: CalibrationPhase): string {
  return PHASE_INSTRUCTIONS[phase];
}

/** The gate's three states in the plan's words. `holding` is a class the gate
 * is satisfied with; `weak` is one it is extending collection for, which is the
 * only thing the gate is allowed to do. */
const GATE_LABELS: Record<GateStatus, string> = {
  [GateStatus.Unknown]: 'collecting',
  [GateStatus.Holding]: 'passed',
  [GateStatus.Weak]: 'extended',
};

export function gateLabel(gate: GateStatus): string {
  return GATE_LABELS[gate];
}

const REJECTION_REASONS: Record<RepRejection, string> = {
  [RepRejection.AtRestBaseline]: 'no gesture detected — the span sat at the rest baseline',
  [RepRejection.LeadOffChannels]: 'an electrode read lead-off across the span',
  [RepRejection.AdcRecoverySettle]: 'the span overlapped a converter recovery settle',
  [RepRejection.FlashOperationOverlap]: 'the span overlapped a flash write',
  [RepRejection.MissingSamples]: 'the acquisition path never produced those windows',
};

export function rejectionReason(rejection: RepRejection): string {
  return REJECTION_REASONS[rejection];
}

const OUTCOME_LABELS: Record<CalibrationOutcome, string> = {
  [CalibrationOutcome.Installed]: 'Installed',
  [CalibrationOutcome.Aborted]: 'Aborted',
  [CalibrationOutcome.FrontEndLost]: 'Front end lost',
  [CalibrationOutcome.GestureFailed]: 'Gesture failed',
  [CalibrationOutcome.StorageFailed]: 'Storage failed',
  [CalibrationOutcome.FitFailed]: 'Fit failed',
};

export function outcomeLabel(outcome: CalibrationOutcome): string {
  return OUTCOME_LABELS[outcome];
}

/** What ended the run, in a sentence, for the terminal card. */
const OUTCOME_DETAILS: Record<CalibrationOutcome, string> = {
  [CalibrationOutcome.Installed]: 'The fitted model is installed and in use.',
  [CalibrationOutcome.Aborted]: 'Stopped from this panel.',
  [CalibrationOutcome.FrontEndLost]: 'The analog front end failed or stalled mid-run.',
  [CalibrationOutcome.GestureFailed]: 'A gesture was re-prompted past its budget without a valid rep.',
  [CalibrationOutcome.StorageFailed]: 'A slot erase, append, or commit failed.',
  [CalibrationOutcome.FitFailed]: 'The fit did not finish.',
};

export function outcomeDetail(outcome: CalibrationOutcome): string {
  return OUTCOME_DETAILS[outcome];
}

/** Permille as a percentage, which is how the four numbers are read. */
export function formatPermille(permille: number): string {
  return `${(permille / 10).toFixed(1)}%`;
}

/** Milliseconds as seconds for anything over a second, so a pass time and a
 * run length read on the same scale. */
export function formatMilliseconds(milliseconds: number): string {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

/** A prior-image hash, as the hex a host tool prints it in. */
export function priorHashText(hash: number): string {
  return `0x${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

/** The row precisions `calibration_rows_dump` reports by number. */
const PRECISION_NAMES = ['float32', 'float16', 'int8'] as const;

export function precisionName(precision: number): string {
  return PRECISION_NAMES[precision] ?? `unknown (${precision})`;
}
