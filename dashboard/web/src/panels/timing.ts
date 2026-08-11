import type {
  CalibrationTimingColor,
  CalibrationTimingState,
  CalibrationTimingStatusFrame,
  GuidedMode,
} from '../lib/protocol';

export interface TimingAvailability {
  readonly selectedDeviceConnected: boolean;
  readonly guidedMode: GuidedMode | null;
}

/** Timing is available only for the exact selected live link and an idle
 * guided coordinator. Measurement quality is deliberately not part of this
 * predicate: a noisy window warns the operator but never hides Start. */
export function timingAvailable(value: TimingAvailability): boolean {
  return value.selectedDeviceConnected && value.guidedMode === null;
}

export interface TimingDisplayProjection {
  readonly state: CalibrationTimingState | 'unknown';
  readonly running: boolean;
  readonly color: CalibrationTimingColor | null;
}

/** Pure rendering projection: no click or timer mutates this state. */
export function timingDisplayProjection(
  status: CalibrationTimingStatusFrame['status'] | null,
): TimingDisplayProjection {
  return {
    state: status?.phase.state ?? 'unknown',
    running: status?.phase.state === 'running',
    // No device acknowledgement means no colour claim. Stopping retains the
    // last observed device colour until the Stopped acknowledgement arrives.
    color:
      status?.phase.state === 'running'
        ? status.phase.observation.color
        : status?.phase.state === 'stopping'
          ? status.phase.last_observation.color
          : status?.phase.state === 'error'
            ? status.phase.last_observation?.color ?? null
            : null,
  };
}

export function timingQualityWarning(
  status: CalibrationTimingStatusFrame['status'] | null,
): string | null {
  if (status === null) return null;
  if (status.estimate.availability === 'no_samples') return null;
  const rtt = status.estimate.median_round_trip_milliseconds;
  const spread = status.estimate.round_trip_spread_milliseconds;
  if (rtt > 50 || spread > 50) {
    return 'Probe timing is noisy. Correction remains available, but repeat the window if the wearer sees drift.';
  }
  return null;
}
