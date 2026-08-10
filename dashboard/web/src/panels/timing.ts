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
    state: status?.state ?? 'unknown',
    running: status?.state === 'running',
    // No device acknowledgement means no colour claim. Stopping retains the
    // last observed device colour until the Stopped acknowledgement arrives.
    color: status?.state === 'running' || status?.state === 'stopping' ? status.color : null,
  };
}

export function timingQualityWarning(
  status: CalibrationTimingStatusFrame['status'] | null,
): string | null {
  if (status === null) return null;
  const rtt = status.median_round_trip_milliseconds;
  const spread = status.round_trip_spread_milliseconds;
  if ((rtt !== null && rtt > 50) || (spread !== null && spread > 50)) {
    return 'Probe timing is noisy; correction remains available, but repeat the window if the wearer sees drift.';
  }
  return null;
}
