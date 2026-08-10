import type {
  GuidedCalibrationSnapshot,
  GuidedMode,
  GuidedSessionAction,
  GuidedSessionIntentFrame,
  GuidedSessionSnapshot,
} from './protocol';

export interface ActiveGuidedSession {
  readonly session_id: number;
  readonly mode: GuidedMode;
  readonly device_id: string | null;
}

export function activeGuidedSession(snapshot: GuidedSessionSnapshot): ActiveGuidedSession | null {
  const lifecycle = snapshot.lifecycle;
  if (lifecycle.state === 'collection') return { ...lifecycle, mode: 'collection' };
  if (lifecycle.state === 'calibration') return { ...lifecycle, mode: 'calibration' };
  return null;
}

export function guidedCalibrationSnapshot(
  snapshot: GuidedSessionSnapshot,
): GuidedCalibrationSnapshot | null {
  const lifecycle = snapshot.lifecycle;
  return lifecycle.state === 'idle' ||
    lifecycle.state === 'calibration' ||
    lifecycle.state === 'calibration_failed'
    ? lifecycle.calibration
    : null;
}

/** Accept only a strictly newer complete projection on one WebSocket. A new
 * socket resets the current value, allowing a restarted backend to begin again
 * at revision zero. */
export function acceptGuidedSnapshot(
  current: GuidedSessionSnapshot | null,
  incoming: GuidedSessionSnapshot,
): GuidedSessionSnapshot {
  if (current === null) return incoming;
  if (incoming.revision <= current.revision) return current;
  if (incoming.run_revision < current.run_revision) return current;
  return incoming;
}

export function guidedIntent(
  snapshot: GuidedSessionSnapshot,
  action: GuidedSessionAction,
): GuidedSessionIntentFrame {
  return {
    type: 'guided_session_intent',
    expected_revision: snapshot.revision,
    expected_run_revision: snapshot.run_revision,
    expected_session_id: activeGuidedSession(snapshot)?.session_id ?? null,
    action,
  };
}
