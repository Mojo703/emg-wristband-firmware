import type {
  GuidedSessionAction,
  GuidedSessionIntentFrame,
  GuidedSessionSnapshot,
} from './protocol';

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
    expected_session_id: snapshot.active?.session_id ?? null,
    action,
  };
}
