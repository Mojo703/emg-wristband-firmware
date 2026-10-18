// Hand-kept mirror of the five classes in `dashboard/config/collection.json`
// (ids and labels only; the dashboard-only display fields like color and
// motion arrows live there, not here). Keep this list in sync by hand, same
// as `protocol.ts` mirrors `ble-media/src/link_bench.rs`.
//
// There is no BLE characteristic on the band yet for writing a gesture
// keymap: the real firmware (`opal-firmware`, built on `ble-media`) only
// exposes BLE HID output today. `applyKeymap` below is a stub that drafts
// the mapping locally so the UI is ready; wiring in a real write is a
// one-function change once that characteristic exists.

import type { MediaKey } from './protocol.ts';

export interface GestureClass {
  readonly id: string;
  readonly label: string;
}

export const GESTURE_CLASSES: readonly GestureClass[] = [
  { id: 'wrist_pronation', label: 'Tip out' },
  { id: 'wrist_supination', label: 'Tip in' },
  { id: 'wrist_radial_deviation', label: 'Tip forward' },
  { id: 'wrist_ulnar_deviation', label: 'Tip back' },
  { id: 'thumb_extension', label: 'Tip center' },
];

const GESTURE_IDS = new Set(GESTURE_CLASSES.map((gesture) => gesture.id));

export function isKnownGesture(id: string): boolean {
  return GESTURE_IDS.has(id.trim());
}

export interface Binding {
  readonly gesture: string;
  readonly key: MediaKey['id'];
}

const STORAGE_KEY = 'opal-gesture-keymap-draft';

export function loadDraft(): Record<string, string> {
  if (typeof localStorage === 'undefined') return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return {};
    const result: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (typeof value === 'string') result[key] = value;
    }
    return result;
  } catch {
    return {};
  }
}

function saveDraft(draft: Record<string, string>): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
}

export function buildBindings(draft: Record<string, string>): Binding[] {
  const bindings: Binding[] = [];
  for (const [key, gesture] of Object.entries(draft)) {
    const trimmed = gesture.trim();
    if (trimmed.length === 0 || !isKnownGesture(trimmed)) continue;
    bindings.push({ gesture: trimmed, key: key as MediaKey['id'] });
  }
  return bindings;
}

/**
 * Stub for the future BLE write. Persists the draft locally and reports
 * that nothing was sent, since the band has no keymap characteristic yet.
 */
export function applyKeymap(draft: Record<string, string>, note: (message: string) => void): Binding[] {
  saveDraft(draft);
  const bindings = buildBindings(draft);
  note(
    bindings.length > 0
      ? `saved ${bindings.length} binding${bindings.length === 1 ? '' : 's'} locally — not sent: the band has no BLE keymap characteristic yet`
      : 'saved an empty keymap locally — not sent: the band has no BLE keymap characteristic yet',
  );
  return bindings;
}
