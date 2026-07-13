// The named colour pool shared by the backend's display descriptors
// (ClassInfo.color, StateInfo.color, EventFrame.color) and the frontend's light/dark
// themes. The backend (dashboard/src/looks.rs) picks a name; the frontend resolves
// it against the active theme via `theme.color()` in `theme.svelte.ts`. Backend and
// frontend never negotiate a literal CSS colour over the wire, so a theme can change
// independently of the backend and an unrecognised name degrades to `gray` instead
// of breaking.
//
// To add a colour: add one entry below. No other file needs to change.
export type ThemeMode = 'light' | 'dark';

interface ColorPair {
  readonly light: string;
  readonly dark: string;
}

const PALETTE: Record<string, ColorPair> = {
  blue: { dark: '#3b82f6', light: '#2563eb' },
  green: { dark: '#22c55e', light: '#16a34a' },
  amber: { dark: '#f59e0b', light: '#b45309' },
  purple: { dark: '#a855f7', light: '#9333ea' },
  pink: { dark: '#ec4899', light: '#db2777' },
  teal: { dark: '#14b8a6', light: '#0f766e' },
  orange: { dark: '#f97316', light: '#c2410c' },
  sky: { dark: '#60a5fa', light: '#2563eb' },
  gray: { dark: '#6b7280', light: '#57606a' },
};

const FALLBACK_NAME = 'gray';

/** Resolves a palette name to a CSS colour for the given theme, falling back to
 * `gray` for a missing or unrecognised name. */
export function resolveColor(name: string | null | undefined, mode: ThemeMode): string {
  const pair = (name !== null && name !== undefined ? PALETTE[name] : undefined) ?? PALETTE[FALLBACK_NAME]!;
  return pair[mode];
}
