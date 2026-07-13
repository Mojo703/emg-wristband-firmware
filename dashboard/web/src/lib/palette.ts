// The named colour pool shared by the backend's display descriptors
// (ClassInfo.color, StateInfo.color, EventFrame.color) and the frontend's light/dark
// themes. The backend (dashboard/src/looks.rs) picks a name; the frontend resolves
// it against the active theme via `theme.color()` in `theme.svelte.ts`. Backend and
// frontend never negotiate a literal CSS colour over the wire, so a theme can change
// independently of the backend and an unrecognised name degrades to `gray` instead
// of breaking.
//
// Values are step 11 ("low-contrast text") from Radix Colors (radix-ui/colors,
// MIT), pulled from its light.ts/dark.ts — the step Radix tunes specifically for
// coloured text/lines against that theme's page background, which is how these are
// used (confidence lines, legend swatches, log levels). Not hand-picked, so a light
// and dark value are never off-hand mixes of different shade levels.
//
// To add a colour: look up the hue's `<name>11` (light.ts) and `<name>Dark11`
// (dark.ts) at https://github.com/radix-ui/colors and add one entry below. No
// other file needs to change.
export type ThemeMode = 'light' | 'dark';

interface ColorPair {
  readonly light: string;
  readonly dark: string;
}

const PALETTE: Record<string, ColorPair> = {
  blue: { light: '#0d74ce', dark: '#70b8ff' },
  green: { light: '#218358', dark: '#3dd68c' },
  amber: { light: '#ab6400', dark: '#ffca16' },
  purple: { light: '#8145b5', dark: '#d19dff' },
  pink: { light: '#c2298a', dark: '#ff8dcc' },
  teal: { light: '#008573', dark: '#0bd8b6' },
  orange: { light: '#cc4e00', dark: '#ffa057' },
  sky: { light: '#00749e', dark: '#75c7f0' },
  gray: { light: '#646464', dark: '#b4b4b4' },
};

const FALLBACK_NAME = 'gray';

/** Resolves a palette name to a CSS colour for the given theme, falling back to
 * `gray` for a missing or unrecognised name. */
export function resolveColor(name: string | null | undefined, mode: ThemeMode): string {
  const pair = (name !== null && name !== undefined ? PALETTE[name] : undefined) ?? PALETTE[FALLBACK_NAME]!;
  return pair[mode];
}
