// Tracks the dashboard's colour theme: a user choice of 'system' | 'light' | 'dark',
// persisted in localStorage, resolved against the OS preference when 'system'. The
// resolved theme is stamped onto <html data-theme="…"> so app.css's per-theme
// variable blocks apply (and so Tailwind's `dark:` variant, repointed at that
// attribute in app.css, tracks the same choice as everything else). Canvas/WebGL
// panels that can't read CSS variables call `theme.color(name)` to resolve a
// backend palette name (see `palette.ts`) against the current theme.
import { resolveColor } from './palette';

export type ThemeChoice = 'system' | 'light' | 'dark';
export type ThemeMode = 'light' | 'dark';

const STORAGE_KEY = 'opal-theme';
const media = typeof window !== 'undefined' ? window.matchMedia('(prefers-color-scheme: dark)') : null;

function loadChoice(): ThemeChoice {
  const stored = typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) : null;
  return stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system';
}

class ThemeManager {
  choice = $state<ThemeChoice>(loadChoice());
  #systemDark = $state(media?.matches ?? true);

  effective = $derived<ThemeMode>(
    this.choice === 'system' ? (this.#systemDark ? 'dark' : 'light') : this.choice,
  );

  constructor() {
    this.#syncAttribute();
    media?.addEventListener('change', (e) => {
      this.#systemDark = e.matches;
      this.#syncAttribute();
    });
  }

  set(choice: ThemeChoice): void {
    this.choice = choice;
    window.localStorage.setItem(STORAGE_KEY, choice);
    this.#syncAttribute();
  }

  /** Resolves a backend palette name (ClassInfo/StateInfo/EventFrame `color`) against
   * the active theme. */
  color(name: string | null | undefined): string {
    return resolveColor(name, this.effective);
  }

  #syncAttribute(): void {
    document.documentElement.setAttribute('data-theme', this.effective);
  }
}

export const theme = new ThemeManager();
