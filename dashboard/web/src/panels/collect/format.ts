// Display formatters shared by the collection setup form and the session summary.
// Every function takes a plain number (branded protocol milliseconds are assignable
// to it) and returns a string meant for a human, never for the wire.

/** `m:ss` — the session length shown in the summary header. */
export function formatMinutesSeconds(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.round(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/** Whole minutes, rounded — the number behind the track-length and estimate lines. */
export function formatWholeMinutes(milliseconds: number): number {
  return Math.max(0, Math.round(milliseconds / 60000));
}

/** Local clock time as `HH:MM`, for the donned stamp and the photo confirmation. */
export function formatClockTime(unixMilliseconds: number): string {
  return new Date(unixMilliseconds).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

/** How long ago an instant was, in the coarse wording the setup form wants. */
export function formatMinutesAgo(unixMilliseconds: number, now: number): string {
  const minutes = Math.floor(Math.max(0, now - unixMilliseconds) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes === 1) return '1 min ago';
  return `${minutes} min ago`;
}

/** Byte counts as a human-readable size; binary units, one decimal above kibibytes. */
export function formatByteSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'] as const;
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const unit = units[unitIndex] ?? 'KiB';
  return `${value.toFixed(value < 10 ? 1 : 0)} ${unit}`;
}

/** Integer millimetres shown as centimetres, the unit the operator measures in. */
export function formatMillimetresAsCentimetres(millimetres: number): string {
  return `${(millimetres / 10).toFixed(1)} cm`;
}

/** A signed offset in milliseconds, always carrying its sign. */
export function formatSignedMilliseconds(milliseconds: number): string {
  if (!Number.isFinite(milliseconds)) return '—';
  const sign = milliseconds < 0 ? '−' : '+';
  return `${sign}${Math.abs(Math.round(milliseconds))} ms`;
}
