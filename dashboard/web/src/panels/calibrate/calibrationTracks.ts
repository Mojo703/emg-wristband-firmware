export interface CalibrationTrack {
  readonly id: string;
  readonly title: string;
  readonly beats_per_minute: number;
  readonly duration_ms: number;
  readonly cue_count: number;
  readonly content_identity: string;
  readonly cue_shortfall: number;
}

const TRACKS_URL = '/calibration/tracks';

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonnegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0;
}

function isCalibrationTrack(value: unknown): value is CalibrationTrack {
  return (
    isObject(value) &&
    typeof value['id'] === 'string' &&
    typeof value['title'] === 'string' &&
    Number.isInteger(value['beats_per_minute']) &&
    (value['beats_per_minute'] as number) > 0 &&
    isNonnegativeInteger(value['duration_ms']) &&
    isNonnegativeInteger(value['cue_count']) &&
    isNonnegativeInteger(value['cue_shortfall']) &&
    typeof value['content_identity'] === 'string' &&
    /^[0-9a-f]{64}$/.test(value['content_identity'])
  );
}

export function parseCalibrationTracks(value: unknown): readonly CalibrationTrack[] {
  if (!Array.isArray(value) || !value.every(isCalibrationTrack)) {
    throw new Error('the dashboard backend returned an unrecognised Calibration track list');
  }
  return value;
}

export async function loadCalibrationTracks(
  signal?: AbortSignal,
): Promise<readonly CalibrationTrack[]> {
  const response = await fetch(TRACKS_URL, signal === undefined ? undefined : { signal });
  if (!response.ok) {
    throw new Error(`the dashboard backend refused the Calibration track list (HTTP ${response.status})`);
  }
  return parseCalibrationTracks(await response.json());
}
