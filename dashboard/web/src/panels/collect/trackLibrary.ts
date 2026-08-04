// The library's non-socket half: importing a Beat Saber map and removing a
// track are plain HTTP against the dashboard backend. A refusal always arrives
// as `{"error": "<one sentence>"}` written for whoever tried it, so the message
// reaches the operator word for word.
//
// Neither call updates local state: the backend broadcasts a fresh catalog
// frame afterwards and the picker follows that.

/** One difficulty-dial position of an imported track. */
export interface LevelSummary {
  readonly name: string;
  readonly cue_count: number;
  readonly cues_per_second: number;
  readonly seconds_held: number;
  readonly column_balance: readonly number[];
}

/** What an import produces: the track's identity and a summary per level. */
export interface ImportReport {
  readonly id: string;
  readonly title: string;
  readonly beats_per_minute: number;
  readonly duration_ms: number;
  readonly difficulty_file: string;
  readonly levels: readonly LevelSummary[];
}

const UPLOAD_URL = '/collection/tracks/import/upload';
const BEATSAVER_URL = '/collection/tracks/import/beatsaver';

function refusalMessage(body: unknown, status: number): string {
  if (
    typeof body === 'object' &&
    body !== null &&
    typeof (body as { error?: unknown }).error === 'string'
  ) {
    return (body as { error: string }).error;
  }
  return `the dashboard backend refused the request (HTTP ${status})`;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function readReport(response: Response): Promise<ImportReport> {
  const body = parseJson(await response.text());
  if (!response.ok) throw new Error(refusalMessage(body, response.status));
  return body as ImportReport;
}

/**
 * Send the bytes of a map archive, reporting how much of it has left the
 * browser. XMLHttpRequest rather than fetch, since only it reports upload
 * progress.
 */
export function importUploadedArchive(
  archive: File,
  onUploadedFraction: (fraction: number) => void,
): Promise<ImportReport> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', UPLOAD_URL);
    request.setRequestHeader('Content-Type', 'application/zip');
    request.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onUploadedFraction(event.loaded / event.total);
      }
    };
    request.onload = () => {
      const body = parseJson(request.responseText);
      if (request.status === 200) {
        resolve(body as ImportReport);
      } else {
        reject(new Error(refusalMessage(body, request.status)));
      }
    };
    request.onerror = () =>
      reject(new Error('the upload did not reach the dashboard backend'));
    request.onabort = () => reject(new Error('the upload was cancelled'));
    request.send(archive);
  });
}

/** Import a map the backend downloads, given a BeatSaver key or a link to one. */
export async function importBeatSaverMap(reference: string): Promise<ImportReport> {
  const response = await fetch(BEATSAVER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reference }),
  });
  return readReport(response);
}

/** Remove one track from the library, audio and chart together. */
export async function deleteTrack(trackId: string): Promise<void> {
  const response = await fetch(`/collection/tracks/${encodeURIComponent(trackId)}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(refusalMessage(parseJson(await response.text()), response.status));
  }
}

/** Whatever a rejected request carried, as a sentence fit to show. */
export function failureText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
