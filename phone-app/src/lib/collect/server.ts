import type { MapNote } from './schedule.ts';
import type { SessionFiles } from './session.ts';

export interface ServerAccess {
  readonly baseUrl: string;
  readonly token: string;
}

export interface TrackSummary {
  readonly id: string;
  readonly title: string;
  readonly duration_ms: number;
}

export interface Track extends TrackSummary {
  readonly levels: Readonly<Record<string, { readonly map_notes: readonly MapNote[] }>>;
}

async function request(access: ServerAccess, path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(`${access.baseUrl.replace(/\/$/, '')}${path}`, {
    ...init,
    headers: { ...init.headers, authorization: `Bearer ${access.token}` },
  });
  if (!response.ok) throw new Error(`${init.method ?? 'GET'} ${path}: ${response.status} ${await response.text()}`);
  return response;
}

export async function listTracks(access: ServerAccess): Promise<TrackSummary[]> {
  return (await request(access, '/tracks')).json();
}

export async function loadTrack(access: ServerAccess, id: string): Promise<{ track: Track; audio: Blob }> {
  const [track, audio] = await Promise.all([
    request(access, `/tracks/${id}/track.json`).then((response) => response.json() as Promise<Track>),
    request(access, `/tracks/${id}/audio.m4a`).then((response) => response.blob()),
  ]);
  return { track, audio };
}

export async function uploadSession(access: ServerAccess, id: string, files: SessionFiles): Promise<void> {
  const order = ['emg.i16', 'emg.missing', 'events.jsonl', 'session.json'] as const;
  for (const name of order) {
    await request(access, `/sessions/${id}/${name}`, { method: 'PUT', body: files[name] });
  }
}
