interface StoredObject {
  readonly body: ReadableStream;
  readonly httpMetadata?: { readonly contentType?: string };
  readonly size: number;
}

interface Listing {
  readonly objects: readonly { readonly key: string; readonly size: number }[];
  readonly delimitedPrefixes: readonly string[];
  readonly truncated: boolean;
  readonly cursor?: string;
}

interface Bucket {
  get(key: string): Promise<StoredObject | null>;
  put(key: string, value: ReadableStream | null, options?: { httpMetadata?: { contentType?: string } }): Promise<unknown>;
  list(options: { prefix: string; delimiter?: string; cursor?: string }): Promise<Listing>;
}

interface Environment {
  readonly STORE: Bucket;
  readonly COLLECTION_TOKEN: string;
}

const SESSION_FILES = new Set(['emg.i16', 'emg.missing', 'events.jsonl', 'session.json']);
const TRACK_FILES = new Set(['track.json', 'audio.m4a']);
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;

const CORS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET, PUT, OPTIONS',
  'access-control-allow-headers': 'authorization, content-type',
  'access-control-max-age': '86400',
};

function respond(body: BodyInit | null, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(body, { status, headers: { ...CORS, ...headers } });
}

function json(value: unknown): Response {
  return respond(JSON.stringify(value), 200, { 'content-type': 'application/json' });
}

function authorized(request: Request, environment: Environment): boolean {
  const expected = `Bearer ${environment.COLLECTION_TOKEN}`;
  const given = request.headers.get('authorization') ?? '';
  if (environment.COLLECTION_TOKEN === undefined || given.length !== expected.length) return false;
  let difference = 0;
  for (let index = 0; index < expected.length; index += 1) {
    difference |= expected.charCodeAt(index) ^ given.charCodeAt(index);
  }
  return difference === 0;
}

async function serve(bucket: Bucket, key: string): Promise<Response> {
  const object = await bucket.get(key);
  if (object === null) return respond('not found', 404);
  return respond(object.body, 200, {
    'content-type': object.httpMetadata?.contentType ?? 'application/octet-stream',
    'content-length': String(object.size),
  });
}

async function sessionIds(bucket: Bucket): Promise<string[]> {
  const ids: string[] = [];
  let cursor: string | undefined;
  do {
    const listing = await bucket.list({ prefix: 'sessions/', delimiter: '/', ...(cursor ? { cursor } : {}) });
    ids.push(...listing.delimitedPrefixes.map((prefix) => prefix.slice('sessions/'.length, -1)));
    cursor = listing.truncated ? listing.cursor : undefined;
  } while (cursor !== undefined);
  return ids;
}

export default {
  async fetch(request: Request, environment: Environment): Promise<Response> {
    if (request.method === 'OPTIONS') return respond(null, 204);
    if (!authorized(request, environment)) return respond('unauthorized', 401);

    const parts = new URL(request.url).pathname.split('/').filter((part) => part !== '');
    const [collection, id, file] = parts;
    const bucket = environment.STORE;

    if (request.method === 'GET' && parts.length === 1 && collection === 'tracks') {
      return serve(bucket, 'tracks/index.json');
    }
    if (request.method === 'GET' && parts.length === 1 && collection === 'sessions') {
      return json(await sessionIds(bucket));
    }
    if (parts.length !== 3 || id === undefined || file === undefined || !IDENTIFIER.test(id)) {
      return respond('not found', 404);
    }
    if (collection === 'tracks' && request.method === 'GET' && TRACK_FILES.has(file)) {
      return serve(bucket, `tracks/${id}/${file}`);
    }
    if (collection === 'sessions' && SESSION_FILES.has(file)) {
      const key = `sessions/${id}/${file}`;
      if (request.method === 'GET') return serve(bucket, key);
      if (request.method === 'PUT') {
        await bucket.put(key, request.body, { httpMetadata: { contentType: 'application/octet-stream' } });
        return respond(null, 204);
      }
    }
    return respond('not found', 404);
  },
};
