import type { SessionFiles } from './session.ts';

export interface StoredSession {
  readonly id: string;
  readonly files: SessionFiles;
  readonly cues: number;
  readonly uploaded: boolean;
}

const DATABASE = 'emg-collection';
const STORE = 'sessions';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function transact<T>(mode: IDBTransactionMode, work: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const database = await open();
  return new Promise((resolve, reject) => {
    const request = work(database.transaction(STORE, mode).objectStore(STORE));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).finally(() => database.close()) as Promise<T>;
}

export function saveSession(session: StoredSession): Promise<IDBValidKey> {
  return transact('readwrite', (store) => store.put(session));
}

export function listSessions(): Promise<StoredSession[]> {
  return transact('readonly', (store) => store.getAll() as IDBRequest<StoredSession[]>);
}

export function deleteSession(id: string): Promise<undefined> {
  return transact('readwrite', (store) => store.delete(id));
}
