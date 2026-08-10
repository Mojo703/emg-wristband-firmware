import assert from 'node:assert/strict';
import test, { after, before } from 'node:test';
import { createServer, type ViteDevServer } from 'vite';

let server: ViteDevServer;
let originalDocument: PropertyDescriptor | undefined;
let originalSetInterval: typeof globalThis.setInterval;

before(async () => {
  originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  originalSetInterval = globalThis.setInterval;
  globalThis.setInterval = ((handler: TimerHandler, timeout?: number, ...arguments_: unknown[]) => {
    const timer = originalSetInterval(handler, timeout, ...arguments_);
    (timer as unknown as { unref?: () => void }).unref?.();
    return timer;
  }) as typeof globalThis.setInterval;
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: { documentElement: { setAttribute(): void {} } },
  });
  server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
});

after(async () => {
  await server.close();
  globalThis.setInterval = originalSetInterval;
  if (originalDocument === undefined) Reflect.deleteProperty(globalThis, 'document');
  else Object.defineProperty(globalThis, 'document', originalDocument);
});

test('authoritative exit projection disables the persistent exit action until device ACK', async () => {
  const [{ render }, componentModule, socketModule] = await Promise.all([
    server.ssrLoadModule('svelte/server'),
    server.ssrLoadModule('/src/panels/Calibrate.svelte'),
    server.ssrLoadModule('/src/lib/socket.svelte.ts'),
  ]);
  const live = socketModule['live'];
  live.status = 'online';
  live.setGuidedSession({
    revision: 9_001,
    run_revision: 1,
    action_authority: { run_revision: 1, session_id: 1, phase_generation: 4 },
    visible_collection_views: 0,
    visible_calibration_views: 1,
    lifecycle: {
      state: 'calibration',
      session_id: 1,
      device_id: 'opal-test',
      calibration: {
        phase: 'exiting',
        detail: 'Stopping the accepted song before discarding calibration…',
      },
    },
  });

  const body = render(componentModule['default']).body;
  const exitButton = body
    .match(/<button\b[^>]*>[\s\S]*?<\/button>/g)
    ?.find((candidate: string) => candidate.includes('Exiting'));
  assert.ok(exitButton, 'the persistent exit control remains visible');
  assert.match(exitButton, /\sdisabled(?:=|(?=\s|>))/);
  assert.match(body, /Waiting for the wristband to confirm that no candidate remains/);
});
