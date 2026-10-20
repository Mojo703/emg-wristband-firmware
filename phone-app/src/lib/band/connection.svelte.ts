// One `Band` for the whole app's product screens (Menu, Gesture
// Configuration), mirroring `bench/connection.svelte.ts`'s shape but
// wrapping the real settings-service connection instead of the bench one.

import { Band, bluetoothAvailable } from './link.ts';
import type { BandSettings, GestureEvent } from './protocol.ts';

function note(log: string[], message: string): string[] {
  const stamp = new Date().toLocaleTimeString([], { hour12: false });
  return [...log.slice(-199), `${stamp} ${message}`];
}

class Connection {
  readonly band = new Band();
  available = $state(true);
  connected = $state(false);
  deviceName = $state('');
  settings = $state<BandSettings | null>(null);
  lastGesture = $state<GestureEvent | null>(null);
  busy = $state<string | null>(null);
  log = $state<string[]>([]);

  constructor() {
    this.available = bluetoothAvailable();
    this.band.subscribe((event) => {
      switch (event.kind) {
        case 'settings':
          this.settings = event.settings;
          break;
        case 'gesture':
          this.lastGesture = event.event;
          break;
        case 'disconnected':
          this.connected = false;
          this.settings = null;
          this.note('disconnected');
          break;
      }
    });
    if (!this.available) {
      this.note(
        typeof window !== 'undefined' && window.isSecureContext
          ? 'Web Bluetooth is not available in this browser'
          : 'Web Bluetooth needs a secure origin: open the page at localhost or over HTTPS',
      );
    }
  }

  note(message: string): void {
    this.log = note(this.log, message);
  }

  async guard(label: string, action: () => Promise<void>): Promise<void> {
    if (this.busy !== null) return;
    this.busy = label;
    try {
      await action();
    } catch (error) {
      this.note(`${label} failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      this.busy = null;
    }
  }

  connect(): Promise<void> {
    return this.guard('connect', async () => {
      await this.band.connect();
      this.connected = true;
      this.deviceName = this.band.name;
      this.note(`connected to ${this.deviceName || 'band'}`);
    });
  }

  disconnect(): void {
    this.band.disconnect();
  }

  writeSettings(settings: BandSettings): Promise<void> {
    return this.guard('save', async () => {
      await this.band.write(settings);
      this.note(`sent ${settings.bindings.length} binding${settings.bindings.length === 1 ? '' : 's'} to the band`);
    });
  }
}

export const connection = new Connection();
