// One `BandLink` for the whole app, so connecting on one screen survives
// navigating to another. Lifted out of the link bench, which used to own
// this state directly.

import { BandLink, bluetoothAvailable, type EchoResult } from './link.ts';
import type { LinkStatus } from './protocol.ts';

function note(log: string[], message: string): string[] {
  const stamp = new Date().toLocaleTimeString([], { hour12: false });
  return [...log.slice(-199), `${stamp} ${message}`];
}

class Connection {
  readonly link = new BandLink();
  available = $state(true);
  connected = $state(false);
  deviceName = $state('');
  status = $state<LinkStatus | null>(null);
  busy = $state<string | null>(null);
  progress = $state(0);
  log = $state<string[]>([]);

  constructor() {
    this.available = bluetoothAvailable();
    this.link.subscribe((event) => {
      switch (event.kind) {
        case 'status':
          this.status = event.status;
          break;
        case 'disconnected':
          this.connected = false;
          this.status = null;
          this.note('disconnected');
          break;
        case 'log':
          this.note(event.message);
          break;
      }
    });
    if (!this.available) {
      this.note(
        typeof window !== 'undefined' && window.isSecureContext
          ? 'Web Bluetooth is not available in this browser. Chromium browsers have it; on Linux enable chrome://flags/#enable-experimental-web-platform-features and relaunch'
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
    this.progress = 0;
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
      await this.link.connect();
      this.connected = true;
      this.deviceName = this.link.name;
      this.note(
        this.link.supportsWriteWithoutResponse
          ? 'command characteristic accepts write without response'
          : 'command characteristic is write with response only',
      );
    });
  }

  disconnect(): void {
    this.link.disconnect();
  }
}

export type { EchoResult };
export const connection = new Connection();
