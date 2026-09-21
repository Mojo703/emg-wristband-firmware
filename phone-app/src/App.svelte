<script lang="ts">
  import { onMount } from 'svelte';
  import { BandLink, bluetoothAvailable, type EchoResult } from '$lib/bench/link.ts';
  import {
    MEDIA_KEYS,
    encodeConnectionParameters,
    encodeForgetBonds,
    encodeMedia,
    encodeRequestFastLink,
    encodeRequestPairing,
    phyName,
    type LinkStatus,
  } from '$lib/bench/protocol.ts';
  import type { StreamResult } from '$lib/bench/statistics.ts';

  const link = new BandLink();
  let available = $state(true);
  let connected = $state(false);
  let deviceName = $state('');
  let status = $state<LinkStatus | null>(null);
  let busy = $state<string | null>(null);
  let progress = $state(0);
  let log = $state<string[]>([]);

  let echoCount = $state(200);
  let echoPayload = $state(8);
  let echoWithResponse = $state(false);
  let echoResults = $state<EchoResult[]>([]);

  let streamPayload = $state(244);
  let streamPackets = $state(2000);
  let streamIntervalMicros = $state(0);
  let streamResults = $state<{ readonly requested: number; readonly result: StreamResult; readonly stalls: number; readonly phy: string; readonly intervalMillis: number }[]>([]);

  const SWEEP_SIZES = [20, 100, 244, 500];
  const RAW_EMG_INT16 = 64;
  const RAW_EMG_24BIT = 96;

  const INTERVAL_PRESETS = [
    { label: '7.5 ms', min: 6, max: 6 },
    { label: '15 ms', min: 12, max: 12 },
    { label: '15 to 30 ms', min: 12, max: 24 },
    { label: '30 ms', min: 24, max: 24 },
  ];

  function note(message: string): void {
    const stamp = new Date().toLocaleTimeString([], { hour12: false });
    log = [...log.slice(-199), `${stamp} ${message}`];
  }

  onMount(() => {
    available = bluetoothAvailable();
    if (!available) {
      note(
        window.isSecureContext
          ? 'Web Bluetooth is not available in this browser. Chromium browsers have it; on Linux enable chrome://flags/#enable-experimental-web-platform-features and relaunch'
          : 'Web Bluetooth needs a secure origin: open the page at localhost or over HTTPS',
      );
    }
    return link.subscribe((event) => {
      switch (event.kind) {
        case 'status':
          status = event.status;
          break;
        case 'disconnected':
          connected = false;
          status = null;
          note('disconnected');
          break;
        case 'log':
          note(event.message);
          break;
      }
    });
  });

  async function guard(label: string, action: () => Promise<void>): Promise<void> {
    if (busy !== null) return;
    busy = label;
    progress = 0;
    try {
      await action();
    } catch (error) {
      note(`${label} failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      busy = null;
    }
  }

  function connect(): Promise<void> {
    return guard('connect', async () => {
      await link.connect();
      connected = true;
      deviceName = link.name;
      note(
        link.supportsWriteWithoutResponse
          ? 'command characteristic accepts write without response'
          : 'command characteristic is write with response only',
      );
    });
  }

  function reconnectKnown(): Promise<void> {
    return guard('reconnect', async () => {
      await link.reconnectKnown();
      connected = true;
      deviceName = link.name;
    });
  }

  function disconnect(): void {
    link.disconnect();
  }

  function runEcho(): Promise<void> {
    return guard('echo', async () => {
      const result = await link.runEcho(
        { count: echoCount, payloadBytes: echoPayload, withResponse: echoWithResponse, timeoutMillis: 2000 },
        (done) => {
          progress = done / echoCount;
        },
      );
      echoResults = [...echoResults, result];
      note(`echo: p50 ${format(result.roundTripMillis.p50)} ms, p95 ${format(result.roundTripMillis.p95)} ms, ${result.timeouts} timeouts`);
    });
  }

  async function streamOnce(payloadBytes: number): Promise<void> {
    const before = status?.streamStalls ?? 0;
    const result = await link.runStream(
      { payloadBytes, packetCount: streamPackets, intervalMicros: streamIntervalMicros, idleTimeoutMillis: 3000 },
      (packets) => {
        progress = streamPackets === 0 ? 0 : packets / streamPackets;
      },
    );
    // The band publishes a final status when the stream ends; give it a beat.
    await new Promise((resolve) => setTimeout(resolve, 600));
    const stalls = Math.max(0, (status?.streamStalls ?? 0) - before);
    streamResults = [
      ...streamResults,
      {
        requested: payloadBytes,
        result,
        stalls,
        phy: status ? phyName(status.transmitPhy) : '?',
        intervalMillis: status?.intervalMillis ?? Number.NaN,
      },
    ];
    note(
      `stream ${result.packetBytes} B: ${format(result.goodputKilobytesPerSecond)} KB/s, ${result.packets} packets, ${result.missing} missing, ${result.corrupt} corrupt`,
    );
  }

  function runStream(): Promise<void> {
    return guard('stream', () => streamOnce(streamPayload));
  }

  function runSweep(): Promise<void> {
    return guard('sweep', async () => {
      for (const size of SWEEP_SIZES) {
        await streamOnce(size);
      }
    });
  }

  function stopStream(): Promise<void> {
    return link.stopStream();
  }

  function sendMedia(usage: number, label: string): Promise<void> {
    return guard('media', async () => {
      await link.write(encodeMedia(usage));
      note(`asked the band to send ${label}`);
    });
  }

  function requestInterval(min: number, max: number, label: string): Promise<void> {
    return guard('interval', async () => {
      await link.write(encodeConnectionParameters({ intervalMin: min, intervalMax: max, latency: 0, supervisionTimeout: 400 }));
      note(`requested a ${label} connection interval`);
    });
  }

  function requestFastLink(): Promise<void> {
    return guard('fast link', async () => {
      await link.write(encodeRequestFastLink());
      note('requested 2M PHY and data length extension');
    });
  }

  function requestPairing(): Promise<void> {
    return guard('pairing', async () => {
      await link.write(encodeRequestPairing());
      note('band asked the phone to pair');
    });
  }

  function forgetBonds(): Promise<void> {
    return guard('forget', async () => {
      await link.write(encodeForgetBonds());
      note('band cleared its bonds; forget it in the phone settings too');
    });
  }

  function format(value: number, digits = 1): string {
    return Number.isFinite(value) ? value.toFixed(digits) : '–';
  }

  function verdict(goodput: number): string {
    if (!Number.isFinite(goodput)) return '';
    if (goodput >= RAW_EMG_24BIT) return 'clears 24-bit raw';
    if (goodput >= RAW_EMG_INT16) return 'clears int16 raw';
    return 'below int16 raw';
  }

  function resultsAsMarkdown(): string {
    const lines: string[] = [];
    lines.push(`# Link bench ${new Date().toISOString()}`);
    lines.push('');
    lines.push(`Device: ${deviceName}; browser: ${navigator.userAgent}`);
    if (status) {
      lines.push(
        `Link: interval ${status.intervalMillis} ms, latency ${status.latency}, timeout ${status.supervisionTimeoutMillis} ms, MTU ${status.mtu}, PHY ${phyName(status.transmitPhy)}/${phyName(status.receivePhy)}, encrypted ${status.encrypted}, bonded ${status.bonded}`,
      );
    }
    if (echoResults.length > 0) {
      lines.push('', '| Echo | Payload | Write | n | min | p50 | p95 | max | timeouts |', '|---|---|---|---|---|---|---|---|---|');
      echoResults.forEach((result, index) => {
        const r = result.roundTripMillis;
        lines.push(
          `| ${index + 1} | ${result.payloadBytes} B | ${result.withResponse ? 'with response' : 'without response'} | ${r.count} | ${format(r.min)} | ${format(r.p50)} | ${format(r.p95)} | ${format(r.max)} | ${result.timeouts} |`,
        );
      });
    }
    if (streamResults.length > 0) {
      lines.push('', '| Stream | Requested | Sent | Packets | Missing | Corrupt | Band span | Receive span | KB/s | pkt/s | Stalls | PHY | Interval |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|');
      streamResults.forEach((entry, index) => {
        const r = entry.result;
        lines.push(
          `| ${index + 1} | ${entry.requested} B | ${r.packetBytes} B | ${r.packets} | ${r.missing} | ${r.corrupt} | ${format(r.bandDurationMillis / 1000, 2)} s | ${format(r.durationMillis / 1000, 2)} s | ${format(r.goodputKilobytesPerSecond)} | ${format(r.packetsPerSecond, 0)} | ${entry.stalls} | ${entry.phy} | ${format(entry.intervalMillis, 2)} ms |`,
        );
      });
    }
    return lines.join('\n');
  }

  async function copyResults(): Promise<void> {
    try {
      await navigator.clipboard.writeText(resultsAsMarkdown());
      note('results copied as markdown');
    } catch (error) {
      note(`copy failed: ${String(error)}`);
    }
  }
</script>

<main>
  <h1>Opal link bench</h1>

  <section>
    <h2>Connection</h2>
    <div class="row">
      {#if connected}
        <button onclick={disconnect}>Disconnect {deviceName}</button>
      {:else}
        <button class="primary" onclick={connect} disabled={!available || busy !== null}>Connect to band</button>
        <button onclick={reconnectKnown} disabled={!available || busy !== null}>Reconnect known band</button>
      {/if}
      {#if busy !== null}
        <span class="muted">{busy}{progress > 0 ? ` ${Math.round(progress * 100)}%` : ''}</span>
      {/if}
    </div>
    {#if status}
      <dl>
        <dt>Link</dt>
        <dd>
          {#if status.encrypted}<span class="good">encrypted</span>{:else}<span class="bad">not encrypted</span>{/if},
          {status.bonded ? 'bonded' : 'not bonded'}
        </dd>
        <dt>Interval</dt>
        <dd>{format(status.intervalMillis, 2)} ms, latency {status.latency}, timeout {status.supervisionTimeoutMillis} ms</dd>
        <dt>MTU</dt>
        <dd>{status.mtu} (notification up to {Math.max(0, status.mtu - 3)} B)</dd>
        <dt>PHY</dt>
        <dd>{phyName(status.transmitPhy)} transmit, {phyName(status.receivePhy)} receive</dd>
        <dt>Band heap</dt>
        <dd>{(status.freeHeap / 1024).toFixed(1)} KB free, largest block {(status.largestFreeBlock / 1024).toFixed(1)} KB</dd>
        <dt>Stream</dt>
        <dd>{status.streamPayload > 0 ? `${status.streamPayload} B packets` : 'idle'}, {status.streamSent} sent, {status.streamStalls} stalls</dd>
        <dt>Last media key</dt>
        <dd>{status.lastMedia}</dd>
        <dt>Uptime</dt>
        <dd>{status.uptimeSeconds} s</dd>
      </dl>
    {/if}
  </section>

  <section>
    <h2>Link requests</h2>
    <div class="row">
      {#each INTERVAL_PRESETS as preset (preset.label)}
        <button disabled={!connected || busy !== null} onclick={() => requestInterval(preset.min, preset.max, preset.label)}>{preset.label}</button>
      {/each}
    </div>
    <div class="row" style="margin-top: 8px">
      <button disabled={!connected || busy !== null} onclick={requestFastLink}>2M PHY + long packets</button>
      <button disabled={!connected || busy !== null} onclick={requestPairing}>Pair from band</button>
      <button disabled={!connected || busy !== null} onclick={forgetBonds}>Forget bonds</button>
    </div>
  </section>

  <section>
    <h2>Round trip</h2>
    <div class="row">
      <label>count <input type="number" min="1" max="5000" bind:value={echoCount} /></label>
      <label>payload B <input type="number" min="0" max="500" bind:value={echoPayload} /></label>
      <label><input type="checkbox" bind:checked={echoWithResponse} /> write with response</label>
      <button class="primary" disabled={!connected || busy !== null} onclick={runEcho}>Run</button>
    </div>
    {#if echoResults.length > 0}
      <table>
        <thead>
          <tr><th>#</th><th>Payload</th><th>Write</th><th>n</th><th>min</th><th>p50</th><th>p95</th><th>max</th><th>lost</th></tr>
        </thead>
        <tbody>
          {#each echoResults as result, index (index)}
            <tr>
              <td>{index + 1}</td>
              <td>{result.payloadBytes} B</td>
              <td>{result.withResponse ? 'ack' : 'no ack'}</td>
              <td>{result.roundTripMillis.count}</td>
              <td>{format(result.roundTripMillis.min)}</td>
              <td>{format(result.roundTripMillis.p50)}</td>
              <td>{format(result.roundTripMillis.p95)}</td>
              <td>{format(result.roundTripMillis.max)}</td>
              <td>{result.timeouts}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>

  <section>
    <h2>Stream</h2>
    <div class="row">
      <label>packet B <input type="number" min="8" max="512" bind:value={streamPayload} /></label>
      <label>packets <input type="number" min="0" max="1000000" bind:value={streamPackets} /></label>
      <label>interval µs <input type="number" min="0" max="10000000" bind:value={streamIntervalMicros} /></label>
    </div>
    <div class="row" style="margin-top: 8px">
      <button class="primary" disabled={!connected || busy !== null} onclick={runStream}>Run</button>
      <button disabled={!connected || busy !== null} onclick={runSweep}>Sweep {SWEEP_SIZES.join(', ')} B</button>
      <button disabled={!connected} onclick={stopStream}>Stop</button>
    </div>
    {#if streamResults.length > 0}
      <table>
        <thead>
          <tr><th>#</th><th>Sent</th><th>Packets</th><th>Missing</th><th>Corrupt</th><th>KB/s</th><th>pkt/s</th><th>Band s</th><th>Here s</th><th>Stalls</th><th>PHY</th><th>Raw EMG</th></tr>
        </thead>
        <tbody>
          {#each streamResults as entry, index (index)}
            <tr>
              <td>{index + 1}</td>
              <td>{entry.result.packetBytes} B</td>
              <td>{entry.result.packets}</td>
              <td>{entry.result.missing}</td>
              <td>{entry.result.corrupt}</td>
              <td>{format(entry.result.goodputKilobytesPerSecond)}</td>
              <td>{format(entry.result.packetsPerSecond, 0)}</td>
              <td>{format(entry.result.bandDurationMillis / 1000, 2)}</td>
              <td>{format(entry.result.durationMillis / 1000, 2)}</td>
              <td>{entry.stalls}</td>
              <td>{entry.phy}</td>
              <td class={entry.result.goodputKilobytesPerSecond >= RAW_EMG_INT16 ? 'good' : 'bad'}>{verdict(entry.result.goodputKilobytesPerSecond)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <p class="muted">Raw EMG needs {RAW_EMG_INT16} KB/s as int16 or {RAW_EMG_24BIT} KB/s as 24-bit codes (16 channels at 2 kHz).</p>
    {/if}
  </section>

  <section>
    <h2>Media keys through HID</h2>
    <div class="row">
      {#each MEDIA_KEYS as key (key.id)}
        <button disabled={!connected || busy !== null} onclick={() => sendMedia(key.usage, key.label)}>{key.label}</button>
      {/each}
    </div>
  </section>

  <section>
    <div class="row" style="justify-content: space-between">
      <h2>Log</h2>
      <button onclick={copyResults} disabled={echoResults.length === 0 && streamResults.length === 0}>Copy results</button>
    </div>
    <pre class="log">{log.join('\n')}</pre>
  </section>
</main>
