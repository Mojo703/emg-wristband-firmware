<script lang="ts">
  import { onMount } from 'svelte';
  import { Band, bluetoothAvailable } from './link.ts';
  import { SENSITIVITIES, gestureLabel, type BandSettings, type GestureEvent } from './protocol.ts';
  import { MEDIA_KEYS } from '$lib/bench/protocol.ts';

  const band = new Band();
  let available = $state(true);
  let connected = $state(false);
  let connecting = $state(false);
  let status = $state('Not connected');
  let settings = $state<BandSettings | null>(null);
  let lastGesture = $state<GestureEvent | null>(null);

  function keyLabel(usage: number): string {
    return MEDIA_KEYS.find((key) => key.usage === usage)?.label ?? `0x${usage.toString(16)}`;
  }

  onMount(() => {
    available = bluetoothAvailable();
    if (!available) status = 'This browser has no Web Bluetooth';
    return band.subscribe((event) => {
      switch (event.kind) {
        case 'settings':
          settings = event.settings;
          break;
        case 'gesture':
          lastGesture = event.event;
          break;
        case 'disconnected':
          connected = false;
          status = 'Not connected';
          break;
      }
    });
  });

  async function connect(): Promise<void> {
    connecting = true;
    try {
      await band.connect();
      connected = true;
      status = band.name;
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    } finally {
      connecting = false;
    }
  }

  async function save(next: BandSettings): Promise<void> {
    try {
      await band.write(next);
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  function setKey(gesture: number, usage: number): void {
    if (settings === null) return;
    void save({
      ...settings,
      bindings: settings.bindings.map((binding) =>
        binding.gesture === gesture ? { gesture, usage } : binding,
      ),
    });
  }

  function setSensitivity(sensitivity: number): void {
    if (settings === null) return;
    void save({ ...settings, sensitivity });
  }
</script>

<main>
  <h1>EMG Wristband</h1>

  <section>
    <div class="row">
      {#if connected}
        <button onclick={() => band.disconnect()}>Disconnect</button>
      {:else}
        <button class="primary" disabled={!available || connecting} onclick={connect}>Connect</button>
      {/if}
      <span class="muted">{status}</span>
    </div>
  </section>

  {#if connected && settings !== null}
    <section>
      <h2>Gestures</h2>
      <dl>
        {#each settings.bindings as binding (binding.gesture)}
          <dt>{gestureLabel(binding.gesture)}</dt>
          <dd>
            <select
              value={binding.usage}
              onchange={(event) => setKey(binding.gesture, Number(event.currentTarget.value))}
            >
              {#each MEDIA_KEYS as key (key.usage)}
                <option value={key.usage}>{key.label}</option>
              {/each}
            </select>
          </dd>
        {/each}
        <dt>Sensitivity</dt>
        <dd>
          <select
            value={settings.sensitivity}
            onchange={(event) => setSensitivity(Number(event.currentTarget.value))}
          >
            {#each SENSITIVITIES as level (level.code)}
              <option value={level.code}>{level.label}</option>
            {/each}
          </select>
        </dd>
      </dl>
    </section>

    <section>
      <h2>Last gesture</h2>
      {#if lastGesture !== null}
        <p>{gestureLabel(lastGesture.gesture)}: {keyLabel(lastGesture.usage)}</p>
      {:else}
        <p class="muted">None yet</p>
      {/if}
    </section>
  {/if}
</main>
