<script lang="ts">
  import { live, api } from '../lib/socket.svelte';
  import { MediaKey, isMediaKey, type Binding } from '../lib/protocol';
  import Icon from '../lib/Icon.svelte';
  import Select from '../lib/ui/Select.svelte';
  import { Button } from '$lib/components/ui/button/index.js';

  const KEYS: readonly { value: MediaKey; label: string }[] = [
    { value: MediaKey.PlayPause, label: 'Play/Pause' },
    { value: MediaKey.NextTrack, label: 'Next track' },
    { value: MediaKey.PrevTrack, label: 'Previous track' },
    { value: MediaKey.VolumeUp, label: 'Volume up' },
    { value: MediaKey.VolumeDown, label: 'Volume down' },
    { value: MediaKey.Mute, label: 'Mute' },
  ];
  const keyOptions = KEYS.map((option) => ({ value: option.value as string, label: option.label }));

  // The discrete settings reflect the backend's live config directly — no local
  // draft. Each change applies immediately; the backend echoes a fresh Hello, so
  // what's shown is always what's active.
  const config = $derived(live.hello);
  const gestures = $derived(config?.gestures ?? 0);
  const keymap = $derived(config?.keymap ?? []);
  const levels = $derived(config?.sensitivity_levels ?? []);
  const levelOptions = $derived(levels.map((level) => ({ value: level.id, label: level.label })));
  const gestureIndices = $derived(Array.from({ length: gestures }, (_, i) => i));

  function keyFor(gesture: number): MediaKey {
    const bound = keymap.find((entry) => entry.gesture === gesture)?.key;
    if (bound !== undefined) return bound;
    const fallback = KEYS[gesture % KEYS.length];
    if (fallback === undefined) return MediaKey.PlayPause;
    return fallback.value;
  }

  function setKey(gesture: number, key: MediaKey): void {
    const next: Binding[] = Array.from({ length: gestures }, (_, g) => ({
      gesture: g,
      key: g === gesture ? key : keyFor(g),
    }));
    api.keymap(next);
  }

  function onKeyChange(gesture: number, value: string): void {
    if (isMediaKey(value)) setKey(gesture, value);
  }

  // WiFi is the exception: free-text credentials are entered as a set and committed
  // together (the password is write-only and never echoed back), so it keeps an
  // explicit Save rather than applying per keystroke.
  let ssid = $state('');
  let psk = $state('');
  let wifiSeeded = false;
  $effect(() => {
    if (wifiSeeded || config === null) return;
    wifiSeeded = true;
    ssid = config.wifi_ssid ?? '';
  });
  const saveWifi = () => api.wifi(ssid, psk);
</script>

<h2>Config</h2>
<p class="muted">Changes apply immediately.</p>

<section>
  <div class="row"><Icon name="sliders" /><strong>Sensitivity</strong></div>
  <div class="row">
    <label>
      Trigger sensitivity
      <Select
        value={config?.sensitivity ?? ''}
        options={levelOptions}
        onChange={(value) => api.sensitivity(value)}
        placeholder="Select…"
      />
    </label>
    <span class="muted">Higher = commands trigger more easily.</span>
  </div>
</section>

<section style="margin-top: 24px;">
  <div class="row"><Icon name="keyboard" /><strong>Gesture → action</strong></div>
  <table>
    <tbody>
      {#each gestureIndices as gesture}
        <tr>
          <td>Gesture {gesture}</td>
          <td>
            <Select
              value={keyFor(gesture)}
              options={keyOptions}
              onChange={(value) => onKeyChange(gesture, value)}
            />
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>

<section style="margin-top: 24px;">
  <div class="row"><Icon name="wifi" /><strong>WiFi</strong></div>
  <div class="row">
    <label>SSID<input type="text" bind:value={ssid} placeholder="network name" /></label>
    <label>Password<input type="password" bind:value={psk} placeholder="••••••" /></label>
  </div>
  <div class="row">
    <Button onclick={saveWifi}>Save WiFi</Button>
    <span class="muted">Credentials are committed together; the password is write-only.</span>
  </div>
</section>
