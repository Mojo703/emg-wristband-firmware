<script lang="ts">
  import { live, api } from '../lib/socket.svelte';
  import { MediaKey, isMediaKey, type Binding } from '../lib/protocol';
  import Icon from '../lib/Icon.svelte';

  const KEYS: readonly { value: MediaKey; label: string }[] = [
    { value: MediaKey.PlayPause, label: 'Play/Pause' },
    { value: MediaKey.NextTrack, label: 'Next track' },
    { value: MediaKey.PrevTrack, label: 'Previous track' },
    { value: MediaKey.VolumeUp, label: 'Volume up' },
    { value: MediaKey.VolumeDown, label: 'Volume down' },
    { value: MediaKey.Mute, label: 'Mute' },
  ];

  // The discrete settings reflect the backend's live config directly — no local
  // draft. Each change applies immediately; the backend echoes a fresh Hello, so
  // what's shown is always what's active.
  const gestures = $derived(live.hello?.gestures ?? 0);
  const keymap = $derived(live.hello?.keymap ?? []);
  const levels = $derived(live.hello?.sensitivity_levels ?? []);
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

  function handleKeyChange(gesture: number, event: Event & { currentTarget: HTMLSelectElement }): void {
    const key = event.currentTarget.value;
    if (!isMediaKey(key)) return;
    setKey(gesture, key);
  }

  // WiFi is the exception: free-text credentials are entered as a set and committed
  // together (the password is write-only and never echoed back), so it keeps an
  // explicit Save rather than applying per keystroke.
  let ssid = $state('');
  let psk = $state('');
  let wifiSeeded = false;
  $effect(() => {
    if (wifiSeeded || live.hello === null) return;
    wifiSeeded = true;
    ssid = live.hello.wifi_ssid ?? '';
  });
  const saveWifi = () => api.wifi(ssid, psk);

  function handleSensitivityChange(event: Event & { currentTarget: HTMLSelectElement }): void {
    api.sensitivity(event.currentTarget.value);
  }
</script>

<h2>Config</h2>
<p class="muted">Changes apply immediately.</p>

<section>
  <div class="row"><Icon name="sliders" /><strong>Sensitivity</strong></div>
  <div class="row">
    <label>
      Trigger sensitivity
      <select value={live.hello?.sensitivity ?? ''} onchange={handleSensitivityChange}>
        {#each levels as level}
          <option value={level.id}>{level.label}</option>
        {/each}
      </select>
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
            <select value={keyFor(gesture)} onchange={(event) => handleKeyChange(gesture, event)}>
              {#each KEYS as option}
                <option value={option.value}>{option.label}</option>
              {/each}
            </select>
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
    <button class="btn" onclick={saveWifi}>Save WiFi</button>
    <span class="muted">Credentials are committed together; the password is write-only.</span>
  </div>
</section>
