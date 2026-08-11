<script lang="ts">
  import { live, api } from '../lib/socket.svelte';
  import {
    CALIBRATION_GESTURE_ORDER,
    MediaKey,
    isMediaKey,
    type Binding,
    type CalibrationGesture,
  } from '../lib/protocol';
  import { gestureLabel } from './calibrate/text';
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

  // The discrete settings reflect the selected device's live config directly — no
  // local draft. Each change applies immediately; the device re-announces and the
  // backend echoes a fresh Hello, so what's shown is always what's active.
  const config = $derived(live.hello?.selection?.config ?? null);
  const keymap = $derived(config?.keymap ?? []);
  const levels = $derived(config?.sensitivity_levels ?? []);
  const levelOptions = $derived(levels.map((level) => ({ value: level.id, label: level.label })));
  const commandRows: readonly {
    readonly gesture: CalibrationGesture;
    readonly index: number;
  }[] = CALIBRATION_GESTURE_ORDER.map((gesture, index) => ({ gesture, index }));

  function keyFor(gesture: number): MediaKey | undefined {
    return keymap.find((entry) => entry.gesture === gesture)?.key;
  }

  function setKey(gesture: number, key: MediaKey): void {
    const next: Binding[] = keymap
      .filter((entry) => entry.gesture !== gesture)
      .concat({ gesture, key })
      .sort((left, right) => left.gesture - right.gesture);
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

  // The dashboard address the device dials over wifi. The backend suggests its own
  // reachable addresses (best guess first); until the user edits the field, mirror the
  // top suggestion, which tracks the backend's live networks (e.g. a hotspot coming
  // up). The value is write-only — the device never echoes it back.
  const serverSuggestions = $derived(live.hello?.server_suggestions ?? []);
  let server = $state('');
  let serverEdited = false;
  $effect(() => {
    if (serverEdited) return;
    const best = serverSuggestions[0];
    if (best !== undefined) server = best;
  });
  const saveServer = () => {
    const addr = server.trim();
    if (addr) api.server(addr);
  };
</script>

<h2>Settings</h2>

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
  </div>
</section>

<section style="margin-top: 24px;">
  <div class="row"><Icon name="keyboard" /><strong>Calibrated command actions</strong></div>
  <table>
    <tbody>
      {#each commandRows as row (row.gesture)}
        <tr>
          <td>{gestureLabel(row.gesture)}</td>
          <td>
            <Select
              value={keyFor(row.index) ?? ''}
              options={keyOptions}
              onChange={(value) => onKeyChange(row.index, value)}
              placeholder="Unassigned"
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
    <label>Password<input type="password" bind:value={psk} placeholder="password" /></label>
  </div>
  <div class="row">
    <Button onclick={saveWifi}>Save WiFi</Button>
  </div>
  <div class="row">
    <label>
      Server
      <input
        type="text"
        bind:value={server}
        oninput={() => (serverEdited = true)}
        list="server-suggestions"
        placeholder="10.42.0.1:9000"
      />
      <datalist id="server-suggestions">
        {#each serverSuggestions as suggestion}
          <option value={suggestion}></option>
        {/each}
      </datalist>
    </label>
  </div>
  <div class="row">
    <Button onclick={saveServer}>Save Server</Button>
  </div>
</section>
