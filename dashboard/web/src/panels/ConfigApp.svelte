<script>
  import { live, api } from '../lib/socket.svelte.js';
  import Icon from '../lib/Icon.svelte';

  const KEYS = [
    { value: 'play_pause', label: 'Play/Pause' },
    { value: 'next_track', label: 'Next track' },
    { value: 'prev_track', label: 'Previous track' },
    { value: 'volume_up', label: 'Volume up' },
    { value: 'volume_down', label: 'Volume down' },
    { value: 'mute', label: 'Mute' },
  ];

  let bindings = $state([]); // [{ gesture, key }]
  let ssid = $state('');
  let psk = $state('');
  let seeded = false;

  // Seed the editable copy from the backend's config once Hello arrives.
  $effect(() => {
    if (seeded || !live.hello) return;
    seeded = true;
    const count = live.hello.gestures;
    const current = live.hello.keymap ?? [];
    bindings = Array.from({ length: count }, (_, gesture) => ({
      gesture,
      key: current.find((entry) => entry.gesture === gesture)?.key ?? KEYS[gesture % KEYS.length].value,
    }));
    ssid = live.hello.wifi_ssid ?? '';
  });

  const saveKeymap = () => api.keymap($state.snapshot(bindings));
  const saveWifi = () => api.wifi(ssid, psk);
</script>

<h2>Config</h2>

<section>
  <div class="row"><Icon name="keyboard" /><strong>Gesture → action</strong></div>
  <table>
    <tbody>
      {#each bindings as binding}
        <tr>
          <td>Gesture {binding.gesture}</td>
          <td>
            <select bind:value={binding.key}>
              {#each KEYS as option}
                <option value={option.value}>{option.label}</option>
              {/each}
            </select>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
  <div class="row" style="margin-top: 10px;">
    <button class="btn" onclick={saveKeymap}>Save keymap</button>
  </div>
</section>

<section style="margin-top: 24px;">
  <div class="row"><Icon name="wifi" /><strong>WiFi</strong></div>
  <div class="row">
    <label>SSID<input type="text" bind:value={ssid} placeholder="network name" /></label>
    <label>Password<input type="password" bind:value={psk} placeholder="••••••" /></label>
  </div>
  <div class="row">
    <button class="btn" onclick={saveWifi}>Save WiFi</button>
    <span class="muted">Stored on the dashboard; pushed to the device once it's connected.</span>
  </div>
</section>
