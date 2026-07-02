<script lang="ts">
  import { onMount } from 'svelte';
  import { Tabs } from 'bits-ui';
  import * as Tooltip from '$lib/components/ui/tooltip/index.js';
  import { panels } from './lib/panels';
  import { connect, live, api } from './lib/socket.svelte';
  import Icon from './lib/Icon.svelte';
  import Select from './lib/ui/Select.svelte';
  import StreamMonitor from './lib/StreamMonitor.svelte';

  const firstPanel = panels[0]!;
  let active = $state<string>(firstPanel.id);

  // Which device the dashboard is viewing. The list and selection are owned by the
  // backend (it tracks every connected device); selecting one routes its stream and
  // our control frames to it.
  const deviceOptions = $derived(
    (live.hello?.devices ?? []).map((device) => ({ value: device.id, label: device.label })),
  );
  const selectedDevice = $derived(live.hello?.selected_device ?? '');

  onMount(connect);
</script>

<Tooltip.Provider delayDuration={200}>
  <Tabs.Root bind:value={active} orientation="vertical" class="app">
    <div class="sidebar">
      <div class="brand">
        <span class="name">Opal</span>
        <span class="company">Cairn Kinetics</span>
      </div>
      <div class="device">
        {#if deviceOptions.length > 0}
          <Select
            value={selectedDevice}
            options={deviceOptions}
            onChange={(id) => api.selectDevice(id)}
            placeholder="Select device…"
          />
        {:else}
          <span class="muted">No devices</span>
        {/if}
      </div>
      <Tabs.List class="nav">
        {#each panels as panel (panel.id)}
          <Tabs.Trigger value={panel.id} class="nav-item">
            <Icon name={panel.icon} />
            <span>{panel.title}</span>
          </Tabs.Trigger>
        {/each}
      </Tabs.List>
      <div class="statusbar">
        <!-- Backend link (not device presence — devices live in the picker above). -->
        <div class="status" class:on={live.connected} title="Dashboard's connection to the backend server">
          <Icon name="server" size={14} />
          {live.connected ? 'backend online' : 'backend offline'}
        </div>
        <StreamMonitor />
      </div>
    </div>

    <main>
      <!-- Render only the active panel. bits-ui Tabs.Content keeps inactive
           content mounted-but-hidden, which mounts every panel's canvas at a zero
           size on load (the 3D pose renderer then initializes broken). Mounting one
           panel at a time also keeps a single canvas/RAF loop running. -->
      {#each panels as panel (panel.id)}
        {#if panel.id === active}
          <Tabs.Content value={panel.id}>
            {@const Panel = panel.component}
            <Panel {...(panel.props ?? {})} />
          </Tabs.Content>
        {/if}
      {/each}
    </main>
  </Tabs.Root>
</Tooltip.Provider>
