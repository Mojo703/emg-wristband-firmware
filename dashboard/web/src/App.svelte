<script lang="ts">
  import { onMount } from 'svelte';
  import { Tabs } from 'bits-ui';
  import * as Tooltip from '$lib/components/ui/tooltip/index.js';
  import { panels } from './lib/panels';
  import { connect, live } from './lib/socket.svelte';
  import Icon from './lib/Icon.svelte';

  const firstPanel = panels[0]!;
  let active = $state<string>(firstPanel.id);

  onMount(connect);
</script>

<Tooltip.Provider delayDuration={200}>
  <Tabs.Root bind:value={active} orientation="vertical" class="app">
    <div class="sidebar">
      <div class="brand">
        <span class="name">Opal</span>
        <span class="company">Cairn Kinetics</span>
      </div>
      <Tabs.List class="nav">
        {#each panels as panel (panel.id)}
          <Tabs.Trigger value={panel.id} class="nav-item">
            <Icon name={panel.icon} />
            <span>{panel.title}</span>
          </Tabs.Trigger>
        {/each}
      </Tabs.List>
      <div class="status" class:on={live.connected}>
        <Icon name="wifi" size={14} />
        {live.connected ? 'connected' : 'offline'}
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
