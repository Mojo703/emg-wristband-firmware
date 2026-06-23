<script lang="ts">
  import { onMount } from 'svelte';
  import { panels } from './lib/panels';
  import { connect, live } from './lib/socket.svelte';
  import Icon from './lib/Icon.svelte';

  const firstPanel = panels[0]!;
  let active = $state<string>(firstPanel.id);
  const current = $derived(panels.find((panel) => panel.id === active) ?? firstPanel);

  onMount(connect);
</script>

<div class="app">
  <nav>
    <div class="brand">EMG Wristband</div>
    {#each panels as panel}
      <button class:active={panel.id === active} onclick={() => (active = panel.id)}>
        <Icon name={panel.icon} />
        <span>{panel.title}</span>
      </button>
    {/each}
    <div class="status" class:on={live.connected}>
      <Icon name="wifi" size={14} />
      {live.connected ? 'connected' : 'offline'}
    </div>
  </nav>

  <main>
    {#key active}
      {@const Panel = current.component}
      <Panel {...(current.props ?? {})} />
    {/key}
  </main>
</div>
