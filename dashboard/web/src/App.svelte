<script lang="ts">
  import { onMount } from 'svelte';
  import { Tabs, ToggleGroup } from 'bits-ui';
  import * as Tooltip from '$lib/components/ui/tooltip/index.js';
  import { panels } from './lib/panels';
  import { api, connect, live } from './lib/socket.svelte';
  import { theme, type ThemeChoice } from './lib/theme.svelte';
  import Icon from './lib/Icon.svelte';
  import DeviceList from './lib/DeviceList.svelte';
  import PhoneToggle from './lib/PhoneToggle.svelte';
  import StreamMonitor from './lib/StreamMonitor.svelte';

  const firstPanel = panels[0]!;
  let active = $state<string>(firstPanel.id);

  const THEME_CHOICES: readonly { readonly value: ThemeChoice; readonly icon: string; readonly title: string }[] = [
    { value: 'system', icon: 'monitor', title: 'Follow system theme' },
    { value: 'light', icon: 'sun', title: 'Light theme' },
    { value: 'dark', icon: 'moon', title: 'Dark theme' },
  ];
  const THEME_INDEX: Record<ThemeChoice, number> = { system: 0, light: 1, dark: 2 };
  const themeIndex = $derived(THEME_INDEX[theme.choice]);

  // bits-ui's single-select ToggleGroup deselects (emits "") when the pressed item
  // is clicked again — fine for a toolbar, wrong for a theme picker, which must
  // always have exactly one of the three selected. Ignore the empty emission.
  function onThemeChange(value: string): void {
    if (value === 'system' || value === 'light' || value === 'dark') theme.set(value);
  }

  onMount(connect);

  // Only the panels that draw waveforms ask for the EMG stream. Sent on every
  // switch (and re-sent on reconnect, because a fresh socket starts subscribed
  // and would otherwise stream the whole of a collection session at a page
  // drawing none of it).
  const drawsEmg = $derived(panels.find((panel) => panel.id === active)?.drawsEmg ?? false);
  $effect(() => {
    if (live.status !== 'online') return;
    api.setEmgStream(drawsEmg);
  });
</script>

<Tooltip.Provider delayDuration={200}>
  <Tabs.Root bind:value={active} orientation="vertical" class="app">
    <div class="sidebar">
      <div class="brand">
        <span class="name">Opal</span>
        <span class="company">Cairn Kinetics</span>
      </div>
      <div class="device">
        <DeviceList />
      </div>
      <PhoneToggle />
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
        <ToggleGroup.Root
          type="single"
          value={theme.choice}
          onValueChange={onThemeChange}
          class="theme-toggle"
          aria-label="Colour theme"
        >
          <span class="theme-toggle-thumb" style:--index={themeIndex}></span>
          {#each THEME_CHOICES as { value, icon, title } (value)}
            <ToggleGroup.Item {value} title={title} class="theme-toggle-item">
              <Icon name={icon} size={14} />
            </ToggleGroup.Item>
          {/each}
        </ToggleGroup.Root>
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
