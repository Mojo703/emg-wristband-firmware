<script lang="ts">
  import Bench from '$lib/bench/Bench.svelte';
  import Collect from '$lib/collect/Collect.svelte';
  import Splash from './lib/screens/Splash.svelte';
  import Menu from './lib/screens/Menu.svelte';
  import Calibration from './lib/screens/Calibration.svelte';
  import GestureConfig from './lib/screens/GestureConfig.svelte';

  type Screen = 'splash' | 'menu' | 'calibration' | 'gesture-config' | 'collect';

  // The link bench stays reachable at #bench for the ble-gatt-bench firmware.
  const hash = typeof location !== 'undefined' ? location.hash : '';

  let screen = $state<Screen>(hash === '#collect' ? 'collect' : 'splash');

  function go(next: Screen): void {
    screen = next;
  }

  function backToMenu(): void {
    screen = 'menu';
  }
</script>

{#if hash === '#bench'}
  <Bench />
{:else if screen === 'splash'}
  <Splash onDone={() => go('menu')} />
{:else if screen === 'menu'}
  <Menu
    onCalibration={() => go('calibration')}
    onGestureConfig={() => go('gesture-config')}
    onCollect={() => go('collect')}
  />
{:else if screen === 'calibration'}
  <Calibration back={backToMenu} />
{:else if screen === 'gesture-config'}
  <GestureConfig back={backToMenu} />
{:else}
  <Collect back={backToMenu} />
{/if}
