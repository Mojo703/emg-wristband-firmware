<script lang="ts">
  import Splash from './lib/screens/Splash.svelte';
  import Menu from './lib/screens/Menu.svelte';
  import Calibration from './lib/screens/Calibration.svelte';
  import GestureConfig from './lib/screens/GestureConfig.svelte';
  import LinkBench from './lib/screens/LinkBench.svelte';

  type Screen = 'splash' | 'menu' | 'calibration' | 'gesture-config' | 'bench';

  let screen = $state<Screen>('splash');

  function go(next: Screen): void {
    screen = next;
  }

  function backToMenu(): void {
    screen = 'menu';
  }
</script>

{#if screen === 'splash'}
  <Splash onDone={() => go('menu')} />
{:else if screen === 'menu'}
  <Menu onCalibration={() => go('calibration')} onGestureConfig={() => go('gesture-config')} onBench={() => go('bench')} />
{:else if screen === 'calibration'}
  <Calibration back={backToMenu} />
{:else if screen === 'gesture-config'}
  <GestureConfig back={backToMenu} />
{:else}
  <LinkBench back={backToMenu} />
{/if}
