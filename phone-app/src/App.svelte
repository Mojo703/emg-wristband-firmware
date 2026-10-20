<script lang="ts">
  import Splash from './lib/screens/Splash.svelte';
  import Menu from './lib/screens/Menu.svelte';
  import Calibration from './lib/screens/Calibration.svelte';
  import GestureConfig from './lib/screens/GestureConfig.svelte';

  type Screen = 'splash' | 'menu' | 'calibration' | 'gesture-config';

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
  <Menu onCalibration={() => go('calibration')} onGestureConfig={() => go('gesture-config')} />
{:else if screen === 'calibration'}
  <Calibration back={backToMenu} />
{:else}
  <GestureConfig back={backToMenu} />
{/if}
