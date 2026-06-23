// Panel registry — the one place panels are wired in. Add a component and one
// entry here to extend the dashboard.
import EmgViewer from '../panels/EmgViewer.svelte';
import ConfigApp from '../panels/ConfigApp.svelte';
import Stub from '../panels/Stub.svelte';

export const panels = [
  // Stream folds in the inference readout: the live confidence track + status.
  { id: 'emg', title: 'Stream', icon: 'activity', component: EmgViewer },
  { id: 'config', title: 'Config', icon: 'sliders', component: ConfigApp },
  { id: 'eval', title: 'Eval', icon: 'chart', component: Stub, props: { title: 'Eval dashboard' } },
  { id: 'pose', title: 'Pose', icon: 'scan', component: Stub, props: { title: 'Pose viewer (SVG-2D)' } },
];
