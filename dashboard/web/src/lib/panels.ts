// Panel registry — the one place panels are wired in. Add a component and one
// entry here to extend the dashboard.
import type { Component } from 'svelte';
import Collect from '../panels/Collect.svelte';
import EmgViewer from '../panels/EmgViewer.svelte';
import ConfigApp from '../panels/ConfigApp.svelte';
import LogViewer from '../panels/LogViewer.svelte';
import PoseViewer from '../panels/PoseViewer.svelte';
import Telemetry from '../panels/Telemetry.svelte';

export interface Panel {
  readonly id: string;
  readonly title: string;
  readonly icon: string;
  readonly component: Component<any, any, any>;
  readonly props?: Record<string, unknown>;
  /** Whether this panel draws raw EMG. The backend is told on every panel
   * change and sends the stream only to a browser that says yes; the panels
   * that read summaries (the electrode check, telemetry) do not need it. */
  readonly drawsEmg?: boolean;
}

export const panels: readonly Panel[] = [
  // Stream folds in the inference readout: the live confidence track + status.
  { id: 'emg', title: 'Stream', icon: 'activity', component: EmgViewer, drawsEmg: true },
  { id: 'config', title: 'Config', icon: 'sliders', component: ConfigApp },
  // Training-data capture: the falling-notes game and its session bookkeeping.
  { id: 'collect', title: 'Collect', icon: 'play', component: Collect },
  { id: 'pose', title: 'Pose', icon: 'scan', component: PoseViewer, drawsEmg: true },
  // Periodic device measurements as live values + session trend plots.
  { id: 'telemetry', title: 'Telemetry', icon: 'chart', component: Telemetry },
  // The device's log console (replaces the serial text monitor).
  { id: 'logs', title: 'Logs', icon: 'terminal', component: LogViewer },
];
