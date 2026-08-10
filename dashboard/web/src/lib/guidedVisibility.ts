import type { GuidedMode } from './protocol';

export function visibleGuidedMode(
  panelId: string,
  documentVisible: boolean,
): GuidedMode | null {
  if (!documentVisible) return null;
  if (panelId === 'collect') return 'collection';
  if (panelId === 'calibrate') return 'calibration';
  return null;
}

export class GuidedPresenceLifecycle {
  #registered: GuidedMode | null | undefined;

  update(
    online: boolean,
    panelId: string,
    documentVisible: boolean,
    send: (mode: GuidedMode | null) => void,
  ): void {
    if (!online) {
      this.#registered = undefined;
      return;
    }
    const mode = visibleGuidedMode(panelId, documentVisible);
    if (mode === this.#registered) return;
    this.#registered = mode;
    send(mode);
  }
}
