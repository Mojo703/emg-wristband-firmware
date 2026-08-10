import { CalibrationGesture } from '../../lib/protocol';

const GESTURE_LABELS: Record<CalibrationGesture, string> = {
  [CalibrationGesture.WristPronation]: 'Pronation',
  [CalibrationGesture.WristSupination]: 'Supination',
  [CalibrationGesture.WristRadialDeviation]: 'Radial deviation',
  [CalibrationGesture.WristUlnarDeviation]: 'Ulnar deviation',
  [CalibrationGesture.ThumbExtension]: 'Tip center',
};

/** Human-readable labels shared by the device configuration and guided views. */
export function gestureLabel(gesture: CalibrationGesture): string {
  return GESTURE_LABELS[gesture];
}
