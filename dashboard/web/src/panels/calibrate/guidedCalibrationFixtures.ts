import { ThumbVariant, VisualLane } from '../../lib/collect/field.ts';
import type {
  CalibrationPlayingSnapshot,
  CalibrationSetupSnapshot,
  CalibrationSongEndSnapshot,
  CalibrationFinalizingSnapshot,
  CalibrationTechnicalFailureSnapshot,
  ActiveCalibrationLanes,
} from './guidedCalibration';
import { asUnixMilliseconds } from '../../lib/protocol.ts';

const contentIdentity = 'a'.repeat(64);

const track = {
  id: 'fixture-track',
  title: 'Fixture Track',
  beats_per_minute: 128,
  duration_ms: 170_000,
  cue_count: 52,
  content_identity: contentIdentity,
  cue_shortfall: 0,
} as const;

const semanticSchedule: number[] = [];
const remaining = [10, 10, 6, 6, 5, 5, 5, 5];
while (remaining.some((count) => count > 0)) {
  for (let command = 0; command < 2; command += 1) {
    for (let modifier = 0; modifier < 4; modifier += 1) {
      const semantic = command + modifier * 2;
      if (remaining[semantic] === 0) continue;
      semanticSchedule.push(semantic);
      remaining[semantic] = remaining[semantic]! - 1;
    }
  }
}

const lanes: ActiveCalibrationLanes = [
  {
    visualLane: VisualLane.Gesture0,
    id: 'wrist_radial_deviation',
    label: 'Tip forward',
    colorName: 'green',
    motion: { arrow: 'up', hint: 'pole tip forward' },
  },
  {
    visualLane: VisualLane.Gesture1,
    id: 'wrist_ulnar_deviation',
    label: 'Tip back',
    colorName: 'purple',
    motion: { arrow: 'down', hint: 'pole tip back' },
  },
  {
    visualLane: VisualLane.Gesture2,
    id: 'center_counterexample',
    label: 'Pole vertical — do not trigger',
    colorName: 'gray',
    motion: null,
  },
];

export const calibrationSetupFixture: CalibrationSetupSnapshot = {
  phase: 'setup',
  tracks: [
    track,
    { ...track, id: 'short-track', title: 'Short Fixture', cue_count: 44, cue_shortfall: 8 },
  ],
  selected_track_id: track.id,
};

export const calibrationPlayingFixture: CalibrationPlayingSnapshot = {
  phase: 'playing',
  track,
  lanes,
  cues: semanticSchedule.map((semanticColumn, index) => ({
    visualLane:
      semanticColumn === 0
        ? lanes[0].visualLane
        : semanticColumn === 1
          ? lanes[1].visualLane
          : VisualLane.Gesture2,
    at: 4_000 + index * 2_100,
    hold: 1_500,
    thumbVariant:
      semanticColumn < 2
        ? ThumbVariant.Up
        : semanticColumn < 4
          ? ThumbVariant.Down
          : semanticColumn < 6
            ? ThumbVariant.Medium
            : ThumbVariant.Hard,
    cueMotion: semanticColumn < 2 ? lanes[semanticColumn]!.motion : null,
    cueLabel:
      semanticColumn < 2
        ? lanes[semanticColumn]!.label
        : `Pole vertical, ${semanticColumn < 4 ? 'soft grip' : semanticColumn < 6 ? 'medium grip' : 'hard grip'}`,
  })),
  position_ms: 18_000,
  position_observed_at_unix_ms: asUnixMilliseconds(1_800_000_018_000),
  valid_reps: 37,
  invalid_reps: 2,
  paused_reason: null,
  counts: lanes.map((lane) => ({
    class_id: lane.id,
    label: lane.label,
    thumb_up: 4,
    thumb_down: 3,
    invalid: lane.visualLane === VisualLane.Gesture1 ? 2 : 0,
  })),
};

export const calibrationSongEndFixture: CalibrationSongEndSnapshot = {
  phase: 'between_songs',
  track_title: track.title,
  tracks: calibrationSetupFixture.tracks,
  selected_track_id: track.id,
  candidate_available: true,
  continue_available: true,
  valid_reps: 72,
  invalid_reps: 6,
  deficits: ['Radial, hard grip: 4/5'],
};

export const calibrationTechnicalFailureFixture: CalibrationTechnicalFailureSnapshot = {
  phase: 'technical_failure',
  detail: 'Fitting stopped before a structurally valid candidate was produced.',
};

export const calibrationFinalizingFixture: CalibrationFinalizingSnapshot = {
  phase: 'finalizing',
  detail: 'Building, validating, and saving calibration on the wristband…',
};
