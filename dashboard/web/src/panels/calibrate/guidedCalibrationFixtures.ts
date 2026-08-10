import { ThumbVariant, VisualLane } from '../../lib/collect/field.ts';
import type {
  CalibrationPlayingSnapshot,
  CalibrationSetupSnapshot,
  CalibrationSongEndSnapshot,
  CalibrationTechnicalFailureSnapshot,
  FiveCalibrationLanes,
} from './guidedCalibration';
import { asUnixMilliseconds } from '../../lib/protocol.ts';

const contentIdentity = 'a'.repeat(64);

const track = {
  id: 'fixture-track',
  title: 'Fixture Track',
  beats_per_minute: 128,
  duration_ms: 280_000,
  cue_count: 130,
  content_identity: contentIdentity,
  cue_shortfall: 0,
} as const;

const pairedCycle = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9] as const;
const antiCycle = [5, 6, 7, 8, 9] as const;
const semanticSchedule = [
  ...Array.from({ length: 10 }, () => pairedCycle).flat(),
  ...Array.from({ length: 6 }, () => antiCycle).flat(),
];

const lanes: FiveCalibrationLanes = [
  {
    visualLane: VisualLane.Gesture0,
    id: 'wrist_pronation',
    label: 'Tip out',
    colorName: 'blue',
    motion: { arrow: 'right', hint: 'pole tip out' },
  },
  {
    visualLane: VisualLane.Gesture1,
    id: 'wrist_supination',
    label: 'Tip in',
    colorName: 'amber',
    motion: { arrow: 'left', hint: 'pole tip in' },
  },
  {
    visualLane: VisualLane.Gesture2,
    id: 'wrist_radial_deviation',
    label: 'Tip forward',
    colorName: 'green',
    motion: { arrow: 'up', hint: 'pole tip forward' },
  },
  {
    visualLane: VisualLane.Gesture3,
    id: 'wrist_ulnar_deviation',
    label: 'Tip back',
    colorName: 'purple',
    motion: { arrow: 'down', hint: 'pole tip back' },
  },
  {
    visualLane: VisualLane.Gesture4,
    id: 'thumb_extension',
    label: 'Lift thumb',
    colorName: 'pink',
    motion: null,
  },
];

export const calibrationSetupFixture: CalibrationSetupSnapshot = {
  phase: 'setup',
  tracks: [
    track,
    { ...track, id: 'short-track', title: 'Short Fixture', cue_count: 74, cue_shortfall: 56 },
  ],
  selected_track_id: track.id,
};

export const calibrationPlayingFixture: CalibrationPlayingSnapshot = {
  phase: 'playing',
  track,
  lanes,
  cues: semanticSchedule.map((semanticColumn, index) => ({
    visualLane: lanes[semanticColumn % 5]!.visualLane,
    at: 4_000 + index * 2_100,
    hold: 1_500,
    thumbVariant: semanticColumn < 5 ? ThumbVariant.Up : ThumbVariant.Down,
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
    invalid: lane.visualLane === VisualLane.Gesture3 ? 2 : 0,
  })),
};

export const calibrationSongEndFixture: CalibrationSongEndSnapshot = {
  phase: 'between_songs',
  track_title: track.title,
  tracks: calibrationSetupFixture.tracks,
  selected_track_id: track.id,
  candidate_available: true,
  continue_available: true,
  valid_reps: 124,
  invalid_reps: 6,
  deficits: ['Tip back, thumb down: 14/16'],
};

export const calibrationTechnicalFailureFixture: CalibrationTechnicalFailureSnapshot = {
  phase: 'technical_failure',
  detail: 'Fitting stopped before a structurally valid candidate was produced.',
};
