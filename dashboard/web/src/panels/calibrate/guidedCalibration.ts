import type {
  GestureMotion,
  GuidedCalibrationSnapshot as WireGuidedCalibrationSnapshot,
  UnixMilliseconds,
} from '../../lib/protocol';
import type { CuePresentation, VisualLane } from '../../lib/collect/field';
import type { CalibrationTrack } from './calibrationTracks';

export interface CalibrationLane {
  readonly visualLane: VisualLane;
  readonly id: string;
  readonly label: string;
  readonly colorName: string;
  readonly motion: GestureMotion | null;
}

export type FiveCalibrationLanes = readonly [
  CalibrationLane,
  CalibrationLane,
  CalibrationLane,
  CalibrationLane,
  CalibrationLane,
];

export interface CalibrationCount {
  readonly class_id: string;
  readonly label: string;
  readonly thumb_up: number;
  readonly thumb_down: number;
  readonly invalid: number;
}

export interface CalibrationSetupSnapshot {
  readonly phase: 'setup';
  readonly tracks: readonly CalibrationTrack[];
  readonly selected_track_id: string | null;
}

export interface CalibrationPlayingSnapshot {
  readonly phase: 'playing';
  readonly track: CalibrationTrack;
  readonly lanes: FiveCalibrationLanes;
  readonly cues: readonly CuePresentation[];
  readonly position_ms: number;
  readonly position_observed_at_unix_ms: UnixMilliseconds;
  readonly valid_reps: number;
  readonly invalid_reps: number;
  readonly paused_reason: string | null;
  readonly counts: readonly CalibrationCount[];
}

export interface CalibrationPreparingSnapshot {
  readonly phase: 'preparing';
  readonly track: CalibrationTrack;
  readonly stage: 'stillness' | 'gain_estimation' | 'ready_for_schedule';
  readonly elapsed_milliseconds: number;
  readonly remaining_milliseconds: number;
}

export interface CalibrationSongEndSnapshot {
  readonly phase: 'between_songs';
  readonly track_title: string;
  readonly tracks: readonly CalibrationTrack[];
  readonly selected_track_id: string | null;
  readonly candidate_available: boolean;
  readonly continue_available: boolean;
  readonly valid_reps: number;
  readonly invalid_reps: number;
  readonly deficits: readonly string[];
}

export interface CalibrationTechnicalFailureSnapshot {
  readonly phase: 'technical_failure';
  readonly detail: string;
}

export interface CalibrationExitingSnapshot {
  readonly phase: 'exiting';
  readonly detail: string;
}

export type GuidedCalibrationSnapshot =
  | CalibrationSetupSnapshot
  | CalibrationPreparingSnapshot
  | CalibrationPlayingSnapshot
  | CalibrationSongEndSnapshot
  | CalibrationExitingSnapshot
  | CalibrationTechnicalFailureSnapshot;

export type CalibrationSongEndAction = 'save' | 'continue' | 'discard';

/** Convert wire naming to the neutral playfield's presentation naming without
 * deriving any session state. The protocol guard has already proved five lanes
 * in visual order before this function runs. */
export function presentGuidedCalibration(
  snapshot: WireGuidedCalibrationSnapshot,
): GuidedCalibrationSnapshot {
  if (snapshot.phase !== 'playing') return snapshot;
  const lane = (index: number): CalibrationLane => {
    const value = snapshot.lanes[index]!;
    return {
      visualLane: value.visual_lane,
      id: value.id,
      label: value.label,
      colorName: value.color_name,
      motion: value.motion,
    };
  };
  return {
    ...snapshot,
    lanes: [lane(0), lane(1), lane(2), lane(3), lane(4)],
    cues: snapshot.cues.map((cue) => ({
      visualLane: cue.visual_lane,
      at: cue.at,
      hold: cue.hold,
      thumbVariant: cue.thumb_variant,
    })),
  };
}

export interface CalibrationSongEndActionState {
  readonly action: CalibrationSongEndAction;
  readonly disabled: boolean;
}

/** Button availability is part of the backend snapshot. The browser does not
 * infer candidate quality, deficits, or whether another song can add rows. */
export function songEndActions(
  snapshot: CalibrationSongEndSnapshot,
): readonly CalibrationSongEndActionState[] {
  return [
    { action: 'save', disabled: !snapshot.candidate_available },
    { action: 'continue', disabled: !snapshot.continue_available },
    { action: 'discard', disabled: false },
  ];
}

export interface CueAnnouncements {
  readonly current: string;
  readonly next: string;
}

/** Accessible narration of the same authoritative cues painted on the canvas. */
export function cueAnnouncements(snapshot: CalibrationPlayingSnapshot): CueAnnouncements {
  const describe = (prefix: string, cue: CuePresentation | undefined): string => {
    if (cue === undefined) return `${prefix}: none`;
    const lane = snapshot.lanes.find((candidate) => candidate.visualLane === cue.visualLane);
    const label = lane?.label ?? 'Unknown gesture';
    return `${prefix}: ${label}, thumb ${cue.thumbVariant}`;
  };
  const current = snapshot.cues.find(
    (cue) => cue.at <= snapshot.position_ms && snapshot.position_ms <= cue.at + cue.hold,
  );
  const next = snapshot.cues.find((cue) => cue.at > snapshot.position_ms);
  return {
    current: describe('Current cue', current),
    next: describe('Next cue', next),
  };
}
