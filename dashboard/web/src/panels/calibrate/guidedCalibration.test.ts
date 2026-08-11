import assert from 'node:assert/strict';
import test from 'node:test';

import {
  cueAnnouncements,
  songEndActions,
  type CalibrationSongEndSnapshot,
} from './guidedCalibration.ts';
import {
  calibrationPlayingFixture,
  calibrationSetupFixture,
  calibrationSongEndFixture,
  calibrationFinalizingFixture,
  calibrationTechnicalFailureFixture,
} from './guidedCalibrationFixtures.ts';
import { isGuidedCalibrationSnapshot } from '../../lib/protocol.ts';

function songEnd(
  candidateAvailable: boolean,
  continueAvailable: boolean,
): CalibrationSongEndSnapshot {
  return {
    phase: 'between_songs',
    track_title: 'Calibration Song',
    tracks: calibrationSetupFixture.tracks,
    selected_track_id: 'fixture-track',
    candidate_available: candidateAvailable,
    continue_available: continueAvailable,
    valid_reps: 94,
    invalid_reps: 6,
    deficits: ['Radial, hard grip: 4/5'],
  };
}

test('song-end actions are projected only from backend availability flags', () => {
  assert.deepEqual(songEndActions(songEnd(true, true)), [
    { action: 'save', disabled: false },
    { action: 'continue', disabled: false },
    { action: 'discard', disabled: false },
  ]);
  assert.deepEqual(songEndActions(songEnd(false, false)), [
    { action: 'save', disabled: true },
    { action: 'continue', disabled: true },
    { action: 'discard', disabled: false },
  ]);
});

test('playing snapshot narrates current and next cues without reading canvas state', () => {
  const announcements = cueAnnouncements(calibrationPlayingFixture);
  assert.match(announcements.current, /^Current cue: .+, (extend index and middle fingers|soft grip|medium grip|hard grip)$/);
  assert.match(announcements.next, /^Next cue: .+, (extend index and middle fingers|soft grip|medium grip|hard grip)$/);
});

test('golden grip variants are narrated as vertical anti poses', () => {
  const center = calibrationPlayingFixture.cues.find((cue) => cue.thumbVariant === 'grip_soft');
  assert.ok(center);
  const announcements = cueAnnouncements({
    ...calibrationPlayingFixture,
    position_ms: center.at,
  });
  assert.match(announcements.current, /Pole vertical, soft grip/);
});

test('fixtures cover every guided calibration presentation phase', () => {
  assert.deepEqual(
    [
      calibrationSetupFixture.phase,
      calibrationPlayingFixture.phase,
      calibrationSongEndFixture.phase,
      calibrationFinalizingFixture.phase,
      calibrationTechnicalFailureFixture.phase,
    ],
    ['setup', 'playing', 'between_songs', 'finalizing', 'technical_failure'],
  );
  assert.equal(calibrationPlayingFixture.lanes.length, 3);
  assert.equal(calibrationPlayingFixture.cues.length, 52);
  assert.deepEqual(
    calibrationPlayingFixture.cues.slice(0, 6).map((cue) => [cue.visualLane, cue.thumbVariant]),
    [
      [0, 'command'], [2, 'grip_soft'], [2, 'grip_medium'], [2, 'grip_hard'],
      [1, 'command'], [2, 'grip_soft'],
    ],
  );
  assert.deepEqual(
    new Set(calibrationPlayingFixture.cues.map((cue) => cue.thumbVariant)),
    new Set(['command', 'grip_soft', 'grip_medium', 'grip_hard']),
  );
});

test('finalizing wire snapshots require a user-facing progress detail', () => {
  assert.equal(isGuidedCalibrationSnapshot(calibrationFinalizingFixture), true);
  assert.equal(isGuidedCalibrationSnapshot({ phase: 'finalizing' }), false);
});

test('calibration lanes carry the same user-facing Collect presentation', () => {
  assert.deepEqual(
    calibrationPlayingFixture.lanes.map(({ id, label, colorName, motion }) => ({
      id,
      label,
      colorName,
      arrow: motion?.arrow ?? null,
    })),
    [
      { id: 'wrist_radial_deviation', label: 'Tip forward', colorName: 'green', arrow: 'up' },
      { id: 'wrist_ulnar_deviation', label: 'Tip back', colorName: 'purple', arrow: 'down' },
      {
        id: 'center_counterexample',
        label: 'Pole vertical — do not trigger',
        colorName: 'gray',
        arrow: null,
      },
    ],
  );
});

test('playing wire snapshots require the source-time playhead pair', () => {
  const wire = {
    ...calibrationPlayingFixture,
    lanes: calibrationPlayingFixture.lanes.map((lane) => ({
      visual_lane: lane.visualLane,
      id: lane.id,
      label: lane.label,
      color_name: lane.colorName,
      motion: lane.motion,
    })),
    cues: calibrationPlayingFixture.cues.map((cue) => ({
      visual_lane: cue.visualLane,
      at: cue.at,
      hold: cue.hold,
      gesture:
        cue.cueLabel === 'Tip forward' ? 'wrist_radial_deviation' : 'wrist_ulnar_deviation',
      modifier: cue.thumbVariant,
    })),
  };
  assert.equal(isGuidedCalibrationSnapshot(wire), true);
  const { position_observed_at_unix_ms: _omitted, ...untimestamped } = wire;
  assert.equal(isGuidedCalibrationSnapshot(untimestamped), false);
  assert.equal(
    isGuidedCalibrationSnapshot({
      ...wire,
      lanes: [...wire.lanes, { ...wire.lanes[0], visual_lane: 3 }],
    }),
    false,
  );
  assert.equal(
    isGuidedCalibrationSnapshot({
      ...wire,
      cues: [{ ...wire.cues[0], visual_lane: 3 }, ...wire.cues.slice(1)],
    }),
    false,
  );
});
