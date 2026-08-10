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
  calibrationTechnicalFailureFixture,
} from './guidedCalibrationFixtures.ts';

function songEnd(
  candidateAvailable: boolean,
  continueAvailable: boolean,
): CalibrationSongEndSnapshot {
  return {
    phase: 'between_songs',
    track_title: 'Calibration Song',
    candidate_available: candidateAvailable,
    continue_available: continueAvailable,
    valid_reps: 94,
    invalid_reps: 6,
    deficits: ['Tip back, thumb down: 14/16'],
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
  assert.match(announcements.current, /^Current cue: .+, thumb (up|down)$/);
  assert.match(announcements.next, /^Next cue: .+, thumb (up|down)$/);
});

test('fixtures cover every guided calibration presentation phase', () => {
  assert.deepEqual(
    [
      calibrationSetupFixture.phase,
      calibrationPlayingFixture.phase,
      calibrationSongEndFixture.phase,
      calibrationTechnicalFailureFixture.phase,
    ],
    ['setup', 'playing', 'between_songs', 'technical_failure'],
  );
  assert.equal(calibrationPlayingFixture.lanes.length, 5);
  assert.equal(calibrationPlayingFixture.cues.length, 130);
  assert.deepEqual(
    calibrationPlayingFixture.cues.slice(0, 10).map((cue) => [cue.visualLane, cue.thumbVariant]),
    [
      [0, 'up'], [0, 'down'], [1, 'up'], [1, 'down'], [2, 'up'],
      [2, 'down'], [3, 'up'], [3, 'down'], [4, 'up'], [4, 'down'],
    ],
  );
  assert.deepEqual(
    calibrationPlayingFixture.cues.slice(100, 105).map((cue) => [cue.visualLane, cue.thumbVariant]),
    [[0, 'down'], [1, 'down'], [2, 'down'], [3, 'down'], [4, 'down']],
  );
  assert.deepEqual(
    new Set(calibrationPlayingFixture.cues.map((cue) => cue.thumbVariant)),
    new Set(['up', 'down']),
  );
});
