import assert from 'node:assert/strict';
import test from 'node:test';

import { parseImportReport } from './trackLibrary.ts';

const valid = {
  id: 'track',
  title: 'Track',
  beats_per_minute: 120,
  duration_ms: 60_000,
  difficulty_file: 'Hard.dat',
  levels: [
    {
      name: 'hard',
      cue_count: 20,
      cues_per_second: 0.3,
      seconds_held: 30,
      column_balance: [4, 4, 3, 3, 3, 3],
    },
  ],
  calibration: {
    availability: 'available',
    cue_count: 20,
    content_identity: 'a'.repeat(64),
    cue_shortfall: 110,
  },
};

test('runtime validation accepts calibration availability metadata', () => {
  assert.deepEqual(parseImportReport(valid), valid);
});

test('runtime validation rejects stale and malformed import reports', () => {
  assert.throws(() => parseImportReport({ ...valid, calibration: undefined }));
  assert.throws(() =>
    parseImportReport({
      ...valid,
      calibration: { ...valid.calibration, cue_shortfall: undefined },
    }),
  );
  assert.throws(() =>
    parseImportReport({
      ...valid,
      calibration: { ...valid.calibration, cue_count: -1 },
    }),
  );
  assert.throws(() =>
    parseImportReport({
      ...valid,
      calibration: { ...valid.calibration, content_identity: 'not-a-hash' },
    }),
  );
});
