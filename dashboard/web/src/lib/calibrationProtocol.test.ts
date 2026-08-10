import assert from 'node:assert/strict';
import test from 'node:test';

import {
  asOffsetMilliseconds,
  asDurationMilliseconds,
  asTrackMilliseconds,
  asIncomingFrame,
  isCalibrationScheduleAcceptedFrame,
  isCalibrationCandidateStatusFrame,
  isCalibrationScheduleCommitDeferredFrame,
  isCalibrationScheduleChunkFrame,
  isCalibrationScheduleUploadAcknowledgedFrame,
  isOutgoingFrame,
  type CalibrationScheduleChunkFrame,
} from './protocol.ts';

const run = { session_id: 1, run_id: 2 };

test('replacement calibration browser mirrors accept 32-entry schedule chunks', () => {
  const frame: CalibrationScheduleChunkFrame = {
    type: 'calibration_schedule_chunk',
    run,
    schedule_revision: 1,
    content_identity: 'a'.repeat(64),
    total_count: 130,
    first_entry: 32,
    entries: Array.from({ length: 32 }, (_, index) => ({
      cue_id: index + 1,
      gesture: 'thumb_extension',
      modifier: index % 2 === 0 ? 'thumb_up' : 'thumb_down',
      track_offset: asTrackMilliseconds(index * 2000),
      hold: asDurationMilliseconds(1500),
    })),
  };

  // The backend, not the browser, owns schedule upload to the device.
  assert.equal(isOutgoingFrame(frame), false);
  assert.equal(isCalibrationScheduleChunkFrame(frame), true);
  assert.equal(isCalibrationScheduleChunkFrame(frame), true);
  assert.equal(
    isOutgoingFrame({ ...frame, entries: [...frame.entries, frame.entries[0]] }),
    false,
  );
  assert.equal(isOutgoingFrame({ ...frame, content_identity: '' }), false);
});

test('upload acknowledgements cannot represent a Begin with a chunk index', () => {
  const begin = {
    type: 'calibration_schedule_upload_acknowledged',
    acknowledgement: {
      run,
      schedule_revision: 1,
      content_identity: 'a'.repeat(64),
      total_count: 90,
      operation: { kind: 'begin', operation_fingerprint: 123 },
    },
  };
  assert.equal(isCalibrationScheduleUploadAcknowledgedFrame(begin), true);
  assert.equal(
    isCalibrationScheduleUploadAcknowledgedFrame({
      ...begin,
      acknowledgement: {
        ...begin.acknowledgement,
        operation: { kind: 'chunk', operation_fingerprint: 123 },
      },
    }),
    false,
  );
});

test('only typed device prerequisites authorize a schedule commit retry', () => {
  const deferred = {
    type: 'calibration_schedule_commit_deferred',
    deferred: { run, schedule_revision: 1, reason: 'preparation_incomplete' },
  };
  assert.equal(isCalibrationScheduleCommitDeferredFrame(deferred), true);
  assert.equal(
    isCalibrationScheduleCommitDeferredFrame({
      ...deferred,
      deferred: { ...deferred.deferred, reason: 'electrodes not making contact' },
    }),
    false,
  );
});

test('an absent calibration candidate cannot carry independent validity', () => {
  const absent = {
    type: 'calibration_candidate_status',
    candidate: { run, schedule_revision: 1, presence: { state: 'absent' } },
  };
  assert.equal(isCalibrationCandidateStatusFrame(absent), true);
  assert.equal(
    isCalibrationCandidateStatusFrame({
      ...absent,
      candidate: {
        ...absent.candidate,
        presence: {
          state: 'present',
          content_identity: 'a'.repeat(64),
          total_count: 90,
          validity: { model_numerically_valid: true, record_crc_valid: true },
        },
      },
    }),
    true,
  );
  assert.equal(
    isCalibrationCandidateStatusFrame({
      ...absent,
      candidate: {
        ...absent.candidate,
        presence: {
          state: 'absent',
          validity: { model_numerically_valid: false, record_crc_valid: false },
        },
      },
    }),
    false,
  );
  assert.equal(
    isCalibrationCandidateStatusFrame({
      ...absent,
      candidate: {
        ...absent.candidate,
        presence: {
          state: 'present',
          validity: { model_numerically_valid: true },
        },
      },
    }),
    false,
  );
});

test('replacement browser mirrors expose bounded integer timing state', () => {
  const status = asIncomingFrame({
    type: 'calibration_timing_status',
    status: {
      phase: {
        state: 'running',
        observation: {
          color: 'blue',
          color_elapsed_milliseconds: 499,
          anchor_device_monotonic_microseconds: 100,
          observed_device_monotonic_microseconds: 101,
        },
      },
      estimate: {
        availability: 'measured',
        automatic_offset_milliseconds: asOffsetMilliseconds(-2147483648),
        median_round_trip_milliseconds: asDurationMilliseconds(12),
        round_trip_spread_milliseconds: asDurationMilliseconds(4),
        sample_count: 11,
        capacity: 11,
      },
      manual_trim_milliseconds: asOffsetMilliseconds(-1000),
    },
  });
  assert.equal(status?.type, 'calibration_timing_status');
  assert.equal(
    asIncomingFrame({
      type: 'calibration_timing_status',
      status: {
        phase: { state: 'running' },
        estimate: { availability: 'no_samples', capacity: 11 },
        manual_trim_milliseconds: 1001,
      },
    }),
    null,
  );
});

test('timing wire guard rejects every formerly nullable invalid combination', () => {
  const valid = {
    type: 'calibration_timing_status',
    status: {
      phase: { state: 'stopped' },
      estimate: { availability: 'no_samples', capacity: 11 },
      manual_trim_milliseconds: 0,
    },
  };
  assert.equal(asIncomingFrame(valid)?.type, 'calibration_timing_status');

  const invalidStatuses = [
    { ...valid.status, phase: { state: 'running' } },
    { ...valid.status, phase: { state: 'stopping', last_observation: null } },
    { ...valid.status, phase: { state: 'error', detail: '', last_observation: null } },
    {
      ...valid.status,
      phase: {
        state: 'stopped',
        observation: {
          color: 'red',
          color_elapsed_milliseconds: 0,
          anchor_device_monotonic_microseconds: 1,
          observed_device_monotonic_microseconds: 1,
        },
      },
    },
    {
      ...valid.status,
      estimate: { availability: 'no_samples', sample_count: 0, capacity: 11 },
    },
    {
      ...valid.status,
      estimate: {
        availability: 'measured',
        automatic_offset_milliseconds: 0,
        median_round_trip_milliseconds: 0,
        round_trip_spread_milliseconds: 0,
        sample_count: 0,
        capacity: 11,
      },
    },
  ];
  for (const status of invalidStatuses) {
    assert.equal(asIncomingFrame({ type: 'calibration_timing_status', status }), null);
  }
});

test('schedule acceptance echoes whole-song identity and candidate state', () => {
  const accepted = {
    type: 'calibration_schedule_accepted',
    accepted: {
      run,
      schedule_revision: 3,
      content_identity: 'b'.repeat(64),
      acknowledged_device_monotonic_microseconds: 1_000,
      anchor_device_monotonic_microseconds: 3_001_000,
      acquisition_sample: 40_000,
    },
  };
  assert.equal(isCalibrationScheduleAcceptedFrame(accepted), true);
  assert.equal(asIncomingFrame(accepted)?.type, 'calibration_schedule_accepted');
  assert.equal(
    isOutgoingFrame({
      type: 'calibration_heartbeat',
      heartbeat: { run, schedule_revision: 3, sequence: 9 },
    }),
    false,
  );
  assert.equal(
    asIncomingFrame({
      type: 'calibration_song_result',
      result: {
        run,
        schedule_revision: 3,
        content_identity: 'b'.repeat(64),
        counts: [
          {
            gesture: 'thumb_extension',
            modifier: 'thumb_down',
            accepted_count: 16,
            rejected_count: 1,
            target_count: 16,
            deficit_count: 0,
          },
        ],
        validity: { model_numerically_valid: true, record_crc_valid: true },
      },
    })?.type,
    'calibration_song_result',
  );
});

test('device preparation status is a closed acquisition-authoritative phase union', () => {
  const settling = asIncomingFrame({
    type: 'calibration_preparation_status',
    status: {
      run,
      schedule_revision: 3,
      phase: {
        phase: 'settling',
        elapsed_milliseconds: 2500,
        remaining_milliseconds: 7500,
      },
    },
  });
  assert.equal(settling?.type, 'calibration_preparation_status');
  assert.equal(
    asIncomingFrame({
      type: 'calibration_preparation_status',
      status: {
        run,
        schedule_revision: 3,
        phase: { phase: 'settling', elapsed_milliseconds: -1, remaining_milliseconds: 1 },
      },
    }),
    null,
  );
});
