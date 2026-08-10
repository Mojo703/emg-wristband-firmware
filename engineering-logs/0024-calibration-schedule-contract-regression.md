# Calibration schedule contract regression

## Failure

Generator version 3 removed the 500 ms recovery interval when a cue changed only
the thumb modifier within one gesture. The ESP32 validator still required 500 ms
after every 1.5 s cue. A real authored schedule supplied 214 ms, so firmware
rejected the first eight-entry upload chunk with `RecoveryTooShort`.

The firmware reported that rejection as an uncorrelated calibration `BenchError`.
The exact-run dashboard actor deliberately ignores uncorrelated diagnostics, retried
the same invalid chunk, and eventually replaced the useful error with a chunk-ACK
timeout and an unavailable candidate.

## Why tests passed

The generator and its tests called the same `recovery_between` helper. The test
fixture was also constructed using that helper, so it proved only that the
generator reproduced its own rule. Firmware's independent `AnchoredSong` validator
was tested separately. No test uploaded a dashboard-generated `CalibrationTrack`
through that validator.

This was a boundary-contract failure hidden by two internally consistent unit-test
suites, followed by an error-correlation failure that obscured the evidence.

## Repair

- Hold and recovery durations now live in `protocol`, imported by both generator
  and validator.
- Generator version 4 restores 500 ms after every cue.
- Persisted version-3 products are regenerated in memory from retained source notes;
  track and audio files are not mutated during catalog load.
- A dashboard integration test uploads the generated track through
  `calibration_flow::AnchoredSong` in the production eight-entry chunk size.
- An exact active upload rejection now emits correlated `CalibrationRunFailed` and
  resets the failed lifecycle, so the dashboard displays the device's real reason
  immediately instead of waiting for an ACK timeout.

## Rule

A producer test may not substitute a copy of a consumer's validation rule. Every
durable cross-process product must pass the actual downstream validator in CI.
