# Opal 269118 calibration artifacts — 2026-09-25

Device MAC observed while reading: `24:ec:4a:26:91:18`. Wearer: Matthew.
Recorded with the dashboard firmware at `3ca4ef1` for the informal demo build
(`demo-ble-firmware`), which reads the newest resident slot.

## Files

### `training-partition-after-recalibration.bin`

- Origin: exact 983,040-byte device training-partition read after both runs.
- SHA-256:
  `4e90ad3b6af10aba024f9671d311352ff0424c700b55b9de2d1a87f20893e5a9`
- Prior hash `8c4e05fa`.
- Slot 0: resident, sequence 7. Slot 1: resident, sequence 8. The newest
  resident, sequence 8, is the one the firmware loads.

### `opal-269118-calibration-7.opal-slot.bin`

- Origin: dashboard resident-slot export, 196,608 bytes. Identical to slot 0 of
  the partition above.
- SHA-256:
  `4a8840906b5effd25333ef96c08e05c5852b8791a2b01a186731052f8961aa27`
- Sequence 7, 10 classes, prior hash `8c4e05fa`.

### `opal-269118-calibration-8.opal-slot.bin`

- Origin: dashboard resident-slot export, 196,608 bytes. Identical to slot 1 of
  the partition above.
- SHA-256:
  `cdd1c99eb04655aa03758b39fba098df3acbeaeecffb1372bf06b967407a87f1`
- Sequence 8, 10 classes, prior hash `8c4e05fa`.
- Wearer's judgement: the best-performing calibration of the session.
