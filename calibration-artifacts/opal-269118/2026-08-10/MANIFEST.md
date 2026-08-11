# Opal 269118 calibration artifacts — 2026-08-10

Device MAC observed while flashing: `24:ec:4a:26:91:18`. Wearer: Matthew,
right wrist. These files span several model revisions and must not be mixed
without respecting each file's prior hash and class map.

## Files

### `164905_three-command-8class_full-partition.bin`

- Origin: exact 983,040-byte device training-partition read.
- SHA-256:
  `cd5981d9d464d45c6634b8f12320db54db2e6381e297dfcf3c2ad5416e0d0579`
- Prior: 5,193 rows, 8 classes, hash `19029904`.
- Resident: slot 0, sequence 1, 702 live rows, CRC valid; slot 1 erased.
- Model at the time: three command classes, three paired anti classes, and two
  prior-rest classes. A complete 78-cue run produced 702 live rows.
- Observation associated with this revision: C0 dominated in stream testing and
  C2 was difficult to activate. The exact glove/pole condition was not encoded
  in the original filename, so do not infer it from the binary.

### `184233_two-command-6class_full-partition.bin`

- Origin: exact 983,040-byte device training-partition read immediately before
  replacing the two-command model.
- SHA-256:
  `351e96ffd4697a0cb12d2c87765a504bf3364cccc90a38320e64ac8e33a50371`
- Prior: 3,934 rows, 6 classes, hash `3b3eb5bd`.
- Resident: slot 1, sequence 2, 468 live rows, CRC valid; slot 0 erased.
- Class map: radial command, ulnar command, one paired anti class for each
  command, static rest, moving rest.
- Live-row counts: 90 rows per command and 144 rows per paired anti class.
- Session context: the two-command runs introduced index and later index+middle
  extension while the remaining hand continued holding the pole. One run used a
  glove. Conversation timing associates this archive with that sequence, but
  the original binary did not encode a glove flag.
- Observation: substantially better command behavior than the three-command
  model, but squeezing or merely holding the pole could produce false fires.

### `191454_two-command-10class_center-grip_resident-slot.bin`

- Origin: byte-for-byte copy of the browser download
  `/home/matthewg/Downloads/opal-269118-calibration-1 pole glove right hand.opal-slot.bin`.
  The original Downloads file was left unchanged.
- Format: exact 196,608-byte resident-slot export, not a full partition.
- SHA-256:
  `938616a60f34e2d0a6b42f5d8232cb1d09dd9dd719fe6200197e94ec9b4aabd0`
- Header: format 2, sequence 1, role Resident, prior hash `8c4e05fa`,
  10 classes, 468 live rows, 72-byte row stride.
- Intended class map: two commands; radial/ulnar soft, medium, and hard grip
  negatives; static and moving rest.
- Actual wearer interpretation: every grip cue in the shared third visual lane
  was performed with the pole vertical at center. The six directional grip
  labels therefore duplicate three physical center-grip strengths.
- Wearer condition recorded in the downloaded filename: pole, glove, right
  hand.
- Observation: worked very well in stream testing except that pole vertical
  with index+middle extended could still activate a command.
- Product status: **golden**. On 2026-08-10 it was restored as the canonical
  firmware/dashboard/model contract rather than admitted through a compatibility
  path.
- Dependency: requires prior hash `8c4e05fa`. The reconstructed partition below
  preserves that prior together with this slot.

### `191454_two-command-10class_center-grip_reconstructed-partition.bin`

- Origin: derived, not read from the device. Regenerated the byte-identical
  10-class prior (`SHA-256 d6fa83516639de0185619dd7ae6af870816e1fba2b5b1d7216a660a4166f2a6e`)
  and placed the exact browser slot above at physical slot 0.
- SHA-256:
  `14dd5b0dc5990aaa70f7a725bb0d1b1376f4c36ad346e34d101406ac2a746cba`
- Inspection: 3,934 prior rows × 10 classes, prior hash `8c4e05fa`;
  slot 0 sequence 1 Resident with 468 rows and valid CRC; slot 1 erased.
- Purpose: self-contained future inspection and same-feature replay. Prefer the
  original slot file when proving what the browser downloaded; prefer this file
  when a tool requires a complete partition.
- Device restore verification: the complete 983,040-byte training-partition
  readback was byte-identical to this file. Runtime boot selected prior
  `8c4e05fa`, slot 0 sequence 1 Resident, 468 rows, CRC `03320e4a`.

### `final_two-command-8class_center-negatives_blank-partition.bin`

- Origin: generated prior and full flash image for the final center-negative
  design; full device readback was byte-identical before this copy was archived.
- SHA-256:
  `a5d58ab8f8cc00bd96056d2e0f5a1c2c1357838040eb8fdb558043d85788d93d`
- Prior: 1,440 historical rest rows × 8 classes, hash `ca001ab0`.
- Slots: both erased. This contains no wearer calibration yet.
- Class map: radial command, ulnar command, center index+middle extension anti,
  center soft grip, center medium grip, center hard grip, static rest, moving
  rest.
- Recipe: 52 cues and 468 eventual live rows: 90 + 90 command rows, 144 center
  extension rows, then 54/45/45 soft/medium/hard grip rows.
- Status: flashed and host-verified, then used for the failed wearer calibration
  archived below. Superseded by the restored golden 10-class partition.

### `204316_two-command-8class_center-extension-ulnar-collision_resident-slot.bin`

- Origin: byte-for-byte copy of the browser download
  `/home/matthewg/Downloads/opal-269118-calibration-1.opal-slot.bin`. The
  original Downloads file was left unchanged.
- Format: exact 196,608-byte resident-slot export.
- SHA-256:
  `51e063eb59cf8909427a9ed6dad34dfca51f69b1f09f55c33d58c2c73844f0e3`
- Header: format 2, sequence 1, role Resident, prior hash `ca001ab0`, 8
  classes, 468 live rows, 72-byte row stride, valid CRC.
- Live-row counts: radial command 90, ulnar command 90, center extension 144,
  center soft/medium/hard grip 54/45/45. Static and moving rest are prior-only.
- Wearer result: **dud**. Centered index+middle extension is too similar to
  ulnar deviation and suppresses the ulnar command.
- Diagnostic replay of the fitted weights over the exact accepted live rows:
  radial was top-ranked on 70/90 radial rows, while ulnar was top-ranked on
  only 10/90 ulnar rows; the center-extension negative was top-ranked on 71/90
  ulnar rows. The failure is present in the saved model itself and is not just
  an inference-stream threshold observation.

### `204316_two-command-8class_center-extension-ulnar-collision_reconstructed-partition.bin`

- Origin: derived, not read from the device. The exact slot above was overlaid
  at physical slot 0 in its matching archived blank partition.
- SHA-256:
  `ffa9212ac7033caf9da578e9f7c867351bd7f6da12de4665f34a9f0bd0e87a8e`
- Inspection: 1,440 prior rows x 8 classes, prior hash `ca001ab0`; slot 0
  sequence 1 Resident with 468 rows and valid CRC; slot 1 erased.
- Purpose: self-contained replay and inspection of the failed center-extension
  experiment. Preserve the resident-slot file above as the authoritative
  browser export.

## Validation commands

Full partitions can be inspected from the repository root with:

```sh
cargo run --quiet --manifest-path firmware-bench/playback-host/Cargo.toml -- \
  inspect-partition --image <full-partition.bin>
sha256sum <file.bin>
```

The resident slot was validated by overlaying it at slot-0 offset `0x90000` in
its regenerated matching blank partition. `inspect-partition` then reported the
whole sequence-1 resident and all 468 rows with a valid CRC.
