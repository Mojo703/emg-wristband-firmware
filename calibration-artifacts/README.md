# Calibration artifacts

Local calibration binaries are organized by device and collection date:

```text
calibration-artifacts/<device-id>/<YYYY-MM-DD>/
```

Each dated directory has a tracked `MANIFEST.md` describing every file, its
provenance, format, model layout, hash, and known wearer conditions. Binary
payloads are intentionally ignored by Git because resident slots contain
wearer-derived packed EMG feature rows. Keep the directory in local backups; do
not assume a fresh clone contains the payloads.

Two binary formats appear here:

- **Full training partition:** 983,040 bytes (`0xF0000`), suitable for
  `playback-host inspect-partition` and flashing at `0x310000`. It contains the
  prior plus two physical resident/candidate slots.
- **Resident slot export:** 196,608 bytes (`0x30000`), downloaded through the
  dashboard. It contains one exact slot but not its matching prior. Preserve the
  prior hash and a compatible full partition alongside it.

Slot exports preserve accepted quantized feature rows and labels, reference
gains, fitted standardization, centroids, spreads, weights, sequence, role, and
CRC. They do not preserve raw EMG, rejected cues, gain-estimation samples, or
cue timing, and therefore cannot be used to recompute another frequency-band
architecture.
