# Engineering logs — ml-bench inference optimization

Iterative log of the effort to cut int8 forward-pass latency on the ESP32-S3,
so the rest of the system (BLE HID, sampling ISR, 3-of-3 smoothing, wake-gate
state machine) fits inside the onset-to-action budget.

Each entry: **Goal → Method → Hypothesis → Measured → Analysis → Next steps.**
Numbers are p50 over 200 timed inferences, `--release`, 240 MHz, real int8 model
(`data/model_int8.bin`, 500-sample window), captured on hardware over
`/dev/ttyACM0`. Correctness gate: SIMD self-test bit-exact vs scalar oracle at
boot, plus cosine ≥ 0.90 vs the Python float32 reference.

## Constraints (from the project owner, 2026-06-21)

- **Accuracy:** anything goes as long as the prototype-matching eval bar holds.
- **Scope:** full stack permitted, but **avoid retraining** unless the gain is
  large; retrains can be batched into an overnight plan.
- **Stopping rule:** minimize until diminishing returns (<~5–10% per step), and
  document the wall.

## Index

| # | Title | Result |
|---|-------|--------|
| [0001](0001-baseline-and-bottleneck.md) | Baseline + bottleneck analysis | 18.23 ms p50; pointwise = 71.6% |
| [0002](0002-pipelined-pointwise-mac.md) | Pipelined pointwise MAC (`accx.ld.ip`) | 17.07 ms p50 (−6.4%); confirms overhead-bound |
