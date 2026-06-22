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
| [0002](0002-pipelined-pointwise-mac.md) | Pipelined pointwise MAC (`accx.ld.ip`) | **17.07 ms p50 (−6.4%)** — current best |
| [0003](0003-qacc-multioutput-pointwise.md) | QACC multi-output pointwise | ❌ correct but +57% slower; reverted. ACCX wins reductions |
| [0004](0004-model-shrink-plan.md) | Model-shrink plan (overnight retrain) | PLAN — kernel floor reached; gains now need a smaller model |
| [0005](0005-sota-review-and-waveformer.md) | Accuracy reality check + SOTA review | pivot to accuracy; reimplement WaveFormer in Rust (`../waveformer/`) |
| [0006](0006-waveformer-rust-port.md) | WaveFormer Rust/candle port | architecture + forward pass OK (2.50M params); data/training next |
| [0007](0007-waveformer-training-gpu-and-pose-pretrain.md) | Training: GPU unblock + pose-pretrain + zero-init fix | CUDA 13.3 fixed; model now learns (was collapsing to mean); finetune numbers pending |
| [0008](0008-waveformer-dead-end-emg2pose-and-tds.md) | WaveFormer dead end → TDS-conv encoder | WaveFormer shelved; DS-conv encoder reaches 80.7% cross-subject 5-class |
| [0009](0009-calibration-methods-tested.md) | Per-user calibration methods | marginal; shelved |
| [0010](0010-augmentation-tested.md) | Data augmentation | **warp σ=0.3 helps (+~4 pt, kills bad-seed collapse)**; rotate hurts |
| [0011](0011-pose-pretrain-transfer-tested.md) | Per-timestep pose pretrain transfer | transfers (+3.8 pt) but redundant with augmentation; augmentation wins |
| [0012](0012-negative-class-training-tested.md) | Negative-class training (false-positive rejection) | explicit reject learnable but pooled negatives too coarse: 60% leak, −10 pt recall |
