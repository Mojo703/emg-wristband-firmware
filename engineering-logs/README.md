# Engineering logs

Why the machine learning and the on-device code turned out the way they did. Each
entry settles one question with a measurement, so a later reader can tell a tested
decision from a guess. Read the entry before reopening the question it closed.

What the entries are about has moved with the project. 0001 through 0004 cut int8
forward-pass latency on the ESP32-S3 and found the kernel floor. 0005 through 0014
went after accuracy: encoder architecture, augmentation, per-user calibration, and
how to handle the negative class. 0015 onward cover the int8 export, the input
conditioning that feeds it, and the timing of the acquisition path.

Each entry: **Goal → Method → Hypothesis → Measured → Analysis → Next steps.**
Every number came from a run. Each entry names the hardware, dataset, and build it
measured, because those changed underneath the series more than once.

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
| [0013](0013-improved-negative-class-training.md) | Improved negative training: grouping vs outlier-exposure vs orthogonal | **grouping wins (AUROC 0.61→0.70)**; outlier-exposure noisy-positive, orthogonal inert; no combination helps |
| [0014](0014-emg2pose-false-activation-mining.md) | Mining false activations on unlabeled emg2pose | model fires on 81% of emg2pose; pronation dominance is a label-space sink, not a bad command — no wrist swap helps; fires unverified (no ground truth) |
| [0015](0015-emg-tds-int8-on-device.md) | In-repo int8 export of `emg-tds` + `ml-bench` retarget | shipping model exports from the current checkpoint; blob format + device runtime updated. `ml-bench` has since been deleted: the runtime is `emg-runtime`, the blob lives in `emg-runtime/data/`, and the checks run as `opal-firmware` device tests |
| [0016](0016-input-conditioning-units-and-direct-current.md) | Input conditioning: the unit error and DC blocking | `input_scale` is normalised units, not µV — DC blocker + amplitude normaliser added before int8; worst correlation 0.996, amplitude error 5.3% |
| [0017](0017-missed-edges-and-honest-gaps.md) | Missed DRDY edges: scheduling, clocks, honest gaps | miss rate 5-6% → <1% (core plan + 8 MHz frame reads); residual gaps on the wire as `Frame::Emg::missing`; telemetry frames replace status logs |
| [0018](0018-the-frame-read-moved-into-the-interrupt.md) | The frame read moved into the DRDY interrupt | 30 min soak: 0 of 7.2M conversions missed, corrupt frames 80/min → 0; IRAM handler owns the bus, driver keeps bring-up/recovery |
| [0019](0019-what-the-band-actually-records.md) | What the band actually records | EMG present at 14-20 µV under a 7-71 µV floor; gain is 6 not 24 and the wire was clipping at ±100 mV; mains guard must be ±12 Hz; the array does not resolve space |
| [0020](0020-why-nine-sessions-recorded-nothing.md) | Why nine sessions recorded nothing | All nine NOT USABLE; six failed on a floor 2-20x the limit because the arm has no reference, three had clean contact and almost no cues; RLDINV traced to a header pin, fix is one wire from J4.1 to BIAS_DRV |
| [0021](0021-what-survives-a-correct-fold.md) | What survives a correct fold | 0020's NOT USABLE overturned: 56-65% five-way within session, offset labels and railing patterns at the null; but a >500 Hz control where EMG cannot exist reaches 50-52%, so the signal is not cleanly myoelectric; 2 commands reach 83-100%; detection collapses under a block-held-out fold |
