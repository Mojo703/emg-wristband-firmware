# Parity fixtures

Everything the firmware bench compares itself against, and what the comparison
is allowed to tolerate.

The reference pipeline is the main checkout's `emg-tds/scripts` —
`score_requirements.py` for the filters, features and reject spine,
`band_learnability.py` for session loading, the per-chip reference and the time
map. Nothing here re-derives them. The arithmetic contract both halves of the
bench implement is `firmware-bench/ARITHMETIC.md`.

## The reproduced golden numbers

At no-op weight 0.4, the float64 reference path reproduces the thumb-modifier
evaluation exactly:

| measurement | result |
| --- | --- |
| modifier false negatives (measurement 3, seed-7 five-fold cue holdout) | 4/50 = 8.0% |
| modifier misclassification | 0/50 = 0.0% |
| same-don calibrated false fires (measurement 4b, seed-11 folds) | 3/80 = 3.8% |
| rest commits in both scored halves (measurement 5) | 0 and 0 |

The feature cache this bench builds is bit-identical to the cache the original
evaluation ran on, across all eight sessions and all of `cue_rows`, `rest_rows`,
`replay_rows`, `replay_starts`, `rest_at`, `cue_group`, `cue_spans` and
`rest_spans`, so the agreement is not a coincidence of two similar pipelines.

Every one of the four numbers is unchanged when the whole path runs through the
float32 device simulation, when the fit itself runs in float32, and when the
training rows are quantized to float16 or to int8.

## Regenerating

```
python3 host/prepare_feature_cache.py    # float64 reference features
python3 host/prepare_device_cache.py     # float32 device-path features
python3 host/reproduce_golden.py         # the four numbers
python3 host/precision_study.py          # every measurement below
python3 host/export_fixtures.py          # everything in this directory
python3 host/verify_fixtures.py          # the four numbers, from the fixtures alone
```

`verify_fixtures.py` reads only the session manifests and the expected commit
sequences, and scores them the way the requirement definitions do. It is the
check that matters: a firmware that reproduces the commit sequences reproduces
the numbers, without needing to agree on a single feature value.

## Recommended parity tolerances

**Do not assert per-feature bit equality, and do not assert a tight per-feature
absolute tolerance.** Two independently written, honest float32 implementations
of this filter cascade — the bench's C kernel and scipy's own float32 `lfilter`
and `sosfilt` — disagree by rms 1.8e-3 and up to 0.106 in log10 units. That is
the same magnitude as the float32-versus-float64 error itself, so a tolerance
tight enough to catch a real firmware bug on a single feature would fail on
correct firmware.

The cascade loses precision because the referenced signal carries a baseline of
order 150 mV while the passband content is order 10 µV, and the high-Q notches
sit right where that baseline lives. This is a property of the filter chain, not
a defect in any implementation of it; the referencing step is clean, at a
relative rms of 5e-8.

Recommended assertions, in the order they should be trusted:

| what | tolerance | why |
| --- | --- | --- |
| commit sequences (window index and command) | exact | identical across both float32 implementations on all four mission sessions |
| the four requirement numbers | exact | invariant under float32 features, float32 fit, float16 and int8 rows |
| features, rms over a session | ≤ 5e-3 log10 units | measured 1.4e-3 to 3.8e-3 against float64; implementation spread 1.8e-3 |
| features, 99.9th percentile | ≤ 5e-2 log10 units | measured 2.0e-2 to 4.0e-2 |
| features, single worst window | ≤ 0.4 log10 units | measured max 0.33; band 0 on high-amplitude channels dominates |
| referenced samples, relative rms | ≤ 1e-6 | measured 5e-8 to 9.6e-8 |
| filter coefficients | exact bits | designed in float64, rounded once; both halves of the bench already agree bit for bit |
| the six sampled feature windows in `expected_features.rs` | the feature tolerances above | provided as exact bits for convenience, not as an equality target |

If the firmware disagrees on commit sequences, the feature statistics are the
place to look: an rms within tolerance but a wildly out-of-tolerance maximum
points at one channel or one band, and band 0 on the highest-amplitude channels
is where float32 hurts most.

## Precision study results

Full output in `precision_study.txt`, machine-readable in `precision_study.json`.

**Feature deltas, float32 device path against float64 reference** (log10 units,
over every replay window of each mission session):

| session | rms | p99 | p99.9 | max |
| --- | --- | --- | --- | --- |
| 22-08-47 modifier | 1.44e-3 | 4.18e-3 | 2.03e-2 | 7.34e-2 |
| 22-16-46 same don | 3.79e-3 | 7.14e-3 | 3.55e-2 | 3.35e-1 |
| 21-22-54 rest static | 3.64e-3 | 1.46e-2 | 4.04e-2 | 1.74e-1 |
| 21-28-08 rest moving | 2.15e-3 | 6.16e-3 | 2.35e-2 | 2.64e-1 |

Per-band rms runs 1.4e-3 to 7.0e-3 in band 0 and falls to 4e-4 to 1.4e-3 in
band 3; the worst single features are almost always band 0.

**Fixed-gain referencing.** In float64, applying the sixteen published gains
reproduces `per_chip_reference` output bit for bit on all four sessions — the
device is not approximating the spec, it is computing the same thing with the
projection hoisted out. In float32 the same path lands within a relative rms of
4.7e-8 to 9.6e-8 of the float64 result.

**The four numbers through each variant.** Reference features with a float64
fit, device features with a float64 fit, device features with a float32 fit,
device features with float16 training rows, and device features with int8
training rows all give 4/50, 0/50, 3/80, and no rest commits.

**Quantization error on the feature rows** (log10 units): float16 costs rms
2.3e-4 to 5.2e-4, max 2.0e-3. int8 with the published constants costs rms 7.5e-3
to 9.4e-3. The int8 maximum reaches 1.0 on sessions whose replay features fall
outside the training matrix range and clip — which is why the constants are for
**stored calibration rows only**. Live replay features must stay in float32;
quantizing them would clip.

## Training matrices

Twelve classes throughout: five commands, five no-ops, static rest, moving rest.

| fit | rows | float32 | float16 | int8 |
| --- | --- | --- | --- | --- |
| measurement 3 fold (4/5 of modifier cues) | 9504 × 64 | 2376.0 KiB | 1188.0 KiB | 594.5 KiB |
| full-data weight-0.4 model | 9654 × 64 | 2413.5 KiB | 1206.8 KiB | 603.9 KiB |

The measurement 4b fold fits are the same shape as the full-data fit minus the
held-out fifth of the same-don cues. Rows per source, for the full-data fit:

| source | role | rows |
| --- | --- | --- |
| 22-08-47 modifier | command | 750 |
| 16-38-35 | no-op | 1200 |
| 16-51-04 | no-op | 1200 |
| 17-14-43 | no-op | 2652 |
| 17-23-35 | no-op | 1212 |
| 22-16-46 same don | no-op | 1200 |
| 21-22-54 rest static | rest | 720 |
| 21-28-08 rest moving | rest | 720 |

The int8 figure includes 512 bytes of shared per-feature offset and scale. Since
int8 costs nothing measurable in outcome, a device that must hold a calibration
matrix should hold it at one byte per feature.

## Fixture inventory

Bulk arrays are `.npy` and are not committed; manifests, coefficient tables and
the Rust-includable samples are small and are.

### Top level

| file | contents |
| --- | --- |
| `filter_coefficients.json` | the exact float32 biquad coefficients, notches then bands, decimal and `u32` bits. Written by the firmware side; `export_fixtures.py` re-derives them from scipy and refuses to run if they disagree |
| `feature_quantization.json` | per-feature `offset` and `scale` for the int8 calibration buffer, decimal and bits, taken from the full-data training matrix |
| `expected_commits.json` | per model, per replayed session, the commit list as `{window, sample, command}`. `window` indexes `replay_starts`; `sample` is the window's end sample, which is what the spine reports |
| `fold_memberships.json` | the cue groups in each fold, for seed 7 over the modifier session and seed 11 over the same-don session |
| `precision_study.json`, `precision_study.txt` | every measurement above |
| `cache/` | the feature caches, rebuilt by the prepare scripts, not committed |

### Per mission session, under `sessions/<session>/`

| file | contents |
| --- | --- |
| `manifest.json` | `scale_uv`, the sixteen reference gains as decimal and float32 bits, the raw stream's sha256 and exact interleaving, sample and record counts, class list, cue spans, rest spans, replay window starts, tau and the 3-of-3 needed count |
| `expected_features.npy` | float32, one row per 500-sample window, 64 features band-major then channel |
| `reference_features_float64.npy` | the same windows down the float64 reference path, for tolerance work |
| `expected_features_sample.json` | six windows — the first three and three spread through the session — as exact float32 bits |
| `expected_features.rs` | the same six windows as `pub const EXPECTED_FEATURES: [(usize, [u32; 64]); 6]`, rebuilt with `f32::from_bits`, for on-device unit tests |

The raw input the device receives is the session's own `emg.i16`, unmodified;
the manifest carries its digest rather than a copy. The file is a sequence of
500-sample window records, and within a record the sixteen channels appear in
slot order with their 500 samples in time order, so channel `c` sample `s` of
record `r` is element `r*16*500 + c*500 + s`. Concatenating records in order
gives the per-channel session stream. **The device must assume this order**, and
it is not the interleaving the name "interleaved" would suggest: channels are
blocked within a record, not interleaved sample by sample.

### Per model, under `models/<model>/`

Eleven models: `full_data_weight_0.4`, `measurement3_fold0` through `fold4`, and
`measurement4b_fold0` through `fold4`. All are fit in float32 on device-path
features at no-op weight 0.4.

| file | contents |
| --- | --- |
| `model.json` | class count, weight shape, training row count and per-source breakdown, held-out cue groups and fold seed, and the weights, standardization mean and deviation as exact float32 bits |
| `weights.npy` | float32, 65 × 12. Row 64 is the bias: `logits = standardized @ weights[:64] + weights[64]` |
| `standardization_mean.npy`, `standardization_deviation.npy` | float32, 64 each. Population deviation, floored at 1e-8 |
| `training_rows.npy` | float32, the raw features the device would buffer |
| `standardized_training_rows.npy` | the same rows after `(x - mean) / deviation` |
| `training_labels.npy` | int32, 0-4 commands, 5-9 no-ops, 10 static rest, 11 moving rest |
| `row_weights.npy` | float64, `class_scale[label] / class_count[label]`. The fit rescales these to sum to the row count before the first step |

## Conformance to ARITHMETIC.md

The float32 simulation follows the contract clause by clause: scaling and the
fixed-gain chip reference with a sequential seven-slot sum in slot order; seven
notches in sequence then four bandpass cascades in parallel off the notch
output, every biquad direct-form II transposed with the update the contract
spells out; filter state in float32, zeroed once per session and never reset
within one; sequential sum of squares per 125-sample quarter, `log10f(sum /
125.0 + 1e-12)`, feature `(q0 + q1 + q2 + q3) / 4.0`; features band-major and
channel-minor; float32 mean and population deviation floored at 1e-8; the 250
step fit at learning rate 1.0 and penalty 1e-2 with row weights normalized once
before the loop and weights starting at zero.

The bandpass section count is read from the coefficient fixture, never assumed —
the simulation applies whatever `butter(4, band, output="sos")` returns, which is
four sections per band.

Two details settled by measurement rather than by reading:

- **Replay softmax precision.** The simulation originally computed the replay
  softmax in float64 even on the device path, which the contract does not allow.
  It now computes it in float32 with a max-subtracted exponential. No commit
  sequence and no requirement number moved.
- **int8 rounding convention.** The contract says `round` without fixing the
  behaviour on exact halves, where numpy rounds to even and Rust's `f32::round`
  rounds away from zero. On the golden training matrix the two agree on all
  617856 codes, so nothing here depends on the choice — but the firmware should
  still pick one deliberately, since the tie is data-dependent. No code clips
  either: the range is -97 to 127, so the published constants cover the matrix.

## Known divergence sources

Ranked by how much they move a feature value.

1. **Biquad accumulation order.** The dominant term, worth up to 0.1 log10 units
   on a single feature. Any two float32 implementations differ here.
2. **`log10f`.** The bench's C kernel calls libm; numpy's `log10` rounds
   differently, at most an ulp — 4.8e-7 in the measured worst case.
3. **Dot product order in the fit and at replay.** The bench's float32 fit uses
   numpy's blocked matrix multiply, not the sequential 65-term dot product
   ARITHMETIC.md specifies. It changes no reported number, but a firmware fit
   that reproduces the bench's weights bit for bit would be a surprise rather
   than a requirement.
4. **Sequential versus pairwise summation** in the feature accumulator. The
   bench sums sequentially, matching a C loop; numpy's default pairwise
   summation would not.
