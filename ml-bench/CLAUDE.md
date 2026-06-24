# CLAUDE.md: ml-bench

ESP32-S3 latency/throughput benchmark for the int8 gesture classifier, and the
home of the hand-written ESP32-S3 SIMD kernels. See `README.md` for measured
numbers, the file layout, and how the kernels work.

Agent notes:

- Firmware. Always run `cargo run --release`; debug numbers are meaningless.
- The startup SIMD self-test (bit-exact vs a scalar oracle) is the correctness
  gate. If you change a kernel in `src/mac.rs` or `src/layers.rs`, that self-test
  must still pass on-device before anything else matters.
- Two model paths: `Model::synthetic()` (random weights, same emg-tds shapes for
  timing) and `Model::real()` (loads `data/model_int8.bin` and is verified against
  the embedded float logits on 32 real labeled windows). The blob now comes from
  `emg-tds export-int8`, not the retired `emg-gesture-class` path.
- `Model::real()` reads the architecture from the blob header (`kernel`, `stride`,
  block channels, `num_classes`). `Model::synthetic()` mirrors the current emg-tds
  shapes (k=25, 16→32→64→128→128, 500-sample window, 5-class head).
- Measured on hardware (ESP32-S3-Zero @ 240 MHz): 13.8 ms p50 on the 500-sample
  emg-tds window, 0.906 device top-1, 0.938 device/float agreement, ~35 KB model
  RAM, 279 KB free heap after load. Details are in `../ml-bench/results.txt` and
  `../engineering-logs/0015`.
- Before re-trying a kernel idea, read `../engineering-logs/` entries 0001 through
  0004 for the dead ends, and 0015 for the current model-export details.
