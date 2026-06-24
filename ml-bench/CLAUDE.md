# CLAUDE.md: ml-bench

ESP32-S3 latency/throughput benchmark for the int8 gesture encoder, and the home of
the hand-written ESP32-S3 SIMD kernels. See `README.md` for measured numbers, the
file layout, and how the kernels work.

Agent notes:

- Firmware. Always run `cargo run --release`; debug numbers are meaningless.
- The startup SIMD self-test (bit-exact vs a scalar oracle) is the correctness
  gate. If you change a kernel in `src/mac.rs` or `src/layers.rs`, that self-test
  must still pass on-device before anything else matters.
- Weights are synthetic on purpose: latency depends on tensor shapes, not values,
  so the benchmark runs before any model export.
- Before re-trying a kernel idea, read `../engineering-logs/` entries 0001 through
  0004 for the dead ends.
