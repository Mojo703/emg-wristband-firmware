# emg-runtime ESP32-S3 tests

This package runs `emg-runtime`'s Xtensa SIMD checks on an ESP32-S3. Off-target
entry points use the scalar fallback, so host tests cannot exercise these kernels.
The package uses the rolling `esp` toolchain from `rust-toolchain.toml`; the
runtime library retains its separate Rust 1.77 minimum. It shares the firmware's
Cargo target directory because both packages use the same ESP-IDF release.

```sh
. ~/export-esp.sh
cargo test-device
```

The command builds the release profile, replaces the application on the attached
board, and opens the serial monitor. Exit after libtest reports the result. Restore
the wristband image afterward with `cargo run --release` from `opal-firmware/`.

For a compile-only check that does not invoke the runner:

```sh
cargo test --release --no-run
```
