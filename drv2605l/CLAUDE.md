# CLAUDE.md: drv2605l

This crate is a reusable, `no_std` DRV2605L driver. The standalone ESP32-S3
hardware bench is the named `drv2605l-bench` example.

Agent notes:

- Keep ESP-IDF code in `examples/drv2605l-bench.rs`. The library depends only on
  `embedded-hal`; production consumers disable the `bench` feature.
- Run host tests from the repository root so `drv2605l/.cargo/config.toml` does
  not select the Xtensa target:

  ```sh
  cargo +stable test --manifest-path drv2605l/Cargo.toml --no-default-features
  ```

- Compile the hardware bench from this crate after loading the ESP environment:

  ```sh
  . ~/export-esp.sh
  cargo build --release --example drv2605l-bench
  ```

- Flash and monitor with `cargo run --release --example drv2605l-bench`. Always
  name the example; this library package has no default binary.
