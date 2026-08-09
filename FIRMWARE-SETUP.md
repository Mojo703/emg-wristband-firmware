# Firmware setup

The ESP32-S3 firmware crates use Espressif's Xtensa Rust toolchain. Keep the
repository at a path without spaces, then install the shared tools once:

```sh
cargo install espup --locked
espup install
cargo install espflash --locked
cargo install ldproxy --locked
```

Source the environment in every shell that builds firmware:

```sh
. ~/export-esp.sh
```

Build from the firmware crate's own directory. Its `rust-toolchain.toml` selects
the `esp` toolchain. `.cargo/config.toml` selects the ESP32-S3 target. The run
command in each crate's README flashes over USB-Serial-JTAG and opens the serial
monitor after building; named bench examples require `cargo run --example <name>`.

## Arch Linux libxml2 workaround

Espressif's bundled `esp-clang` expects `libxml2.so.2`. Current Arch provides
`libxml2.so.16`. The first build creates the crate's `.embuild/` directory. If
the build reports that `libxml2.so.2` is missing, run this from that crate:

```sh
ln -sf /usr/lib/libxml2.so.16 \
  .embuild/espressif/tools/esp-clang/*/esp-clang/lib/libxml2.so.2
```

Each crate owns its generated `.embuild/` checkout. Apply the workaround to every
affected crate. Deleting `.embuild/` also removes the symlink.
