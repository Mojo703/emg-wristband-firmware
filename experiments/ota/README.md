# OTA experiment

This directory archives the verified over-the-air update proof:

- [`client/`](client) is the ESP32-S3 dual-slot update client.
- [`server/`](server) is the host HTTP image server used by the client.

The proof was verified on hardware from client v1.0.0 in `ota_0` to v1.0.1 in
`ota_1` without reflashing. It is deferred experimental work, not integrated
Opal functionality: `opal-firmware` still uses its own non-OTA flash layout and
does not call this client. See the client README for the preserved verification
procedure and [`../../FIRMWARE-SETUP.md`](../../FIRMWARE-SETUP.md) for the active
shared firmware toolchain setup.
