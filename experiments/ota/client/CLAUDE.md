# CLAUDE.md: OTA client experiment

Deferred ESP32-S3 OTA update client proof (`std` / `esp-idf-svc`). It is not
integrated into Opal. See `README.md` for the boot flow, flash layout, and staged
end-to-end verification; shared toolchain setup lives at
`../../../FIRMWARE-SETUP.md`.

Agent notes:

- Firmware. It builds and flashes to hardware over USB-Serial-JTAG; you cannot run
  it on this host.
- The flash runner must pass `--partition-table partitions.csv`. Without it
  espflash writes a default single-`factory` table and there are no OTA slots.
- Two behaviours look like bugs but are intentional: the client updates on every
  boot (so identical slots ping-pong), and `wifi::connect(...)?` exits on failure
  rather than retrying. Don't "fix" them silently; they are known next-hardening
  steps.
- The esp-rs version pins (`esp-idf-svc 0.52`, ESP-IDF `v5.2.2`) are deliberate;
  older versions hit `c_char` signedness errors. Don't bump without testing.
