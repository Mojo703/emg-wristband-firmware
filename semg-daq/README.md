# semg-daq

ESP32-S3 firmware for sEMG data acquisition over two daisy-chained TI ADS1298
8-channel ADCs (16 channels total), connected over a shared SPI bus. Built on `std` / `esp-idf-svc`

```sh
. ~/export-esp.sh
cargo run        # builds, flashes, opens the serial monitor
```

The flash runner passes `--partition-table partitions.csv` (required, same
reason as `ota-client`).


