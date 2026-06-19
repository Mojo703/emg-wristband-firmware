# ota-server

Minimal `axum` server that hosts firmware images for the OTA client to download.
Runs on your dev machine; no ESP toolchain needed (plain `x86_64` Rust).

## Run

```sh
cargo run                         # binds 0.0.0.0:8080
OTA_SERVER_ADDR=0.0.0.0:9000 cargo run   # custom port
```

Drop firmware images in `./firmware/`. They're served at
`http://<host>:8080/firmware/<name>.bin`.

Find the IP the ESP32 should hit (use a LAN IP, not 127.0.0.1):

```sh
ip -4 addr show | grep inet
```

Put that IP + port into `../ota-client/cfg.toml` (`ota_url`).

## Firewall

The server binds all interfaces, but a host firewall will block the device's
inbound connection while your own `curl` to the same address still works (it's
local). On Arch, firewalld is active by default; the device then sees a connect
timeout rather than a response. Open the port:

```sh
sudo firewall-cmd --add-port=8080/tcp     # runtime rule, clears on reboot
sudo firewall-cmd --list-ports            # confirm 8080/tcp
```

Verify reachability from another device on the same WiFi:
`curl http://<dev-ip>:8080/` should return the liveness line.
