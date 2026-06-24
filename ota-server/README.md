# ota-server

A minimal `axum` server that hosts firmware images for [`ota-client`](../ota-client)
to download. It runs on your dev machine and needs no ESP toolchain. It is plain
`x86_64` Rust.

```sh
cargo run                                 # binds 0.0.0.0:8080
OTA_SERVER_ADDR=0.0.0.0:9000 cargo run    # custom bind address
```

Drop firmware images into `./firmware/`; they are served at
`http://<host>:8080/firmware/<name>.bin`. Produce an image with `espflash
save-image`; see [`../ota-client/README.md`](../ota-client/README.md). Find the
address the ESP32 should hit (a LAN IP, not `127.0.0.1`) with `ip -4 addr show |
grep inet`, and put it into `../ota-client/cfg.toml` as `ota_url`.

## Firewall

The server binds all interfaces, but a host firewall blocks the device's inbound
connection even while your own local `curl` to the same address succeeds. On Arch,
firewalld is active by default, and the device then sees a connect timeout rather
than a response. Open the port:

```sh
sudo firewall-cmd --add-port=8080/tcp     # runtime rule, clears on reboot
sudo firewall-cmd --list-ports            # confirm
```

Verify from another device on the same WiFi: `curl http://<dev-ip>:8080/` should
return the liveness line.
