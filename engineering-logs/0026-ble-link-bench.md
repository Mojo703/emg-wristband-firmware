# 0026 — BLE link bench: what the radio grants, and the log line that cost 80% of it

**Date:** 2026-09-20
**Hardware:** bare ESP32-S3 (the Opal board) running `ble-media`'s
`ble-gatt-bench` example; ESP-IDF v5.2.2, esp32-nimble 0.12. Centrals: a Linux
desktop (kernel 7.2.6, BlueZ 5.87, MediaTek MT7922 adapter, Chrome with the
experimental web platform flag) and an iPhone running Bluefy.
**Code:** the link bench in `ble-media` (`link_bench.rs`, `nimble.rs`, the
`ble-gatt-bench` example) and the page in `phone-app/`.
**Data:** the page's copy-as-markdown tables and the band's serial log, both
pasted into the session; the desktop's `btmon` trace for one 2000-packet run.

## Goal

Semester 2 makes the band BLE only. The August demo proved the HID media keys
and nothing else. This entry measures the pieces nobody had run:

- a custom GATT service beside the bonded HID service on one connection;
- the phone-to-band write path;
- the connection parameters a real central grants;
- the write-to-notify round trip;
- the notification goodput ceiling.

The goodput bar is deliberately high. Raw 16-channel EMG at 2 kHz is 64 KB/s
as int16 and 96 KB/s as 24-bit codes. Embedding rows, the product's actual
payload, are about 2 KB/s.

## Method

The bench firmware keeps the HID remote unchanged and adds one service with
four characteristics. The central writes commands to the first. The second
echoes a command back from inside the write callback. The third streams
numbered packets as fast as the stack accepts. The fourth is a status record
the band notifies on change: encrypted and bonded flags, granted interval,
MTU, PHY each way, free heap, and stream counters. The service UUID rides in
the scan response, which leaves the primary advertisement iOS bonds against
untouched.

The page connects over Web Bluetooth. It runs sequential echo round trips and
streams of 2000 packets at 20, 100, 244 and 500 bytes. Each stream packet
carries the band's microsecond clock, so the page can show the band's send
span beside its own receive span. A stall is a notify call the stack refused
for want of a packet buffer. The band backs off one millisecond and retries.

## Hypothesis

The band would not be the limit. The central's interval and its packets per
connection event would set the ceiling. The 2M PHY with data length extension
would roughly double 1M goodput. The desktop would grant more than the phone.

## Measured

### The first ceiling was a log line

Every size that fit one link-layer frame arrived at 101 packets per second,
with zero stalls, at a 48.75 ms interval and at 7.5 ms, on the 1M PHY and on
2M. The band's send span equalled the receive span at 19.44 s for 2000 packets.
The pacing was therefore on the band. The `btmon` trace showed one
notification per 7.5 ms event with every fourth event skipped. That is a
9.7 ms handoff against a 7.5 ms schedule.

The handoff was NimBLE's info-level log. `ble_gatts_notify_custom` prints two
lines per notification from inside the call, on the sending thread. The
default console was UART0 at 115200 baud, on pins this board does not route.
About 90 bytes at that rate is 8 ms, plus the USB copy.

The fix is two lines in `sdkconfig.defaults`: `CONFIG_BT_NIMBLE_LOG_LEVEL_WARNING`
and the console on USB-Serial-JTAG. After it, the notify call averaged under
a millisecond and the controller was ready for the next packet within 44 µs.

| Desktop, 7.5 ms, 2M + long packets | Packets/s | KB/s | Stalls |
|---|---:|---:|---:|
| 244 B, log on | 101 | 24.7 | 0 |
| 244 B, log off | 532 | 129.9 | 1347 |

Zero stalls had hidden the throttle. When the controller is merely full,
NimBLE blocks the sending thread on a semaphore rather than refusing. The
stall counter only rises once the host's packet pool is empty. Stalls now
mean the producer outran the link. That is the state the bench wants, and
the reading to keep: a run with zero stalls and a low rate has something
upstream of the radio throttling it.

### Desktop sweeps after the fix

| Interval, PHY | 20 B | 100 B | 244 B | 500 B |
|---|---:|---:|---:|---:|
| 7.5 ms, 2M + long | 946/s, 18.9 KB/s | 800/s, 80.0 KB/s | 533/s, 130.1 KB/s | 199/s, 99.4 KB/s |
| 15 to 30 ms preset, 1M, 27-byte frames | 974/s, 19.5 KB/s | 317/s, 31.7 KB/s | 129/s, 31.5 KB/s | 63/s, 31.6 KB/s |
| 15 to 30 ms preset, 2M + long | 1007/s, 20.1 KB/s | 928/s, 92.8 KB/s | 623/s, 152.1 KB/s | 266/s, 132.8 KB/s |

The 1M row without long packets is flat in bytes across sizes. That is air
time. The 2M rows are flat in packets per event. That is the controller's
scheduling. The best packet on this adapter is 244 bytes, the largest
notification that fits one 251-byte frame. A 500-byte packet costs a second
frame, and the band spends 57% of its time inside the call.

| Round trip, 8 B, write without response, 200 tries | min | median | p95 | max | lost |
|---|---:|---:|---:|---:|---:|
| Desktop, 2M | 28.3 ms | 45.0 ms | 45.3 ms | 67.1 ms | 0 |
| iPhone, 2M | 50 ms | 60 ms | 63 ms | 90 ms | 0 |

### iPhone through Bluefy

The page found the band in Bluefy's chooser. iOS raised its pairing prompt
on connect, and the link encrypted with no trip through Settings. iOS
negotiated the MTU to 517 and the packet length on its own. It granted the 2M
PHY only when the band asked.

| iPhone | 20 B | 100 B | 244 B | 500 B |
|---|---:|---:|---:|---:|
| 1M, before the request | 821/s, 16 KB/s | 213/s, 21 KB/s | 84/s, 21 KB/s | 46/s, 23 KB/s |
| 2M + long packets | 560/s, 11 KB/s | 687/s, 69 KB/s | 473/s, 115 KB/s | 188/s, 94 KB/s |

Later the same day, with the band asking for its own link settings on every
encrypted connection: a fresh connection from the page came up at the full
rate with no button pressed, media keys sent from the page arrived on that
app-opened link, and keys pressed at the band's console arrived with the
phone locked. The interval iOS granted was not read off.

Discovery has a limit. After the first pairing, iOS reclaims the bonded
band as a HID device the moment it advertises, and a connected peripheral
goes silent. Bluefy's chooser is a scan, so it never lists the band again;
the page gets in only after the phone forgets it in Settings. Native
CoreBluetooth can list system-connected peripherals without scanning. Web
Bluetooth's door is `getDevices()`, and Bluefy has it: after a reload it
returned the granted band by name, then failed to connect with its own
error 2 while iOS held the link. Bluefy evidently reaches a device by
scanning, which a silent one defeats. Grants belong to an origin, so a
deploy to a fresh preview URL starts with none.

## Analysis

The hypothesis held once the log was out of the way. The iPhone takes
244-byte notifications at 115 KB/s. That is above the 96 KB/s that 24-bit raw
EMG would need, and 76% of the desktop's 152 KB/s. Embedding rows have
fifty-fold headroom. Raw EMG over BLE is possible on this phone. It stays a
bench-only option, because it leaves no margin for a weaker link and the
product does not need it.

Every gain above 21 KB/s on the phone came from the band asking for the 2M
PHY and long packets. Both requests die with the connection: a reconnect on
either central came back at 1M with 27-byte frames, and the desktop's flat
31.6 KB/s row is that state measured. Whatever owns the link must re-send
them after every connect. The 60 ms median
round trip is two events at a 30 ms interval, the interval iOS holds unless
asked. The 15 to 30 ms request is the one Apple's rules allow.

Pairing from the page is the demo path for the first connection: scan the
code, tap the band, accept the prompt. Every later connection is the problem.
iOS owns the bond, and the page needs a way to reach a peripheral that is
not advertising. That favours the Capacitor build, which can ask
CoreBluetooth for connected peripherals, over staying in Bluefy. The
manual must also name the bond-mismatch loop seen on the desktop. A central
holding a key the band no longer has reconnects and drops forever, and only
the phone can clear it.

Bluefy warns that the connection may drop in the background. The warning is
about its own GATT session, which iOS suspends with the page. The HID link
belongs to iOS and stayed up with the phone locked. A Capacitor app with the
Bluetooth background mode keeps the data channel too.

## Next steps

- Firmware: request the 2M PHY, data length extension, and a 15 to 30 ms
  interval on every connection, not from a button. Landed the same day in
  `Phone::tick`, once per encrypted link; not yet re-measured on the phone.
- Firmware: `CONFIG_BT_NIMBLE_LOG_LEVEL_WARNING` in `opal-firmware` too. Its
  console is off, so it does not pay the UART cost, but it still formats two
  lines per HID key.
- Phone: read the granted interval. `getDevices()` is tested and does not
  reach a bonded band in Bluefy.
- Protocol: 244-byte notifications are the packet size to design the embedding
  stream around; frames larger than that need chunking in `protocol/`.
