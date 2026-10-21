# phone-app

The phone side of the band, as a website: a splash screen ("Opal"), a main
menu showing whether the band is connected over BLE, a Calibration screen
(empty for now), and a Gesture Configuration screen.

Gesture Configuration writes for real: it shows one box per media command
(Play/Pause, Next/Previous Track, Volume up/down) and a scrollable tray of
gesture chips below. Dragging a chip onto a box assigns it (tap a filled box
to clear it back to the tray, or drag it to another box). Opening the screen
while connected loads whatever mapping the band already reports; **Save**
writes the whole mapping to the band's settings characteristic and the boxes
update again once the band echoes back what it kept — there is no local
draft or `localStorage` involved, the boxes always reflect the band's own
state. This talks to the real BLE "settings service" in
`src/lib/band/{protocol.ts,link.ts}` (ported from a teammate's
`demo-ble-firmware` branch — see that branch's `ble-media/src/settings_service.rs`
for the device side), **not** the same firmware as the link bench below.

The tray only offers the 2 gestures the band actually recognizes today
(Radial deviation, Ulnar deviation) — the protocol and `collection.json`
define 5, but the band's `calibration-flow::ACTIVE_CALIBRATION_GESTURES`
only trains/accepts these 2 so far. Offering the other 3 would look like a
working feature that silently does nothing on save, which is worse than not
offering them. `GestureConfig.svelte` has a `TODO` at the gesture list
marking exactly what to change once the band supports more.

To test this against real hardware, flash `demo-ble-firmware`'s firmware
(a stripped-down demo build — it does not include calibration, playback, or
the dashboard's USB-serial link) onto a bare ESP32-S3, e.g. from a separate
worktree so this checkout is undisturbed:

```sh
git worktree add ../emg-wristband-firmware-demo-ble demo-ble-firmware
cd ../emg-wristband-firmware-demo-ble/opal-firmware
. ~/export-esp.sh
cargo run
```

Then run `phone-app` as usual (`pnpm run dev`, connect from the Menu screen)
against that board. Connecting from a laptop over `localhost` works fine for
this — Web Bluetooth only needs a secure origin, which `localhost` already
is, and the connection itself is a direct Bluetooth link to the board, not
something that goes over Wi-Fi.

The link bench that used to be the only screen in this app is no longer
reachable from the menu — it talks to a different firmware image
(`ble-gatt-bench`, see below) than the one Gesture Configuration now uses,
and isn't something an actual user of the band needs. Its code
(`src/lib/screens/LinkBench.svelte`, `src/lib/bench/*`) is untouched and
still fully working; it's just not wired into `App.svelte`'s navigation
today. Re-adding a way to reach it (e.g. its own menu entry) is a small
change if it's needed again.

## Run

```sh
pnpm install
pnpm run dev      # localhost, Chrome on the laptop; localhost counts as secure
pnpm run check    # svelte-check
pnpm run test     # node --test over src/**/*.test.ts
pnpm run build    # dist/, static
```

A phone needs HTTPS, so the phone path is the built `dist/` on a static host.
Cloudflare Pages: `pnpm run build`, then either upload `dist/` in the Pages
dashboard or connect the repository with build command
`cd phone-app && pnpm install && pnpm run build` and output directory
`phone-app/dist`.

On iPhone open the URL in Bluefy, which puts Web Bluetooth on CoreBluetooth, the
same stack a Capacitor build would use. Safari has no Web Bluetooth. On Android
use Chrome. On Linux, Chrome exposes Web Bluetooth only behind
`chrome://flags/#enable-experimental-web-platform-features`, and needs BlueZ
5.41 or later; it works for developing the page, but its pairing behaviour and
connection parameters say nothing about iOS.

## What the bench measures

Flash the band with `. ~/export-esp.sh && cargo run --example ble-gatt-bench`
from `ble-media/`. It advertises as the HID media remote at boot with the bench
service in its scan response.

- **Connection** reads the band's status record: encrypted and bonded flags,
  the granted connection interval, latency and supervision timeout, the
  negotiated MTU, the PHY each way, the band's free heap, and the outcome of
  the last media key. The band notifies it on change and once a second.
- **Link requests** ask the phone for a connection interval, for the 2M PHY
  with data length extension, to pair from the band's side, and to clear the
  band's bonds. Each is a request; the status shows what was granted. The
  band itself asks for the 2M PHY, long packets and a 15 to 30 ms interval
  once on every encrypted link, so the buttons are for re-asking or for
  trying an interval the band would not choose.
- **Round trip** writes an echo command and times the matching notification,
  sequentially, and reports min, p50, p95 and max. The write-with-response
  box adds the acknowledged write for comparison.
- **Stream** asks the band to notify numbered packets of a requested size, as
  fast as its stack accepts or at an interval, and reports goodput, packet
  rate, missing and corrupt packets, and how often the band's stack refused a
  packet. The band clamps the size to the MTU; the table shows the size that
  arrived. Sweep runs 20, 100, 244 and 500 bytes in turn. Raw EMG at 16
  channels and 2 kHz needs 64 KB/s as int16 or 96 KB/s as 24-bit codes.
  Read the packet rate against the interval: the same rate at every size up
  to one link-layer packet means a fixed number of notifications per
  connection event, and the interval the central granted is the ceiling. The
  stall count only rises when the band's packet buffers run out; when the
  controller is merely full, NimBLE blocks the sending thread, so a run with
  zero stalls can still be sitting on the link's limit.
- Apple accepts an interval request only when the minimum is at least 15 ms
  and the maximum is at least 15 ms above it, so **15 to 30 ms** is the
  preset an iPhone can grant; the others are there to see what it refuses.
- **Media keys** send a command the band turns into the same HID consumer key
  a gesture commit sends. The status says whether the band sent it or dropped
  it for want of an encrypted link.
- Two pages can bench one band in turn. The band answers the connection
  that wrote the last command, so a stream started from one page is counted
  there and the other page's status shows its own link.
- **Copy results** puts every table on the clipboard as markdown for the
  engineering log.

## Orderings worth testing on the iPhone

Which of these works decides how the manual tells a stranger to connect.

1. Pair through Settings first, so iOS holds the HID link, then open the page
   and connect. The band keeps advertising while connected, once a second,
   so the chooser should list it; it may take a few seconds to appear.
2. Connect from the page first, then press **Pair from band**. The question is
   whether the pairing dialog appears and whether iOS then attaches its HID
   driver on the same link, so the media buttons work without a trip through
   Settings.
3. Reboot the band while both are up and see which side reconnects, and how
   long it takes.

If a phone or desktop reconnects and drops with `Authentication Failure` over
and over, it still holds a long-term key the band no longer has. The band
cannot clear the other side's key, so forget the device there and pair again.
Anything that clears the band's NVS, and the **Forget bonds** button, will
cause this on a phone that has not forgotten the band.

Also worth a run each: media keys with the phone locked and with Bluefy in the
background, a walk out of range and back, and **Forget bonds** followed by a
fresh pairing.
