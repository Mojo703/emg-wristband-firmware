# phone-app

The phone side of the band, as a website: a splash screen ("Opal"), a main
menu showing whether the band is connected over BLE, a Calibration screen
(empty for now), and a Gesture Configuration screen. The link bench that used
to be the only screen is still here, reachable from the menu's "Link bench"
link; it drives `ble-media`'s `ble-gatt-bench` firmware over Web Bluetooth and
reports what the phone's radio stack grants.

Gesture Configuration is a UI mockup today, not wired to the band: it shows
one empty box per media command (Play/Pause, Next/Previous Track, Volume
up/down) and a scrollable tray of placeholder gesture chips below. Dragging a
chip onto a box assigns it (tap a filled box to clear it back to the tray);
this is in-memory component state only, reset on reload, with no BLE write
and no persistence. There is also no BLE characteristic on the band yet for
writing a gesture keymap — the real firmware only exposes BLE HID output —
so wiring this up is future work. `src/lib/bench/keymap.ts` has groundwork
for that next step (gesture-name validation, a `localStorage` draft, and a
stub `applyKeymap`) but nothing in the UI calls it yet.

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
