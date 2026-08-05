# Onboarding

For teammates joining the EMG wristband work. By the end you will have the
dashboard running on your own machine with a song imported, and you will know how
to record a session with the band and check it afterwards.

There is one band and it gets passed around, so most of this runs on your laptop
with nothing attached. Set that part up before your turn comes. Doing it during
your turn wastes everyone else's.

Everything below assumes Arch. Other distributions differ mostly in package
names. Ask Matthew when something does not match.

## Who needs what

Section 1, the dashboard, is for everyone. It builds and runs with no hardware
plugged in, and the collection game plays in practice mode, so you can learn the
interface before you ever touch the band.

Section 3, the firmware toolchain, is only for people who flash. It downloads a
couple of gigabytes of Espressif tooling and does nothing without a board in your
hand, so skip it until you need it.

## 1. The dashboard

You need Rust, either stable or nightly, and pnpm. Nobody here builds or tests
with npm. Add ffmpeg to `PATH` if you want the webcam, since nothing else uses
it.

```sh
sudo pacman -S rustup pnpm ffmpeg
rustup default stable

git clone <repo> EMG-Wristband
cd EMG-Wristband/dashboard
./run.sh
```

Open <http://localhost:8090>. `run.sh` builds the frontend into `web/dist`, then
serves the app and its WebSocket on port 8090. The first build takes a while and
later ones do not.

There is no shared Cargo workspace here. Every Rust project carries its own
`Cargo.toml` with an empty `[workspace]` table, so build and run from inside the
subproject directory rather than from the repo root.

With no band connected, the Collect panel runs the game in practice mode: it
plays on the real schedule and writes nothing to disk. The setup form tells you
so before you press the button. Use it to learn what the lanes look like.

## 2. Put a track in

A track is one song together with the cue schedule built from it. The library is
personal music and `dashboard/tracks/` is gitignored, so everyone builds their
own. One track is enough to start.

Tracks come from hand-made Beat Saber maps. In the Collect panel's Import card
you can paste a BeatSaver key or map address, such as `858` or
`https://beatsaver.com/maps/858`, and the backend fetches the archive itself.
Uploading a `.zip` from a map site does the same thing. The converter picks a
difficulty, keeps the right hand's notes, and builds one cue schedule per level
of this game's own difficulty dial.

Pick something you can stand hearing several times.

## 3. The firmware toolchain

Only if you are flashing the band. The one-time setup lives in
[`ota-client/README.md`](ota-client/README.md) and every firmware project here
shares it:

```sh
cargo install espup --locked
espup install
cargo install espflash --locked
cargo install ldproxy --locked
```

Every shell that builds firmware has to source the exports first:

```sh
. ~/export-esp.sh
```

### The Arch libxml2 workaround

Espressif's bundled `esp-clang` links against `libxml2.so.2` while Arch ships
`libxml2.so.16`, so the first build of each firmware crate dies with
`libxml2.so.2: cannot open shared object file`. Each crate keeps its own ESP-IDF
checkout under `.embuild`, which means you hit this once per crate rather than
once per machine. Run this from inside the crate after the build fails, then
build again:

```sh
ln -sf /usr/lib/libxml2.so.16 \
  .embuild/espressif/tools/esp-clang/*/esp-clang/lib/libxml2.so.2
```

### Building the band

```sh
cd opal-firmware
. ~/export-esp.sh
cargo run          # builds, flashes over USB-Serial-JTAG, opens the monitor
cargo test-device  # the unit tests, which run on the device
```

Two things will confuse you otherwise. The first is `cfg.toml`, which holds wifi
credentials. Git ignores it, so a fresh clone has none. It is optional:
without it the defaults leave `wifi_ssid` empty, and an empty SSID boots the
device into the USB serial link that you want for recording anyway.

The second is the serial port. Opening it resets the device, since the port
asserts DTR, and the dashboard opens that same port. Stop the dashboard before
flashing, or start it with `EMG_NO_SERIAL=1`. Skip that and the two will fight
over the port while you chase a hardware fault that is not there.

## 4. Taking a turn with the band

Recording is a procedure and the order matters. The usual way to waste a session
is to skip ahead to the game and find out afterwards that nothing was recordable.

Prepare the skin first. Shave the area, abrade it, wipe it with alcohol and let
it dry, and give the dorsal side the most attention. We recorded every unusable
session so far without this step, which is most of why they came out unusable.

Don the band next, then measure how far up the forearm it sits from the ulnar
styloid, the bony bump on the wrist's pinky side. That distance and the rotation
both go into the setup form, and they are what makes two sessions comparable.

Now read the electrode check, before anything else on screen. It sits in the
Collect panel's band card with one row per channel, and the number that decides
everything is the noise floor, which has to stay under 10 µV RMS. Surface EMG on
this band runs 14–20 µV RMS, so a channel at the limit still puts its gestures
above the noise. A channel well above the limit cannot show a gesture at all.

If the floor fails, stop and fix it before you go on. Re-prep the skin, re-seat
the band, or unplug the laptop charger. A take recorded above the limit gives you
nothing, and it uses up a subject's turn to do it.

Then run the session. Fill in the metadata, say which board and harness the
device is on if that changed, choose whether to record video, take a placement
photo, then pick a track and a difficulty and press Start.

Starting arms the session, and the recorder begins writing at that moment rather
than when the track begins. The event log marks that stretch as an
`armed_prefix`, and we keep those samples as baseline. The start gate shows how
much has reached disk so far.

The track begins on your tap, after a lead-in that gives the first notes time to
fall. When it ends, or when you finish early, the backend finalizes the files and
shows a summary. You keep or discard the take there with that summary in front of
you.

### The five gestures

All five work with a fist closed on a pole and none of them takes the hand off
the grip. The arrows in the game show a plan view of the pole tip from above,
with up the screen meaning down-track. They assume the right arm.

| Lane | Arrow | What the subject does                                                     |
|---|---|---------------------------------------------------------------------------|
| Tip out | → | Rotate the forearm outward; the pole tip swings away from you to the side |
| Tip in | ← | Rotate the forearm inward; the pole tip swings toward you into your legs  |
| Tip forward | ↑ | Cock the wrist toward the thumb; the tip swings out from you forward      |
| Tip back | ↓ | Drop the wrist toward the pinky; the tip swings behind you                |
| Lift thumb | none | Lift the thumb off the grip; nothing else moves                           |

These motions are large. We picked them to sit above the current noise floor, and
a skier would choose smaller ones. Once the analog front end works we will swap
to those.

## 5. Check the take

```sh
cd dashboard
cargo run --release --bin session_report -- sessions/<session id>
```

It opens with a verdict, usable for training or not, and the biggest single
reason when it is not. After that come file and clock integrity, the per-channel
noise floor and unused input range for both converters, the cue timeline against
the schedule the converter authored, and whether each gesture produced a response
locked to its cue.

Read the verdict before the band comes off. A session that fails on noise sends
you back to the skin and the band. That is far easier while the subject is still
sitting there.

## Where things live

| Path | What it is |
|---|---|
| `dashboard/sessions/` | recorded sessions, gitignored |
| `dashboard/tracks/` | the track library, gitignored, yours to build |
| `dashboard/config/collection.json` | the class list and vocabularies, in the repo |
| `dashboard/config/provenance.cbor` | board revisions and don counts the host remembers, gitignored |
| `opal-firmware/cfg.toml` | wifi credentials, gitignored, optional |
| `*/.embuild/` | the ESP-IDF checkout, gigabytes, gitignored |
| `engineering-logs/` | why the model and the kernels turned out the way they did |

The backend reads `config/collection.json` once at startup, so editing the class
list or adding a subject needs a restart. Someone very nearly recorded a session
under stale labels this way.

## The state of the rig, August 2026

Read this before your first session so you know what you are looking at.

All nine recordings so far come back NOT USABLE, and the noise floor is usually
why. Chip medians run from 5.6 to 219 µV against a 10 µV limit, and only two of
the nine put any channel under it. Mains pickup spans 6 µV on the quietest
session to 5.9 mV on the worst. Between none and seven of the sixteen channels
rail, and which ones move each time the band goes back on.

Gestures do sometimes show through. One session produced seven channel-by-gesture
cells above the significance threshold for forearm rotation, and six of those
peaked at the cue's real time rather than beside it, so the rig is not blind. The two sessions with clean contact
are also the two that were cut short with almost no cues, which is the pattern to
break: nobody has yet recorded a full track on well-prepared skin.

We are working through three causes, in this order.

The arm has no body reference. Nothing sits on the electrode connector's ground
pins, so the arm floats. That is the leading explanation for the DC offsets that
rail channels, and for the way a different set rails on each re-don. Putting an
electrode on one of those pins changes nothing else about the rig.

Nobody has prepared the skin, ever. Section 4 says what to do about it.

The bias drive is switched off, and the board wires its compensation network to
the wrong pin, so turning the drive on is not enough by itself. One jumper wire
fixes the board, and the firmware change is separate. Engineering log 0019 has
the detail.

Expect early sessions to fail the electrode check, and treat a passing floor as
the news it is. Neither the model nor the software is the limiting factor right
now.
