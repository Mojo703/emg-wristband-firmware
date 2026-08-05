# dashboard

The host-side web dashboard for the EMG wristband. A Rust/axum backend relays
CBOR frames between the wristband and a Svelte 5 frontend, and runs the
training-data collection game. Devices reach the backend two ways: over wifi,
dialing the TCP port, or over USB serial, which is discovered automatically. Both
carry the same framing.

The backend is a relay. It owns no model and no functional configuration, only
the cosmetic projection in `src/looks.rs`, the signal-quality estimator, and the
collection session state. A pose-inference service can be attached, and when one
is configured the backend forwards every `Emg` frame to it and proxies the
resulting `Pose` frames to the Pose panel; if that service drops, the backend
reconnects with exponential backoff.

## Pieces

`src/` is the axum backend: device intake over TCP and serial, the device
registry, the `/ws` browser session loop, the optional pose-service proxy,
`src/signal_quality.rs` (the per-channel electrode estimator), and
`src/collect/`, the collection game's backend half: the session recorder, the
webcam capture, the track catalog, the Beat Saber map importer, the host-side
provenance store, and the session state machine.

`src/session_report/` is the offline take-quality report, run from
`src/bin/session_report.rs`. It shares `src/signal_quality.rs` with the live
panel, so the two cannot measure a channel differently.

`web/` is the Svelte 5 and TypeScript frontend (Vite): a panel shell plus a
registry in `web/src/lib/panels.ts`. The panels are Stream (the EMG scope, with
the device's own decisions folded in: the per-class confidence track, the
threshold line, wake-gate status and streak, and event markers), Config (the
keymap and wifi), Collect (the electrode check, the session setup form, the
game, and the track library), Pose (a code-split Three.js hand viewer),
Telemetry (device measurements as live values and trend plots), and Logs (the
device's log console).

Shared wire types come from the sibling [`protocol`](../protocol) crate (CBOR via
ciborium and cbor-x). The optional Python pose service lives in
[`../pose-service`](../pose-service).

## What you need

- Rust, stable or nightly. `cargo build` pulls everything else and needs no
  system libraries: the HTTPS clients use rustls rather than OpenSSL.
- pnpm, for the frontend. Not npm; nobody here builds or tests with it.
- `ffmpeg` on `PATH`, for the collection game's webcam recording and its
  placement photos. Nothing else uses it, so the rest of the dashboard runs
  without it.
- A v4l2 webcam, likewise only for collection, and only when a session asks for
  video. It defaults to `/dev/video0`; `EMG_CAMERA_DEVICE` points elsewhere. The
  setup form's Video row turns recording on and shows a live preview of the same
  camera, streamed as motion JPEG from `/collection/camera/preview`. A session
  that asks for video and cannot get it refuses to start, saying why.

  Capture defaults to 240p30 (`EMG_CAMERA_SIZE`, `EMG_CAMERA_FRAMERATE`). The
  video is there to check a hand against a label, not to look good, and a
  smaller capture keeps sessions small and keeps the camera off a USB bus the
  device link shares. Cameras vary in which modes they offer: if the default is
  refused, ffmpeg's error names what the driver wanted instead, and a 16:9-only
  camera wants `EMG_CAMERA_SIZE=426x240` or `640x360`. Above 240p, set
  `EMG_CAMERA_INPUT_FORMAT=mjpeg` — raw frames cost bus bandwidth in proportion
  to their size, and raw 720p30 needs more than USB 2.0 has.

## Run

```sh
./run.sh          # build frontend, serve app and /ws on :8090 (add --release)
./dev.sh          # backend plus Vite hot-reload dev server on :5173
```

`run.sh` is the normal path; open <http://localhost:8090>. `dev.sh` is for
frontend iteration; open <http://localhost:5173>, which proxies `/ws` to the
backend. Both start a local pose service when `../pose-service/.venv` exists and
`EMG_POSE_URL` is unset.

The manual steps are:

```sh
cd web && pnpm install && pnpm run check && pnpm run build   # type-check, emit web/dist
cd .. && cargo run --bin dashboard                           # serve app and /ws on :8090
```

The crate builds two binaries, `dashboard` and `session_report`, so `cargo run`
needs the `--bin` to know which one you meant.

Devices announce themselves. The backend recognizes a wristband on USB by its USB
identity and probes it, with no configuration; over wifi, point the wristband at
this machine's address on port 9000. Set `EMG_NO_SERIAL=1` to keep the backend off the serial
ports while flashing firmware.

## The collection game

A collection session records raw EMG and a label log while the subject follows
falling notes in the browser, plus a webcam video when the operator asks for one.
The backend writes everything that carries a timestamp, and it plays the audio,
so the playhead the cues are logged against is the mixer's own sample cursor. The
browser sends the operator's playback intents and draws the positions it is
sent.

### Getting a song in

Each track in the library is one directory under `tracks/`, holding `audio.ogg`,
a `track.json` cue schedule, and the map files it was converted from under
`source/`. The track's id is in `track.json`; the directory is that id with every
`_` written as `-`, so `kamome_sano_citrus` lives in `tracks/kamome-sano-citrus/`.
Everything that names a track over the wire or over HTTP names it by id.

Tracks come from hand-made Beat Saber maps, imported from the Collect panel two
ways:

- Upload a map archive. The `.zip` a map site hands you goes straight in.
- Give a BeatSaver key or map address, either `858` or
  `https://beatsaver.com/maps/858`. The backend downloads the archive itself.

Both paths run the same converter. `src/collect/beatsaber.rs` reads map formats
v2, v3, and v4, picks a difficulty by the preference order in
`DIFFICULTY_PREFERENCE` (Normal first, then the nearest by density), keeps only
the right hand's notes, and turns them into one cue schedule per difficulty level
of this game's own dial. A map is far denser than a hand forming gestures can
follow, so a level's schedule keeps a subset of the notes as cue roots rather
than all of them; the times it keeps are the map's own, verbatim. The audio is
copied without being decoded. A map's audio therefore has to be the Ogg Vorbis
that Beat Saber maps ship, which is what the backend's mixer decodes at session
start.

An import that fails leaves nothing behind: the track is built in a temporary
directory beside the library and renamed into place. The importer refuses a track
whose directory name the library already holds rather than overwriting it; delete
that track first.

The same routes serve the frontend and a terminal:

```sh
curl --data-binary @map.zip http://localhost:8090/collection/tracks/import/upload
curl -H 'content-type: application/json' -d '{"reference":"858"}' \
  http://localhost:8090/collection/tracks/import/beatsaver
curl -X DELETE http://localhost:8090/collection/tracks/kamome_sano_citrus
```

Either import answers with the track's id, title, tempo, duration, the difficulty
file it chose, and a per-level summary; a refusal answers with one message saying
why. BeatSaver refuses requests that do not name themselves, so the client sends
a `User-Agent`; nothing about that needs configuring.

### Running a session

The vocabularies the setup form offers (subjects, gesture classes, activities,
sweat levels) come from `config/collection.json`, which is read once at startup —
changing it needs a backend restart. Edit that file to add a teammate or a
condition; the gesture classes are the game's lanes, in the order listed.

The classes are Hyser 10, 11, 16, 17 and 1: forearm rotation both ways, wrist
deviation both ways, and lifting the thumb. All five are performable with a fist
closed on a pole and none of them takes the hand off the grip. The first four
swing the pole tip along two axes — rotation moves it sideways, deviation moves
it fore and aft — and each carries a `motion` in the config: an arrow and a line
of explanation, drawn by one component wherever the class is shown. The arrows
are a **plan view of the pole tip seen from above**, with up the screen meaning
down-track, and the lateral pair is written for the right arm. Lifting the thumb
moves no pole, so it has no `motion` and draws no arrow.

They are **larger than the gestures this product wants**, chosen to clear the
current analog front end's noise floor rather than to sit apart from ordinary
skiing motion; the `note` at the top of `collection.json` says so, and they
should be replaced with subtler ones once the front end is fixed. Trained on
held-out Hyser subjects this set reaches 57.4%, against 61.5% for the same four
without the thumb: the fifth command costs four points, and almost all of it
comes out of radial deviation, which falls to 18%. Expect that lane to look
worst in the data. Thumb extension was the best of six candidate fifth classes
measured, and the only one where the hand never leaves the pole.

While no session is running, the Collect panel is four cards over two columns:
subject and session tags, the band and its signal, the track library with the
audio controls, and track import. A ready bar along the bottom holds Start and
says what is blocking it. The band card carries the electrode check: per channel, the broadband noise floor, the mains figure, the
DC offset and the headroom it leaves, how much of the last ten seconds sat at
the rail, and the device's own lead-off comparator. The floor is the number that
decides whether a take is worth making, and it has to stay under 10 µV RMS;
surface EMG on this band sits at 14–20 µV RMS, so a channel at the limit still
has its gestures well clear of the noise. The floor is interharmonic band power
over 20–450 Hz, not a plain RMS: mains hum is nearly all the in-band power on
this rig, so a plain RMS fails every channel identically and says nothing about
what to fix. The mains column is what says which way to go when the floor fails.
Both come from the backend, threshold included.

In the Collect panel: fill in the session's metadata, say which board and
harness the device is on if that has changed, choose whether to record video,
take a placement photo of the donned band, pick a track and a difficulty, and
start. Starting arms the session: the backend creates the session directory, and
the recorder and camera begin writing straight away. The game then waits for you
to start the track, with a lead-in before the first cue so the notes have time to
fall. A session ends when the track does, or earlier if you finish it by
hand; either way the backend finalizes the files and the summary appears. You
keep or discard the take there, with that summary in view. Discarding deletes
the session directory.

If the device stops sending EMG for more than a second and a half, the backend
pauses the session: the audio stops, the cue timeline freezes, and the field says
which device went quiet and for how long. Press play to carry on once it is back;
the track resumes from exactly where it froze and the cues follow it, so what the
log says you were asked to do still lines up with the samples. A cue the pause
landed inside is cut short at the pause and not re-issued. The pause button does
the same thing on request, and the log records which of the two it was.

An armed session nobody starts is not thrown away. It keeps recording, and after
ten minutes it finalizes itself and goes to review like any other take; the
stretch before the track began is marked in the event log as an `armed_prefix`,
so the samples nothing was cued in are labelled as such rather than left to be
inferred. The top bar carries the
recorded seconds and the sample count throughout, which is what stops moving when
something goes wrong.

Starting with no device selected gives a practice run: the game plays and
nothing is recorded. The setup form says so before you press the button, and the
session is named with a `_practice` suffix.

Each recorded session leaves one directory under `sessions/`, named for the
local time it was created and the subject:

| File | What it holds |
|------|---------------|
| `emg.i16` | the raw sample blobs, verbatim, little-endian `i16`, channel-major within each window and concatenated in `seq` order |
| `emg.missing` | the windows' gap bit planes, in the same window order, so a placeholder zero is never mistaken for a measurement |
| `events.jsonl` | one JSON event per line: every cue, every window, every phase change, and every pause with the resume that ended it |
| `session.json` | the manifest: the metadata, the hardware identity and device provenance, the track, the don count, the audio output and its measured latency, and `completed` |
| `video.mkv` | the webcam recording, when the session asked for video |
| `placement.jpg` | the placement photo, if one was taken at setup |

`events.jsonl` describes every byte of `emg.i16`, so a windowing tool needs
nothing else to label the stream. A session that crashed mid-recording is
recognizable by its manifest alone, whose `completed` stays false.

The manifest also records the rig itself: the firmware build the device reported
on connect, and the ADS1298 registers read back off each chip after configuration
rather than copied from what the firmware meant to write. Beside them sit the two
facts neither the device nor the form can supply: the board revision the operator
entered for that device id, and which don of that subject's arm the session was
recorded on. The host keeps both between runs in `config/provenance.cbor`.

### Checking a take

```
cargo run --release --bin session_report -- sessions/<session id>
```

Prints, to standard output, whether that session is usable for training and the
biggest single reason if it is not. It leads with the verdict, then covers file
and clock integrity, the gap mask, per-channel noise floor and headroom (the two
ADS1298s reported separately), the cue timeline against the track's authored
schedule, and whether each gesture produced a cue-locked response.

The response test reports a family-wise p — the null is the largest effect
anywhere in the channel-by-class grid under a shuffle of the cues' class labels
— and a lag sweep, so a claimed response has to peak at true alignment rather
than merely exist there. It closes with a detection limit: the EMG amplitude
this recording could have resolved, which is what separates "the subject did
nothing" from "the signal was below what this take could see".

The noise floor is the Collect panel's electrode-check estimator
(`src/signal_quality.rs`) run over the whole recording, so a session's floor and
the number the operator saw before pressing record cannot drift apart.
`--tracks <directory>` points the schedule cross-check at a track library other
than the one beside `sessions/`.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `DASHBOARD_ADDR` | `0.0.0.0:8090` | browser and static-file bind |
| `EMG_DEVICE_ADDR` | `0.0.0.0:9000` | the TCP port devices dial over wifi |
| `DASHBOARD_WEB` | `web/dist` | static frontend directory |
| `EMG_NO_SERIAL` | (unset) | set to disable USB serial discovery, e.g. while flashing |
| `EMG_SERIAL_PORT` | (unset) | force serial discovery onto one port instead of selecting by USB identity |
| `EMG_DEVICE_LOG` | (unset) | set to echo device log frames onto the backend's own tty |
| `EMG_POSE_URL` | (unset) | optional pose service WebSocket |
| `EMG_COLLECTION_CONFIG` | `config/collection.json` | collection vocabularies |
| `EMG_PROVENANCE_STORE` | `config/provenance.cbor` | board revisions and don counts the host remembers between sessions |
| `EMG_TRACKS_DIR` | `tracks` | the track library |
| `EMG_SESSIONS_DIR` | `sessions` | where recorded sessions land |
| `EMG_CAMERA_DEVICE` | `/dev/video0` | the webcam |
| `EMG_CAMERA_SIZE` | `320x240` | capture size, `WIDTHxHEIGHT` |
| `EMG_CAMERA_FRAMERATE` | `30` | capture frame rate |
| `EMG_CAMERA_INPUT_FORMAT` | (driver's choice) | v4l2 pixel format, e.g. `mjpeg` |
| `EMG_AUDIO_OUTPUT` | (default output device) | game audio sink: part of a device name, or `silent` to run the mixer against no device |

Without a pose URL the Pose panel renders but receives no frames.

## The pose service

To set it up once:

```sh
cd ../pose-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# for the Meta emg2pose model, also:
pip install -r requirements-emg2pose.txt
python setup_emg2pose.py          # clone repo, download checkpoint
```

`run.sh` then starts it automatically, using the Meta `emg2pose` model if its
checkpoint is present under `../pose-service/checkpoints/` and the mock
otherwise. To point at an external service or force the mock:

```sh
EMG_POSE_URL=ws://localhost:8081 ./run.sh   # external service
POSE_MODEL=mock ./run.sh                     # force the mock estimator
```

See [`../pose-service/README.md`](../pose-service/README.md) for the estimator env
vars and the streaming-window size.

## Wire format

Wire types live in [`../protocol`](../protocol); the dashboard adds no framing of
its own. Three dashboard-specific notes. Browser-to-backend numeric controls are
integers (for example `tau_permille`) so whole-number JavaScript values survive
cbor-x's integer encoding. `Pose` frames are only passed through: the backend
forwards them and the frontend renders them, while the pose service owns the
model and the `format` tag (`mock_21` or `emg2pose_21`). And a device packs the
`Emg` sample blob on its way in; the backend unpacks it before fanning out, so
the browser and the pose service see the plain little-endian `i16` blob,
channel-major within the window.
