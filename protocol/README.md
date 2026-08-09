# protocol

The CBOR wire-protocol types shared by the wristband firmware, the dashboard
backend, the browser, and the pose service. The crate is `no_std` plus `alloc`, so
the same definitions compile for the ESP32-S3. `src/lib.rs` is the whole crate, and
there is one central enum, `Frame`. Plain host Rust: `cargo test` runs the
round-trip encode/decode tests, and `cargo build` cross-checks the `no_std` build.

`Frame` is `#[serde(tag = "type", rename_all = "snake_case")]`, so every frame is
an internally-tagged CBOR map with string keys and a generic viewer shows
`{type: "emg", seq: 0, ...}` rather than an opaque positional array. The tag string
is the variant name in snake_case.

## The links

Three byte pipes carry frames, and one enum covers all of them. A device reaches
the backend over a TCP socket it dials on wifi, or over the USB serial port; both
use the framing described under [Byte-pipe framing](#byte-pipe-framing). The
browser talks to the backend over the `/ws` WebSocket, one CBOR frame per binary
message and no framing of its own. The pose service sits on a third WebSocket: the
backend forwards each `Emg` frame and the service answers with `Pose`.

The backend is a relay. It forwards a device's data frames to the browsers viewing
it and forwards browser control frames back to the selected device.

## The frames

### Device and backend

| Frame | Direction | What it carries |
|-------|-----------|-----------------|
| `DeviceHello` | device → backend | `device_id`, the device's own `DeviceConfig`, and its `DeviceProvenance`. Sent once on connect. |
| `Emg` | device → backend → browser | One bulk sample window. See [The EMG window](#the-emg-window). |
| `Prediction` | device → backend → browser | One window's classifier output: `logits`, `softmax`, `reject_score`, `argmax`, `accepted`, `wake_state`, `streak`, and the `tau` in force. The device decides; the backend relays and the browser paints. |
| `Event` | device → backend → browser | A discrete decision, as `t_us` on the `Emg` timeline plus an opaque `kind`, an optional `label`, and an optional named palette `color`. The frontend draws a labelled line and knows nothing of what `kind` means, so a new event kind needs no frontend change. |
| `Log` | device → backend → browser | One device log record: `t_us` since device boot, `level` (`LogLevel`), `message`. The USB byte pipe carries only frames, so logs ride the protocol instead of a serial text console. |
| `Telemetry` | device → backend → browser | Periodic numeric device measurements: `t_us`, an emitting `source` (`"chip0"`, `"aligner"`, `"inference"`, …), and a list of `TelemetryMetric` name/value pairs. Self-describing (the unit is a suffix on the name, e.g. `edge_period_mean_us`) and loss-tolerant, so counters are cumulative since boot. |
| `Probe` | backend → device | "A dashboard is now on this link: announce yourself and make it the active data link." The device replies with `DeviceHello`. Serial only: dialing a TCP socket is already the claim, but a serial port has no connection semantics, so this invents them. |
| `Heartbeat` | backend → device | Keepalive for a probed serial link, every couple of seconds. Silence releases the serial claim. The current runtime then has no dashboard data link because it does not start Wi-Fi automatically. The reply direction needs none: the data stream is its own liveness signal. |

### Browser view and control

| Frame | Direction | What it carries |
|-------|-----------|-----------------|
| `Hello` | backend → browser | The complete view state: `devices` (the picker list), `selection` (`None` when nothing is selected), the `states` render hints, and `server_suggestions`, this host's reachable IPv4 addresses paired with the device-listener port, so the config UI can pre-fill an address. Re-sent whenever the device set, the selection, or the selected device's config changes. |
| `Pose` | pose service → backend → browser | A 3-D hand pose: `t_us`, `joints`, `confidence`, and the `format` naming the joint convention. Passed through; the backend owns no part of it. |
| `SignalQuality` | backend → browser | What the electrodes look like right now: the measured `mains_fundamental_hertz`, the `noise_floor_limit_microvolts` a channel has to stay under, and one `ChannelQuality` per channel in channel order. The backend measures this from the live stream with the same estimator the offline session report uses. |
| `SelectDevice` | browser → backend | Which connected device to view. |
| `DismissDevice` | browser → backend | Drop a disconnected device from the picker. Ignored while its session is live; only reconnecting revives an entry. |
| `SetSensitivity` | browser → backend → device | A preset `level` id. The device owns the preset → threshold mapping. |
| `SetKeymap` | browser → backend → device | The gesture → media-key `bindings`. |
| `SetWifi` | browser → backend → device | `ssid` and `psk`. The device persists them for an explicit future Wi-Fi-mode transition; stored credentials do not start Wi-Fi at boot. |
| `SetServer` | browser → backend → device | The dashboard address an explicit Wi-Fi mode would dial, e.g. `"10.42.0.1:9000"`. |
| `SetPhone` | browser → backend → device | Start or stop advertising the resident BLE HID peripheral. Not persisted: the device starts dormant on every boot and reports each transition through `PhoneState`. |
| `PhoneState` | device → backend → browser | What the phone peripheral is doing, as one `PhoneStatus`: `dormant`, `standby`, `advertising`, `connecting`, `paired`, or `unavailable` with a reason. Sent on every transition rather than on the telemetry interval, because a button needs prompt feedback. `connecting` is a connected but unencrypted link, in which HID input is silently discarded — it must not render as connected. The off states say nothing about memory: the stack is resident from boot. |
| `SetBoardRevision` | browser → backend | Which board and harness a device is soldered to. The firmware cannot know this, so the backend remembers it per device id and stamps it into every later session manifest. It goes no further than the backend. |

### The collection game

These frames travel only between browser and backend; the device never sees them.
Collection records the same `Emg` stream the device already sends. Instants ride as
`UnixMilliseconds` on the one wall clock the browser and backend share, and
positions inside a track as `TrackMilliseconds`.

| Frame | Direction | What it carries |
|-------|-----------|-----------------|
| `CollectionCatalog` | backend → browser | Everything the session-setup form offers: the `subjects` roster, the playable `tracks`, the `collection_classes` being collected with their lane colours and their optional `motion` (an arrow to draw and a line to read, `null` for a gesture that moves nothing), and the `activities` and `sweat_levels` vocabularies. The collection classes are the target set and are unrelated to the device's trained model classes in `DeviceConfig`. |
| `StartCollection` | browser → backend | `metadata`, `track_id`, `difficulty`, and `record_video`. The operator chooses video per session; a session that asked for it and cannot get it fails to start rather than quietly recording EMG alone. |
| `StartTrack` | browser → backend | The operator tapped Start. The backend plays the audio, so it begins playback and the cue timeline together and the browser has nothing to report back about when the track began. |
| `PauseTrack` | browser → backend | Freeze playback and the cue timeline where they stand. The session stays open and keeps recording whatever EMG arrives. |
| `ResumeTrack` | browser → backend | Unfreeze from the frozen position. Answers both an operator pause and one the backend declared because the device fell silent. |
| `FinishCollection` | browser → backend | End the running session now. Recording is finalized as if the track had played out. |
| `StopCollection` | browser → backend | Resolve a session in the reviewing phase: `save: true` keeps the directory, `save: false` deletes it. |
| `CapturePlacementPhoto` | browser → backend | Capture a webcam still of the donned band. The backend holds the most recent photo and writes it into the next session's directory. |
| `SetEmgStream` | browser → backend | Whether this browser needs the raw EMG stream. Only the panels that draw waveforms do; a page showing the collection game draws none of it. The backend keeps consuming the stream either way, so the electrode check is unaffected. A browser that never sends this gets the stream. |
| `SetAudioVolume` | browser → backend | The music's level in thousandths. Applies to the next buffer the mixer renders, so it moves a running session, and it moves only the track — the cue clicks keep their own level. |
| `SetAudioOutput` | browser → backend | Which output device the game plays through; `null` asks for the host's default. A running session switches sinks in place and re-anchors its cue timeline on the new device. A backend started with `EMG_AUDIO_OUTPUT=silent` ignores it. |
| `AudioSettings` | backend → browser | The devices the host offers, the chosen `output` (`null` for the default) and the current `volume_permille`. Sent on connect and after every change. |
| `CollectionState` | backend → browser | The authoritative `phase` (`CollectionPhase`) plus the capture instant of any held `placement_photo`. Sent on every phase change and periodically while recording, so the browser's recording tripwire reflects bytes reaching disk. |
| `Beatmap` | backend → browser | The armed session's complete note schedule: `session_id`, `track`, `notes`, the track's measured `beat_times` for the debug metronome, and the `lead_in` silence before audio t = 0. The browser renders it and never invents notes; the backend logs the same schedule as cue events, so labels never depend on the browser. |
| `PlaybackPosition` | backend → browser | Where the backend's audio output stands: `position_ms` is what the subject hears at `at_unix_ms`, which sits a little ahead of the send because it accounts for the output device's latency. The browser extrapolates between these and never derives the timeline itself; the same pair is the anchor every cue is logged against. |
| `NoteResult` | backend → browser | One cued note's verdict from the activity detector: `session_id`, `index`, `hit`. It says muscle activity landed inside the note's window and makes no claim about which gesture was made. |

## The EMG window

`Frame::Emg` carries `seq`, `t0_us`, `channels`, `sample_rate`, `scale_uv`, the
`samples` blob and the `missing` mask.

`samples` is little-endian `i16`, channel-major. All of channel 0's samples for the
window come first, then all of channel 1's, and so on. The blob holds
`channels * samples_per_channel` values laid out as `samples[channel * time + t]`,
not `samples[t * channels + channel]`, so a 16-channel window of 500 samples is 500
samples of channel 0, then 500 of channel 1. Read it as sample-interleaved and it
still decodes without error, producing nonsense, so check any new reader against
one that already works: the Rust recorder, the browser's
`decodeEmg`, and the pose service's `np.frombuffer(...).reshape(channels, time)`
all agree on this layout.

The values are raw ADC counts, not model input: unfiltered, DC offsets and all.
`scale_uv` converts counts to microvolts. It is fixed by the front end's reference
and gain, which is what lets a recorded stream be read in real units afterwards;
the firmware's `MICROVOLTS_PER_WIRE_COUNT` is where the figure comes from. A
consumer that wants a centred trace has to remove the DC itself, since electrode
offsets of tens of millivolts are normal.

`missing` marks time steps that carry no measurement. The sixteen channels arrive
as two eight-channel acquisition sources (channel blocks `0..8` and `8..16`), and a
source that misses a grid tick has zeros written into `samples` as placeholders,
which against an electrode's DC offset read as full-scale spikes. The field is one
bit plane per source in channel-block order, each `missing_plane_stride(...)` bytes
long; step `t` is byte `t / 8`, bit `t % 8`, and a set bit means that source's eight
samples at that step are placeholders. Use `missing_at` rather than indexing by
hand. An absent or short mask reads as all-data.

## Encoding notes

Bulk EMG rides as a raw byte blob (`serde_bytes`) rather than a CBOR number array:
half the bytes of `f32` and a fraction of a number array.

On the device → backend hop only, `pack_samples` packs that blob further: lossless
delta, zigzag, varint, O(1) state. The backend calls `unpack_samples` before
fanning out, so the browser and the pose service receive the plain `i16` blob and
need no knowledge of the packing. `pack_sample_stream` and
`pack_sample_stream_into` produce the same format straight from sample values, for
the device, which cannot spare the heap to materialize the byte blob first.

Browser-to-backend numeric controls are integers (`tau_permille` and the like) so
whole-number JavaScript values survive cbor-x's integer encoding. Epoch
milliseconds are the exception in the other direction: cbor-x encodes JavaScript
integers that large as CBOR float64, so `UnixMilliseconds` accepts an integral
float as well as an integer, and rejects one with a fractional part.

The collection units and identifiers are all `#[serde(transparent)]`:
`UnixMilliseconds`, `TrackMilliseconds`, `DurationMilliseconds`,
`OffsetMilliseconds`, `NoteIndex`, `Millimeters`, `Degrees`, `BeatsPerMinute`, and
the string ids `SubjectId`, `TrackId`, `ClassId`, `ActivityId`, `SweatId` and
`SessionId`. On the wire they are the bare value. In Rust they are distinct types,
so a track position cannot be handed to something expecting a wall-clock instant.

Two shapes validate at deserialization rather than in use. `Beatmap` decodes as a
bare note array and rejects one whose onsets regress or whose holds overlap: one
hand performs one gesture at a time. `BeatsPerMinute` wraps a `NonZeroU16`, so a
track claiming 0 bpm is refused rather than dividing by zero mid-session.

## Byte-pipe framing

TCP and serial carry `FRAME_MAGIC ++ u32 little-endian length ++ that many CBOR
bytes`. The magic exists for the serial path: the ESP32-S3's ROM bootloader prints
text on the USB CDC at every reset, so a reader has to be able to resynchronize
mid-stream rather than trust the next byte to be a length. Both magic bytes sit
outside printable ASCII, so console text can never begin a frame.

`FrameScanner` is the incremental parser every reader shares. Feed bytes with
`extend`, take complete payloads with `next_frame`. It skips garbage between frames
by scanning to the next magic, and treats a length above `FRAME_MAX_LEN` as a
failed resync rather than a real header. `frame_bytes` wraps one encoded frame for
sending, in a single buffer so `TCP_NODELAY` sees one frame per write.

## Ownership

The device is the source of functional truth and runs standalone, so it owns
`DeviceConfig` (gestures, keymap, sensitivity presets and their thresholds, the
active threshold, the streak goal). The backend owns only the cosmetics the
firmware has no reason to carry: the colours and labels in `ClassInfo` and
`StateInfo`. It layers those on top when projecting `DeviceConfig` into the browser
`Hello`.

`DeviceProvenance` sits beside `DeviceConfig` rather than inside it: it is the
firmware build that is running (`FirmwareBuild`) and the analog front ends it
brought up (`AnalogFrontEnd` with its `RegisterReadback` list, read back off each
chip rather than copied from what the firmware meant to write). Nothing in it is
settable, which is exactly what makes it worth recording with a session.
`BoardRevision` is the one identity fact neither side can see, so it comes from the
operator and stops at the backend.

## One source, two hand-kept copies

The firmware and the dashboard backend both compile this crate, so they cannot
drift from it. The browser and the pose service each hold their own transcription,
and nothing generates either one. When you change a frame here, update both by
hand: the browser (cbor-x) in `../dashboard/web/src/lib/protocol.ts`, and the
Python service (cbor2) in `../pose-service`. `protocol.ts` also carries hand-written runtime validators,
so a new frame needs an entry there as well as a type.
