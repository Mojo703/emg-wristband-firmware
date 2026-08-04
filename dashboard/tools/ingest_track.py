#!/usr/bin/env python3
"""Add a song to the collection game's track catalog.

A track's cues come from a Beat Saber map — either a hand-made one:

    .venv/bin/python ingest_track.py --beatsaber-map ~/.local/share/BSManager/\\
        SharedContent/SharedMaps/CustomLevels/"Flares - Joetastic & nasafrasa"

which brings its own audio and title, or one the InfernoSaber automapper
writes for a song that has no map:

    .venv/bin/python ingest_track.py ~/Music/song.opus --title "Song Name"

Either way the audio is transcoded to the mono Ogg Vorbis the dashboard
serves, the map's right hand becomes the cue stream, and the entry stores a
schedule per difficulty level plus the column split points that balance the
lanes. The model runs in its own interpreter (.venv-beatsaber, set up per
beatsaber/README.md) and its output is cached, so re-running conversion after
a rule change costs nothing; `--remap` forces a fresh model run and
`--notes-per-second` is its density knob.
"""

import argparse
import itertools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import soundfile

TOOLS_DIRECTORY = Path(__file__).resolve().parent
BEATSABER_DIRECTORY = TOOLS_DIRECTORY / "beatsaber"


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if not slug:
        raise SystemExit(f"cannot derive a track id from title {title!r}")
    return slug


def transcode_to_ogg(audio_path: Path, ogg_path: Path) -> None:
    """Mono 44.1 kHz Ogg Vorbis, the shape every catalog track ships in."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "libvorbis",
            "-qscale:a",
            "3",
            str(ogg_path),
        ],
        check=True,
    )


def cached_map(stem: str) -> Path | None:
    """The model's cached output for this song, if a prior run left one.
    Conversion rules change far more often than the model's opinion of a song,
    so step 1 (the model) is skipped whenever its output already exists."""
    output_root = BEATSABER_DIRECTORY / "Data" / "prediction" / "new_map"
    if not output_root.exists():
        return None
    produced = [
        directory
        for directory in output_root.iterdir()
        if directory.is_dir() and directory.name.endswith(stem)
    ]
    return produced[0] if produced else None


def run_infernosaber(ogg_path: Path, stem: str, notes_per_second: float) -> Path:
    """Map one song and return the directory holding info.dat and Expert.dat.

    InfernoSaber reads every song staged in songs_predict/, so the staging
    directory is cleared first — one song per run keeps attribution obvious.
    """
    interpreter = TOOLS_DIRECTORY / ".venv-beatsaber" / "bin" / "python"
    checkout = BEATSABER_DIRECTORY / "InfernoSaber"
    runner = BEATSABER_DIRECTORY / "run_infernosaber.py"
    if not interpreter.exists() or not checkout.exists():
        raise SystemExit(
            "the InfernoSaber toolchain is missing; set it up per "
            f"{BEATSABER_DIRECTORY / 'README.md'}"
        )

    staging = BEATSABER_DIRECTORY / "Data" / "prediction" / "songs_predict"
    output_root = BEATSABER_DIRECTORY / "Data" / "prediction" / "new_map"
    staging.mkdir(parents=True, exist_ok=True)
    for stale in staging.iterdir():
        stale.unlink()
    if output_root.exists():
        for stale in output_root.iterdir():
            if stale.is_dir() and stale.name.endswith(stem):
                shutil.rmtree(stale)
    # InfernoSaber's loader keys on the .egg extension Beat Saber uses.
    shutil.copyfile(ogg_path, staging / f"{stem}.egg")

    subprocess.run(
        [
            str(interpreter),
            str(runner),
            "--difficulty",
            str(notes_per_second),
        ],
        cwd=checkout,
        check=True,
    )

    produced = [
        directory
        for directory in output_root.iterdir()
        if directory.is_dir() and directory.name.endswith(stem)
    ]
    if not produced:
        raise SystemExit(f"InfernoSaber produced no map directory for {stem}")
    return produced[0]


MAXIMUM_COLUMNS = 6

# Beat Saber's blue hand — the right one, and the only one this game reads.
RIGHT_HAND = 1


class Level(NamedTuple):
    """One position of the difficulty dial. `hold_minimum` and `hold_maximum`
    bracket every gesture's length; `rest` is the gap between one release and
    the next onset, which is also the labeled rest segment in the recording."""

    name: str
    hold_minimum: int
    hold_maximum: int
    rest: int


DIFFICULTY_LEVELS = (
    Level("easy", 1_000, 3_000, 1_000),
    Level("medium", 1_000, 2_000, 750),
    Level("hard", 750, 1_500, 500),
)

# The cycle (hold + rest) is a whole number of beats, so every repeat onset
# lands on a quarter note however long a sustain runs. An even count is
# preferred where one fits nearly as well, so cycles tend to align with
# half-bars rather than drifting against the bar line.
MAXIMUM_CYCLE_BEATS = 16
ODD_CYCLE_PENALTY = 1.15


def cell_of(note: dict) -> int:
    """Flatten the 4x3 lattice left-to-right dominant: cells 0..2 are the
    leftmost Beat Saber column bottom to top, 3..5 the next, and so on."""
    return 3 * note["x"] + note["y"]


def split_points(histogram: list[int], column_count: int) -> list[int]:
    """The cell indices where one of our columns ends and the next begins:
    `column_count - 1` boundaries over the flattened 1D cell array, placed so
    the note counts in the resulting contiguous ranges come out as even as the
    song's spatial distribution allows (best effort, whole song).

    A note in cell `c` lands in the column counting how many split points are
    at or below `c`. With 12 cells there are at most C(11,5) boundary sets, so
    the search is exhaustive: minimize the summed squared deviation from a
    perfectly even split.
    """
    if column_count == 1:
        return []
    total = sum(histogram)
    target = total / column_count
    prefix = [0]
    for count in histogram:
        prefix.append(prefix[-1] + count)

    best: list[int] | None = None
    best_cost = float("inf")
    for boundaries in itertools.combinations(range(1, 12), column_count - 1):
        edges = [0, *boundaries, 12]
        cost = sum(
            (prefix[edges[index + 1]] - prefix[edges[index]] - target) ** 2
            for index in range(column_count)
        )
        if cost < best_cost:
            best_cost = cost
            best = list(boundaries)
    assert best is not None
    return best


def read_info(map_directory: Path) -> dict:
    """`Info.dat` whatever its capitalization (BSManager levels use both)."""
    for name in ("info.dat", "Info.dat"):
        path = map_directory / name
        if path.exists():
            return json.loads(path.read_text())
    raise SystemExit(f"no info.dat in {map_directory}")


def choose_difficulty(info: dict) -> str:
    """The difficulty file to convert: Normal when the map has it, else the
    nearest by density — pinch gestures track far fewer notes than sabers."""
    preference = ["Normal", "Hard", "Easy", "Expert", "ExpertPlus"]
    available: dict[str, str] = {}
    for beatmap_set in info.get("_difficultyBeatmapSets", []):
        if beatmap_set.get("_beatmapCharacteristicName") != "Standard":
            continue
        for beatmap in beatmap_set.get("_difficultyBeatmaps", []):
            available[beatmap["_difficulty"]] = beatmap["_beatmapFilename"]
    for difficulty in preference:
        if difficulty in available:
            return available[difficulty]
    raise SystemExit("the map has no Standard difficulty")


def beat_clock(initial_beats_per_minute: float, bpm_events: list[dict]):
    """beats → milliseconds under a piecewise tempo (v3 `bpmEvents`). Most
    maps have zero or one event; the general integration costs nothing."""
    segments: list[tuple[float, float, float]] = []  # (start_beat, start_ms, ms_per_beat)
    current_beat, current_milliseconds = 0.0, 0.0
    current_period = 60_000.0 / initial_beats_per_minute
    for event in sorted(bpm_events, key=lambda entry: entry["b"]):
        beat = float(event["b"])
        current_milliseconds += (beat - current_beat) * current_period
        current_beat = beat
        current_period = 60_000.0 / float(event["m"])
        segments.append((current_beat, current_milliseconds, current_period))
    if not segments or segments[0][0] > 0.0:
        segments.insert(0, (0.0, 0.0, 60_000.0 / initial_beats_per_minute))

    def to_milliseconds(beat: float) -> float:
        start_beat, start_milliseconds, period = segments[0]
        for segment in segments:
            if segment[0] > beat:
                break
            start_beat, start_milliseconds, period = segment
        return start_milliseconds + (beat - start_beat) * period

    return to_milliseconds


def convert_map(
    map_directory: Path, duration_milliseconds: int, difficulty_file: str = "Expert.dat"
) -> tuple[list[dict], dict, float]:
    """One Beat Saber difficulty as a cue schedule per difficulty level.

    Reads both map generations: v3 (`colorNotes`/`sliders`, InfernoSaber and
    newer custom maps) and v2 (`_notes`, older custom maps — no arcs there, so
    every cue reads as the level's minimum hold). Timing is the map's own,
    verbatim; v3 `bpmEvents` are honoured by integrating the tempo timeline.

    Only the right hand's notes are used. Each Beat Saber hand plays its own
    half of the lattice, so one hand spans the grid the way our single
    instrumented hand should, and the split points then balance the columns
    without any of the two-hand interleaving.
    """
    info = read_info(map_directory)
    beats_per_minute = float(info["_beatsPerMinute"])
    difficulty = json.loads((map_directory / difficulty_file).read_text())
    if float(info.get("_songTimeOffset", 0) or 0) != 0.0:
        print("warning: map declares a nonzero _songTimeOffset; ignoring it")

    if "colorNotes" in difficulty:
        # v3 serializers omit zero-valued fields, so every read defaults to 0.
        def normalized(entry: dict) -> dict:
            return {
                "b": entry.get("b", 0),
                "x": entry.get("x", 0),
                "y": entry.get("y", 0),
                "c": entry.get("c", 0),
                "tb": entry.get("tb", 0),
            }

        raw_notes = [normalized(note) for note in difficulty.get("colorNotes", [])]
        sliders = [normalized(slider) for slider in difficulty.get("sliders", [])]
        bpm_events = [
            {"b": event.get("b", 0), "m": event["m"]}
            for event in difficulty.get("bpmEvents", [])
            if event.get("m")
        ]
        to_milliseconds = beat_clock(beats_per_minute, bpm_events)
    else:
        # v2: bombs ride in `_notes` as `_type` 3; hands are 0/1.
        raw_notes = [
            {"b": note["_time"], "x": note["_lineIndex"], "y": note["_lineLayer"], "c": note["_type"]}
            for note in difficulty.get("_notes", [])
            if note.get("_type") in (0, 1)
        ]
        sliders = []
        to_milliseconds = beat_clock(beats_per_minute, [])

    # Arc heads mark the mapper's own sustain intent: hold until the tail beat.
    arc_by_head: dict[tuple[float, int, int, int], float] = {}
    for slider in sliders:
        head = (round(slider["b"], 4), slider["x"], slider["y"], slider["c"])
        arc_by_head[head] = max(arc_by_head.get(head, 0.0), slider["tb"] - slider["b"])

    # The right hand's notes on the *beat* timeline, each with the span its
    # author gave it: an arc's length in beats, or zero for a plain tap.
    # Staying in beats is what keeps every cue on the grid — the millisecond
    # conversion happens once, at the end, through the tempo timeline.
    source: list[tuple[float, int, float]] = []  # (beat, cell, arc_beats)
    for note in raw_notes:
        if note["c"] != RIGHT_HAND:
            continue
        if not (0 <= note["x"] <= 3 and 0 <= note["y"] <= 2):
            continue
        beat = float(note["b"])
        arc_beats = arc_by_head.get(
            (round(beat, 4), note["x"], note["y"], note["c"]), 0.0
        )
        source.append((beat, cell_of(note), float(arc_beats)))
    source.sort()

    levels = {}
    for level in DIFFICULTY_LEVELS:
        notes = schedule_level(source, to_milliseconds, duration_milliseconds, level)
        histogram = [0] * 12
        for note in notes:
            histogram[note["cell"]] += 1
        levels[level.name] = {
            "map_notes": notes,
            "column_splits": {
                str(column_count): split_points(histogram, column_count)
                for column_count in range(1, MAXIMUM_COLUMNS + 1)
            },
        }
    return levels, beats_per_minute


def cycle_beats(period_milliseconds: float, level: "Level") -> int | None:
    """How many beats one hold-and-rest cycle spans at this tempo.

    The rest is exactly the level's value and the hold takes the remainder of
    the cycle, so the count is chosen to land `beats·period − rest` nearest the
    middle of the level's hold range. Counts outside the range are a last
    resort, and an odd count has to beat an even one by a clear margin
    ([`ODD_CYCLE_PENALTY`]) to win — cycles that align with half-bars read as
    part of the music rather than against it.

    `None` when no count produces a usable hold, which needs a tempo under
    about 60 bpm (below that the quarter note alone outlasts a hard-mode cue).
    """
    middle = (level.hold_minimum + level.hold_maximum) / 2
    best, best_cost = None, float("inf")
    for beats in range(1, MAXIMUM_CYCLE_BEATS + 1):
        hold = beats * period_milliseconds - level.rest
        if hold < 100:
            continue
        cost = abs(hold - middle) * (1.0 if beats % 2 == 0 else ODD_CYCLE_PENALTY)
        if not level.hold_minimum <= hold <= level.hold_maximum:
            cost += 1_000_000
        if cost < best_cost:
            best, best_cost = beats, cost
    return best


def schedule_level(
    source: list[tuple[float, int, float]],
    to_milliseconds,
    duration_milliseconds: int,
    level: "Level",
) -> list[dict]:
    """The cue schedule for one difficulty level, placed greedily on the beat.

    Beat Saber maps are far denser than a hand forming pinches can follow, so
    the schedule takes roots rather than notes: walk the hand's notes in beat
    order, skip whatever the current cue still covers, and make the next
    reachable note a root.

    A root's cue spans one cycle — a whole number of beats — of which the tail
    is the level's rest and the rest is the hold. Advancing by whole beats is
    what keeps every onset on a quarter note no matter how long a sustain runs
    or how the tempo moves; a root whose authored span outlasts one cycle
    simply repeats in the same cell, turning a written sustain into several
    labeled samples of one gesture rather than a single unwieldy one.
    """
    notes: list[dict] = []
    cursor = 0.0
    for beat, cell, arc_beats in source:
        if beat < cursor:
            continue
        position = 0.0
        while True:
            onset = to_milliseconds(beat + position)
            # The local beat period: with a tempo map this differs along the
            # track, so it is read where the cue actually falls.
            period = to_milliseconds(beat + position + 1) - onset
            beats = cycle_beats(period, level)
            if beats is None:
                break
            hold = beats * period - level.rest
            if onset + hold + level.rest >= duration_milliseconds:
                break
            notes.append(
                {
                    "time_ms": int(round(onset)),
                    "cell": cell,
                    "hold_ms": int(round(hold)),
                }
            )
            position += beats
            # The root keeps going while its own authored span does.
            if position >= arc_beats:
                break
        cursor = beat + position
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",
        help="the song to ingest (optional with --beatsaber-map: the level's own audio is used)",
    )
    parser.add_argument("--title", help="display title (default: the audio filename)")
    parser.add_argument(
        "--id", dest="track_id", help="catalog id (default: a slug of the title)"
    )
    parser.add_argument(
        "--notes-per-second",
        type=float,
        default=5.0,
        help="InfernoSaber's density target (default 5.0)",
    )
    parser.add_argument(
        "--remap",
        action="store_true",
        help="re-run the model even when its cached output exists",
    )
    parser.add_argument(
        "--beatsaber-map",
        type=Path,
        help="convert this existing Beat Saber map directory (info.dat + "
        "Expert.dat) instead of running the model — for hand-made maps",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=TOOLS_DIRECTORY.parent / "tracks",
        help="the track library to write into (default: dashboard/tracks)",
    )
    arguments = parser.parse_args()

    # An imported map is self-contained: audio, title, and difficulty all come
    # from the level directory unless overridden.
    difficulty_file = "Expert.dat"
    audio_source = arguments.audio
    default_title = None
    if arguments.beatsaber_map is not None:
        info = read_info(arguments.beatsaber_map)
        difficulty_file = choose_difficulty(info)
        if audio_source is None:
            audio_source = arguments.beatsaber_map / info["_songFilename"]
        artist = str(info.get("_songAuthorName", "")).strip()
        song_name = str(info.get("_songName", "")).strip()
        default_title = f"{artist} - {song_name}".strip(" -")
    if audio_source is None:
        raise SystemExit("an audio file is required unless --beatsaber-map provides one")
    if not audio_source.exists():
        raise SystemExit(f"no such audio file: {audio_source}")

    title = arguments.title or default_title or re.sub(
        r"\s+", " ", audio_source.stem.replace("_", " ")
    ).strip()
    track_id = arguments.track_id or slugify(title)
    # One directory per track: the schedule, the audio the dashboard serves,
    # and a copy of the map it came from, so a track is added or removed as a
    # single unit and can be re-converted without the original library.
    track_directory = arguments.tracks / track_id.replace("_", "-")
    track_directory.mkdir(parents=True, exist_ok=True)
    ogg_path = track_directory / "audio.ogg"

    if audio_source.resolve() != ogg_path.resolve():
        transcode_to_ogg(audio_source, ogg_path)
    audio_information = soundfile.info(str(ogg_path))
    duration_milliseconds = int(round(audio_information.frames / audio_information.samplerate * 1000))

    if arguments.beatsaber_map is not None:
        map_directory = arguments.beatsaber_map
        print(f"importing {map_directory.name} [{difficulty_file}]")
    else:
        map_directory = None if arguments.remap else cached_map(track_directory.name)
        if map_directory is not None:
            print(f"reusing cached model output {map_directory.name}")
        else:
            map_directory = run_infernosaber(
                ogg_path, track_directory.name, arguments.notes_per_second
            )

    levels, beats_per_minute = convert_map(
        map_directory, duration_milliseconds, difficulty_file
    )
    for name, level in levels.items():
        notes = level["map_notes"]
        if len(notes) < 2:
            raise SystemExit(f"only {len(notes)} playable notes at level {name}")
        splits = level["column_splits"]["4"]
        columns = [0] * 4
        for note in notes:
            columns[sum(note["cell"] >= boundary for boundary in splits)] += 1
        seconds = sum(note["hold_ms"] for note in notes) / 1000
        print(
            f"  {name:7} {len(notes):5} cues  "
            f"{len(notes) / (duration_milliseconds / 1000):.2f}/s  "
            f"{seconds:6.1f} s held  columns {columns}"
        )

    entry = {
        "id": track_id,
        "title": title,
        "beats_per_minute": round(beats_per_minute, 2),
        "duration_ms": duration_milliseconds,
        "levels": levels,
    }
    (track_directory / "track.json").write_text(json.dumps(entry, indent=2) + "\n")

    # The map itself, so a conversion-rule change can be replayed from here
    # rather than from wherever the map originally came from. The audio is
    # already transcoded beside it, so the source audio is not copied.
    source_directory = track_directory / "source"
    if source_directory.exists():
        shutil.rmtree(source_directory)
    source_directory.mkdir()
    for name in {"info.dat", "Info.dat", difficulty_file}:
        source_file = map_directory / name
        if source_file.is_file():
            shutil.copyfile(source_file, source_directory / source_file.name)

    print(f"wrote {track_directory}/")


if __name__ == "__main__":
    sys.exit(main())
