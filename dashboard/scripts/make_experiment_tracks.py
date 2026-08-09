"""Write the thumb-modifier experiment track into the (untracked) track library.

Unlike an imported song, this carries no music: the schedule is a plain grid of
evenly spaced holds over a metronome click, so every class gets the same number
of cues and the same amount of rest between them.

    modifier-a   5 lanes, 10 cues per class, 1.4 s holds  config/collection-modifier.json

A track's lane count is the class count of the config the backend started with,
so the track is only meaningful under its own config.

    python3 scripts/make_experiment_tracks.py
"""

import json
import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

TRACKS = Path(__file__).resolve().parent.parent / "tracks"
SAMPLE_RATE = 44100
CLICK_FREQUENCY_HZ = 660.0
CLICK_SECONDS = 0.05
CLICK_AMPLITUDE = 0.18
LEVELS = ("easy", "medium", "hard")
MAXIMUM_COLUMNS = 6

# The browser needs the first block to fall the whole way, and the backend
# stops the audio at duration_ms, so the schedule sits clear of both ends.
HEAD_SECONDS = 8
TAIL_SECONDS = 6


def assign_columns(map_notes, column_count):
    """The port of `assign_columns` in src/collect/beatmap.rs, cue for cue.

    Kept as a port rather than a shortcut because the backend validates what it
    finds here: cues rank by lattice cell, rank `rank` of `total` draws in
    column `rank · column_count / total`, and a cell spanning several columns
    deals its cues to whichever is furthest behind its share.
    """
    total = len(map_notes)
    if total == 0:
        return []

    cues_in_cell = {}
    for index, note in enumerate(map_notes):
        cues_in_cell.setdefault(note["cell"], []).append(index)

    assignments = [0] * total
    first_rank = 0
    for cell in sorted(cues_in_cell):
        cues = cues_in_cell[cell]
        quotas = []
        for rank in range(first_rank, first_rank + len(cues)):
            column = rank * column_count // total
            if quotas and quotas[-1][0] == column:
                quotas[-1][1] += 1
            else:
                quotas.append([column, 1])
        dealt = [0] * len(quotas)
        for index in cues:
            turn = min(
                (column for column in range(len(quotas)) if dealt[column] < quotas[column][1]),
                key=lambda column: (
                    (2 * dealt[column] + 1) / quotas[column][1],
                    column,
                ),
            )
            assignments[index] = quotas[turn][0]
            dealt[turn] += 1
        first_rank += len(cues)
    return assignments


def build_schedule(lane_count, cues_per_lane, hold_ms, spacing_ms):
    """Cues cycling the lanes in turn, evenly spaced.

    Cue `index` names lattice cell `index % lane_count`, which makes the cell
    ranking hand cell `c` exactly column `c` at `lane_count` columns — so the
    lanes come round robin and every class gets `cues_per_lane` cues.
    """
    return [
        {
            "time_ms": HEAD_SECONDS * 1000 + index * spacing_ms,
            "cell": index % lane_count,
            "hold_ms": hold_ms,
        }
        for index in range(lane_count * cues_per_lane)
    ]


def write_track_json(directory, track_id, title, beats_per_minute, map_notes):
    last = map_notes[-1]
    duration_ms = last["time_ms"] + last["hold_ms"] + TAIL_SECONDS * 1000
    level = {
        "map_notes": map_notes,
        "column_assignments": {
            str(count): assign_columns(map_notes, count)
            for count in range(1, MAXIMUM_COLUMNS + 1)
        },
    }
    entry = {
        "id": track_id,
        "title": title,
        "beats_per_minute": beats_per_minute,
        "duration_ms": duration_ms,
        "levels": {name: level for name in LEVELS},
    }
    (directory / "track.json").write_text(json.dumps(entry, indent=1) + "\n")
    return duration_ms


def write_audio(directory, duration_ms, beats_per_minute):
    """A click on every beat, quiet enough to pace against without masking the
    operator's voice."""
    total = int(duration_ms / 1000 * SAMPLE_RATE)
    samples = bytearray(2 * total)
    click_length = int(CLICK_SECONDS * SAMPLE_RATE)
    click = [
        int(
            CLICK_AMPLITUDE
            * (1.0 - index / click_length)
            * math.sin(2.0 * math.pi * CLICK_FREQUENCY_HZ * index / SAMPLE_RATE)
            * 32767
        )
        for index in range(click_length)
    ]
    period = int(60.0 / beats_per_minute * SAMPLE_RATE)
    for start in range(0, total - click_length, period):
        for offset, value in enumerate(click):
            struct.pack_into("<h", samples, 2 * (start + offset), value)

    with tempfile.NamedTemporaryFile(suffix=".wav") as raw:
        with wave.open(raw.name, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(bytes(samples))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw.name,
             str(directory / "audio.ogg")],
            check=True,
        )


def main():
    beats_per_minute = 60.0
    plans = [
        ("modifier-a", "modifier_a", "Modifier A", 5, 10, 1400, 5_000),
    ]
    for name, track_id, title, lanes, per_lane, hold_ms, spacing_ms in plans:
        directory = TRACKS / name
        directory.mkdir(parents=True, exist_ok=True)
        map_notes = build_schedule(lanes, per_lane, hold_ms, spacing_ms)
        duration_ms = write_track_json(
            directory, track_id, title, beats_per_minute, map_notes
        )
        write_audio(directory, duration_ms, beats_per_minute)
        print(
            f"{directory}: {len(map_notes)} cues over {lanes} lanes, "
            f"{duration_ms / 60_000:.1f} min"
        )


if __name__ == "__main__":
    main()
