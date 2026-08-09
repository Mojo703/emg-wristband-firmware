"""Check the generated experiment tracks against what the backend's loader
demands, without starting the backend.

Mirrors `load_levels` and `TrackCatalog::from_parts` in src/collect/beatmap.rs:
every difficulty present, at least two cues, holds that neither overlap the next
onset nor run past the audio, lattice cells in range, a column resolved for every
cue at all six column counts, and a config of at most six classes. It also checks
what those rules do not: that at its own config's lane count each track deals the
cues equally, so no class is short of reps.

    python3 scripts/check_experiment_tracks.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parent.parent
LEVELS = ("easy", "medium", "hard")
LATTICE_CELLS = 12
MAXIMUM_COLUMNS = 6

# Each generated track and the config whose classes become its lanes.
PAIRINGS = [
    ("modifier-a", "config/collection-modifier.json", 10),
]


def check_track(directory, lane_count, cues_per_lane, complain):
    entry = json.loads((directory / "track.json").read_text())
    if not (directory / "audio.ogg").exists():
        complain(f"{directory.name}: no audio.ogg")
    if not (isinstance(entry["beats_per_minute"], float) and entry["beats_per_minute"] > 0):
        complain(f"{directory.name}: tempo {entry['beats_per_minute']}")
    duration_ms = entry["duration_ms"]
    rest = entry.get("rest")

    for level in LEVELS:
        loaded = entry["levels"].get(level)
        if loaded is None:
            complain(f"{directory.name}: no {level} schedule")
            continue
        notes = loaded["map_notes"]
        where = f"{directory.name} at {level}"
        if rest is None and len(notes) < 2:
            complain(f"{where}: {len(notes)} cues")
        for first, second in zip(notes, notes[1:]):
            if first["time_ms"] + first["hold_ms"] >= second["time_ms"]:
                complain(f"{where}: hold at {first['time_ms']} ms overlaps the next onset")
        for note in notes:
            if not 0 <= note["cell"] < LATTICE_CELLS:
                complain(f"{where}: lattice cell {note['cell']}")
            if note["hold_ms"] == 0:
                complain(f"{where}: zero-length hold at {note['time_ms']} ms")
            if note["time_ms"] + note["hold_ms"] >= duration_ms:
                complain(f"{where}: cue at {note['time_ms']} ms runs past the audio")

        for count in range(1, MAXIMUM_COLUMNS + 1):
            columns = loaded["column_assignments"].get(str(count))
            if columns is None:
                complain(f"{where}: no column assignment for {count} columns")
                continue
            if len(columns) != len(notes):
                complain(f"{where}: {len(columns)} columns for {len(notes)} cues at {count}")
            if any(not 0 <= column < count for column in columns):
                complain(f"{where}: a cue drawn outside the {count} columns")

        columns = loaded["column_assignments"].get(str(lane_count)) or []
        per_lane = Counter(columns)
        if sorted(per_lane) != list(range(lane_count)):
            complain(f"{where}: {len(per_lane)} of {lane_count} lanes used")
        for lane, count in sorted(per_lane.items()):
            if count != cues_per_lane:
                complain(f"{where}: lane {lane} has {count} cues, wanted {cues_per_lane}")


def check_config(path, lane_count, complain):
    config = json.loads(path.read_text())
    classes = config["collection_classes"]
    if not classes:
        complain(f"{path.name}: no collection_classes")
    if len(classes) > MAXIMUM_COLUMNS:
        complain(f"{path.name}: {len(classes)} classes, the game supports {MAXIMUM_COLUMNS}")
    if len(classes) != lane_count:
        complain(f"{path.name}: {len(classes)} classes for a {lane_count} lane track")
    identifiers = [entry["id"] for entry in classes]
    if len(set(identifiers)) != len(identifiers):
        complain(f"{path.name}: a class id is repeated")
    for entry in classes:
        if entry.get("motion") is not None and "hint" not in entry["motion"]:
            complain(f"{path.name}: {entry['id']} has an arrow with no hint")


def main():
    problems = []
    for name, config_path, cues_per_lane in PAIRINGS:
        config = DASHBOARD / config_path
        lane_count = len(json.loads(config.read_text())["collection_classes"])
        check_config(config, lane_count, problems.append)
        check_track(DASHBOARD / "tracks" / name, lane_count, cues_per_lane, problems.append)
        print(f"{name}: {lane_count} lanes, {cues_per_lane} cues per class")

    for problem in problems:
        print(f"problem: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
