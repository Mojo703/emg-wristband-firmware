"""Write the two rest tracks into the (untracked) track library: static rest is
silence, moving rest clicks at 55 beats per minute to pace pole swings. Their
`rest` field makes the backend log the playback as one rest span.

    python3 scripts/make_rest_tracks.py [--minutes 3]
"""

import argparse
import json
import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

TRACKS = Path(__file__).resolve().parent.parent / "tracks"
SAMPLE_RATE = 44100
CLICK_BEATS_PER_MINUTE = 55.0
CLICK_FREQUENCY_HZ = 660.0
CLICK_SECONDS = 0.06
CLICK_AMPLITUDE = 0.35
LEVELS = ("easy", "medium", "hard")


def empty_levels():
    level = {
        "map_notes": [],
        "column_assignments": {str(count): [] for count in range(1, 7)},
    }
    return {name: level for name in LEVELS}


def write_track_json(directory, track_id, title, minutes, rest_label,
                     beats_per_minute):
    entry = {
        "id": track_id,
        "title": title,
        "beats_per_minute": beats_per_minute,
        "duration_ms": int(minutes * 60_000),
        "levels": empty_levels(),
        "rest": rest_label,
    }
    (directory / "track.json").write_text(json.dumps(entry, indent=1) + "\n")


def write_audio(directory, minutes, click):
    seconds = int(minutes * 60)
    total = seconds * SAMPLE_RATE
    samples = bytearray()
    period = 60.0 / CLICK_BEATS_PER_MINUTE
    for index in range(total):
        t = index / SAMPLE_RATE
        value = 0.0
        if click:
            into_beat = t % period
            if into_beat < CLICK_SECONDS:
                envelope = 1.0 - into_beat / CLICK_SECONDS
                value = (CLICK_AMPLITUDE * envelope
                         * math.sin(2.0 * math.pi * CLICK_FREQUENCY_HZ * into_beat))
        samples += struct.pack("<h", int(value * 32767))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=3.0)
    arguments = parser.parse_args()

    plans = [
        ("rest-static", "rest_static", "Static Rest", "static", False),
        ("rest-moving", "rest_moving", "Moving Rest", "moving", True),
    ]
    for directory_name, track_id, title, label, click in plans:
        directory = TRACKS / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        write_track_json(directory, track_id, title, arguments.minutes, label,
                         CLICK_BEATS_PER_MINUTE)
        write_audio(directory, arguments.minutes, click)
        print(f"{directory}: {title}, {arguments.minutes:g} min")


if __name__ == "__main__":
    main()
