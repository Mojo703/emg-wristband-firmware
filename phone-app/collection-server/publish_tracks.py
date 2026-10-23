import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACKS = ROOT.parent.parent / "dashboard" / "tracks"
BUCKET = "emg-collection"
LEVEL = "hard"


def wrangler_put(key, path, content_type, location):
    subprocess.run(
        ["pnpm", "dlx", "wrangler", "r2", "object", "put", f"{BUCKET}/{key}",
         "--file", str(path), "--content-type", content_type, location],
        cwd=ROOT, check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", nargs="*", type=Path)
    parser.add_argument("--local", action="store_true")
    arguments = parser.parse_args()
    location = "--local" if arguments.local else "--remote"

    directories = arguments.tracks or sorted(path for path in TRACKS.iterdir() if path.is_dir())
    index = []
    with tempfile.TemporaryDirectory() as scratch:
        for directory in directories:
            track = json.loads((directory / "track.json").read_text())
            notes = track.get("levels", {}).get(LEVEL, {}).get("map_notes", [])
            if track.get("rest") is not None or not notes:
                continue
            identifier = directory.name
            audio = Path(scratch) / f"{identifier}.m4a"
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(directory / "audio.ogg"),
                 "-vn", "-c:a", "aac", "-b:a", "160k", str(audio)],
                check=True,
            )
            summary = {"id": identifier, "title": track["title"], "duration_ms": track["duration_ms"]}
            slim = dict(summary, levels={LEVEL: {"map_notes": [{"time_ms": n["time_ms"]} for n in notes]}})
            description = Path(scratch) / f"{identifier}.json"
            description.write_text(json.dumps(slim))
            wrangler_put(f"tracks/{identifier}/audio.m4a", audio, "audio/mp4", location)
            wrangler_put(f"tracks/{identifier}/track.json", description, "application/json", location)
            index.append(summary)
            print(f"published {identifier}: {len(notes)} notes")
        listing = Path(scratch) / "index.json"
        listing.write_text(json.dumps(index))
        wrangler_put("tracks/index.json", listing, "application/json", location)


if __name__ == "__main__":
    main()
