import argparse
import json
import os
import urllib.request
from pathlib import Path

SESSIONS = Path(__file__).resolve().parent.parent.parent / "dashboard" / "sessions"
FILES = ("session.json", "events.jsonl", "emg.missing", "emg.i16")


def fetch(base, token, path):
    request = urllib.request.Request(f"{base.rstrip('/')}{path}", headers={"authorization": f"Bearer {token}", "user-agent": "emg-pull-sessions"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="https://collect.underfloor.ca")
    parser.add_argument("--token", default=os.environ.get("COLLECTION_TOKEN"))
    arguments = parser.parse_args()
    if not arguments.token:
        parser.error("pass --token or set COLLECTION_TOKEN")

    for identifier in json.loads(fetch(arguments.base, arguments.token, "/sessions")):
        directory = SESSIONS / identifier
        if (directory / "session.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            (directory / name).write_bytes(fetch(arguments.base, arguments.token, f"/sessions/{identifier}/{name}"))
        print(f"pulled {identifier}")


if __name__ == "__main__":
    main()
