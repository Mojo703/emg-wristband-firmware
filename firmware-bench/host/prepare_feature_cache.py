"""Build the float64 feature cache for every session the bench uses.

    python3 prepare_feature_cache.py [session ...]
"""

import pickle
import sys

from bench_sessions import ALL_SESSIONS, FEATURE_CACHE
from reference_pipeline import prepare


def cache_path(name):
    return FEATURE_CACHE / f"{name}.pkl"


def load(name):
    with cache_path(name).open("rb") as handle:
        return pickle.load(handle)


def main():
    names = sys.argv[1:] or ALL_SESSIONS
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        out = cache_path(name)
        if out.exists():
            print(f"cached  {name}", flush=True)
            continue
        data = prepare(name)
        with out.open("wb") as handle:
            pickle.dump(data, handle)
        print(f"{name}: {len(data['cue_rows'])} cue windows over "
              f"{len(data['cue_spans'])} cues, {len(data['rest_rows'])} rest, "
              f"{len(data['replay_rows'])} replay, align {data['align_ms']:.1f} ms",
              flush=True)


if __name__ == "__main__":
    main()
