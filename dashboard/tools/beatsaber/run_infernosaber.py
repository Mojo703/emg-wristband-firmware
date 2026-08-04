"""Headless InfernoSaber runner.

Generates a Beat Saber v3 map for every audio file staged in
beatsaber/Data/prediction/songs_predict/ and leaves the result in
beatsaber/Data/prediction/new_map/.

Run from the InfernoSaber checkout with the .venv-beatsaber interpreter:

    cd dashboard/tools/beatsaber/InfernoSaber
    ../../.venv-beatsaber/bin/python ../run_infernosaber.py --difficulty 5
"""

import argparse
import os
import sys
import time

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_directory, "InfernoSaber"))

parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", type=float, default=5.0,
                    help="target notes per second (Beat Saber 'bps' difficulty)")
parser.add_argument("--mapper", default="fav_15",
                    help="huggingface model branch / mapper style")
parser.add_argument("--single-hand", action="store_true",
                    help="restrict output to one note at a time")
arguments = parser.parse_args()

from app_helper.set_app_paths import set_app_paths  # noqa: E402
from tools.config import config  # noqa: E402

config.use_mapper_selection = arguments.mapper
set_app_paths(os.path.join(script_directory, "Data"))

from main import main  # noqa: E402

start = time.time()
main(use_model=arguments.mapper,
     diff=arguments.difficulty * 4.0,
     export_results_to_bs=False,
     single_mode=arguments.single_hand or None,
     legacy_mode=False)
print(f"Total wall time: {time.time() - start:.1f}s")
