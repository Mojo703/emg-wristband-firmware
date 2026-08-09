"""Session names, class lists and pipeline constants shared by the bench.

The reference pipeline lives in the main checkout under emg-tds/scripts; this
module only fixes which sessions and classes the firmware bench is built on.
"""

import sys
from pathlib import Path

REFERENCE_SCRIPTS = Path(
    "/home/matthewg/Documents/Projects/EMG-Wristband/emg-tds/scripts")
if str(REFERENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REFERENCE_SCRIPTS))

BENCH = Path(__file__).resolve().parent.parent
FIXTURES = BENCH / "fixtures"
FEATURE_CACHE = FIXTURES / "cache"

MODIFIER = "2026-08-07T22-08-47_Matthew"
SAME_DON = "2026-08-07T22-16-46_Matthew"
REST_STATIC = "2026-08-07T21-22-54_Matthew"
REST_MOVING = "2026-08-07T21-28-08_Matthew"
BASE = [
    "2026-08-07T16-38-35_Matthew",
    "2026-08-07T16-51-04_Matthew",
    "2026-08-07T17-14-43_Matthew",
    "2026-08-07T17-23-35_Matthew",
]
RESTS = {"static": REST_STATIC, "moving": REST_MOVING}
ALL_SESSIONS = [MODIFIER, SAME_DON] + BASE + [REST_STATIC, REST_MOVING]

# The four sessions the firmware bench replays on device.
MISSION_SESSIONS = [MODIFIER, SAME_DON, REST_STATIC, REST_MOVING]

COMMANDS = [
    "thumb_up_pronation",
    "thumb_up_supination",
    "thumb_up_radial_deviation",
    "thumb_up_ulnar_deviation",
    "thumb_up_hold",
]
BASE_CLASSES = [
    "wrist_pronation",
    "wrist_supination",
    "wrist_radial_deviation",
    "wrist_ulnar_deviation",
    "thumb_extension",
]

NUMBER_OF_COMMANDS = 5
CHANNELS = 16
SAMPLES_PER_WINDOW_RECORD = 500
DEVICE_WINDOW_SAMPLES = 500
SAMPLE_RATE = 2000.0
TAU = 0.5
NEEDED = 3
GRACE = 1000.0 * SAMPLE_RATE / 1000.0

ONSET_SKIP = 500
CUE_STRIDE = 125
REST_STRIDE = 250

MODIFIER_FOLD_SEED = 7
SAME_DON_FOLD_SEED = 11
FOLDS = 5
NO_OP_WEIGHT = 0.4
