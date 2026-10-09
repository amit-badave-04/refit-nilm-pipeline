"""Project-wide paths and constants.

Every threshold used anywhere in the analysis lives here so that a
reader can see, in one place, which numbers were chosen and where they came from.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data layout. Archives and caches live in data/ and are not versioned (see data/README.md).
DATA_DIR = PROJECT_ROOT / "data"
EXTRACT_DIR = DATA_DIR / "extracted"
RAW_CSV_DIR = EXTRACT_DIR / "raw"
CLEAN_CSV_DIR = EXTRACT_DIR / "clean"
PARQUET_DIR = DATA_DIR / "parquet"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIG_DIR = ARTIFACTS_DIR / "figures"
TABLE_DIR = ARTIFACTS_DIR / "tables"
METRIC_DIR = ARTIFACTS_DIR / "metrics"
MODEL_DIR = ARTIFACTS_DIR / "models"

# Official REFIT release files (University of Strathclyde PURE, CC BY 4.0). ``mirror`` is an
# unmodified copy attached to this repository's GitHub release, used only if the portal fails;
# the SHA-256 check applies to both.
MIRROR = "https://github.com/amit-badave-04/refit-nilm-pipeline/releases/download/refit-data-081116"
# The portal links in the brief (pureportal.strath.ac.uk/files/...) sit behind an interactive
# browser check; the equivalent "ws/portalfiles" links serve the same files directly.
ARCHIVES = {
    "clean": {
        "filename": "CLEAN_REFIT_081116.7z",
        "mirror": f"{MIRROR}/CLEAN_REFIT_081116.7z",
        "url": "https://pure.strath.ac.uk/ws/portalfiles/portal/62090184/CLEAN_REFIT_081116.7z",
        "sha256": "4dc4dde763da1c843e9d565fc68f1a63752d858980615635994d7f701fdfcc4b",
        "bytes": 514_265_481,
    },
    "raw": {
        "filename": "REFIT_RAW_081116.7z",
        "mirror": f"{MIRROR}/REFIT_RAW_081116.7z",
        "url": "https://pure.strath.ac.uk/ws/portalfiles/portal/62090177/REFIT_RAW_081116.7z",
        "sha256": "f294fe673532220dc527b5e7d7a25e99568bd24acf9fb5be0a0def14af64b2cf",
        "bytes": 698_108_465,
    },
    "clean_readme": {
        "filename": "CLEAN_READ_ME_081116.txt",
        "mirror": f"{MIRROR}/CLEAN_READ_ME_081116.txt",
        "url": "https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt",
        "sha256": "c5f5236fd5a7b29771913480fcef6e9be8a85d07511939c5a35160b7afc97bfb",
        "bytes": 12_020,
    },
}

HOUSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]
CHANNELS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
LOCAL_TZ = "Europe/London"

# --- Washing-machine activation rule (Kelly & Knottenbelt 2015, Table 4) -----------------
WM_ON_THRESHOLD_W = 20.0      # on if power >= 20 W
WM_MIN_OFF_MIN = 3            # bridge off-gaps shorter than 160 s (-> 3 one-minute samples)
WM_MIN_ON_MIN = 30            # discard activations shorter than 1800 s
WM_MAX_POWER_W = 3120.0       # 13 A x 240 V: the most a plug-in appliance can draw. Kelly's 2500 W
                              # flags real 2.55 kW heaters (House 4), so the physical limit is used.

# --- 1-minute protocol (Petralia et al., KDD 2025) ----------------------------------------
RESAMPLE_RULE = "1min"
SHORT_GAP_FILL_MIN = 3        # forward-fill gaps up to 3 minutes; longer gaps stay missing
AGG_CLIP_W = 10_000.0
APPLIANCE_CLIP_W = 4_000.0    # REFIT IAM spike rule (> 4000 W is a sensor fault)

# --- Canonical REFIT washing-machine split (D'Incecco et al. 2020; Zhong's reference code) --
TRAIN_HOUSES = [2, 5, 7, 9, 15, 16, 17]
VAL_HOUSE = 18
TEST_HOUSE = 8
# Further unseen homes: one WM, no PV, no documented WM change, not used for training/selection.
EXTRA_UNSEEN_HOUSES = [1, 6, 10, 19, 20]
# Appliance column of the washing machine used by the reference code, for cross-checking
# the mapping parsed from the README.
REFERENCE_WM_CHANNEL = {2: 2, 5: 3, 7: 5, 8: 4, 9: 3, 15: 3, 16: 5, 17: 4, 18: 5}

SEEDS = [10, 20, 42]
