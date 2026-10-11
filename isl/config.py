"""Configuration constants and paths for ISL (Indian Sign Language) mode."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISL_DIR = PROJECT_ROOT / "isl"
MODEL_DIR = ISL_DIR / "models"
DATA_DIR = ISL_DIR / "data"
SAMPLE_DIR = DATA_DIR / "samples"
WEIGHTS_PATH = MODEL_DIR / "isl_mlp.pt"
LABELS_PATH = MODEL_DIR / "labels.json"
LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Manual Alphabet: ISLRTC standard 2-handed alphabet
ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Phase 1 static letters (excluding dynamic motion letters J and Z)
STATIC_LETTERS = [c for c in ALL_LETTERS if c not in ("J", "Z")]
SPECIAL_SIGNS = ["SPACE"]

# Static ISL Words (iconic signs requiring no hand movement)
STATIC_WORDS = [
    "NAMASTE",  # Both flat hands joined at chest in prayer/greeting [2 hands]
    "GOOD",     # Thumbs up gesture [1 hand]
    "BAD",      # Pinky finger extended upright [1 hand]
    "PEACE",    # V-sign with index & middle fingers [1 hand]
    "OK",       # Thumb & index forming a circle/ring, 3 fingers up [1 hand]
    "HOUSE",    # Both hands with fingertips touching to form a roof [2 hands]
    "BOOK",     # Both open palms facing upward touching pinky edges [2 hands]
    "TIME",     # Dominant index pointing at base wrist [2 hands]
    "LOVE",     # Thumb, index, and pinky extended [1 hand]
    "STOP",     # Open flat palm facing forward [1 hand]
]

TRAIN_LABELS = STATIC_LETTERS + SPECIAL_SIGNS + STATIC_WORDS

# Expected number of hands per sign (1 or 2):
# - 1-hand: C, L, V, SPACE, GOOD, BAD, PEACE, OK, LOVE, STOP
# - 2-hands: A, B, D-I, K, M-U, W-Y, NAMASTE, HOUSE, BOOK, TIME
EXPECTED_HANDS = {
    # Letters (24 static + dynamic J, Z)
    "A": 2, "B": 2, "C": 1, "D": 2, "E": 2, "F": 2, "G": 2, "H": 2,
    "I": 2, "J": 2, "K": 2, "L": 1, "M": 2, "N": 2, "O": 2, "P": 2,
    "Q": 2, "R": 2, "S": 2, "T": 2, "U": 2, "V": 1, "W": 2, "X": 2,
    "Y": 2, "Z": 2, "SPACE": 1,
    # Static Words
    "NAMASTE": 2,
    "GOOD": 1,
    "BAD": 1,
    "PEACE": 1,
    "OK": 1,
    "HOUSE": 2,
    "BOOK": 2,
    "TIME": 2,
    "LOVE": 1,
    "STOP": 1,
}

IS_STATIC = {
    k: (k not in ("J", "Z")) for k in ALL_LETTERS + SPECIAL_SIGNS + STATIC_WORDS
}

# MediaPipe landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP = 9
MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 10, 11, 12
RING_MCP = 13
RING_PIP, RING_DIP, RING_TIP = 14, 15, 16
PINKY_MCP = 17
PINKY_PIP, PINKY_DIP, PINKY_TIP = 18, 19, 20

DIGIT_CHAINS = [
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Hardware & Pipeline
CAMERA_INDEX = int(os.environ.get("SIGNLANG_CAMERA", "0"))
SOURCE = os.environ.get("SIGNLANG_SOURCE", str(CAMERA_INDEX))
CAPTURE_WIDTH = int(os.environ.get("SIGNLANG_WIDTH", "640"))
CAPTURE_HEIGHT = int(os.environ.get("SIGNLANG_HEIGHT", "480"))
DETECT_HEIGHT = int(os.environ.get("SIGNLANG_DETECT_HEIGHT", str(CAPTURE_HEIGHT)))
MIRROR = os.environ.get("SIGNLANG_MIRROR", "1") not in ("0", "false", "False")
DOMINANT_HAND = os.environ.get("SIGNLANG_DOMINANT", "right").strip().lower()
NUM_HANDS = 2
READ_TIMEOUT_S = float(os.environ.get("SIGNLANG_READ_TIMEOUT", "3.0"))
GPU_DELEGATE = os.environ.get("SIGNLANG_GPU", "1") not in ("0", "false", "False")
DETECT_EVERY = max(1, int(os.environ.get("SIGNLANG_DETECT_EVERY", "1")))

# Feature dimensionality
SINGLE_HAND_DIM = 74
INTER_HAND_DIM = 14
PRESENCE_DIM = 2
FEATURE_DIM = SINGLE_HAND_DIM * 2 + INTER_HAND_DIM + PRESENCE_DIM  # 164

# Dwell & Recognition timings
BURST_FRAMES = 14
DWELL_MS = 1000
SPACE_DWELL_MS = 800
SMOOTH_MS = 420
CONFIRM_THRESHOLD = 0.70
STABLE_MARGIN = 0.15
REPEAT_COOLDOWN_MS = 900
SPACE_COOLDOWN_MS = 900

# Audio / TTS
TTS_VOICE = "en_US-lessac-medium"
