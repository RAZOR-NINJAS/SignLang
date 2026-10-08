"""Configuration constants for ASL Words recognition mode.

Optimized for low-power CPU-only execution (AMD A6-9220 dual-core, 4GB RAM, no GPU).
Uses MediaPipe Holistic for motion-based landmark sequence capture across body and hands.
"""

import os
from pathlib import Path

# Paths
WORDS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORDS_DIR.parent
DATA_DIR = WORDS_DIR / "data"
MODEL_DIR = WORDS_DIR / "models"

# Video capture and CPU-friendly processing defaults
DEFAULT_SOURCE = "0"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Detection frame-skip: run MediaPipe Holistic every Nth frame.
# Dwell/segmentation timing is wall-clock based, so recognition is unaffected.
# 1 = every frame, 2 = every other frame (default), 3+ = more skip.
FRAME_SKIP = max(1, int(os.environ.get("SIGNLANG_WORDS_DETECT_EVERY", "2")))
SEQUENCE_LENGTH = 30

# 24-word ASL motion vocabulary
WORDS = [
    "HELLO",
    "THANKYOU",
    "PLEASE",
    "YES",
    "NO",
    "HELP",
    "SORRY",
    "GOOD",
    "BAD",
    "NAME",
    "MORE",
    "LOVE",
    "EAT",
    "DRINK",
    "WHERE",
    "FINISHED",
    # Greetings / time-of-day, plus common polite phrases.
    "GOOD_MORNING",
    "MORNING",
    "AFTERNOON",
    "NIGHT",
    "HOW",
    "WELCOME",
    "HAVE",
    "WATER",
    "FOOD",
]
VOCABULARY = WORDS
NUM_WORDS = len(WORDS)

WORD_GLOSS = {
    "HELLO": "hello",
    "THANKYOU": "thank you",
    "PLEASE": "please",
    "YES": "yes",
    "NO": "no",
    "HELP": "help",
    "SORRY": "sorry",
    "GOOD": "good",
    "BAD": "bad",
    "NAME": "name",
    "MORE": "more",
    "LOVE": "love",
    "EAT": "eat",
    "DRINK": "drink",
    "WHERE": "where",
    "FINISHED": "finished",
    "GOOD_MORNING": "good morning",
    "MORNING": "morning",
    "AFTERNOON": "afternoon",
    "NIGHT": "night",
    "HOW": "how",
    "WELCOME": "welcome",
    "HAVE": "have",
    "WATER": "water",
    "FOOD": "food",
}

# MediaPipe Holistic landmark indices
# Hand landmarks (21 points per hand)
NUM_HAND_LANDMARKS = 21
HAND_WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
HAND_LANDMARK_INDICES = list(range(NUM_HAND_LANDMARKS))
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Pose / Body reference landmarks (33 pose points in MediaPipe)
NUM_POSE_LANDMARKS = 33
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24

# Key body reference points used for motion normalization and relative positions
BODY_REFERENCE_INDICES = [
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
]
HOLISTIC_BODY_INDICES = BODY_REFERENCE_INDICES

# CPU performance tunables
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# GPU delegate for MediaPipe Holistic (experimental, falls back to CPU)
# Set SIGNLANG_WORDS_GPU=1 to enable. Requires mediapipe 0.10+ with GPU support.
USE_GPU_DELEGATE = os.environ.get("SIGNLANG_WORDS_GPU", "0") not in ("0", "false", "False")

# Draw landmark skeleton overlay (costs ~2-3ms/frame). Disable for max FPS.
DRAW_LANDMARKS = os.environ.get("SIGNLANG_WORDS_DRAW_LANDMARKS", "1") not in ("0", "false", "False")

# Landmark feature shapes
NUM_BODY_LANDMARKS = len(BODY_REFERENCE_INDICES)  # 9
BODY_LEFT_SHOULDER_IDX = 1
BODY_RIGHT_SHOULDER_IDX = 2
TOTAL_LANDMARKS = NUM_BODY_LANDMARKS + (2 * NUM_HAND_LANDMARKS)  # 9 + 42 = 51
COORDS_PER_LANDMARK = 3  # (x, y, z)
FEATURE_DIM = TOTAL_LANDMARKS * COORDS_PER_LANDMARK  # 153

# Classifier & recognition tunables
#
# CONFIDENCE_THRESHOLD applies to a confidence value that is a *relative*
# separation between the winning class and the runner-up class, not an absolute
# distance. Measured on this dataset: random-noise input scores 0.000-0.003,
# correct real webcam matches score 0.29-0.84, correct synthetic matches
# 0.56-0.96. A threshold of 0.18 sits far above the noise floor and below the
# weakest genuine match, so it separates "real sign" from "not a real sign"
# without needing to know how far apart the classes happen to be.
CONFIDENCE_THRESHOLD = 0.18

# Secondary safety net: a query whose nearest reference is farther than this
# multiple of the training set's typical cross-class distance is rejected as
# "unlike anything we know", even if the relative gap looks large. Measured
# ratios: synthetic 0.1x, real webcam 2.7-4.2x, random noise 13.8x.
MAX_ACCEPT_DISTANCE_RATIO = 8.0

KNN_NEIGHBORS = 3

# Motion energy thresholds, in shoulder-width units, measured against the
# hand-centroid motion of this dataset. Real webcam signs have a peak centroid
# energy of 0.12-0.79 (median 0.35); 0.020 is below every real sequence while
# staying well above idle hand jitter.
ENERGY_ACTIVE_THRESHOLD = 0.020
ENERGY_QUIET_THRESHOLD = 0.010
COOLDOWN_FRAMES = 15

# Landmark dropout handling. MediaPipe Holistic frequently loses a hand for a
# frame or two; missing landmarks arrive as exact zeros, which after
# shoulder-normalization become a large fixed artifact. A hand is considered
# present only if at least this many of its 21 landmarks are non-zero.
LANDMARK_EPSILON = 1e-4
MIN_PRESENT_HAND_LANDMARKS = 10
MIN_PRESENT_BODY_LANDMARKS = 5

# MediaPipe marks pose landmarks it inferred rather than observed with a low
# visibility score; treating those as tracked data adds noise.
POSE_VISIBILITY_MIN = 0.5

# A captured segment is discarded if more than this fraction of its frames are
# missing either shoulder, since those frames cannot be normalized.
MAX_DROPOUT_FRAME_FRACTION = 0.25

