"""Configuration constants for ASL Words recognition mode.

Optimized for low-power CPU-only execution (AMD A6-9220 dual-core, 4GB RAM, no GPU).
Uses MediaPipe Holistic for motion-based landmark sequence capture across body and hands.
"""

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
FRAME_SKIP = 2
SEQUENCE_LENGTH = 30

# 16-word ASL motion vocabulary
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

# Landmark feature shapes
NUM_BODY_LANDMARKS = len(BODY_REFERENCE_INDICES)  # 9
BODY_LEFT_SHOULDER_IDX = 1
BODY_RIGHT_SHOULDER_IDX = 2
TOTAL_LANDMARKS = NUM_BODY_LANDMARKS + (2 * NUM_HAND_LANDMARKS)  # 9 + 42 = 51
COORDS_PER_LANDMARK = 3  # (x, y, z)
FEATURE_DIM = TOTAL_LANDMARKS * COORDS_PER_LANDMARK  # 153

# Classifier & recognition tunables
CONFIDENCE_THRESHOLD = 0.65
KNN_NEIGHBORS = 3
ENERGY_ACTIVE_THRESHOLD = 0.035
ENERGY_QUIET_THRESHOLD = 0.015
COOLDOWN_FRAMES = 15

