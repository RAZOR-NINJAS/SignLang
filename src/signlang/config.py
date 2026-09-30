from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "samples"
WEIGHTS_PATH = MODEL_DIR / "signs_mlp.pt"
LABELS_PATH = MODEL_DIR / "labels.json"
LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

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

TRAIN_LABELS = LETTERS + WORDS

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

CAMERA_INDEX = int(os.environ.get("SIGNLANG_CAMERA", "0"))
SOURCE = os.environ.get("SIGNLANG_SOURCE", str(CAMERA_INDEX))
CAPTURE_WIDTH = int(os.environ.get("SIGNLANG_WIDTH", "640"))
CAPTURE_HEIGHT = int(os.environ.get("SIGNLANG_HEIGHT", "480"))
DETECT_HEIGHT = int(os.environ.get("SIGNLANG_DETECT_HEIGHT", str(CAPTURE_HEIGHT)))
MIRROR = os.environ.get("SIGNLANG_MIRROR", "1") not in ("0", "false", "False")
MAX_HANDS = int(os.environ.get("SIGNLANG_MAX_HANDS", "1"))
READ_TIMEOUT_S = float(os.environ.get("SIGNLANG_READ_TIMEOUT", "3.0"))

DWELL_MS = 620
SMOOTH_MS = 420
CONFIRM_THRESHOLD = 0.72
STABLE_MARGIN = 0.18
REPEAT_COOLDOWN_MS = 900
SWIPE_MIN_PX = 110
SWIPE_MAX_MS = 420

TTS_VOICE = "en_US-lessac-medium"
