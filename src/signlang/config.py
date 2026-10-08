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

SPECIAL_SIGNS = ["SPACE"]

TRAIN_LABELS = LETTERS + SPECIAL_SIGNS + WORDS

# Frames captured per SPACE press. Samples land in runs of this many near-identical
# frames, so any honest validation split must hold out whole runs -- see eval_burst.py.
BURST_FRAMES = 14

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

# --- Performance levers (verified on this machine, see scripts/bench_landmarker.py) ---
#
# GPU delegate: runs MediaPipe on the GPU via the OpenGL/EGL delegate instead of the
# TensorFlow Lite CPU (XNNPACK) path. On the AMD Radeon iGPU here this drops the
# ~182 ms/frame detection to ~70 ms/frame (~2.6x) with identical landmarks, and it
# still works headlessly (Mesa surfaceless EGL). Falls back to CPU automatically if
# the GPU delegate cannot be created. Set "0" to force the CPU path.
GPU_DELEGATE = os.environ.get("SIGNLANG_GPU", "1") not in ("0", "false", "False")

# Detection frame-skip: run the light MediaPipe hand detector only every Nth frame,
# reusing the last landmarks in between. The dwell timer is wall-clock based
# (engine.py), so recognition timing is unaffected -- only the landmark refresh rate
# drops. 2 nearly doubles the effective frame rate at the cost of slightly slower
# hand tracking; 1 keeps the original every-frame behaviour. Measured on a synthetic
# frame: every=1 -> 5.5 fps, every=2 -> ~11 fps (CPU), GPU + every=1 -> ~14 fps.
DETECT_EVERY = max(1, int(os.environ.get("SIGNLANG_DETECT_EVERY", "1")))

DWELL_MS = 1000
SPACE_DWELL_MS = 800
SMOOTH_MS = 420
CONFIRM_THRESHOLD = 0.72
STABLE_MARGIN = 0.18
REPEAT_COOLDOWN_MS = 900
SPACE_COOLDOWN_MS = 900
SWIPE_MIN_PX = 110
SWIPE_MAX_MS = 420

TTS_VOICE = "en_US-lessac-medium"
