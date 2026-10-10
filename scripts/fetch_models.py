"""Download the MediaPipe hand landmarker and Piper TTS voice models.

Windows equivalent of scripts/fetch_models.sh:

    python scripts/fetch_models.py

Uses only the standard library, so it runs before pip install.
Idempotent: skips files that are already present.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

FILES_TO_FETCH = [
    (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODELS_DIR / "hand_landmarker.task",
        "hand landmarker (~7.8 MB)",
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        ROOT / "en_US-lessac-medium.onnx",
        "Piper TTS voice (~61 MB)",
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        ROOT / "en_US-lessac-medium.onnx.json",
        "Piper TTS voice config",
    ),
]


def fetch(url: str, dest: Path, desc: str) -> bool:
    if dest.exists():
        print(f"already present: {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {desc}...")

    tmp = dest.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(dest)
        print(f"saved to {dest.name}")
        return True
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"download failed for {desc}: {exc}", file=sys.stderr)
        print(f"download it manually from:\n  {url}", file=sys.stderr)
        return False


def main():
    success = True
    for url, dest, desc in FILES_TO_FETCH:
        if not fetch(url, dest, desc):
            success = False

    if success:
        print("All models verified and ready.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
