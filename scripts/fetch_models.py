"""Download the MediaPipe hand landmarker into models/.

Windows equivalent of scripts/fetch_models.sh:

    python scripts/fetch_models.py

Uses only the standard library, so it runs before pip install.
Idempotent: skips the file if it is already present.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def main():
    if DEST.exists():
        print(f"already present: {DEST}")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("downloading hand landmarker (~7.8 MB)...")

    tmp = DEST.with_suffix(".part")
    try:
        with urllib.request.urlopen(URL) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(DEST)
    except Exception as exc:  # noqa: BLE001 - user-facing script
        tmp.unlink(missing_ok=True)
        print(f"download failed: {exc}", file=sys.stderr)
        print(f"download it manually from:\n  {URL}", file=sys.stderr)
        return 1

    print(f"saved to {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
