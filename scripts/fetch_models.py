"""Download MediaPipe hand landmarker and Piper TTS voice models.

Cross-platform Python downloader using standard library.
Idempotent: skips files that are already present.
"""

import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

FILES_TO_FETCH = [
    (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODELS_DIR / "hand_landmarker.task",
        "Hand Landmarker Model (~7.8 MB)",
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        ROOT / "en_US-lessac-medium.onnx",
        "Piper TTS Voice (~61 MB)",
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        ROOT / "en_US-lessac-medium.onnx.json",
        "Piper TTS Voice Config",
    ),
]


def fetch(
    url: str,
    dest: Path,
    desc: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Download url to dest with optional progress callback (bytes_downloaded, total_bytes)."""
    if dest.exists() and dest.stat().st_size > 0:
        if progress_cb:
            size = dest.stat().st_size
            progress_cb(size, size)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SignLang-Installer/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 65536
            with tmp.open("wb") as out:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    out.write(block)
                    downloaded += len(block)
                    if progress_cb:
                        progress_cb(downloaded, total)
        tmp.replace(dest)
        return True
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"Download failed for {desc}: {exc}", file=sys.stderr)
        return False


def main():
    print("Verifying model assets...")
    success = True
    for url, dest, desc in FILES_TO_FETCH:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [OK] Already present: {dest.name}")
            continue
        print(f"  Downloading {desc}...")
        if not fetch(url, dest, desc):
            success = False
            print(f"  [FAIL] Could not download {dest.name}")
        else:
            print(f"  [OK] Downloaded: {dest.name}")

    if success:
        print("All required models are present.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
