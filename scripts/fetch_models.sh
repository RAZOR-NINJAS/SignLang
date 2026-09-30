#!/usr/bin/env bash
# Downloads the MediaPipe hand landmarker into models/.
# Idempotent: skips the file if it is already present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "$DEST"

if [ -f "$DEST/hand_landmarker.task" ]; then
  echo "already present: models/hand_landmarker.task"
  exit 0
fi

echo "downloading hand landmarker (~7.8 MB)..."
curl -fL --progress-bar "$URL" -o "$DEST/hand_landmarker.task"
echo "saved to models/hand_landmarker.task"
