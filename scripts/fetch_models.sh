#!/usr/bin/env bash
# Downloads the MediaPipe hand landmarker and Piper TTS voice models.
# POSIX systems (Linux, macOS, WSL). On native Windows use fetch_models.py.
# Idempotent: skips files that are already present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

# 1. MediaPipe Hand Landmarker
LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if [ -f "$MODELS_DIR/hand_landmarker.task" ]; then
  echo "already present: models/hand_landmarker.task"
else
  echo "downloading hand landmarker (~7.8 MB)..."
  curl -fL --progress-bar "$LANDMARKER_URL" -o "$MODELS_DIR/hand_landmarker.task"
  echo "saved to models/hand_landmarker.task"
fi

# 2. Piper TTS Voice Model
VOICE_ONNX_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
VOICE_JSON_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

if [ -f "$ROOT/en_US-lessac-medium.onnx" ]; then
  echo "already present: en_US-lessac-medium.onnx"
else
  echo "downloading Piper TTS voice model (~61 MB)..."
  curl -fL --progress-bar "$VOICE_ONNX_URL" -o "$ROOT/en_US-lessac-medium.onnx"
  echo "saved to en_US-lessac-medium.onnx"
fi

if [ -f "$ROOT/en_US-lessac-medium.onnx.json" ]; then
  echo "already present: en_US-lessac-medium.onnx.json"
else
  echo "downloading Piper TTS voice config..."
  curl -fL --progress-bar "$VOICE_JSON_URL" -o "$ROOT/en_US-lessac-medium.onnx.json"
  echo "saved to en_US-lessac-medium.onnx.json"
fi

echo "All models verified and ready."
