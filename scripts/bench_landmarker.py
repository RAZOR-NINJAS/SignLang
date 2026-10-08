#!/usr/bin/env python
"""Benchmark the MediaPipe hand landmarker inference cost on a synthetic frame.

No camera needed. Times detect_for_video() on a blank 640x480 RGB image so we can
measure the model's CPU cost in isolation (the thing that caps the frame rate).

Usage: .venv/bin/python scripts/bench_landmarker.py [model_path] [iters]
"""
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/hand_landmarker.task"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
WARM = 5

frame = np.zeros((480, 640, 3), dtype=np.uint8)
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

lm = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

ts = 0
for _ in range(WARM):
    lm.detect_for_video(img, ts); ts += 1

times = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    lm.detect_for_video(img, ts); ts += 1
    times.append(time.perf_counter() - t0)

times.sort()
avg = sum(times) / len(times)
print(f"model: {MODEL}")
print(f"  mean {avg*1000:.1f} ms  p50 {times[len(times)//2]*1000:.1f} ms  p95 {times[int(len(times)*0.95)]*1000:.1f} ms")
print(f"  -> {1.0/avg:.1f} fps ceiling (CPU-only, synthetic blank frame)")
lm.close()