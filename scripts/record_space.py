#!/usr/bin/env python3
"""Record or generate landmark training samples for the custom SPACE gesture.

Sign: Open flat palm (all 5 fingers extended and spread, palm facing the camera).
This sign is natural, intuitive, and does not clash with any ASL letters A–Z:
- Distinct from 'B': in 'B' the thumb is folded across the palm and 4 fingers are tightly held.
  In open flat palm, the thumb is fully extended outward and fingers are spread.
- Distinct from '5': 5 is a number sign not in A–Z fingerspelling.
- Distinct from 'W': only 3 fingers are upright in 'W' (pinky and thumb tucked).

Usage:
    # Generate high-fidelity synthetic landmark samples (200+ samples, default 336):
    .venv/bin/python scripts/record_space.py --generate

    # Record interactively from camera:
    .venv/bin/python scripts/record_space.py --camera
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signlang import dataset
from signlang.config import BURST_FRAMES, SAMPLE_DIR
from signlang.features import normalize_hand


def generate_open_palm_samples(n_bursts=24, frames_per_burst=BURST_FRAMES, seed=42):
    """Generate realistic open flat palm landmark coordinates.

    All 5 fingers (thumb, index, middle, ring, pinky) are extended and spread wide.
    Coordinates match MediaPipe 21-landmark normalized hand format.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((21, 3), dtype=np.float32)

    # Landmark 0: Wrist at origin
    base[0] = [0.0, 0.0, 0.0]

    # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (TIP) - extended outward and upward
    base[1] = [0.24, -0.28, -0.06]
    base[2] = [0.42, -0.62, -0.12]
    base[3] = [0.60, -0.92, -0.18]
    base[4] = [0.75, -1.18, -0.24]

    # Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP) - extended upright, spread slightly right
    base[5] = [0.22, -0.96, -0.01]
    base[6] = [0.27, -1.33, -0.03]
    base[7] = [0.29, -1.58, -0.06]
    base[8] = [0.30, -1.78, -0.09]

    # Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP) - extended upright along central axis
    base[9] = [0.06, -0.98, -0.02]
    base[10] = [0.06, -1.40, -0.03]
    base[11] = [0.06, -1.68, -0.06]
    base[12] = [0.06, -1.90, -0.09]

    # Ring: 13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP) - extended upright, spread slightly left
    base[13] = [-0.10, -0.92, -0.04]
    base[14] = [-0.14, -1.32, -0.06]
    base[15] = [-0.16, -1.58, -0.09]
    base[16] = [-0.18, -1.78, -0.12]

    # Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP) - extended upright and spread outward
    base[17] = [-0.26, -0.82, -0.07]
    base[18] = [-0.34, -1.16, -0.10]
    base[19] = [-0.39, -1.38, -0.13]
    base[20] = [-0.44, -1.56, -0.15]

    samples = []
    for b in range(n_bursts):
        # Vary orientation per burst: pitch, yaw, roll, finger splay, thumb extension
        b_roll = np.deg2rad(rng.uniform(-12, 12))
        b_pitch = np.deg2rad(rng.uniform(-16, 16))
        b_yaw = np.deg2rad(rng.uniform(-16, 16))
        b_spread = rng.uniform(0.92, 1.14)
        b_thumb = rng.uniform(0.90, 1.15)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(b_pitch), -np.sin(b_pitch)],
            [0, np.sin(b_pitch), np.cos(b_pitch)],
        ], dtype=np.float32)

        Ry = np.array([
            [np.cos(b_yaw), 0, np.sin(b_yaw)],
            [0, 1, 0],
            [-np.sin(b_yaw), 0, np.cos(b_yaw)],
        ], dtype=np.float32)

        Rz = np.array([
            [np.cos(b_roll), -np.sin(b_roll), 0],
            [np.sin(b_roll), np.cos(b_roll), 0],
            [0, 0, 1],
        ], dtype=np.float32)

        R = Rz @ Ry @ Rx

        burst_base = base.copy()
        burst_base[[1, 2, 3, 4], 0] *= b_thumb
        burst_base[[5, 6, 7, 8], 0] *= b_spread
        burst_base[[13, 14, 15, 16], 0] *= b_spread
        burst_base[[17, 18, 19, 20], 0] *= b_spread

        for f in range(frames_per_burst):
            p = burst_base.copy()
            # Frame-level sensor noise/jitter
            p += rng.normal(0, 0.007, p.shape).astype(np.float32)
            p = p @ R.T
            # Re-normalize: wrist at 0, unit scale to middle MCP
            p = p - p[0]
            scale = np.linalg.norm(p[9] - p[0])
            p = p / max(scale, 1e-6)
            samples.append(p)

    return np.array(samples, dtype=np.float32)


def record_from_camera(n_bursts=24, frames_per_burst=BURST_FRAMES):
    """Interactively record open palm landmark bursts from webcam."""
    import cv2
    from signlang.hands import HandPipeline, draw_landmarks

    pipe = HandPipeline()
    print("Hold an OPEN FLAT PALM (all 5 fingers extended and spread) facing the camera.")
    print("Press SPACE to record a burst, Q to finish early.")

    recorded_all = []
    try:
        burst = 0
        while burst < n_bursts:
            frame, landmarks, handedness = pipe.read(timeout=3.0)
            if frame is None:
                break
            if landmarks:
                draw_landmarks(frame, landmarks)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)
                norm = normalize_hand(pts, handedness)
            else:
                norm = None

            cv2.putText(
                frame, f"SPACE gesture (Open Flat Palm) - Burst {burst}/{n_bursts}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("Record SPACE", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" ") and norm is not None:
                # Capture burst
                burst_frames = []
                for _ in range(frames_per_burst):
                    f, lms, hand = pipe.read(timeout=1.0)
                    if lms:
                        p = np.array([[lm.x, lm.y, lm.z] for lm in lms], np.float32)
                        burst_frames.append(normalize_hand(p, hand))
                        time.sleep(0.03)
                if len(burst_frames) >= 5:
                    recorded_all.extend(burst_frames)
                    burst += 1
                    print(f"Captured burst {burst}/{n_bursts} ({len(burst_frames)} frames)")
    finally:
        pipe.close()
        cv2.destroyAllWindows()

    return np.array(recorded_all, dtype=np.float32) if recorded_all else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="Record from camera interactively")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic landmark samples")
    parser.add_argument("--bursts", type=int, default=24, help="Number of bursts to record/generate (default: 24)")
    parser.add_argument("--frames-per-burst", type=int, default=BURST_FRAMES, help="Frames per burst (default: 14)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SPACE samples")
    args = parser.parse_args(argv)

    is_interactive = sys.stdin is not None and sys.stdin.isatty()
    use_camera = args.camera or (not args.generate and is_interactive)

    if use_camera:
        print("Starting camera recording for SPACE gesture...")
        try:
            data = record_from_camera(args.bursts, args.frames_per_burst)
            source_tag = "camera"
        except Exception as exc:
            print(f"Camera recording failed ({exc}), falling back to generation.")
            data = None
    else:
        data = None

    if data is None or len(data) == 0:
        print(f"Generating {args.bursts} bursts x {args.frames_per_burst} frames of open flat palm landmarks...")
        data = generate_open_palm_samples(args.bursts, args.frames_per_burst)
        source_tag = "synthetic"

    if args.force:
        dataset.reset_label("SPACE")

    total = dataset.append_samples("SPACE", data, source=source_tag)
    print(f"Successfully recorded {len(data)} samples for 'SPACE' (total stored: {total}).")
    print(f"Stored at: {dataset._path_for('SPACE')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
