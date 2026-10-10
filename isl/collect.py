"""Interactive data collection for Indian Sign Language (ISL) two-handed signs."""

import argparse
import select
import shutil
import sys
import tempfile
import time
from typing import List, Optional

if sys.platform == "win32":
    import msvcrt

import cv2
import numpy as np

from signlang.hands import HandPipeline, draw_landmarks

from . import dataset
from .config import (
    BURST_FRAMES,
    DOMINANT_HAND,
    EXPECTED_HANDS,
    SAMPLE_DIR,
    TRAIN_LABELS,
)
from .features import extract_features

BURSTS_PER_LABEL = 4
WINDOW = "signlang - ISL collect"
BAR_H = 150


def poll_stdin_keys():
    """Non-blocking terminal key reader."""
    keys = []
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return keys
        if sys.platform == "win32":
            while msvcrt.kbhit():
                keys.append(msvcrt.getwche())
            return keys
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys.append(ch)
    except (ValueError, OSError):
        pass
    return keys


def _panel(frame, lines, origin_y=None):
    h, w = frame.shape[:2]
    y0 = h - BAR_H if origin_y is None else origin_y
    cv2.rectangle(frame, (0, y0), (w, h), (18, 18, 22), -1)
    for i, (text, scale, color, thickness) in enumerate(lines):
        cv2.putText(
            frame, text, (18, y0 + 34 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness, cv2.LINE_AA,
        )
    return frame


def _synthetic_hand(seed=0.0):
    pts = np.zeros((21, 3), np.float32)
    rng = np.random.default_rng(int(seed * 1000) % 9973)
    for i in range(21):
        pts[i] = [
            0.5 + (rng.random() - 0.5) * 0.30,
            0.55 + (rng.random() - 0.5) * 0.30,
            rng.random() * 0.05,
        ]
    return pts


class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


def _selftest():
    """Headless selftest for ISL collection and dataset round-tripping."""
    saved_dir = dataset.SAMPLE_DIR
    saved_before = sum(dataset.counts().values())
    tmp = tempfile.mkdtemp(prefix="isl-selftest-")
    dataset.SAMPLE_DIR = __import__("pathlib").Path(tmp)

    try:
        for label in TRAIN_LABELS[:3]:
            # Generate 6 synthetic 164-dim feature vectors
            feats = [
                extract_features(
                    [_synthetic_hand(i), _synthetic_hand(i + 10)],
                    ["Left", "Right"],
                )
                for i in range(6)
            ]
            n = dataset.append_samples(label, feats, source="selftest_cam0")
            assert n == 6

        X, y, kept = dataset.load_all()
        assert X.shape == (18, 164), f"Unexpected shape {X.shape}"
        assert len(kept) == 3

        report, _ = dataset.class_report()
        assert sum(report.values()) == 18

        for label in TRAIN_LABELS[:3]:
            dataset.reset_label(label)
        assert sum(dataset.counts().values()) == 0

        print(
            f"ISL SELFTEST OK: round-tripped 18 samples (164-dim) in temp directory. "
            f"Real data untouched ({saved_before} samples present)."
        )
        return 0
    finally:
        dataset.SAMPLE_DIR = saved_dir
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    args_list = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in args_list:
        return _selftest()

    parser = argparse.ArgumentParser(prog="signlang isl collect")
    parser.add_argument("--only", default=None, help="Comma-separated subset of signs, e.g. A,B,C")
    parser.add_argument("--session", default="session1", help="Session tag for evaluation splitting (e.g. session1, session2)")
    parser.add_argument("--bursts", type=int, default=BURSTS_PER_LABEL, help="Bursts per sign (default: 4)")
    args = parser.parse_args(args_list)

    labels = list(TRAIN_LABELS)
    if args.only:
        wanted = [s.strip().upper() for s in args.only.split(",") if s.strip() in TRAIN_LABELS]
        if wanted:
            labels = wanted
            print(f"  Restricted to {len(labels)} sign(s): {' '.join(labels)}")

    idx = 0
    bursts_done = 0
    recording = None
    last_msg = ""
    msg_until = 0.0
    running = True
    session_tag = dataset.source_tag(session=args.session)

    print("=" * 64)
    print(f"  ISL DATA COLLECTOR (Session: {args.session})")
    print("=" * 64)
    print("  Hold the sign clearly in view, then press SPACE to capture a burst.")
    print("  Note expected hand count: 1 hand for single-hand signs, both hands for 2-hand signs.")
    print()
    print("  SPACE capture burst      N / B   next / previous sign")
    print("  R     clear this sign    Q       quit and save")
    print("=" * 64)

    pipe = HandPipeline(num_hands=2)
    pipe.started.wait(timeout=10)
    if pipe.error is not None:
        print(f"Camera error: {pipe.error}", file=sys.stderr)
        return 1

    try:
        while running:
            frame, landmarks, handedness = pipe.read(timeout=3.0)
            if frame is None:
                continue

            target = labels[idx]
            expected = EXPECTED_HANDS.get(target, 2)

            # Count visible hands
            visible_hands = 0
            if landmarks:
                if isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], list):
                    visible_hands = len(landmarks)
                    draw_landmarks(frame, landmarks)
                else:
                    visible_hands = 1
                    draw_landmarks(frame, landmarks)

            hand_count_ok = (visible_hands >= expected) if expected == 2 else (visible_hands >= 1)
            ready = hand_count_ok and landmarks is not None
            have = dataset.count_for(target)

            if recording is not None:
                recorded, rec_label, misses, wanted = recording
                if ready and len(recorded) < wanted:
                    # Parse into lists
                    lms_list = landmarks if isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], list) else [landmarks]
                    hnd_list = handedness if isinstance(handedness, list) else [handedness]
                    feat = extract_features(lms_list, hnd_list, dominant=DOMINANT_HAND)
                    recorded.append(feat)
                    misses = 0
                else:
                    misses += 1
                    if misses > 4 or len(recorded) >= wanted:
                        if len(recorded) >= 4:
                            total = dataset.append_samples(rec_label, recorded, source=session_tag)
                            last_msg = f"Saved {len(recorded)}/{wanted} ({rec_label} now {total} in {args.session})"
                            bursts_done += 1
                        else:
                            last_msg = f"Only {len(recorded)} frames captured; hold steady and retry"
                        print(f"\n  {last_msg}")
                        msg_until = time.monotonic() + 2.5
                        recording = None
                        if bursts_done >= args.bursts:
                            bursts_done = 0
                            idx = (idx + 1) % len(labels)

            disp = cv2.resize(frame, (frame.shape[1] // 2 * 2, frame.shape[0] // 2 * 2))
            h, w = disp.shape[:2]

            # HUD display lines
            target_str = f"ISL: {target} ({expected}-handed)"
            color_state = (0, 255, 140) if ready else (120, 120, 130)
            lines = [
                (target_str, 1.2, (0, 255, 140) if recording else color_state, 2),
                (f"Sign {idx + 1}/{len(labels)}  Bursts {bursts_done}/{args.bursts}  Stored {have}  Session: {args.session}",
                 0.5, (170, 170, 180), 1),
            ]
            if not hand_count_ok:
                hint = "Show BOTH hands clearly" if expected == 2 else "Show single hand clearly"
                lines.append((hint, 0.6, (0, 160, 255), 2))
            elif ready:
                lines.append(("Ready! Press SPACE to capture burst", 0.55, (0, 220, 120), 1))

            if time.monotonic() < msg_until:
                lines.append((last_msg, 0.55, (0, 220, 255), 1))
            lines.append(("SPACE capture   N/B next/prev   R clear   Q quit", 0.45, (110, 110, 120), 1))
            _panel(disp, lines)

            cv2.imshow(WINDOW, disp)
            keys = poll_stdin_keys()
            win_key = cv2.waitKey(1) & 0xFF
            if win_key not in (255, -1):
                keys.append(chr(win_key))

            for key in keys:
                if key in ("q", "\x1b"):
                    running = False
                    break
                elif key == " ":
                    if ready and recording is None:
                        recording = ([], target, 0, BURST_FRAMES)
                    elif not hand_count_ok:
                        last_msg = f"Cannot capture: {target} requires {expected} hand(s) (saw {visible_hands})"
                        msg_until = time.monotonic() + 2.0
                        print(f"\n  WARNING: {last_msg}")
                elif key in ("n", "N"):
                    idx = (idx + 1) % len(labels)
                    bursts_done = 0
                elif key in ("b", "B"):
                    idx = (idx - 1) % len(labels)
                    bursts_done = 0
                elif key in ("r", "R"):
                    dataset.reset_label(target)
                    bursts_done = 0
                    last_msg = f"Cleared stored samples for {target}"
                    msg_until = time.monotonic() + 2.0
                    print(f"  cleared {target}")
    finally:
        pipe.close()
        cv2.destroyAllWindows()

    report, kept = dataset.class_report()
    total = sum(report.values())
    print("\n" + "=" * 64)
    print(f"  COLLECTION SUMMARY: {total} samples across {len(kept)} signs")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
