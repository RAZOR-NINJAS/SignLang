import select
import shutil
import sys
import time

import cv2
import numpy as np

from . import dataset
from .config import TRAIN_LABELS
from .features import normalize_hand
from .hands import HandPipeline, draw_landmarks

BURSTS_PER_LABEL = 4
BURST_FRAMES = 14
MAX_BURST_MISSES = 4
MIN_HAND_SCALE = 0.055
WINDOW = "signlang - collect"

BAR_H = 150


def poll_stdin_keys():
    keys = []
    try:
        if not (sys.stdin is not None and sys.stdin.isatty()):
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


def _synthetic_hand(letter_seed=0.0):
    pts = np.zeros((21, 3), np.float32)
    rng = np.random.default_rng(int(letter_seed * 1000) % 9973)
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


def _selftest(frames=40):
    import tempfile

    import signlang.dataset as ds_mod

    saved_dir = ds_mod.SAMPLE_DIR
    saved_before = sum(ds_mod.counts().values())
    tmp = tempfile.mkdtemp(prefix="signlang-selftest-")
    ds_mod.SAMPLE_DIR = __import__("pathlib").Path(tmp)

    class FakePipe:
        def __init__(self):
            self.n = 0

        def read(self, timeout=2.0):
            self.n += 1
            frame = np.zeros((480, 640, 3), np.uint8)
            pts = _synthetic_hand(self.n)
            lms = [_FakeLandmark(*p) for p in pts]
            return frame, lms, "Right"

        def close(self):
            pass

    import signlang.hands as hands_mod

    real = hands_mod.HandPipeline
    hands_mod.HandPipeline = FakePipe
    try:
        burst_ok = 0
        for label in TRAIN_LABELS[:3]:
            coords = [normalize_hand(_synthetic_hand(i), "Right") for i in range(6)]
            n = ds_mod.append_samples(label, coords)
            burst_ok += 1
            assert n >= 6
        raw = np.load(ds_mod._path_for(TRAIN_LABELS[0]))
        assert raw.ndim == 3 and raw.shape[1:] == (21, 3), f"collector shape drifted: {raw.shape}"
        X, y, kept = ds_mod.load_all()
        assert X.shape[1] == 63, X.shape
        assert X.shape[0] == 18, X.shape
        assert len(kept) == burst_ok, f"{kept}"
        from .features import landmark_features

        landmark_features(X[0].reshape(21, 3))
        for label in TRAIN_LABELS[:3]:
            ds_mod.reset_label(label)
        after = sum(ds_mod.counts().values())
        assert after == 0, f"temp dir not clean: {after}"
        print(
            f"SELFTEST OK  {burst_ok} labels round-tripped in temp dir, "
            f"real data untouched ({saved_before} samples still present)"
        )
        return 0
    finally:
        hands_mod.HandPipeline = real
        ds_mod.SAMPLE_DIR = saved_dir
        real_after = sum(ds_mod.counts().values())
        assert real_after == saved_before, (
            f"REAL DATA CHANGED: {saved_before} -> {real_after}"
        )
        shutil.rmtree(tmp, ignore_errors=True)
        cv2.destroyAllWindows()


def _main():
    if "--selftest" in sys.argv:
        return _selftest()

    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = [s.strip().upper() for s in a.split("=", 1)[1].split(",") if s.strip()]
        elif a == "--only":
            pass
    if len(argv) >= 2 and argv[0] == "--only":
        only = [s.strip().upper() for s in argv[1].split(",") if s.strip()]

    labels = list(TRAIN_LABELS)
    if only:
        wanted = [l for l in only if l in TRAIN_LABELS]
        missing = [l for l in only if l not in TRAIN_LABELS]
        if missing:
            print(f"unknown sign(s), ignored: {' '.join(missing)}")
        if not wanted:
            print("no valid signs selected")
            return 1
        labels = wanted
        print(f"  restricted to {len(labels)} sign(s): {' '.join(labels)}")
    idx = 0
    bursts_done = 0
    recording = None
    last_msg = ""
    msg_until = 0.0
    running = True
    next_tick = 0.0
    interactive = sys.stdin is not None and sys.stdin.isatty()

    print("=" * 62)
    print("  SIGNLANG DATA COLLECTOR")
    print("=" * 62)
    print("  Hold each sign clearly in view, then press SPACE to capture a burst.")
    print("  Vary position/angle slightly between bursts for better training.")
    print()
    print("  SPACE  capture burst      N / B   next / previous sign")
    print("  R      clear this sign    Q      quit and save")
    if not interactive:
        print()
        print("  NOTE: stdin is not a terminal, so keyboard control is disabled.")
    print("=" * 62)
    print("  FOCUS THE VIDEO WINDOW ONLY IF YOU WANT TO USE ITS KEYS;")
    print("  otherwise just type in this terminal.")
    print("=" * 62)

    pipe = HandPipeline()
    try:
        while running:
            frame, landmarks, handedness = pipe.read(timeout=3.0)
            if frame is None:
                print("camera stalled; aborting", file=sys.stderr)
                break

            scale = None
            if landmarks is not None:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], np.float32)
                scale = float(np.linalg.norm(pts[9] - pts[0]))
                draw_landmarks(frame, landmarks)
                cx, cy = int(landmarks[0].x * frame.shape[1]), int(landmarks[0].y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)

            target = labels[idx]
            ready = landmarks is not None and scale is not None and scale >= MIN_HAND_SCALE
            have = dataset.count_for(target)

            if recording is not None:
                recorded, rec_label, misses, wanted = recording
                if ready and len(recorded) < wanted:
                    recorded.append(normalize_hand(pts, handedness))
                    misses = 0
                else:
                    misses += 1
                    if misses > MAX_BURST_MISSES or len(recorded) >= wanted:
                        if len(recorded) >= 3:
                            total = dataset.append_samples(rec_label, recorded)
                            last_msg = f"saved {len(recorded)}/{wanted}  ({rec_label} now {total})"
                        else:
                            last_msg = f"only {len(recorded)} frames - check lighting / framing"
                        print(f"\n  {last_msg}")
                        msg_until = time.monotonic() + 2.5
                        if len(recorded) >= 3:
                            bursts_done += 1
                        recording = None
                        if bursts_done >= BURSTS_PER_LABEL:
                            bursts_done = 0
                            idx = (idx + 1) % len(labels)

            disp = cv2.resize(frame, (frame.shape[1] // 2 * 2, frame.shape[0] // 2 * 2))
            h, w = disp.shape[:2]

            if recording is not None:
                got, want = len(recording[0]), recording[3]
                frac = got / max(want, 1)
                cv2.rectangle(disp, (0, 0), (w, 8), (60, 60, 60), -1)
                cv2.rectangle(disp, (0, 0), (int(w * frac), 8), (0, 220, 120), -1)
                target_text = f"RECORDING  {target}"
                tcol = (0, 255, 140)
            else:
                target_text = target
                tcol = (255, 255, 255) if ready else (120, 120, 130)

            big = 1.5 if len(target) < 4 else 1.0
            lines = [
                (target_text, big, tcol, 3 if big > 1.2 else 2),
                (f"sign {idx + 1}/{len(labels)}   bursts {bursts_done}/{BURSTS_PER_LABEL}   stored {have}",
                 0.5, (170, 170, 180), 1),
            ]
            if not ready:
                lines.append(("show one hand clearly, palm to camera", 0.55, (80, 160, 255), 1))
            if time.monotonic() < msg_until:
                lines.append((last_msg, 0.55, (0, 220, 255), 1))
            lines.append(("SPACE capture   N/B next/prev   R clear   Q quit",
                          0.45, (110, 110, 120), 1))
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
                    elif not ready:
                        last_msg = "no hand detected"
                        msg_until = time.monotonic() + 1.5
                elif key in ("n", "N"):
                    idx = (idx + 1) % len(labels)
                    bursts_done = 0
                elif key in ("b", "B"):
                    idx = (idx - 1) % len(labels)
                    bursts_done = 0
                elif key in ("r", "R"):
                    dataset.reset_label(target)
                    bursts_done = 0
                    last_msg = f"cleared stored samples for {target}"
                    msg_until = time.monotonic() + 2.0
                    print(f"  cleared {target}")
            if not running:
                break

            if recording is not None:
                got, want = len(recording[0]), recording[3]
                sys.stdout.write("\r  recording... %2d/%d" % (got, want))
                sys.stdout.flush()
            elif time.monotonic() > next_tick:
                next_tick = time.monotonic() + 0.25
                sys.stdout.write(
                    "\r  next: %-10s  stored=%-5d burst=%d/%d   (SPACE=capture)  "
                    % (target, have, bursts_done, BURSTS_PER_LABEL)
                )
                sys.stdout.flush()
    finally:
        pipe.close()
        cv2.destroyAllWindows()
        sys.stdout.write("\n")

    report, kept = dataset.class_report()
    total = sum(report.values())
    print("\n" + "=" * 62)
    print(f"  COLLECTION COMPLETE   {total} samples across {len(kept)} signs")
    print("=" * 62)
    empty = [k for k, v in report.items() if v == 0]
    if empty:
        print("  MISSING SIGNS: " + " ".join(empty))
    low = [f"{k}({v})" for k, v in report.items() if 0 < v < 40]
    if low:
        print("  THIN SIGNS: " + " ".join(low))
    print("  Next: uv pip run signlang train" if total else "  No data recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
