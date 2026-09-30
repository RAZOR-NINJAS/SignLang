import time

import cv2

from .config import DETECT_HEIGHT, MIRROR, SOURCE
from .hands import HandPipeline


def main(argv=None):
    argv = list(argv or [])
    source = argv[0] if argv else SOURCE
    print(f"source   : {source}")
    print(f"detect H : {DETECT_HEIGHT}px    mirror: {MIRROR}")
    print("probing for 10s...\n")

    pipe = HandPipeline(source=source)
    pipe.started.wait(timeout=10)
    if pipe.error is not None:
        print(f"FAILED: {pipe.error}")
        pipe.close()
        return 1

    n, hands, t0, sizes = 0, 0, time.time(), set()
    try:
        while time.time() - t0 < 10:
            frame, lms, handedness = pipe.read(timeout=3.0)
            if frame is None:
                continue
            n += 1
            sizes.add((frame.shape[1], frame.shape[0]))
            if lms is not None:
                hands += 1
    except BaseException as exc:
        print(f"FAILED mid-stream: {exc}")
        pipe.close()
        return 1

    elapsed = time.time() - t0
    pipe.close()

    if n == 0:
        print("FAILED: no frames arrived")
        return 1
    print(f"frames   : {n} in {elapsed:.1f}s  ->  {n/elapsed:.1f} fps")
    print(f"sizes    : {sorted(sizes)}")
    print(f"hands    : detected in {hands}/{n} frames ({100*hands/n:.0f}%)")
    if hands == 0:
        print()
        print("  no hand detected. That is expected with nobody in frame, but if you")
        print("  are in frame: check lighting, and that your hand fills ~1/3 of the view.")
    else:
        print()
        print("  source is good. Hold your hand in view, then run `signlang collect`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
