import sys

from . import collect, train


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    rest = argv[1:]

    if cmd == "collect":
        return collect._main()
    if cmd == "train":
        return train.main(rest)
    if cmd == "cameras":
        from .cameras import main as cam_main

        return cam_main()
    if cmd == "live":
        from .live import main as live_main

        return live_main(rest)
    if cmd == "check":
        from .check import main as check_main

        return check_main(rest)
    if cmd in ("help", "-h", "--help"):
        print(
            """
signlang - real-time ASL fingerspelling and word-sign translator

  signlang collect    record training samples from your webcam
  signlang train      train the classifier on recorded samples
  signlang live        terminal recognition (no browser needed)
  signlang cameras    list detected video devices
  signlang check      verify a video source works (fps + hand detection)
  signlang assess     evaluate the trained model

Collect first, then train, then live. Nothing is shipped pre-trained:
the model learns your hand, your angle, and your lighting.

Environment:
  SIGNLANG_SOURCE=http://ip:8080/video   any OpenCV-readable source (incl. HTTP MJPEG)
  SIGNLANG_CAMERA=1        video device index (alternative to SIGNLANG_SOURCE)
  SIGNLANG_DETECT_HEIGHT=720   landmark resolution (higher = better, no extra cost)
  SIGNLANG_MIRROR=0        disable selfie mirroring
"""
        )
        return 0
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
