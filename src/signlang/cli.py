import argparse
import sys

from . import collect, train


def _argparse(rest):
    ap = argparse.ArgumentParser(prog="signlang game")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 to reach it from a phone)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--source", default=None,
                    help="video source; defaults to the configured camera")
    ap.add_argument("--letters", default=None,
                    help="comma-separated letters to allow, e.g. C,D,F")
    ap.add_argument("--no-camera", action="store_true",
                    help="do not open the camera at startup")
    args = ap.parse_args(rest)
    if args.letters:
        args.letters = [c.strip().upper() for c in args.letters.split(",") if c.strip()]
    return args


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
    if cmd == "game":
        from .game.server import serve

        args = _argparse(rest)
        return serve(
            host=args.host,
            port=args.port,
            source=args.source,
            include=args.letters,
            camera=not args.no_camera,
        )
    if cmd in ("export", "import"):
        from .transfer import main as transfer_main

        # transfer.main takes the subcommand as argv[0], like train.main does
        return transfer_main([cmd, *rest])
    if cmd in ("help", "-h", "--help"):
        print(
            """
signlang - real-time ASL fingerspelling and word-sign translator

  signlang collect    record training samples from your webcam
  signlang train      train the classifier on recorded samples
  signlang live        terminal recognition (no browser needed)
  signlang game        shadow-play fingerspelling game in the browser
  signlang cameras    list detected video devices
  signlang check      verify a video source works (fps + hand detection)
  signlang assess     evaluate the trained model
  signlang export     package recordings + model to move to another machine
  signlang import     merge a bundle from another machine

Collect, train, then live. A model for A-H and K-N ships with the repo, but it
was trained on someone else's hands: collect your own samples and retrain for
the best accuracy.

Moving between machines (e.g. train on a fast laptop, demo on another):
  signlang export            writes signlang-transfer.tar.gz
  <copy it>                  transfer recordings + trained model
  signlang import            merges it into the other project

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
