import sys

import cv2

from .config import (
    CAMERA_INDEX,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DETECT_HEIGHT,
    MIRROR,
)


def probe(max_index=6):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        ok, frame = None, None
        for _ in range(8):
            ok, frame = cap.read()
            if not ok:
                break
        if ok and frame is not None:
            h, w = frame.shape[:2]
            found.append((i, w, h))
        cap.release()
    return found


def main():
    print("scanning video devices...")
    devs = probe()
    if not devs:
        print("  no working cameras found")
        return 1
    print()
    if sys.platform == "win32":
        for i, w, h in devs:
            mark = "  <-- active" if i == CAMERA_INDEX else ""
            print(f"  camera {i}       {w}x{h}{mark}")
    else:
        for i, w, h in devs:
            mark = "  <-- active" if i == CAMERA_INDEX else ""
            print(f"  /dev/video{i:<2d}  {w}x{h}{mark}")
    print()
    print(f"  detect height : {DETECT_HEIGHT}px (landmark accuracy scales with this)")
    print(f"  mirror        : {MIRROR}")
    print()
    print("  switch camera:  SIGNLANG_CAMERA=2 signlang collect")
    print("  use more res :  SIGNLANG_DETECT_HEIGHT=720 signlang collect")
    if len(devs) < 2:
        print()
        print("  only one camera found. to add a phone as a webcam, either:")
        print("    - DroidCam / IP Webcam on the LAN (set SIGNLANG_CAMERA to its index)")
        if sys.platform != "win32":
            print("    - USB tether + 'PTP' camera over gphoto2 / gtkam")
        print("    - a capture card if the phone outputs HDMI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
