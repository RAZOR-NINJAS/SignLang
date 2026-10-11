import sys

from .config import (
    CAMERA_INDEX,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DETECT_HEIGHT,
    MIRROR,
)
from .platform_compat import is_windows, probe_cameras


def probe(max_index=6):
    """Probe connected cameras and return [(index, width, height)]."""
    return probe_cameras(max_index=max_index)


def main():
    print("scanning video devices...")
    devs = probe()
    if not devs:
        print("  no working cameras found")
        return 1
    print()
    if is_windows():
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
    if is_windows():
        print("  switch camera:  set SIGNLANG_CAMERA=1 && signlang collect")
    else:
        print("  switch camera:  SIGNLANG_CAMERA=2 signlang collect")
    print("  use more res :  SIGNLANG_DETECT_HEIGHT=720 signlang collect")
    if len(devs) < 2:
        print()
        print("  only one camera found. to add a phone as a webcam, either:")
        print("    - DroidCam / IP Webcam on the LAN (set SIGNLANG_CAMERA to its index)")
        if not is_windows():
            print("    - USB tether + 'PTP' camera over gphoto2 / gtkam")
        print("    - a capture card if the phone outputs HDMI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
