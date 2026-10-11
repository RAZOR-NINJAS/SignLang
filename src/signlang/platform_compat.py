"""Cross-platform compatibility abstractions for SignLang.

Handles Windows (DirectShow, Edge/Chrome, msvcrt) and Linux (V4L2, Chromium, POSIX)
differences transparently.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2


def is_windows() -> bool:
    """Return True if running on Microsoft Windows."""
    return sys.platform == "win32"


def is_linux() -> bool:
    """Return True if running on Linux."""
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    """Return True if running on macOS (Darwin)."""
    return sys.platform == "darwin"


def get_camera_backend() -> int:
    """Return the preferred OpenCV capture API preference for the current platform."""
    if is_windows():
        return cv2.CAP_DSHOW
    if is_linux():
        # Prefer V4L2 on Linux, default if unavailable
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def get_camera_capture(
    source: Union[int, str] = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    buffer_size: int = 1,
) -> cv2.VideoCapture:
    """Open a video capture device or file with the optimal backend.

    On Windows, uses cv2.CAP_DSHOW to prevent 5-10s init delays and internal frame buffering.
    On Linux, uses cv2.CAP_V4L2.
    For video files or RTSP streams, uses cv2.CAP_FFMPEG.
    """
    is_cam_idx = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

    if is_cam_idx:
        idx = int(source)
        if is_windows():
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to default backend if DSHOW fails
                cap.release()
                cap = cv2.VideoCapture(idx)
        elif is_linux():
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)
    else:
        # File or network stream
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(str(source))

    if cap.isOpened():
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        except Exception:
            pass

    return cap


def probe_cameras(max_index: int = 6) -> List[Tuple[int, int, int]]:
    """Scan and return list of working camera indices with (index, width, height)."""
    found = []
    for i in range(max_index):
        cap = get_camera_capture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = None, None
        for _ in range(6):
            ok, frame = cap.read()
            if not ok:
                break
        if ok and frame is not None:
            h, w = frame.shape[:2]
            found.append((i, w, h))
        cap.release()
    return found


def find_browser_executable() -> Optional[str]:
    """Locate an installed browser capable of app/kiosk fullscreen display."""
    if is_windows():
        candidates = [
            # Microsoft Edge (default on Win 10/11)
            os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe"),
            # Google Chrome
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            # Brave
            os.path.expandvars(r"%ProgramFiles%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        # Check PATH
        for exe in ("msedge.exe", "chrome.exe"):
            found = shutil.which(exe)
            if found:
                return found
    else:
        # Linux / Unix
        for exe in (
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "brave-browser",
            "microsoft-edge",
        ):
            found = shutil.which(exe)
            if found:
                return found

    return None
