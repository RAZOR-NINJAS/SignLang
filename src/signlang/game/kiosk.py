"""Browser kiosk launcher for SignLang Game and Learning Mode.

Launches Microsoft Edge or Google Chrome in kiosk or app mode across Windows and Linux.
"""

import subprocess
import webbrowser
from typing import Optional

from ..platform_compat import find_browser_executable, is_windows


def launch_browser_kiosk(url: str, kiosk: bool = True) -> Optional[subprocess.Popen]:
    """Launch a browser in app or kiosk mode pointing to the given URL.

    Args:
        url: The web URL to open (e.g. http://127.0.0.1:8000).
        kiosk: If True, uses --kiosk or --app mode to run fullscreen.

    Returns:
        The subprocess.Popen instance if an external browser was launched, or None.
    """
    browser_exe = find_browser_executable()

    if browser_exe:
        flags = [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-translate",
        ]
        if kiosk:
            # --app creates a minimal window without address bar/tabs;
            # --kiosk creates true fullscreen.
            flags.extend([f"--app={url}", "--start-maximized"])
        else:
            flags.append(url)

        cmd = [browser_exe] + flags
        try:
            # On Windows, avoid creating an extra console window
            creationflags = 0
            if is_windows():
                creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            proc = subprocess.Popen(cmd, creationflags=creationflags)
            return proc
        except Exception as exc:
            print(f"[kiosk] Failed to launch {browser_exe} ({exc}), falling back to default browser.")

    # Fallback to standard Python webbrowser module
    webbrowser.open(url)
    return None
