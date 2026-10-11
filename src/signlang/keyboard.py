"""Cross-platform non-blocking console keyboard input polling.

Abstracts msvcrt on Windows and select on POSIX so that interactive data
collection and controls work identically without throwing socket errors.
"""

import sys
from typing import List

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import select
except ImportError:
    select = None


def is_interactive() -> bool:
    """Return True if stdin is attached to an interactive terminal."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def poll_stdin_keys() -> List[str]:
    """Read pending single characters from stdin without blocking.

    On Windows: uses msvcrt.kbhit() and msvcrt.getwche().
    On POSIX: uses select.select([sys.stdin], ...) when isatty().
    Returns an empty list in headless or pipe environments.
    """
    keys: List[str] = []

    if sys.platform == "win32":
        if msvcrt is not None:
            try:
                while msvcrt.kbhit():
                    ch = msvcrt.getwche()
                    if ch:
                        keys.append(ch)
            except Exception:
                pass
        return keys

    # Linux / macOS / POSIX
    if select is not None and is_interactive():
        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                keys.append(ch)
        except Exception:
            pass

    return keys
