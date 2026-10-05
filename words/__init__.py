"""ASL Words recognition module (motion-based holistic sequence classification)."""

from typing import List, Optional


def main(source: str = "0", extra_args: Optional[List[str]] = None) -> int:
    """Entry point for ASL words mode.

    Args:
        source: Video source (device index like "0" or path to video file).
        extra_args: Additional command line arguments forwarded to words.live.

    Returns:
        Exit code (0 for clean success).
    """
    from .live import run_live

    args_list = list(extra_args or [])
    headless = "--headless" in args_list
    max_frames = None
    for a in args_list:
        if a.startswith("--max-frames="):
            max_frames = int(a.split("=", 1)[1])

    return run_live(
        source=source,
        max_frames=max_frames,
        headless=headless,
    )


__all__ = ["main"]
