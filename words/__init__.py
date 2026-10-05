"""ASL Words recognition module (motion-based holistic sequence classification)."""

from typing import List, Optional


def main(source: Optional[str] = None, extra_args: Optional[List[str]] = None) -> int:
    """Entry point for ASL words mode.

    Supports subcommands:
      - record: Interactive or synthetic sequence recording (words.record)
      - eval: Model cross-validation and evaluation (words.eval)
      - live: Real-time recognition (words.live) - default if no subcommand

    Args:
        source: Video source (device index like "0" or path to video file).
        extra_args: Additional command line arguments.

    Returns:
        Exit code (0 for clean success).
    """
    args = list(extra_args or [])
    subcmd = args[0] if args and not args[0].startswith("-") else None

    if subcmd == "record":
        from .record import main as record_main
        sub_args = args[1:]
        if source is not None and "--source" not in sub_args:
            sub_args = ["--source", str(source)] + sub_args
        return record_main(sub_args)

    elif subcmd == "eval":
        from .eval import main as eval_main
        return eval_main(args[1:])

    else:
        # Default to live recognition
        from .live import main as live_main
        live_args = args[1:] if subcmd == "live" else args
        if source is not None and "--source" not in live_args:
            live_args = ["--source", str(source)] + live_args
        return live_main(live_args)


__all__ = ["main"]
