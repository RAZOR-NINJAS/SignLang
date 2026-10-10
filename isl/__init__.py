"""Indian Sign Language (ISL) recognition package for SignLang."""

import os
import sys
from typing import List, Optional


def main(source: Optional[str] = None, extra_args: Optional[List[str]] = None) -> int:
    """Entry point for ISL mode.

    Supports subcommands:
      - collect: Interactive two-hand data collection
      - train: Train the ISL classifier on recorded samples
      - eval: Cross-session model evaluation and confusion matrix
      - live: Real-time recognition (default)
    """
    args = list(extra_args or [])
    subcmd = args[0] if args and not args[0].startswith("-") else None

    # Strict isolation: ensure words is not imported
    if "words" in sys.modules:
        del sys.modules["words"]

    if source is not None:
        os.environ["SIGNLANG_SOURCE"] = str(source)
        if str(source).isdigit():
            os.environ["SIGNLANG_CAMERA"] = str(source)

    if subcmd in ("help", "-h", "--help"):
        print(
            """
signlang (ISL mode) - Indian Sign Language recognition

  python main.py --mode isl live       real-time ISL recognition in HUD
  python main.py --mode isl collect    record ISL two-handed training samples
  python main.py --mode isl train      train the ISL classifier on recorded samples
  python main.py --mode isl eval       evaluate trained ISL model across sessions

Grounding:
  Follows the standardized two-handed manual alphabet established by the
  Indian Sign Language Research and Training Centre (ISLRTC).
"""
        )
        return 0

    if subcmd == "collect":
        from .collect import main as collect_main
        return collect_main(args[1:])
    elif subcmd == "train":
        from .train import main as train_main
        return train_main(args[1:])
    elif subcmd == "eval":
        from .eval import main as eval_main
        return eval_main(args[1:])
    else:
        from .live import main as live_main
        live_args = args[1:] if subcmd == "live" else args
        return live_main(live_args)


__all__ = ["main"]
