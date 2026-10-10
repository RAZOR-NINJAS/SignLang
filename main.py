#!/usr/bin/env python3
"""Main dispatcher for SignLang.

Selects between letters mode (ASL fingerspelling) and words mode (ASL holistic signs).

Strict Isolation Guarantee:
Letters mode NEVER imports the words/ package or any of its submodules.
"""

import argparse
import os
import sys
from typing import List, Optional


def dispatch(argv: Optional[List[str]] = None) -> int:
    """Parse dispatcher arguments and run the chosen mode."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SignLang: Real-time ASL fingerspelling (letters) and motion signs (words).",
        add_help=False,  # Allow --help to be handled or passed through
    )
    parser.add_argument(
        "--mode",
        choices=["letters", "words", "game", "isl"],
        default="letters",
        help="Recognition mode: 'letters' (fingerspelling), 'words' (holistic signs), 'game' (shadow-play), or 'isl' (Indian Sign Language). Default: letters.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video capture source (camera device index like '0' or video file path).",
    )
    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show this help message and exit.",
    )

    args, rest = parser.parse_known_args(argv)

    if args.help:
        if not rest:
            parser.print_help()
            print("\nExamples:")
            print("  python main.py                      # Run letters live recognition (default)")
            print("  python main.py --mode letters       # Run letters mode")
            print("  python main.py --mode words         # Run words live mode")
            print("  python main.py --mode words record  # Record words sequences")
            print("  python main.py --mode words eval    # Evaluate words model")
            print("  python main.py --mode game          # Run shadow-play Dino game")
            print("  python main.py --mode isl           # Run ISL live recognition")
            print("  python main.py --mode isl collect   # Record ISL two-handed samples")
            print("  python main.py --mode isl train     # Train ISL classifier")
            print("  python main.py --mode isl eval      # Evaluate ISL model")
            return 0
        else:
            rest = list(rest) + ["--help"]

    # Strict isolation: letters mode MUST NOT import words
    if args.mode == "letters":
        # Ensure words is not imported
        if "words" in sys.modules:
            del sys.modules["words"]

        # Configure video source environment if supplied
        if args.source is not None:
            os.environ["SIGNLANG_SOURCE"] = str(args.source)
            if str(args.source).isdigit():
                os.environ["SIGNLANG_CAMERA"] = str(args.source)

        # Dispatch to signlang package
        # If user passed a subcommand like 'game', 'train', 'collect', use cli.main
        known_cmds = {"live", "game", "train", "collect", "cameras", "check", "export", "import", "help"}
        if rest and rest[0] in known_cmds:
            from signlang.cli import main as cli_main
            return cli_main(rest)
        else:
            from signlang.live import main as live_main
            return live_main(rest)

    elif args.mode == "words":
        # Dynamically import words package
        import words
        src = args.source if args.source is not None else "0"
        return words.main(source=src, extra_args=rest)

    elif args.mode == "game":
        from signlang.cli import main as cli_main
        game_args = ["game"]
        if args.source is not None:
            game_args.extend(["--source", str(args.source)])
        game_args.extend(rest)
        return cli_main(game_args)

    elif args.mode == "isl":
        # Strict isolation: ensure words is not imported
        if "words" in sys.modules:
            del sys.modules["words"]
        import isl
        return isl.main(source=args.source, extra_args=rest)

    return 1


if __name__ == "__main__":
    sys.exit(dispatch(sys.argv[1:]))
