"""Verification of isolation guarantees for ISL mode."""

import subprocess
import sys


def test_isl_modules_never_import_words():
    """Verify that importing isl packages does not pull in the words package."""
    code = """
import sys
import isl
import isl.config
import isl.features
import isl.model
assert "words" not in sys.modules, f"Isolation failure: words imported into {list(sys.modules.keys())}"
print("ISL isolation passed.")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ISL isolation passed." in result.stdout


def test_main_dispatcher_isl_help_works():
    """Verify that main.py --mode isl help executes cleanly."""
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "isl", "help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "signlang (ISL mode) - Indian Sign Language recognition" in result.stdout


def test_main_dispatcher_help_includes_isl():
    """Verify that main.py --help lists ISL mode."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "isl" in result.stdout
    assert "Indian Sign Language" in result.stdout
