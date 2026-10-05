"""Unit tests verifying strict module isolation between letters and words modes."""

import os
import subprocess
import sys
import pytest


def test_letters_modules_never_import_words():
    """Verify that importing any letters mode module never loads words/."""
    # Run in a fresh python subprocess to test cold imports
    code = """
import sys
import signlang
import signlang.cli
import signlang.engine
import signlang.features
import signlang.model
import signlang.train
import signlang.dataset

words_modules = [m for m in sys.modules if m == "words" or m.startswith("words.")]
assert len(words_modules) == 0, f"Isolation failure! Words modules imported: {words_modules}"
print("Isolation check passed: zero words modules in sys.modules.")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Isolation check passed" in result.stdout


def test_main_dispatcher_letters_mode_never_imports_words():
    """Verify that main.py in letters mode never imports words."""
    code = """
import sys
import os
from unittest.mock import patch

# Mock live_main and cli_main so they do not start camera or GUI
with patch("signlang.live.main", return_value=0) as mock_live:
    import main
    exit_code = main.dispatch(["--mode", "letters", "--source", "0"])
    assert exit_code == 0
    words_modules = [m for m in sys.modules if m == "words" or m.startswith("words.")]
    assert len(words_modules) == 0, f"Words imported in letters mode: {words_modules}"
    print("Dispatcher letters isolation passed.")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Dispatcher letters isolation passed." in result.stdout


def test_main_dispatcher_help_works():
    """Verify that main.py --help exits with 0."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "SignLang: Real-time ASL fingerspelling" in result.stdout
    assert "--mode {letters,words}" in result.stdout


def test_words_standalone_execution():
    """Verify words module can be loaded independently."""
    code = """
import words
import words.config
import words.normalize
import words.classifier
import words.dataset

assert len(words.config.WORDS) == 16
print("Words loaded independently.")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Words loaded independently." in result.stdout


def test_main_dispatcher_words_subcommands_help():
    """Verify that main.py --mode words forwards help to record, eval, and live."""
    for subcmd in ["record", "eval", "live"]:
        result = subprocess.run(
            [sys.executable, "main.py", "--mode", "words", subcmd, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"words.{subcmd}" in result.stdout


def test_main_dispatcher_default_mode_never_imports_words():
    """Verify that main.py default mode (letters) never imports words."""
    code = """
import sys
from unittest.mock import patch

with patch("signlang.live.main", return_value=0):
    import main
    exit_code = main.dispatch([])
    assert exit_code == 0
    words_modules = [m for m in sys.modules if m == "words" or m.startswith("words.")]
    assert len(words_modules) == 0, f"Words imported in default letters mode: {words_modules}"
    print("Default mode isolation passed.")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Default mode isolation passed." in result.stdout
