"""Unit tests for the signlang doctor diagnostic module."""

import sys
from pathlib import Path
import pytest

from signlang.doctor import (
    DoctorCheck,
    check_browser_kiosk,
    check_network_port,
    check_operating_system,
    check_python_environment,
    main,
    run_all_checks,
)


def test_doctor_check_instantiation():
    chk = DoctorCheck("Test Component", "Test Category")
    assert chk.name == "Test Component"
    assert chk.category == "Test Category"
    assert not chk.passed
    assert not chk.warning
    assert chk.remediation is None


def test_check_python_environment():
    chk = check_python_environment()
    assert isinstance(chk, DoctorCheck)
    assert chk.category == "Environment"
    # Current test runner is Python 3.11
    if sys.version_info.major == 3 and sys.version_info.minor in (11, 12):
        assert chk.passed


def test_check_operating_system():
    chk = check_operating_system()
    assert isinstance(chk, DoctorCheck)
    assert chk.passed
    assert sys.platform in chk.message.lower() or "windows" in chk.message.lower() or "linux" in chk.message.lower()


def test_check_network_port():
    chk = check_network_port()
    assert isinstance(chk, DoctorCheck)
    assert chk.passed


def test_run_all_checks():
    checks = run_all_checks()
    assert len(checks) >= 8
    categories = {c.category for c in checks}
    assert "Environment" in categories
    assert "Models" in categories
    assert "Network" in categories


def test_doctor_main_check_only():
    code = main(["--check-only"])
    assert code == 0


def test_doctor_cli_dispatch():
    import subprocess
    result = subprocess.run(
        [sys.executable, "main.py", "doctor", "--check-only"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
