"""System diagnostic and self-test utility for SignLang (signlang doctor).

Verifies cross-platform hardware, models, dependencies, and all recognition modes.
Outputs human-readable diagnostics and actionable remediation instructions.
"""

import argparse
import os
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .platform_compat import (
    find_browser_executable,
    get_camera_capture,
    is_linux,
    is_windows,
    probe_cameras,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class DoctorCheck:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.passed: bool = False
        self.warning: bool = False
        self.message: str = ""
        self.remediation: Optional[str] = None


def check_python_environment() -> DoctorCheck:
    chk = DoctorCheck("Python Runtime", "Environment")
    v = sys.version_info
    chk.message = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if v.major == 3 and v.minor in (11, 12):
        chk.passed = True
    elif v.major == 3 and v.minor >= 13:
        chk.warning = True
        chk.passed = True
        chk.remediation = "Python 3.13+ may lack precompiled MediaPipe wheels; Python 3.11 is strongly recommended."
    else:
        chk.passed = False
        chk.remediation = "SignLang requires Python 3.11 or 3.12 for MediaPipe compatibility. Run install.ps1 or install.sh."
    return chk


def check_operating_system() -> DoctorCheck:
    chk = DoctorCheck("Operating System", "Environment")
    chk.passed = True
    arch = "x86_64" if sys.maxsize > 2**32 else "x86_32"
    if is_windows():
        chk.message = f"Windows ({sys.platform}, {arch}) - DirectShow backend"
    elif is_linux():
        chk.message = f"Linux ({sys.platform}, {arch}) - V4L2 backend"
    else:
        chk.message = f"{sys.platform} ({arch})"
    return chk


def check_camera_subsystem() -> DoctorCheck:
    chk = DoctorCheck("Webcam Capture", "Hardware")
    cams = probe_cameras(max_index=4)
    if cams:
        chk.passed = True
        desc = ", ".join(f"Cam {idx} ({w}x{h})" for idx, w, h in cams)
        backend_name = "DirectShow" if is_windows() else "V4L2"
        chk.message = f"Found {len(cams)} camera(s) via {backend_name}: {desc}"
    else:
        chk.passed = True
        chk.warning = True
        chk.message = "No physical webcams currently accessible."
        chk.remediation = (
            "Connect a USB webcam or set SIGNLANG_SOURCE to a video file / IP stream.\n"
            "If using Windows, ensure Camera Privacy Settings allow desktop apps to access the camera."
        )
    return chk


def check_mediapipe_landmarker() -> DoctorCheck:
    chk = DoctorCheck("MediaPipe Landmarker", "Vision")
    model_path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    if not model_path.exists():
        chk.passed = False
        chk.message = f"Missing model asset: {model_path.name}"
        chk.remediation = "Run `python scripts/fetch_models.py` to download the MediaPipe landmarker."
        return chk

    try:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        landmarker.close()
        chk.passed = True
        chk.message = "Loaded hand_landmarker.task (Tasks API verified)"
    except Exception as exc:
        chk.passed = False
        chk.message = f"Failed to instantiate landmarker: {exc}"
        chk.remediation = "Verify that mediapipe is installed properly: `pip install mediapipe==0.10.21`"

    return chk


def check_isl_model() -> DoctorCheck:
    chk = DoctorCheck("ISL Mode (Two-Handed)", "Models")
    model_path = PROJECT_ROOT / "isl" / "models" / "isl_mlp.pt"
    labels_path = PROJECT_ROOT / "isl" / "models" / "labels.json"
    if model_path.exists() and labels_path.exists():
        chk.passed = True
        chk.message = f"Trained weights present ({model_path.name})"
    else:
        chk.passed = True
        chk.warning = True
        chk.message = "No custom ISL weights trained yet."
        chk.remediation = "Run `python main.py collect` to record two-handed signs, then `python main.py train`."
    return chk


def check_asl_letters_model() -> DoctorCheck:
    chk = DoctorCheck("ASL Letters Mode", "Models")
    model_path = PROJECT_ROOT / "models" / "signs_mlp.pt"
    labels_path = PROJECT_ROOT / "models" / "labels.json"
    if model_path.exists() and labels_path.exists():
        chk.passed = True
        chk.message = f"Weights present ({model_path.name})"
    else:
        chk.passed = False
        chk.message = "Missing models/signs_mlp.pt"
        chk.remediation = "Run `python main.py --mode letters train` to train ASL letters."
    return chk


def check_asl_words_model() -> DoctorCheck:
    chk = DoctorCheck("ASL Words Mode", "Models")
    model_path = PROJECT_ROOT / "words" / "models" / "words_classifier.npz"
    if model_path.exists():
        chk.passed = True
        chk.message = f"Classifier present ({model_path.name})"
    else:
        chk.passed = True
        chk.warning = True
        chk.message = "Optional words classifier not yet trained."
        chk.remediation = "Run `python main.py --mode words eval` to initialize word models."
    return chk


def check_audio_tts() -> DoctorCheck:
    chk = DoctorCheck("Text-to-Speech (Piper)", "Audio")
    onnx_path = PROJECT_ROOT / "en_US-lessac-medium.onnx"
    json_path = PROJECT_ROOT / "en_US-lessac-medium.onnx.json"
    if not (onnx_path.exists() and json_path.exists()):
        chk.passed = True
        chk.warning = True
        chk.message = "Piper TTS model files not downloaded."
        chk.remediation = "Run `python scripts/fetch_models.py` to enable spoken audio feedback."
        return chk

    try:
        import sounddevice as sd
        devices = sd.query_devices()
        chk.passed = True
        chk.message = "Piper voice model present & SoundDevice initialized"
    except Exception as exc:
        chk.passed = True
        chk.warning = True
        chk.message = f"TTS audio device unavailable: {exc}"
        chk.remediation = "Audio output is muted; visual HUD recognition remains fully functional."

    return chk


def check_browser_kiosk() -> DoctorCheck:
    chk = DoctorCheck("Browser Kiosk (Game Mode)", "Display")
    browser = find_browser_executable()
    if browser:
        chk.passed = True
        chk.message = f"Detected: {Path(browser).name} ({browser})"
    else:
        chk.passed = True
        chk.warning = True
        chk.message = "No dedicated Edge/Chrome kiosk browser found."
        chk.remediation = "The game will open using your standard default browser."
    return chk


def check_network_port() -> DoctorCheck:
    chk = DoctorCheck("Localhost Loopback", "Network")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        chk.passed = True
        chk.message = f"Loopback network stack operational (tested ephemeral port {port})"
    except Exception as exc:
        chk.passed = False
        chk.message = f"Cannot bind to 127.0.0.1: {exc}"
        chk.remediation = "Check local firewall and loopback interface permissions."
    return chk


def run_all_checks() -> List[DoctorCheck]:
    """Execute all system diagnostic checks."""
    return [
        check_python_environment(),
        check_operating_system(),
        check_camera_subsystem(),
        check_mediapipe_landmarker(),
        check_isl_model(),
        check_asl_letters_model(),
        check_asl_words_model(),
        check_audio_tts(),
        check_browser_kiosk(),
        check_network_port(),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="signlang doctor", description="SignLang System Diagnostics")
    parser.add_argument("--check-only", action="store_true", help="Silent mode returning exit code 0 on pass, 1 on fail")
    args = parser.parse_args(argv)

    checks = run_all_checks()
    critical_failures = [c for c in checks if not c.passed and not c.warning]

    if args.check_only:
        return 0 if len(critical_failures) == 0 else 1

    console = Console() if _HAS_RICH else None

    if console:
        console.print("\n[bold cyan]=== SignLang System Doctor & Compatibility Diagnostics ===[/bold cyan]\n")
        table = Table(border_style="cyan")
        table.add_column("Category", style="bold white", width=14)
        table.add_column("Component", style="bold white", width=26)
        table.add_column("Status", width=10)
        table.add_column("Details", style="dim")

        for c in checks:
            if not c.passed:
                status = "[bold red]FAIL[/bold red]"
            elif c.warning:
                status = "[bold yellow]WARN[/bold yellow]"
            else:
                status = "[bold green]PASS[/bold green]"
            table.add_row(c.category, c.name, status, c.message)

        console.print(table)
        console.print()

        issues = [c for c in checks if c.remediation]
        if issues:
            console.print("[bold yellow]Recommendations & Remediation Tips:[/bold yellow]")
            for issue in issues:
                tag = "[bold red]FAIL[/bold red]" if not issue.passed else "[bold yellow]INFO[/bold yellow]"
                console.print(f"  • {tag} [bold white]{issue.name}[/bold white]: {issue.remediation}")
            console.print()

        if critical_failures:
            console.print(f"[bold red]Doctor reported {len(critical_failures)} critical issue(s).[/bold red]\n")
            return 1
        else:
            console.print("[bold green]All core systems are operational and ready for use![/bold green]\n")
            return 0
    else:
        print("=== SignLang System Doctor ===")
        for c in checks:
            tag = "FAIL" if not c.passed else ("WARN" if c.warning else "PASS")
            print(f"[{tag:4s}] {c.category:12s} | {c.name:24s} | {c.message}")
            if c.remediation:
                print(f"       -> Fix: {c.remediation}")

        return 0 if len(critical_failures) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
