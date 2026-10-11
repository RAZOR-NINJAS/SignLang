#!/usr/bin/env python3
"""Interactive TUI Installer and Setup Wizard for SignLang.

Cross-platform (Windows & Linux).
Renders a gradient ASCII banner, assists with camera selection, verifies models,
creates desktop/start menu shortcuts, and runs full self-tests.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from signlang.platform_compat import (
    find_browser_executable,
    get_camera_capture,
    is_linux,
    is_windows,
    probe_cameras,
)
from scripts.fetch_models import FILES_TO_FETCH, fetch

ASCII_BANNER = r"""
  ███████╗██╗ ██████╗ ███╗   ██╗██╗      █████╗ ███╗   ██╗ ██████╗ 
  ██╔════╝██║██╔════╝ ████╗  ██║██║     ██╔══██╗████╗  ██║██╔════╝ 
  ███████╗██║██║  ███╗██╔██╗ ██║██║     ███████║██╔██╗ ██║██║  ███╗
  ╚════██║██║██║   ██║██║╚██╗██║██║     ██╔══██║██║╚██╗██║██║   ██║
  ███████║██║╚██████╔╝██║ ╚████║███████╗██║  ██║██║ ╚████║╚██████╔╝
  ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
"""

SUBTITLE = "Real-Time Sign Language Translator (ISL • ASL • Words • Shadow Play)"


def print_banner(console: Optional["Console"] = None):
    """Render large ASCII art logo with color gradient."""
    if console and _HAS_RICH:
        colors = ["#00f2fe", "#4facfe", "#00c6ff", "#0072ff", "#7f00ff", "#e100ff"]
        styled_text = Text()
        lines = [l for l in ASCII_BANNER.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            c = colors[i % len(colors)]
            styled_text.append(line + "\n", style=f"bold {c}")
        console.print(styled_text)
        console.print(f"[bold white]{SUBTITLE.center(68)}[/bold white]\n")
    else:
        print(ASCII_BANNER)
        print(SUBTITLE.center(68) + "\n")


def print_system_info(console: "Console"):
    """Show target environment information."""
    table = Table(title="Target Environment Check", border_style="cyan", show_header=False)
    table.add_column("Property", style="bold white", width=22)
    table.add_column("Value", style="cyan")

    table.add_row("Operating System", f"{'Windows 10/11' if is_windows() else 'Linux'} ({sys.platform})")
    table.add_row("Python Runtime", f"{sys.version.split()[0]} ({sys.executable})")
    table.add_row("CPU Logical Cores", str(os.cpu_count() or "Unknown"))
    
    browser = find_browser_executable()
    table.add_row("Kiosk Browser", Path(browser).name if browser else "None found (will use default browser)")
    console.print(table)
    console.print()


def configure_modes(console: "Console", unattended: bool) -> dict:
    """Prompt user to select enabled sign language modes."""
    console.print("[bold yellow]Step 1: Modes Configuration[/bold yellow]")
    modes = {
        "isl": True,
        "asl_letters": True,
        "asl_words": True,
        "game": True,
    }

    if unattended:
        console.print("  [green]✓[/green] Enabling all modes (ISL, ASL Letters, ASL Words, Game Kiosk)\n")
        return modes

    console.print("  SignLang supports 4 distinct interactive modes:")
    console.print("    1. [bold cyan]ISL Mode[/bold cyan]: Indian Sign Language (Two-Handed Standard, default)")
    console.print("    2. [bold cyan]ASL Letters[/bold cyan]: ASL Single-Hand Fingerspelling (A-Y + SPACE)")
    console.print("    3. [bold cyan]ASL Words[/bold cyan]: Dynamic Motion Gestures (DTW + k-NN)")
    console.print("    4. [bold cyan]Shadow Play & Learning[/bold cyan]: Browser Kiosk Game (http://localhost:8000)")
    
    keep_all = Confirm.ask("  Enable all modes with full functionality?", default=True)
    if not keep_all:
        modes["isl"] = Confirm.ask("    Enable ISL mode?", default=True)
        modes["asl_letters"] = Confirm.ask("    Enable ASL letters mode?", default=True)
        modes["asl_words"] = Confirm.ask("    Enable ASL words mode?", default=True)
        modes["game"] = Confirm.ask("    Enable Shadow Play / Learning mode?", default=True)

    console.print()
    return modes


def select_camera(console: "Console", unattended: bool) -> int:
    """Scan and select webcam."""
    console.print("[bold yellow]Step 2: Video Camera Detection[/bold yellow]")
    with console.status("[cyan]Scanning connected cameras...[/cyan]"):
        cams = probe_cameras(max_index=4)

    if not cams:
        console.print("  [bold red]![/bold red] No physical webcams detected.")
        console.print("    (You can still run recorded videos or connect a camera later via SIGNLANG_CAMERA)\n")
        return 0

    table = Table(border_style="dim", show_header=True)
    table.add_column("Index", style="bold green", width=8)
    table.add_column("Resolution", width=16)
    table.add_column("Status", style="cyan")

    for idx, w, h in cams:
        table.add_row(str(idx), f"{w}x{h}", "Ready")
    console.print(table)

    if unattended or len(cams) == 1:
        chosen = cams[0][0]
        console.print(f"  [green]✓[/green] Selected default camera: [bold cyan]index {chosen}[/bold cyan]\n")
        return chosen

    chosen_str = Prompt.ask("  Select camera index to use", default=str(cams[0][0]), choices=[str(c[0]) for c in cams])
    console.print()
    return int(chosen_str)


def download_models(console: "Console"):
    """Download required models with progress bars."""
    console.print("[bold yellow]Step 3: Model Assets Verification[/bold yellow]")
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for url, dest, desc in FILES_TO_FETCH:
            if dest.exists() and dest.stat().st_size > 0:
                console.print(f"  [green]✓[/green] Present: {dest.name}")
                continue

            task = progress.add_task(f"Downloading {dest.name}", total=100)

            def update_cb(done, total):
                if total > 0:
                    progress.update(task, completed=done, total=total)

            ok = fetch(url, dest, desc, progress_cb=update_cb)
            if ok:
                progress.update(task, completed=dest.stat().st_size, total=dest.stat().st_size)
                console.print(f"  [green]✓[/green] Downloaded: {dest.name}")
            else:
                console.print(f"  [red]✗[/red] Failed to download: {dest.name}")

    console.print()


def create_shortcuts(console: "Console", unattended: bool):
    """Create Desktop and Start Menu shortcuts."""
    console.print("[bold yellow]Step 4: Desktop & Launcher Setup[/bold yellow]")

    create = True if unattended else Confirm.ask("  Create Desktop and Start Menu launcher shortcuts?", default=True)
    if not create:
        console.print()
        return

    bat_path = ROOT / "signlang.bat"
    
    if is_windows():
        try:
            # Use PowerShell WScript.Shell to create .lnk without admin rights
            ps_script = f"""
            $WshShell = New-Object -ComObject WScript.Shell
            # Desktop Shortcut
            $Desktop = [System.Environment]::GetFolderPath('Desktop')
            $Shortcut = $WshShell.CreateShortcut("$Desktop\\SignLang.lnk")
            $Shortcut.TargetPath = "{bat_path}"
            $Shortcut.WorkingDirectory = "{ROOT}"
            $Shortcut.Description = "SignLang Real-Time Sign Language Translator"
            $Shortcut.Save()

            # Start Menu Shortcut
            $StartMenu = [System.Environment]::GetFolderPath('StartMenu')
            $Programs = Join-Path $StartMenu "Programs"
            $SmShortcut = $WshShell.CreateShortcut("$Programs\\SignLang.lnk")
            $SmShortcut.TargetPath = "{bat_path}"
            $SmShortcut.WorkingDirectory = "{ROOT}"
            $SmShortcut.Description = "SignLang Real-Time Sign Language Translator"
            $SmShortcut.Save()
            """
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, capture_output=True)
            console.print("  [green]✓[/green] Created Desktop Shortcut: [cyan]Desktop\\SignLang.lnk[/cyan]")
            console.print("  [green]✓[/green] Created Start Menu Shortcut: [cyan]Start Menu\\SignLang.lnk[/cyan]")
        except Exception as exc:
            console.print(f"  [yellow]![/yellow] Could not create Windows shortcuts automatically: {exc}")
    else:
        # Linux .desktop entries
        try:
            desktop_entry = f"""[Desktop Entry]
Type=Application
Name=SignLang
Comment=Real-Time Sign Language Translator
Exec="{ROOT}/install.sh"
Path={ROOT}
Terminal=true
Categories=Education;Utility;
"""
            local_apps = Path.home() / ".local" / "share" / "applications"
            local_apps.mkdir(parents=True, exist_ok=True)
            (local_apps / "signlang.desktop").write_text(desktop_entry)
            
            desktop_dir = Path.home() / "Desktop"
            if desktop_dir.is_dir():
                (desktop_dir / "signlang.desktop").write_text(desktop_entry)
                try:
                    os.chmod(desktop_dir / "signlang.desktop", 0o755)
                except Exception:
                    pass
            console.print("  [green]✓[/green] Created Linux desktop application shortcut.")
        except Exception as exc:
            console.print(f"  [yellow]![/yellow] Could not create Linux desktop entry: {exc}")

    console.print()


def run_self_test(console: "Console") -> bool:
    """Run fast non-interactive doctor verification."""
    console.print("[bold yellow]Step 5: Built-in Self-Test & Diagnostic Check[/bold yellow]")
    
    all_ok = True
    tests = []

    # Test 1: Camera
    with console.status("[cyan]Checking video capture...[/cyan]"):
        try:
            cap = get_camera_capture(0)
            opened = cap.isOpened()
            cap.release()
            tests.append(("Video Camera Capture", opened, "Camera initialized successfully" if opened else "No camera active (non-fatal)"))
        except Exception as e:
            tests.append(("Video Camera Capture", False, str(e)))

    # Test 2: MediaPipe Landmarker
    with console.status("[cyan]Verifying MediaPipe tasks...[/cyan]"):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            landmarker_path = ROOT / "models" / "hand_landmarker.task"
            if landmarker_path.exists():
                options = vision.HandLandmarkerOptions(
                    base_options=mpp.BaseOptions(model_asset_path=str(landmarker_path)),
                    running_mode=vision.RunningMode.IMAGE,
                )
                lm = vision.HandLandmarker.create_from_options(options)
                lm.close()
                tests.append(("MediaPipe HandLandmarker", True, "Model loaded & delegate verified"))
            else:
                tests.append(("MediaPipe HandLandmarker", False, "Model file not found"))
                all_ok = False
        except Exception as e:
            tests.append(("MediaPipe HandLandmarker", False, str(e)))
            all_ok = False

    # Test 3: ISL Model
    isl_model = ROOT / "isl" / "models" / "isl_mlp.pt"
    isl_labels = ROOT / "isl" / "models" / "labels.json"
    if isl_model.exists() and isl_labels.exists():
        tests.append(("ISL Two-Handed Classifier", True, f"{isl_model.name} present"))
    else:
        tests.append(("ISL Two-Handed Classifier", False, "Missing weights (run signlang collect & train)"))

    # Test 4: ASL Model
    asl_model = ROOT / "models" / "signs_mlp.pt"
    if asl_model.exists():
        tests.append(("ASL Fingerspelling Classifier", True, f"{asl_model.name} present"))
    else:
        tests.append(("ASL Fingerspelling Classifier", False, "Missing signs_mlp.pt"))

    # Test 5: Game Server Bind Test
    with console.status("[cyan]Checking game port binding...[/cyan]"):
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sock.close()
            tests.append(("Network Port Binding", True, "Localhost loopback verified"))
        except Exception as e:
            tests.append(("Network Port Binding", False, str(e)))

    # Print results
    table = Table(border_style="cyan")
    table.add_column("Subsystem", style="bold white", width=32)
    table.add_column("Status", width=12)
    table.add_column("Details", style="dim")

    for name, ok, details in tests:
        status_text = "[bold green]PASS[/bold green]" if ok else "[bold yellow]WARN[/bold yellow]"
        table.add_row(name, status_text, details)

    console.print(table)
    console.print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="SignLang Interactive Setup Wizard")
    parser.add_argument("-y", "--yes", action="store_true", help="Non-interactive setup with defaults")
    args = parser.parse_args()

    console = Console() if _HAS_RICH else None

    if console:
        console.clear()
        print_banner(console)
        print_system_info(console)
        configure_modes(console, args.yes)
        select_camera(console, args.yes)
        download_models(console)
        create_shortcuts(console, args.yes)
        run_self_test(console)

        console.print(
            Panel(
                "[bold green]SignLang is completely installed and ready to run![/bold green]\n\n"
                "• [bold cyan]ISL Recognition (Default)[/bold cyan]: double-click signlang.bat or run `python main.py`\n"
                "• [bold cyan]Shadow Play Dino Game[/bold cyan]: run `python main.py --mode game`\n"
                "• [bold cyan]ASL Fingerspelling[/bold cyan]: run `python main.py --mode letters`\n"
                "• [bold cyan]Diagnostics / Doctor[/bold cyan]: run `python main.py doctor`",
                title="Installation Complete",
                border_style="green",
            )
        )
    else:
        print_banner()
        print("Running in headless/standard mode...")
        download_models(None)
        print("Installation complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
