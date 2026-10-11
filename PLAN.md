# Implementation Plan: Full Windows Support & Interactive Installer for `signlang`

## Overview & Objectives

Enable seamless, first-class cross-platform execution of `signlang` on **Windows 10/11** and **Linux** (Omarchy/Arch, Ubuntu, Debian), with zero prerequisites on target Windows machines (handles bare laptops without Python, Git, or admin rights).

### Core Modes Supported Cross-Platform
1. **ISL Mode (Primary)**: Dual-hand static and alphabet recognition
2. **ASL Letters Mode**: Single-hand fingerspelling (A–Y + custom SPACE sign)
3. **ASL Words Mode**: Dynamic motion gesture recognition (DTW + k-NN)
4. **Game Mode (Shadow Play)**: Browser-based Dino obstacle game
5. **Browser Learning Mode**: Fullscreen kiosk browser pointed at `http://localhost:8000`

---

## 1. Workstream File Ownership Matrix (Strict Isolation)

To ensure no two subagents edit the same file at once, the following disjoint file ownership is enforced:

| Workstream | Subagent Role | Scope / Assigned Tasks | Owned Files |
| :--- | :--- | :--- | :--- |
| **Stream 1** | **Codebase Auditor** | Static analysis & codebase scan for Linux-only assumptions, terminal I/O, device paths, and platform-specific calls. | *Read-only* (writes findings to `chat.md`) |
| **Stream 2** | **Windows Compatibility Dev** | Core camera backend (`cv2.CAP_DSHOW`), path standardization, kiosk launcher abstraction (`msedge`/`chrome`), dependency pins with Windows wheels. | `src/signlang/platform_compat.py` *(new)*<br>`src/signlang/cameras.py`<br>`src/signlang/game/server.py`<br>`src/signlang/game/kiosk.py` *(new)*<br>`pyproject.toml` |
| **Stream 3** | **Installer Dev** | PowerShell zero-prereq bootstrap (`install.ps1`), Python TUI installer (`scripts/install_tui.py`) with gradient ASCII banner & shortcut creation, Linux installer (`install.sh`), and launcher (`signlang.bat`). | `install.ps1` *(new)*<br>`install.sh` *(new)*<br>`signlang.bat` *(new)*<br>`scripts/install_tui.py` *(new)*<br>`scripts/fetch_models.py` |
| **Stream 4** | **Modes Dev** | Keyboard polling abstraction (`msvcrt` vs `select`), ASL/ISL/Words live & collection loops cross-platform testing, audio/TTS resilience (piper-tts ONNX / fallback). | `src/signlang/keyboard.py` *(new)*<br>`src/signlang/hands.py`<br>`src/signlang/live.py`<br>`src/signlang/collect.py`<br>`isl/collect.py`<br>`isl/live.py`<br>`words/record.py`<br>`words/live.py`<br>`src/signlang/tts.py` |
| **Stream 5** | **QA / CI Agent** | `signlang doctor` system verification command, multi-platform GitHub Actions CI workflow, doctor unit tests, README.md documentation update. | `src/signlang/doctor.py` *(new)*<br>`src/signlang/cli.py`<br>`main.py`<br>`src/signlang/test_doctor.py` *(new)*<br>`.github/workflows/ci.yml` *(new)*<br>`README.md` |

---

## 2. Core Architectural Abstractions & Components

### 2.1 Camera Backend Abstraction (`src/signlang/platform_compat.py`)
- **DirectShow on Windows**: `cv2.VideoCapture(index, cv2.CAP_DSHOW)` avoids 5-10s hang and latency from MSMF.
- **V4L2 on Linux**: `cv2.VideoCapture(index, cv2.CAP_V4L2)`.
- **Buffer size**: Set `cv2.CAP_PROP_BUFFERSIZE = 1` across platforms to prevent frame queue lag.
- **Graceful Error Handling**: Return actionable error message when camera index is unavailable or blocked by OS permissions.

### 2.2 Terminal Keyboard Input Abstraction (`src/signlang/keyboard.py`)
- **Windows**: Use `msvcrt.kbhit()` and `msvcrt.getwch()` (non-blocking single character read).
- **POSIX**: Use `select.select([sys.stdin], [], [], 0)` when `sys.stdin.isatty()`.
- **Headless Fallback**: Safe fallback when running without a tty / in CI pipelines.

### 2.3 Browser Kiosk Launcher (`src/signlang/game/kiosk.py`)
- **Windows**: Detect Microsoft Edge (`msedge.exe`) or Google Chrome (`chrome.exe`) in `Program Files`, `Program Files (x86)`, `LocalAppData`.
- Launch with `--app=http://localhost:8000` or `--kiosk http://localhost:8000` for a true fullscreen distraction-free app experience.
- **Linux**: Detect `google-chrome`, `chromium`, `chromium-browser`, or fallback to `xdg-open` / stdlib `webbrowser.open()`.

### 2.4 Zero-Prerequisite PowerShell Bootstrap (`install.ps1`)
- Runnable via: `irm https://raw.githubusercontent.com/RAZOR-NINJAS/signlang/main/install.ps1 | iex`
- **Steps**:
  1. Check for Python 3.11 (or 3.12). If missing:
     - Attempt `winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements`.
     - Fallback: download official embeddable/installer Python 3.11 to user directory (no admin rights needed!).
  2. Check for Git. If missing:
     - Attempt `winget install Git.Git` or download zip archive of repository from GitHub.
  3. Ensure repository is cloned / extracted to `%USERPROFILE%\Projects\signlang` (or current directory).
  4. Create virtual environment `.venv` using Python 3.11.
  5. Upgrade pip, install wheel/setuptools, install dependencies (`pip install -e .` + `rich`).
  6. Execute `python scripts/install_tui.py` to launch interactive setup.

### 2.5 Interactive Python TUI Installer (`scripts/install_tui.py`)
- **Visuals**: Large ASCII art "SIGNLANG" banner with terminal color gradient.
- **Features**:
  - Mode selector (ISL, ASL Letters, ASL Words, Game / Learning Mode).
  - Camera selector with live preview test.
  - Shortcut creator: creates Desktop and Start Menu shortcuts (`signlang.bat`).
  - Progress bars and spinners for model downloads (`fetch_models.py`).
  - Built-in self-test running quick assertions: camera open, landmarker init, port 8000 check.
  - Non-interactive flag `--yes` for CI/unattended script execution.

### 2.6 Diagnostic Engine (`signlang doctor`)
- Built into `src/signlang/doctor.py` (and registered in `signlang/cli.py` and `main.py`):
  - [x] Python version check (3.11 supported, <=3.12).
  - [x] OS platform & architecture detection.
  - [x] Camera access check with backend report (`DirectShow` / `V4L2`).
  - [x] MediaPipe Hand Landmarker model file and inference check.
  - [x] Sign classifier models check (`signs_mlp.pt`, `isl_mlp.pt`, `words_classifier.npz`).
  - [x] Audio output & TTS check (Piper voice ONNX + sounddevice / fallback).
  - [x] Browser kiosk candidate check (Edge / Chrome).
  - [x] Summary report with green checkmarks or actionable remediation tips.

### 2.7 Multi-OS CI Workflow (`.github/workflows/ci.yml`)
- Runs on push / PR for:
  - `ubuntu-latest`
  - `windows-latest`
- Test Matrix:
  - Setup Python 3.11
  - Install dependencies (`pip install -e .` + pytest)
  - Download test models (`python scripts/fetch_models.py`)
  - Run pytest (`pytest`)
  - Run doctor in headless mode (`python main.py doctor --check-only`)

---

## 3. Implementation Phasing & Integration

- **Phase 1: Codebase Audit & Baseline Verification** (Agent 1)
- **Phase 2: Platform Abstraction & Core Compatibility** (Agent 2)
- **Phase 3: Installer & Bootstrap Scripting** (Agent 3)
- **Phase 4: Modes Harmonization & Runtime Resiliency** (Agent 4)
- **Phase 5: Diagnostics & CI Matrix** (Agent 5)
- **Phase 6: Integration Pass & Validation** (Orchestrator)
