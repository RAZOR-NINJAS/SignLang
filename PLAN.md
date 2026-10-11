# Implementation Plan: Full Windows Compatibility & Cross-Platform Support for `signlang`

## Overview & Objectives

Enable seamless, first-class cross-platform execution of the `signlang` repository on **both Windows and Linux** (Ubuntu/Arch/Debian) across all supported recognition engines and interactive experiences:
- **ISL Mode (Primary)**: Dual-hand static and alphabet recognition
- **ASL Letters Mode**: Single-hand fingerspelling (A–Y, SPACE)
- **ASL Words Mode**: Dynamic motion gesture recognition (DTW + k-NN)
- **Game Mode (Shadow Play)**: Browser-based Dino obstacle game
- **Interactive Collection & Training**: Terminal key polling and HUD loops

---

## 1. Workstream File Ownership Matrix (Strict Isolation)

To ensure no two subagents edit the same file at once, the following disjoint file ownership is enforced:

| Workstream | Subagent Role | Scope / Assigned Tasks | Owned Files |
| :--- | :--- | :--- | :--- |
| **Stream 1** | **Codebase Auditor** | Static analysis & codebase scan for Linux-only assumptions, terminal I/O, device paths, and platform-specific calls. | *Read-only* (writes findings to `chat.md`) |
| **Stream 2** | **Windows Compatibility Dev** | Core camera backend (`cv2.CAP_DSHOW`), path standardization, kiosk launcher abstraction, dependency pins. | `src/signlang/platform_compat.py` *(new)*<br>`src/signlang/cameras.py`<br>`src/signlang/game/server.py`<br>`src/signlang/game/kiosk.py` *(new)*<br>`pyproject.toml` |
| **Stream 3** | **Installer Dev** | PowerShell bootstrap (`install.ps1`), Python TUI installer with ASCII banner logo, Windows CMD launcher (`signlang.bat`). | `install.ps1` *(new)*<br>`signlang.bat` *(new)*<br>`scripts/install_tui.py` *(new)*<br>`scripts/fetch_models.py` |
| **Stream 4** | **Modes Dev** | Keyboard polling abstraction (`msvcrt` vs `select`), ASL/ISL/Words live & collection loops cross-platform testing, audio/TTS resilience. | `src/signlang/keyboard.py` *(new)*<br>`src/signlang/hands.py`<br>`src/signlang/live.py`<br>`src/signlang/collect.py`<br>`isl/collect.py`<br>`isl/live.py`<br>`words/record.py`<br>`words/live.py`<br>`src/signlang/tts.py` |
| **Stream 5** | **QA / CI Agent** | `signlang doctor` system verification command, multi-platform GitHub Actions CI workflow, doctor unit tests. | `src/signlang/doctor.py` *(new)*<br>`src/signlang/cli.py`<br>`main.py`<br>`src/signlang/test_doctor.py` *(new)*<br>`.github/workflows/ci.yml` *(new)* |

---

## 2. Core Architectural Abstractions

### 2.1 Camera Backend Abstraction (`src/signlang/platform_compat.py`)
- **Problem**: On Windows, OpenCV default camera capture often defaults to MSMF or VFW, causing slow 5–10s initialization, frame lag, or failure. On Linux, V4L2 is standard.
- **Solution**: Provide `get_camera_capture(index_or_source, width=None, height=None, fps=None, buffer_size=1)`:
  - If source is an integer index:
    - On Windows (`sys.platform == 'win32'`): Open with `cv2.CAP_DSHOW`. Fall back to default if unavailable.
    - On Linux: Open with `cv2.CAP_V4L2` or default.
  - If source is a video file or stream URL: Open with `cv2.CAP_FFMPEG` or default.
  - Set `cv2.CAP_PROP_BUFFERSIZE = 1` to ensure zero frame lag.

### 2.2 Terminal Keyboard Input Abstraction (`src/signlang/keyboard.py`)
- **Problem**: `select.select([sys.stdin], [], [], 0)` crashes on Windows with `OSError: [WinError 10038] An operation was attempted on something that is not a socket`.
- **Solution**: Centralize non-blocking stdin key reading:
  - Windows: Use `msvcrt.kbhit()` and `msvcrt.getwch()`.
  - POSIX: Use `select.select([sys.stdin], ...)` if `sys.stdin.isatty()`.
  - Non-interactive / headless fallback: return empty list without crashing.

### 2.3 Browser Kiosk Launcher (`src/signlang/game/kiosk.py`)
- **Problem**: Exhibition game mode needs to launch the browser in kiosk/app mode across different operating systems.
- **Solution**:
  - Windows: Check Edge (`msedge.exe` in `Program Files` or Registry) and Chrome (`chrome.exe`), launching with `--app=http://127.0.0.1:8000` or `--kiosk`.
  - Linux: Check `google-chrome`, `chromium`, `chromium-browser`, or fallback to `webbrowser.open()`.

### 2.4 Diagnostic Engine (`signlang doctor`)
- Built into `src/signlang/doctor.py` and invokable via `python main.py doctor` or `signlang doctor`:
  - [x] Python version check (>=3.11, <3.13)
  - [x] Operating System & architecture detection
  - [x] OpenCV & video capture backend probe (`cv2.CAP_DSHOW` / V4L2)
  - [x] MediaPipe Hand Landmarker model presence & initialization
  - [x] Model weights presence (`models/signs_mlp.pt`, `isl/models/isl_mlp.pt`, `words/models/words_classifier.npz`)
  - [x] Audio output device and Piper TTS ONNX readiness
  - [x] Browser detection for Shadow Play game

### 2.5 Multi-OS CI Workflow (`.github/workflows/ci.yml`)
- Runs on push / PR for:
  - `ubuntu-latest`
  - `windows-latest`
- Steps:
  1. Checkout code
  2. Set up Python 3.11
  3. Install dependencies (`pip install -e .` + pytest)
  4. Fetch required test models via `python scripts/fetch_models.py`
  5. Run `pytest`
  6. Run `python main.py doctor --check-only`

---

## 3. Execution Plan & Phasing

- **Phase 1: Codebase Audit & Baseline Verification** (Agent 1)
  - Run comprehensive audit across all files; record all OS-specific patterns into `chat.md`.
- **Phase 2: Platform Abstraction & Core Compatibility** (Agent 2)
  - Implement `platform_compat.py`, update `cameras.py`, implement kiosk launcher in `server.py`, update `pyproject.toml`.
- **Phase 3: Installer & Bootstrap Scripting** (Agent 3)
  - Build `install.ps1`, `signlang.bat`, and interactive Python TUI installer (`scripts/install_tui.py`).
- **Phase 4: Modes Harmonization & Runtime Resiliency** (Agent 4)
  - Implement `keyboard.py`, patch `hands.py`, `live.py`, `collect.py`, `isl/collect.py`, `words/record.py`, ensure TTS fallback.
- **Phase 5: Diagnostics & CI Matrix** (Agent 5)
  - Implement `src/signlang/doctor.py`, wire CLI/main entry points, create `.github/workflows/ci.yml`, add unit tests.
- **Phase 6: Integration Pass & Validation** (Orchestrator)
  - Run full test suite (`pytest`), test `doctor` command, review git diff, and create clean commits on `windows-support`.
