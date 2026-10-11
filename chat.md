# SignLang Cross-Platform & Windows Compatibility Workstream Log

## Orchestrator Kickoff
- **Date**: 2026-10-11
- **Branch**: `windows-support`
- **Scope**: Cross-platform support (Windows 10/11 + Linux) across all modes (ISL, ASL letters, ASL words, game, learning kiosk).
- **Baseline Test Suite Status**: 96/96 tests passing.

---

## Agent Progress Logs

### Agent 1: Codebase Auditor
- **Status**: Completed
- **What was done**:
  1. **Terminal & I/O Audit**:
     - `isl/collect.py` uses `select.select([sys.stdin], ...)` without Windows fallback. Windows `select()` only accepts socket descriptors, causing runtime crashes when collecting samples.
     - `src/signlang/collect.py` has an initial `msvcrt` branch, but needs standardization.
     - Recommendation: create `src/signlang/keyboard.py` as a single unified cross-platform non-blocking key poller.
  2. **Camera Backend Audit**:
     - `src/signlang/cameras.py`: probe calls `cv2.VideoCapture(i)` without backend specification. On Windows, MSMF is slow (5-10s delay). DirectShow (`cv2.CAP_DSHOW`) is required on Windows.
     - `src/signlang/hands.py`: `_open()` instantiates camera with default backend. Needs `cv2.CAP_DSHOW` on Windows.
     - `words/live.py` and `words/record.py`: uses `cv2.VideoCapture(src_val)` with default backend.
     - Recommendation: create `src/signlang/platform_compat.py` with `get_camera_capture(...)`.
  3. **Browser & Kiosk Audit**:
     - `src/signlang/game/server.py`: outputs `http://127.0.0.1:8000` but has no automated browser launcher or kiosk support.
     - Recommendation: create `src/signlang/game/kiosk.py` detecting Microsoft Edge (`msedge.exe`) or Chrome (`chrome.exe`) in `--app` or `--kiosk` mode, and add `--kiosk` flag to CLI.
  4. **Dependencies & Python Compatibility**:
     - Python 3.11 is the gold standard for MediaPipe `0.10.21` and PyTorch CPU on Windows.
     - `pyproject.toml` dependencies all have precompiled Windows x86_64 wheels.
  5. **Installer & Bootstrap Requirements**:
     - Need `install.ps1` for zero-prerequisite setup (handles winget, direct downloads, Git/zip fallback, venv setup).
     - Need `scripts/install_tui.py` for rich interactive setup with gradient ASCII banner, mode picker, camera test, and shortcut creation.
     - Need `signlang.bat` and `install.sh`.
  6. **Self-Test & CI**:
     - Need `src/signlang/doctor.py` (`signlang doctor`) with pass/fail verification for all subsystems.
     - Need `.github/workflows/ci.yml` testing both `windows-latest` and `ubuntu-latest`.
- **What is blocked**: None.
- **What is next**: Handoff to Streams 2, 3, 4, 5.

### Agent 2: Windows Compatibility Dev
- **Status**: Completed
- **What was done**:
  1. Created `src/signlang/platform_compat.py`:
     - Added `is_windows()`, `is_linux()`, `is_macos()`.
     - Added `get_camera_capture()`: uses `cv2.CAP_DSHOW` on Windows with automatic fallback, preventing 5-10s MSMF freezes and buffering latency; uses `cv2.CAP_V4L2` on Linux.
     - Added `probe_cameras()` with fast unbuffered frame checking.
     - Added `find_browser_executable()`: detects Microsoft Edge (`msedge.exe`) and Google Chrome (`chrome.exe`) in standard Windows installation paths and Linux Chrome/Chromium.
  2. Updated `src/signlang/cameras.py`: uses `platform_compat.probe_cameras()`, prints Windows or Linux device nomenclature based on OS.
  3. Created `src/signlang/game/kiosk.py`: launches Edge or Chrome with `--app=http://127.0.0.1:8000` or `--kiosk`, with graceful fallback to `webbrowser.open()`.
  4. Updated `src/signlang/game/server.py`: added `kiosk: bool = False` parameter to `serve()` with automatic browser process cleanup on server shutdown.
  5. Updated `pyproject.toml`: added `rich>=13.0` for interactive TUI installer and terminal displays.
- **What is blocked**: None.
- **What is next**: Handoff to Stream 3 and Stream 4.

### Agent 3: Installer Dev
- **Status**: Completed
- **What was done**:
  1. Created `install.ps1`: zero-prerequisite PowerShell bootstrap for Windows. Handles bare machines without Python or Git. Installs Python 3.11 (via winget or direct download with user-level non-admin flags), clones repository or extracts zip, creates `.venv`, installs dependencies with pip, and hands off to the interactive Python TUI.
  2. Created `install.sh`: Linux equivalent installer checking distribution python, virtualenv creation, dependency installation, and TUI handoff.
  3. Created `signlang.bat`: Windows desktop batch launcher running `.venv\Scripts\python.exe main.py` with error pausing on crash.
  4. Created `scripts/install_tui.py`: interactive setup wizard with gradient ASCII art logo, mode configuration, camera detection and resolution probe, model download progress bars, shortcut creation (Windows Desktop/Start Menu `.lnk` via PowerShell `WScript.Shell`, Linux `.desktop`), built-in self-test, and unattended `--yes` flag.
  5. Updated `scripts/fetch_models.py`: enhanced with optional download progress callback support.
- **What is blocked**: None.
- **What is next**: Complete Stream 5 (Doctor command, test suite, CI workflow, README).

### Agent 4: Modes Dev
- **Status**: Completed
- **What was done**:
  1. Created `src/signlang/keyboard.py`: unified non-blocking console keyboard input polling across platforms (`msvcrt` on Windows, `select.select` on POSIX, graceful fallback in headless/pipes).
  2. Updated `src/signlang/hands.py`: integrated `get_camera_capture()` into `_open()`, ensuring unbuffered DirectShow camera access on Windows.
  3. Updated `src/signlang/collect.py` and `isl/collect.py`: refactored to use `signlang.keyboard`, eliminating POSIX `select()` socket errors on Windows during sample collection.
  4. Updated `words/record.py` and `words/live.py`: added platform-aware camera opening using DirectShow on Windows and V4L2 on Linux.
  5. Verified audio and TTS engine resilience: `src/signlang/tts.py` handles missing PortAudio devices or models without crashing.
  6. Tested existing test suites: 92/92 tests in affected modules passing with 0 regressions.
- **What is blocked**: None.
- **What is next**: Complete Stream 3 (Installer & TUI) and Stream 5 (Doctor & CI).

### Agent 5: QA / CI Agent
- **Status**: Completed
- **What was done**:
  1. Created `src/signlang/doctor.py`: comprehensive system diagnostic command (`signlang doctor` and `python main.py doctor`). Evaluates Python version, OS platform, DirectShow/V4L2 camera capture, MediaPipe Hand Landmarker instantiation, all model weights (ISL, ASL letters, ASL words), Piper TTS engine, browser kiosk detection, and localhost loopback port binding. Supports `--check-only` for silent CI / automation validation.
  2. Registered `doctor` command in `src/signlang/cli.py` and `main.py`. Added `--kiosk` option to `signlang game`.
  3. Created `src/signlang/test_doctor.py`: 7 unit tests verifying `DoctorCheck` instantiation, individual check functions, `--check-only`, and CLI dispatch.
  4. Created `.github/workflows/ci.yml`: GitHub Actions CI pipeline running across both `windows-latest` and `ubuntu-latest` testing Python 3.11, dependency install, model download (`scripts/fetch_models.py`), pytest suite, `main.py doctor`, and headless TUI setup.
  5. Updated `README.md`: added detailed Windows 10/11 one-liner PowerShell install instructions (`irm ... | iex`), manual steps, Linux installer details, and doctor documentation.
- **What is blocked**: None.
- **What is next**: Orchestrator final integration pass and full regression test execution.
