#!/usr/bin/env bash
# SignLang Linux installer bootstrap
set -e

REPO_URL="https://github.com/RAZOR-NINJAS/SignLang.git"
BRANCH="windows-support"
TAR_URL="https://github.com/RAZOR-NINJAS/SignLang/archive/refs/heads/${BRANCH}.tar.gz"

echo "=== SignLang Linux Installer Bootstrap ==="

# 1. Determine Target Directory
INSTALL_DIR=""
if [ -f "./pyproject.toml" ] && grep -q 'name = "signlang"' "./pyproject.toml" 2>/dev/null; then
    INSTALL_DIR="$(pwd)"
elif [ -n "${BASH_SOURCE[0]}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/pyproject.toml" ]; then
    INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -f "${HOME}/Projects/signlang/pyproject.toml" ]; then
    INSTALL_DIR="${HOME}/Projects/signlang"
else
    INSTALL_DIR="${HOME}/Projects/signlang"
fi

# 2. Check or Clone Repository Source
if [ ! -f "${INSTALL_DIR}/pyproject.toml" ]; then
    echo "[1/4] Fetching SignLang repository into ${INSTALL_DIR}..."
    mkdir -p "$(dirname "$INSTALL_DIR")"
    if command -v git >/dev/null 2>&1; then
        echo "      Cloning via git (branch: ${BRANCH})..."
        git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR" 2>/dev/null || git clone "$REPO_URL" "$INSTALL_DIR"
    else
        echo "      Git not found. Downloading repository archive via curl..."
        mkdir -p "$INSTALL_DIR"
        curl -fsSL "$TAR_URL" | tar -xz -C "$INSTALL_DIR" --strip-components=1
    fi
else
    echo "[1/4] Repository files detected in: ${INSTALL_DIR}"
fi

cd "$INSTALL_DIR"

# 3. Check for suitable Python 3 (3.11 or 3.12 preferred for MediaPipe)
echo "[2/4] Detecting Python runtime..."
PYTHON_BIN=""
for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$("$candidate" -c "import sys; print(sys.version_info.major)" 2>/dev/null || true)
        minor=$("$candidate" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || true)
        if [ "$major" = "3" ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
            PYTHON_BIN="$candidate"
            echo "      Found suitable Python: $candidate ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3.11 or 3.12 is required by MediaPipe."
    echo "Please install it with your distribution's package manager:"
    echo "  Arch/Omarchy: sudo pacman -S python"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip"
    exit 1
fi

# 4. Set Up Virtual Environment inside repository directory
echo "[3/4] Configuring virtual environment (.venv)..."
VENV_DIR="${INSTALL_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"

NEED_CREATE_VENV=0
if [ ! -f "$VENV_PY" ]; then
    NEED_CREATE_VENV=1
elif ! "$VENV_PY" -c "import pip" >/dev/null 2>&1; then
    echo "      Existing .venv is missing pip. Attempting ensurepip bootstrap..."
    "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || NEED_CREATE_VENV=1
fi

if [ "$NEED_CREATE_VENV" -eq 1 ]; then
    echo "      Creating virtual environment in ${VENV_DIR}..."
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || {
        echo "Error: Failed to create virtual environment. If on Ubuntu/Debian, install python3-venv:"
        echo "  sudo apt install -y python3-venv"
        exit 1
    }
fi

# Ensure pip is functional inside the venv
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "      Bootstrapping pip inside venv via ensurepip..."
    "$VENV_PY" -m ensurepip --upgrade || {
        echo "Error: pip could not be bootstrapped inside .venv."
        exit 1
    }
fi

echo "[4/4] Installing dependencies into .venv..."
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m pip install -e . --quiet
"$VENV_PY" -m pip install rich --quiet

echo "=========================================================="
echo "Launching interactive setup TUI..."
echo "=========================================================="
exec "$VENV_PY" "${INSTALL_DIR}/scripts/install_tui.py" "$@"
