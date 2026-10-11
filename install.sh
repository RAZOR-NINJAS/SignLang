#!/usr/bin/env bash
# SignLang Linux installer bootstrap
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SignLang Linux Installer Bootstrap ==="

# Check Python 3
PYTHON_BIN=""
for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$candidate" -c "import sys; print(sys.version_info.major)")
        minor=$("$candidate" -c "import sys; print(sys.version_info.minor)")
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
            PYTHON_BIN="$candidate"
            echo "Found suitable Python: $candidate ($ver)"
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

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    "$PYTHON_BIN" -m venv .venv
fi

VENV_PY=".venv/bin/python"

echo "Installing/updating dependencies..."
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m pip install -e . --quiet
"$VENV_PY" -m pip install rich --quiet

echo "Launching interactive TUI installer..."
exec "$VENV_PY" scripts/install_tui.py "$@"
