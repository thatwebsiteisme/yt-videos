#!/usr/bin/env bash
set -euo pipefail

# Resolve folder where Download.command lives (portable across macOS & Linux)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# 1. Check GitHub for updates on load (if git is available)
if command -v git >/dev/null 2>&1; then
  if [ ! -d ".git" ]; then
    echo "[GitHub] Linking folder to https://github.com/thatwebsiteisme/yt-videos.git ..."
    git init -q
    git remote add origin https://github.com/thatwebsiteisme/yt-videos.git || true
    git fetch origin main -q && git branch -M main && git reset --mixed origin/main -q || true
  else
    echo "[GitHub] Checking for updates from origin/main..."
    git fetch origin main -q && git pull --ff-only origin main -q || true
  fi
fi

# 2. Auto-install uv if missing on a new machine
if ! command -v uv >/dev/null 2>&1; then
  echo "[Setup] 'uv' not found on this machine. Installing uv locally..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 3. If .venv was copied from another computer/path, rebuild it cleanly
if [ -d ".venv" ] && [ -f ".venv/pyvenv.cfg" ]; then
  if [ ! -x ".venv/bin/python" ] || ! ".venv/bin/python" -c "import sys" >/dev/null 2>&1; then
    echo "[Setup] Detected .venv from another machine/path. Rebuilding local environment..."
    rm -rf .venv
  fi
fi

exec uv run --python 3.11 python server.py
