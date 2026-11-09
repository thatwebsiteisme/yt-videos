#!/usr/bin/env bash
set -euo pipefail

# Resolve folder where run.sh lives (portable across macOS & Linux)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_URL="https://github.com/thatwebsiteisme/yt-videos.git"

# 1. Auto-update from GitHub on load (preserves ignored files: downloads/, urls.txt, settings.json)
if command -v git >/dev/null 2>&1; then
  if [ ! -d ".git" ]; then
    git init -q -b main 2>/dev/null || git init -q
    git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
  fi
  if git fetch origin main -q 2>/dev/null; then
    LOCAL_SHA="$(git rev-parse HEAD 2>/dev/null || echo "")"
    REMOTE_SHA="$(git rev-parse origin/main 2>/dev/null || echo "")"
    if [ -n "$REMOTE_SHA" ] && [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
      echo "[GitHub] Auto-updating to latest version (${REMOTE_SHA:0:7})..."
      git checkout -B main origin/main -f -q
    fi
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

# 4. Run pipeline (uv automatically installs Python 3.11 + dependencies if needed)
exec uv run --python 3.11 python process.py "$@"
