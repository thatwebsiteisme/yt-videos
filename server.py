#!/usr/bin/env python3
"""Lightweight local web server for managing urls.txt, running process.py, tracking watch state, and checking GitHub updates."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from process import (
    APP_VERSION,
    BASE_DIR,
    DEFAULT_DOWNLOADS_DIR,
    DEFAULT_URLS_FILE,
    parse_duration_seconds,
    read_settings,
    render_root_dashboard,
    save_settings,
)

HOST = "127.0.0.1"
PORT = 8765
GITHUB_REPO = "https://github.com/thatwebsiteisme/yt-videos.git"


class PipelineJobState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.logs: list[str] = [
            "Ready. Paste URLs on the left and click 'Save & Download Now'."
        ]
        self.update_info: dict[str, Any] = {
            "behind": 0,
            "local_sha": "",
            "remote_sha": "",
        }

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 400:
                self.logs = self.logs[-400:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            queue_text = (
                DEFAULT_URLS_FILE.read_text(encoding="utf-8")
                if DEFAULT_URLS_FILE.exists()
                else ""
            )
            return {
                "version": APP_VERSION,
                "running": self.running,
                "logs": "\n".join(self.logs),
                "queue_text": queue_text,
                "update_info": self.update_info,
                "quality": read_settings()["quality"],
            }


STATE = PipelineJobState()


def check_git_updates(auto_pull: bool = False) -> dict[str, Any]:
    """Check GitHub origin/main for updates (and optionally update tracked files)."""
    git_dir = BASE_DIR / ".git"
    try:
        if not git_dir.exists():
            subprocess.run(
                ["git", "init", "-q", "-b", "main"],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                check=False,
            )
            subprocess.run(
                ["git", "remote", "add", "origin", GITHUB_REPO],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                check=False,
            )

        subprocess.run(
            ["git", "fetch", "origin", "main", "--quiet"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        local_res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        remote_res = subprocess.run(
            ["git", "rev-parse", "--short", "origin/main"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        behind_res = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        local_sha = local_res.stdout.strip()
        remote_sha = remote_res.stdout.strip()
        behind = (
            int(behind_res.stdout.strip())
            if behind_res.stdout.strip().isdigit()
            else (1 if (remote_sha and local_sha != remote_sha) else 0)
        )

        if behind > 0 and auto_pull:
            STATE.append_log(
                f"[Update] Pulling {behind} new update(s) from GitHub ({remote_sha})..."
            )
            pull_res = subprocess.run(
                ["git", "checkout", "-B", "main", "origin/main", "-f", "-q"],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                check=False,
            )
            if pull_res.returncode == 0:
                local_sha = remote_sha
                behind = 0
                STATE.append_log("✓ Updated to latest GitHub version.")

        info = {"behind": behind, "local_sha": local_sha, "remote_sha": remote_sha}
        with STATE.lock:
            STATE.update_info = info
        return info
    except Exception:
        return {"behind": 0, "local_sha": "", "remote_sha": ""}


def update_video_watch_metadata(
    video_id: str,
    watched_seconds: float,
    duration_seconds: float,
    watch_status: str | None = None,
    downloads_dir: Path = DEFAULT_DOWNLOADS_DIR,
) -> bool:
    """Persist watch time & status into downloads/<folder>/metadata.json and rebuild root index.html."""
    if not video_id or not downloads_dir.exists():
        return False

    for folder in downloads_dir.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        meta_path = folder / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        folder_vid = meta.get("video_id") or (
            folder.name[-11:]
            if len(folder.name) >= 12 and folder.name[-12] == "_"
            else (folder.name.split("_")[-1] if "_" in folder.name else "")
        )
        if folder_vid != video_id:
            continue

        dur = float(
            duration_seconds
            or meta.get("duration_seconds")
            or parse_duration_seconds(str(meta.get("duration") or ""))
            or 1.0
        )
        watched = max(0.0, float(watched_seconds))
        pct = min(100.0, round((watched / dur) * 100.0, 1)) if dur > 0 else 0.0
        if not watch_status:
            if pct >= 90.0:
                watch_status = "watched"
            elif watched >= 5.0:
                watch_status = "in_progress"
            else:
                watch_status = "unwatched"

        meta["video_id"] = video_id
        meta["duration_seconds"] = round(dur, 2)
        meta["watched_seconds"] = round(watched, 2)
        meta["watched_percent"] = pct
        meta["watch_status"] = watch_status
        meta["last_watched_at"] = datetime.now().isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        render_root_dashboard(downloads_dir, DEFAULT_URLS_FILE)
        return True

    return False


def run_pipeline_worker() -> None:
    with STATE.lock:
        if STATE.running:
            return
        STATE.running = True
        STATE.logs = ["==> Starting local download & transcription pipeline..."]

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(BASE_DIR / "process.py")],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            STATE.append_log(line.rstrip("\n"))
        proc.wait()
        render_root_dashboard(DEFAULT_DOWNLOADS_DIR, DEFAULT_URLS_FILE)
        if proc.returncode == 0:
            STATE.append_log("\n✓ All queued videos finished! Refreshing library...")
        else:
            STATE.append_log(f"\n⚠ Pipeline exited with code {proc.returncode}")
    except Exception as exc:
        STATE.append_log(f"\n[Error] {exc}")
    finally:
        with STATE.lock:
            STATE.running = False


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        if (
            self.path.startswith("/api/status")
            or self.path.startswith("/api/watch")
            or self.path.startswith("/favicon.ico")
            or self.path.startswith("/robots.txt")
        ):
            return
        super().log_message(format, *args)

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        clean_path = self.path.split("?", 1)[0]
        if clean_path == "/api/status":
            self._send_json(STATE.snapshot())
            return
        if clean_path in ("/favicon.ico", "/robots.txt"):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if clean_path in ("/", "/index.html"):
            threading.Thread(
                target=lambda: check_git_updates(auto_pull=True), daemon=True
            ).start()
            render_root_dashboard(DEFAULT_DOWNLOADS_DIR, DEFAULT_URLS_FILE)
        super().do_GET()

    def do_POST(self) -> None:
        clean_path = self.path.split("?", 1)[0]
        if clean_path not in ("/api/queue", "/api/run", "/api/watch", "/api/update"):
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))

        if clean_path == "/api/watch":
            ok = update_video_watch_metadata(
                video_id=str(data.get("video_id") or ""),
                watched_seconds=float(data.get("watched_seconds") or 0.0),
                duration_seconds=float(data.get("duration_seconds") or 0.0),
                watch_status=data.get("watch_status"),
            )
            self._send_json({"ok": ok})
            return

        if clean_path == "/api/update":
            info = check_git_updates(auto_pull=True)
            render_root_dashboard(DEFAULT_DOWNLOADS_DIR, DEFAULT_URLS_FILE)
            self._send_json({"ok": True, "update_info": info})
            return

        if "quality" in data and data["quality"]:
            save_settings(str(data["quality"]))

        if "queue_text" in data:
            DEFAULT_URLS_FILE.write_text(str(data["queue_text"]), encoding="utf-8")
            render_root_dashboard(DEFAULT_DOWNLOADS_DIR, DEFAULT_URLS_FILE)

        if clean_path == "/api/run":
            thread = threading.Thread(target=run_pipeline_worker, daemon=True)
            thread.start()

        self._send_json(STATE.snapshot())


def main() -> int:
    # Check GitHub for updates on load in background
    threading.Thread(
        target=lambda: check_git_updates(auto_pull=True), daemon=True
    ).start()
    render_root_dashboard(DEFAULT_DOWNLOADS_DIR, DEFAULT_URLS_FILE)
    url = f"http://{HOST}:{PORT}/index.html"

    # If server is already running on PORT, just open the browser to it and exit cleanly
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((HOST, PORT)) == 0:
            print(f"🎬 Server is already running at {url} — opening browser...")
            webbrowser.open(url)
            return 0

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print("============================================================")
    print("🎬 Local YouTube Library & Downloader UI running at:")
    print(f"   {url}")
    print("   (Keep this window open while using the live downloader)")
    print("   Press Ctrl+C to stop.")
    print("============================================================")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
