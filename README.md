# Local YouTube Downloader, Transcriber & Library Viewer

100% local, zero-API-call YouTube pipeline.

## Features
- **Video & Audio Naming**: Saves `<Title-Slug>.mp4` and `<Title-Slug>.m4a` inside `downloads/<YYYY-MM-DD>_<Title-Slug>_<ID>/` ready for external apps.
- **3 Split Markdown Files**:
  - `<Title-Slug>.md` (Main details, metadata, description, links)
  - `transcripts/transcript-clean.md` (Clean paragraph reading view)
  - `transcripts/transcript-timestamped.md` (Timestamped blocks + inline scene screenshots)
- **Context-Aware Scene Screenshots**: Uses `ffmpeg` keyframe scene detection + YouTube chapters (`screenshots/frame_HH-MM-SS.jpg`).
- **Interactive HTML Library & Player (`index.html`)**:
  - Tracks watch time, percentage, and `Unwatched` / `In Progress` / `Watched` status.
  - Filter by watch status and sort by Watch Progress, Recently Watched, Upload Date, Duration, or Title.
  - Auto-checks GitHub for updates on launch.

## Quick Start
1. Double-click **`Download.command`** (or run `./run.sh` in terminal).
2. On first launch on a new machine, it automatically installs `uv`, Python 3.11, `yt-dlp`, `ffmpeg`, and Whisper.
