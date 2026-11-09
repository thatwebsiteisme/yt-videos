#!/usr/bin/env python3
"""Local YouTube Downloader, Audio Extractor, Scene Screenshotter, Markdown & HTML Viewer Pipeline."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DEFAULT_URLS_FILE = BASE_DIR / "urls.txt"
DEFAULT_DOWNLOADS_DIR = BASE_DIR / "downloads"
WHISPER_MODEL_REPO = "mlx-community/whisper-turbo"
BLOCK_INTERVAL_SECONDS = 30.0
SCENE_THRESHOLD = 0.28
MIN_SCREENSHOT_GAP_SECONDS = 25.0
FALLBACK_SCREENSHOT_INTERVAL = 60.0
MAX_SCREENSHOTS = 40

YT_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{11}$")
YT_URL_ID_REGEX = re.compile(
    r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})"
)
PTS_TIME_REGEX = re.compile(r"pts_time:([0-9.]+)")


def ensure_ffmpeg_in_path() -> None:
    """Ensure ffmpeg and ffprobe binaries are available on PATH."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except ImportError:
        pass


def normalize_youtube_target(raw_entry: str) -> str:
    """Convert raw 11-character YouTube ID or URL into a canonical YouTube watch URL."""
    entry = raw_entry.strip()
    if not entry:
        raise ValueError("Empty URL or video ID")

    if YT_ID_REGEX.match(entry):
        return f"https://www.youtube.com/watch?v={entry}"

    if entry.startswith(("http://", "https://")):
        return entry

    if entry.startswith(("www.youtube.com/", "youtube.com/", "youtu.be/")):
        return f"https://{entry}"

    match = YT_URL_ID_REGEX.search(entry)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"

    raise ValueError(f"Invalid YouTube URL or ID: {entry}")


def read_queue(urls_file: Path) -> list[str]:
    """Read active non-empty, non-comment entries from urls.txt."""
    if not urls_file.exists():
        return []
    entries: list[str] = []
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def remove_from_queue(urls_file: Path, completed_entry: str) -> None:
    """Atomically remove the first matching entry from urls.txt, preserving comments."""
    if not urls_file.exists():
        return
    lines = urls_file.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines: list[str] = []
    removed = False
    target = completed_entry.strip()

    for line in lines:
        if not removed and line.strip() == target:
            removed = True
            continue
        new_lines.append(line)

    temp_file = urls_file.with_suffix(".tmp")
    temp_file.write_text("".join(new_lines), encoding="utf-8")
    temp_file.replace(urls_file)


def slugify(value: str, max_length: int = 70) -> str:
    """Create a filesystem-safe slug from video title."""
    cleaned = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip()
    slug = re.sub(r"[-\s]+", "-", cleaned)
    slug = slug[:max_length].strip("-")
    return slug or "video"


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_secs = max(0, int(seconds))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_duration_seconds(duration_str: str) -> float:
    """Parse HH:MM:SS string back into seconds."""
    if not duration_str:
        return 0.0
    parts = [int(p) for p in duration_str.split(":") if p.isdigit()]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return 0.0


def format_upload_date(raw_date: str | None) -> str:
    """Format YYYYMMDD string from yt-dlp into YYYY-MM-DD."""
    if raw_date and len(raw_date) == 8 and raw_date.isdigit():
        return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    return datetime.now().strftime("%Y-%m-%d")


def parse_json3_subtitles(json3_path: Path) -> list[dict[str, Any]]:
    """Parse YouTube json3 subtitle file into a list of {'start': float, 'text': str} segments."""
    data = json.loads(json3_path.read_text(encoding="utf-8"))
    events = data.get("events", [])
    raw_segments: list[dict[str, Any]] = []

    for event in events:
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(seg.get("utf8", "") for seg in segs)
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue

        start_ms = event.get("tStartMs", 0)
        start_sec = float(start_ms) / 1000.0

        if raw_segments and raw_segments[-1]["text"] == text:
            continue

        raw_segments.append({"start": start_sec, "text": text})

    return raw_segments


def capture_frame_at(video_path: Path, timestamp: float, output_file: Path) -> bool:
    """Capture a single JPEG frame at exact timestamp using fast input seeking."""
    ensure_ffmpeg_in_path()
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, timestamp):.2f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(output_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0


def extract_context_screenshots(
    video_path: Path,
    screenshots_dir: Path,
    info: dict[str, Any],
    scene_threshold: float = SCENE_THRESHOLD,
    min_gap: float = MIN_SCREENSHOT_GAP_SECONDS,
    max_screenshots: int = MAX_SCREENSHOTS,
) -> list[dict[str, Any]]:
    """Extract context-aware screenshots into screenshots_dir using chapters + fast keyframe scene detection."""
    ensure_ffmpeg_in_path()
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    duration = float(info.get("duration") or 0.0)

    for old_raw in screenshots_dir.glob("_raw_*.jpg"):
        old_raw.unlink(missing_ok=True)

    raw_chapters = info.get("chapters") or []
    chapter_map: dict[float, str] = {}
    for ch in raw_chapters:
        st = float(ch.get("start_time") or 0.0)
        title = str(ch.get("title") or "").strip()
        target_ts = min(st + 1.0, max(0.0, duration - 0.5)) if duration > 2.0 else st
        chapter_map[round(target_ts, 1)] = title

    select_expr = (
        f"isnan(prev_selected_t)+gte(t-prev_selected_t,{min_gap:.1f})"
        f"*gt(scene,{scene_threshold:.2f})"
    )
    temp_pattern = str(screenshots_dir / "_raw_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-skip_frame",
        "nokey",
        "-i",
        str(video_path),
        "-vf",
        f"select='{select_expr}',showinfo",
        "-vsync",
        "vfr",
        "-frames:v",
        str(max_screenshots),
        "-q:v",
        "3",
        temp_pattern,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    pts_times = [float(m.group(1)) for m in PTS_TIME_REGEX.finditer(res.stderr)]
    raw_files = sorted(screenshots_dir.glob("_raw_*.jpg"))

    captured: list[dict[str, Any]] = []
    for raw_file, ts in zip(raw_files, pts_times):
        ts_label = format_timestamp(ts)
        safe_ts = ts_label.replace(":", "-")
        final_name = f"frame_{safe_ts}.jpg"
        final_path = screenshots_dir / final_name
        raw_file.replace(final_path)
        captured.append(
            {
                "timestamp": ts,
                "label": ts_label,
                "rel_path": f"screenshots/{final_name}",
                "caption": f"Scene @ {ts_label}",
            }
        )

    for leftover in screenshots_dir.glob("_raw_*.jpg"):
        leftover.unlink(missing_ok=True)

    for ch_ts, ch_title in chapter_map.items():
        if any(abs(c["timestamp"] - ch_ts) < 10.0 for c in captured):
            closest = min(captured, key=lambda c: abs(c["timestamp"] - ch_ts))
            if ch_title:
                closest["caption"] = f"{ch_title} ({closest['label']})"
            continue
        ts_label = format_timestamp(ch_ts)
        safe_ts = ts_label.replace(":", "-")
        final_name = f"frame_{safe_ts}.jpg"
        final_path = screenshots_dir / final_name
        if capture_frame_at(video_path, ch_ts, final_path):
            captured.append(
                {
                    "timestamp": ch_ts,
                    "label": ts_label,
                    "rel_path": f"screenshots/{final_name}",
                    "caption": f"{ch_title} ({ts_label})" if ch_title else f"Chapter @ {ts_label}",
                }
            )

    if len(captured) <= 2 and duration > 5.0:
        step = min(FALLBACK_SCREENSHOT_INTERVAL, max(10.0, duration / 4.0))
        t = min(2.0, duration * 0.1)
        while t < duration - 1.0 and len(captured) < max_screenshots:
            if not any(abs(c["timestamp"] - t) < (step * 0.5) for c in captured):
                ts_label = format_timestamp(t)
                safe_ts = ts_label.replace(":", "-")
                final_name = f"frame_{safe_ts}.jpg"
                final_path = screenshots_dir / final_name
                if capture_frame_at(video_path, t, final_path):
                    captured.append(
                        {
                            "timestamp": t,
                            "label": ts_label,
                            "rel_path": f"screenshots/{final_name}",
                            "caption": f"Frame @ {ts_label}",
                        }
                    )
            t += step

    captured.sort(key=lambda item: float(item["timestamp"]))
    return captured[:max_screenshots]


def build_structured_transcript(
    segments: list[dict[str, Any]],
    screenshots: list[dict[str, Any]] | None = None,
    interval_seconds: float = BLOCK_INTERVAL_SECONDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build structured blocks (with attached screenshots) and clean reading paragraphs."""
    screenshots = screenshots or []
    if not segments:
        return ([], [])

    raw_blocks: list[tuple[float, list[str]]] = []
    current_start = float(segments[0]["start"])
    current_texts: list[str] = []

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))

        if current_texts and (start - current_start) >= interval_seconds:
            raw_blocks.append((current_start, current_texts))
            current_start = start
            current_texts = [text]
        else:
            if not current_texts:
                current_start = start
            current_texts.append(text)

    if current_texts:
        raw_blocks.append((current_start, current_texts))

    block_screenshots: dict[int, list[dict[str, Any]]] = {
        i: [] for i in range(len(raw_blocks))
    }
    for shot in screenshots:
        shot_ts = float(shot["timestamp"])
        assigned_idx = 0
        for idx, (b_start, _) in enumerate(raw_blocks):
            if b_start <= shot_ts + 2.0:
                assigned_idx = idx
            else:
                break
        block_screenshots[assigned_idx].append(shot)

    structured_blocks: list[dict[str, Any]] = []
    clean_block_texts: list[str] = []

    for idx, (start_sec, texts) in enumerate(raw_blocks):
        block_text = re.sub(r"\s+", " ", " ".join(texts)).strip()
        next_start = (
            raw_blocks[idx + 1][0] if (idx + 1) < len(raw_blocks) else start_sec + 99999.0
        )
        structured_blocks.append(
            {
                "start": round(start_sec, 2),
                "end": round(next_start, 2),
                "label": format_timestamp(start_sec),
                "text": block_text,
                "screenshots": block_screenshots.get(idx, []),
            }
        )
        clean_block_texts.append(block_text)

    reading_paragraphs: list[str] = []
    for i in range(0, len(clean_block_texts), 3):
        reading_paragraphs.append(" ".join(clean_block_texts[i : i + 3]))

    return (structured_blocks, reading_paragraphs)


def group_segments_into_blocks(
    segments: list[dict[str, Any]],
    screenshots: list[dict[str, Any]] | None = None,
    interval_seconds: float = BLOCK_INTERVAL_SECONDS,
    screenshot_prefix: str = "../",
) -> tuple[str, str]:
    """Group fine-grained segments into (clean_reading_text, timestamped_markdown_with_screenshots)."""
    screenshots = screenshots or []
    structured_blocks, reading_paragraphs = build_structured_transcript(
        segments, screenshots=screenshots, interval_seconds=interval_seconds
    )

    if not structured_blocks:
        if not screenshots:
            return ("_No transcript available._", "_No transcript available._")
        gallery = "\n\n".join(
            f"![{s['caption']}]({screenshot_prefix}{s['rel_path']})" for s in screenshots
        )
        return ("_No transcript available._", gallery)

    timestamped_lines: list[str] = []
    for block in structured_blocks:
        lines_for_block: list[str] = []
        for shot in block["screenshots"]:
            lines_for_block.append(
                f"![{shot['caption']}]({screenshot_prefix}{shot['rel_path']})"
            )
        lines_for_block.append(f"**[{block['label']}]** {block['text']}")
        timestamped_lines.append("\n\n".join(lines_for_block))

    return ("\n\n".join(reading_paragraphs), "\n\n".join(timestamped_lines))


def transcribe_with_local_whisper(audio_path: Path) -> list[dict[str, Any]]:
    """Run local Whisper on extracted audio when YouTube subtitles are missing (Apple Silicon MLX or CPU/CUDA faster-whisper)."""
    try:
        import mlx_whisper

        print(f"  [Whisper] Running local MLX Whisper ({WHISPER_MODEL_REPO}) on {audio_path.name}...")
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=WHISPER_MODEL_REPO,
            verbose=False,
        )
        segments = result.get("segments", [])
        parsed: list[dict[str, Any]] = []
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if text:
                parsed.append({"start": float(seg.get("start", 0.0)), "text": text})
        return parsed
    except ImportError:
        from faster_whisper import WhisperModel

        print(f"  [Whisper] Running local faster-whisper (base) on {audio_path.name}...")
        model = WhisperModel("base", compute_type="int8")
        segments_gen, _ = model.transcribe(str(audio_path))
        return [
            {"start": float(seg.start), "text": seg.text.strip()}
            for seg in segments_gen
            if seg.text.strip()
        ]


def build_markdown_files(
    info: dict[str, Any],
    transcript_source: str,
    clean_text: str,
    timestamped_text: str,
    video_filename: str,
    audio_filename: str,
    screenshots_count: int = 0,
    html_filename: str = "index.html",
    clean_rel_path: str = "transcripts/transcript-clean.md",
    timestamped_rel_path: str = "transcripts/transcript-timestamped.md",
    main_md_filename: str = "details.md",
) -> tuple[str, str, str]:
    """Build the 3 Markdown documents: (main_details_md, clean_transcript_md, timestamped_transcript_md)."""
    title = info.get("title") or "Untitled Video"
    video_id = info.get("id") or "unknown"
    webpage_url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
    channel = info.get("uploader") or info.get("channel") or "Unknown Channel"
    upload_date = format_upload_date(info.get("upload_date"))
    duration_secs = float(info.get("duration") or 0)
    duration_str = format_timestamp(duration_secs)
    description = (info.get("description") or "").strip() or "_No description provided._"

    main_md = (
        f"# {title}\n\n"
        f"- **URL:** {webpage_url}\n"
        f"- **Video ID:** `{video_id}`\n"
        f"- **Channel:** {channel}\n"
        f"- **Upload Date:** {upload_date}\n"
        f"- **Duration:** {duration_str}\n"
        f"- **Video File:** `{video_filename}`\n"
        f"- **Audio File:** `{audio_filename}`\n"
        f"- **HTML Player:** [{html_filename}]({html_filename})\n"
        f"- **Screenshots:** `{screenshots_count}` in `screenshots/`\n"
        f"- **Transcript Source:** {transcript_source}\n\n"
        f"## Transcripts\n\n"
        f"- 📖 [Transcript (Clean Reading View)]({clean_rel_path})\n"
        f"- ⏱️ [Transcript (Timestamped & Visuals)]({timestamped_rel_path})\n\n"
        f"---\n\n"
        f"## Description\n\n"
        f"{description}\n"
    )

    clean_md = (
        f"# {title} — Transcript (Clean Reading View)\n\n"
        f"- **Back to Details:** [{main_md_filename}](../{main_md_filename})\n"
        f"- **Timestamped View:** [transcript-timestamped.md](transcript-timestamped.md)\n"
        f"- **Source:** {transcript_source}\n\n"
        f"---\n\n"
        f"{clean_text}\n"
    )

    timestamped_md = (
        f"# {title} — Transcript (Timestamped)\n\n"
        f"- **Back to Details:** [{main_md_filename}](../{main_md_filename})\n"
        f"- **Clean Reading View:** [transcript-clean.md](transcript-clean.md)\n"
        f"- **Source:** {transcript_source}\n\n"
        f"---\n\n"
        f"{timestamped_text}\n"
    )

    return (main_md, clean_md, timestamped_md)


def build_markdown(
    info: dict[str, Any],
    transcript_source: str,
    clean_text: str,
    timestamped_text: str,
    video_filename: str,
    audio_filename: str,
    screenshots_count: int = 0,
    html_filename: str = "index.html",
) -> str:
    """Backwards-compatible helper returning main details markdown."""
    main_md, _, _ = build_markdown_files(
        info=info,
        transcript_source=transcript_source,
        clean_text=clean_text,
        timestamped_text=timestamped_text,
        video_filename=video_filename,
        audio_filename=audio_filename,
        screenshots_count=screenshots_count,
        html_filename=html_filename,
    )
    return main_md


def render_html_viewer(
    info: dict[str, Any],
    transcript_source: str,
    blocks: list[dict[str, Any]],
    clean_paragraphs: list[str],
    screenshots: list[dict[str, Any]],
    video_filename: str,
    audio_filename: str,
    markdown_filename: str,
    templates_dir: Path = TEMPLATES_DIR,
) -> str:
    """Render standalone index.html from templates/viewer.html using Jinja2."""
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("viewer.html")

    title = info.get("title") or "Untitled Video"
    video_id = info.get("id") or "unknown"
    webpage_url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
    channel = info.get("uploader") or info.get("channel") or "Unknown Channel"
    upload_date = format_upload_date(info.get("upload_date"))
    duration_secs = float(info.get("duration") or 0)
    duration_str = format_timestamp(duration_secs)
    description = (info.get("description") or "").strip() or "No description provided."

    return template.render(
        title=title,
        video_id=video_id,
        webpage_url=webpage_url,
        channel=channel,
        upload_date=upload_date,
        duration=duration_str,
        duration_seconds=round(duration_secs, 2),
        description=description,
        transcript_source=transcript_source.replace("`", ""),
        video_filename=video_filename,
        audio_filename=audio_filename,
        markdown_filename=markdown_filename,
        blocks=blocks,
        clean_paragraphs=clean_paragraphs,
        screenshots=screenshots,
    )


def collect_downloaded_videos(downloads_dir: Path = DEFAULT_DOWNLOADS_DIR) -> list[dict[str, Any]]:
    """Scan downloads/ directory and return metadata + watch progress for all completed video folders."""
    if not downloads_dir.exists():
        return []

    videos: list[dict[str, Any]] = []
    for folder in sorted(downloads_dir.iterdir(), reverse=True):
        if not folder.is_dir() or folder.name.startswith("."):
            continue

        meta_file = folder / "metadata.json"
        md_files = [p for p in folder.glob("*.md") if p.is_file()]
        details_md_name = md_files[0].name if md_files else "details.md"
        shots = (
            sorted((folder / "screenshots").glob("*.jpg"))
            if (folder / "screenshots").exists()
            else []
        )
        chosen_shot = shots[1] if len(shots) > 1 else (shots[0] if shots else None)
        thumb_rel = (
            f"downloads/{folder.name}/screenshots/{chosen_shot.name}"
            if chosen_shot
            else ""
        )

        # Find actual video and audio filenames inside the folder
        video_candidates = [
            p for p in folder.glob("*.mp4") if not p.name.startswith(".")
        ]
        audio_candidates = [
            p for p in folder.glob("*.m4a") if not p.name.startswith(".")
        ]
        video_filename = video_candidates[0].name if video_candidates else "video.mp4"
        audio_filename = audio_candidates[0].name if audio_candidates else "audio.m4a"

        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                duration_str = meta.get("duration", "00:00:00")
                duration_seconds = float(
                    meta.get("duration_seconds") or parse_duration_seconds(duration_str)
                )
                watched_seconds = float(meta.get("watched_seconds") or 0.0)
                watched_percent = float(meta.get("watched_percent") or 0.0)
                watch_status = str(meta.get("watch_status") or "unwatched")
                last_watched_at = str(meta.get("last_watched_at") or "")

                videos.append(
                    {
                        "folder_name": folder.name,
                        "title": meta.get("title", folder.name),
                        "video_id": meta.get("video_id", ""),
                        "channel": meta.get("channel", "Unknown"),
                        "upload_date": meta.get("upload_date", ""),
                        "duration": duration_str,
                        "duration_seconds": duration_seconds,
                        "watched_seconds": watched_seconds,
                        "watched_label": format_timestamp(watched_seconds),
                        "watched_percent": round(watched_percent, 1),
                        "watch_status": watch_status,
                        "last_watched_at": last_watched_at,
                        "video_filename": meta.get("video_filename", video_filename),
                        "audio_filename": meta.get("audio_filename", audio_filename),
                        "screenshots_count": len(shots),
                        "thumb_rel": thumb_rel,
                        "viewer_rel": f"downloads/{folder.name}/index.html",
                        "details_md_rel": f"downloads/{folder.name}/{details_md_name}",
                        "clean_md_rel": f"downloads/{folder.name}/transcripts/transcript-clean.md",
                        "ts_md_rel": f"downloads/{folder.name}/transcripts/transcript-timestamped.md",
                    }
                )
                continue
            except Exception:
                pass

        # Fallback: parse basic fields from the main .md file
        title = folder.name
        channel = "YouTube"
        upload_date = folder.name[:10] if len(folder.name) >= 10 else ""
        duration_str = "00:00:00"
        video_id = folder.name.split("_")[-1] if "_" in folder.name else ""
        if md_files:
            md_text = md_files[0].read_text(encoding="utf-8", errors="ignore")
            for line in md_text.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("- **Channel:**"):
                    channel = line.split(":**", 1)[-1].strip()
                elif line.startswith("- **Duration:**"):
                    duration_str = line.split(":**", 1)[-1].strip()
                elif line.startswith("- **Upload Date:**"):
                    upload_date = line.split(":**", 1)[-1].strip()

        duration_seconds = parse_duration_seconds(duration_str)
        videos.append(
            {
                "folder_name": folder.name,
                "title": title,
                "video_id": video_id,
                "channel": channel,
                "upload_date": upload_date,
                "duration": duration_str,
                "duration_seconds": duration_seconds,
                "watched_seconds": 0.0,
                "watched_label": "00:00:00",
                "watched_percent": 0.0,
                "watch_status": "unwatched",
                "last_watched_at": "",
                "video_filename": video_filename,
                "audio_filename": audio_filename,
                "screenshots_count": len(shots),
                "thumb_rel": thumb_rel,
                "viewer_rel": f"downloads/{folder.name}/index.html",
                "details_md_rel": f"downloads/{folder.name}/{details_md_name}",
                "clean_md_rel": f"downloads/{folder.name}/transcripts/transcript-clean.md",
                "ts_md_rel": f"downloads/{folder.name}/transcripts/transcript-timestamped.md",
            }
        )

    return videos


def render_root_dashboard(
    downloads_dir: Path = DEFAULT_DOWNLOADS_DIR,
    urls_file: Path = DEFAULT_URLS_FILE,
    output_html: Path = BASE_DIR / "index.html",
    templates_dir: Path = TEMPLATES_DIR,
) -> Path:
    """Render root index.html library hub from templates/dashboard.html."""
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("dashboard.html")
    videos = collect_downloaded_videos(downloads_dir)
    queue_text = urls_file.read_text(encoding="utf-8") if urls_file.exists() else ""

    html = template.render(
        videos=videos,
        queue_text=queue_text,
    )
    output_html.write_text(html, encoding="utf-8")
    return output_html


def extract_audio_from_video(video_path: Path, audio_path: Path) -> None:
    """Extract audio stream from downloaded video locally using ffmpeg (zero re-download)."""
    ensure_ffmpeg_in_path()
    cmd_copy = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "copy",
        str(audio_path),
    ]
    res = subprocess.run(cmd_copy, capture_output=True, text=True, check=False)
    if res.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
        return

    cmd_encode = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(audio_path),
    ]
    subprocess.run(cmd_encode, capture_output=True, text=True, check=True)


def process_single_video(raw_entry: str, downloads_dir: Path) -> Path:
    """Download video, extract audio, capture scene screenshots, and generate Markdown + HTML viewer."""
    import yt_dlp

    ensure_ffmpeg_in_path()
    url = normalize_youtube_target(raw_entry)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    meta_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    print(f"\n==> Fetching metadata for: {url}")
    with yt_dlp.YoutubeDL(meta_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise RuntimeError(f"Failed to extract video metadata for {url}")

    video_id = info.get("id", "unknown")
    title = info.get("title", "video")
    upload_date = format_upload_date(info.get("upload_date"))
    slug = slugify(title)
    folder_name = f"{upload_date}_{slug}_{video_id}"
    video_dir = downloads_dir / folder_name
    video_dir.mkdir(parents=True, exist_ok=True)

    # Name video and audio after the video title slug so external apps identify them cleanly
    video_path = video_dir / f"{slug}.mp4"
    audio_path = video_dir / f"{slug}.m4a"
    legacy_video_path = video_dir / "video.mp4"
    legacy_audio_path = video_dir / "audio.m4a"

    # Migrate legacy video.mp4 / audio.m4a if re-running on an older folder
    if legacy_video_path.exists() and not video_path.exists():
        legacy_video_path.replace(video_path)
    if legacy_audio_path.exists() and not audio_path.exists():
        legacy_audio_path.replace(audio_path)

    screenshots_dir = video_dir / "screenshots"
    transcripts_dir = video_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    md_path = video_dir / f"{slug}.md"
    clean_md_path = transcripts_dir / "transcript-clean.md"
    ts_md_path = transcripts_dir / "transcript-timestamped.md"
    html_path = video_dir / "index.html"

    dl_opts: dict[str, Any] = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(video_dir / f"{slug}.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3",
        "ignoreerrors": "only_download",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
    }

    print(f"  [Download] Saving video & checking captions -> {video_dir.name}/")
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        info = ydl.extract_info(url, download=True) or info

    if not video_path.exists():
        candidates = [
            p
            for p in video_dir.glob(f"{slug}.*")
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        ]
        if not candidates:
            raise FileNotFoundError(f"Downloaded video file not found in {video_dir}")
        video_path = candidates[0]

    print(f"  [Audio] Extracting audio track -> {audio_path.name}")
    extract_audio_from_video(video_path, audio_path)

    print("  [Screenshots] Detecting visual scene changes & chapters -> screenshots/")
    screenshots = extract_context_screenshots(video_path, screenshots_dir, info)
    print(f"  [Screenshots] Captured {len(screenshots)} screenshot(s)")

    sub_files = sorted(video_dir.glob("*.json3"))
    segments: list[dict[str, Any]] = []
    transcript_source = "None"

    if sub_files:
        chosen_sub = sub_files[0]
        for sf in sub_files:
            if sf.name.endswith(".en.json3"):
                chosen_sub = sf
                break
        print(f"  [Subtitles] Found YouTube captions ({chosen_sub.name}) - parsing locally (0 AI compute)")
        segments = parse_json3_subtitles(chosen_sub)
        transcript_source = "YouTube Subtitles (`en`)"
        for sf in sub_files:
            sf.unlink(missing_ok=True)

    if not segments:
        segments = transcribe_with_local_whisper(audio_path)
        transcript_source = f"Local Apple Silicon Whisper (`{WHISPER_MODEL_REPO}`)"

    blocks, clean_paragraphs = build_structured_transcript(segments, screenshots=screenshots)
    clean_text, timestamped_text = group_segments_into_blocks(
        segments,
        screenshots=screenshots,
        screenshot_prefix="../",
    )

    main_md, clean_md, ts_md = build_markdown_files(
        info=info,
        transcript_source=transcript_source,
        clean_text=clean_text,
        timestamped_text=timestamped_text,
        video_filename=video_path.name,
        audio_filename=audio_path.name,
        screenshots_count=len(screenshots),
        html_filename=html_path.name,
        clean_rel_path="transcripts/transcript-clean.md",
        timestamped_rel_path="transcripts/transcript-timestamped.md",
        main_md_filename=md_path.name,
    )
    md_path.write_text(main_md, encoding="utf-8")
    clean_md_path.write_text(clean_md, encoding="utf-8")
    ts_md_path.write_text(ts_md, encoding="utf-8")
    print(
        f"  [Markdown] Created {md_path.name} + transcripts/{clean_md_path.name} & transcripts/{ts_md_path.name}"
    )

    html_content = render_html_viewer(
        info=info,
        transcript_source=transcript_source,
        blocks=blocks,
        clean_paragraphs=clean_paragraphs,
        screenshots=screenshots,
        video_filename=video_path.name,
        audio_filename=audio_path.name,
        markdown_filename=md_path.name,
    )
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  [HTML] Created {html_path.name} (double-click in Finder to view)")

    # Preserve existing watch progress if re-running on an existing folder
    meta_path = video_dir / "metadata.json"
    existing_meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}

    duration_secs = float(info.get("duration") or 0.0)
    meta_summary = {
        "title": title,
        "video_id": video_id,
        "webpage_url": info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        "channel": info.get("uploader") or info.get("channel") or "Unknown Channel",
        "upload_date": upload_date,
        "duration": format_timestamp(duration_secs),
        "duration_seconds": round(duration_secs, 2),
        "video_filename": video_path.name,
        "audio_filename": audio_path.name,
        "screenshots_count": len(screenshots),
        "details_md": md_path.name,
        "watched_seconds": float(existing_meta.get("watched_seconds") or 0.0),
        "watched_percent": float(existing_meta.get("watched_percent") or 0.0),
        "watch_status": str(existing_meta.get("watch_status") or "unwatched"),
        "last_watched_at": str(existing_meta.get("last_watched_at") or ""),
    }
    meta_path.write_text(
        json.dumps(meta_summary, indent=2), encoding="utf-8"
    )

    return video_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos from urls.txt, extract audio & scene screenshots, and generate Markdown + HTML viewer."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="Optional YouTube URLs or IDs to process directly (if omitted, reads urls.txt)",
    )
    parser.add_argument(
        "--queue-file",
        type=Path,
        default=DEFAULT_URLS_FILE,
        help="Path to queue file containing YouTube URLs/IDs (default: urls.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DOWNLOADS_DIR,
        help="Directory where per-video folders are created (default: downloads/)",
    )
    parser.add_argument(
        "--rebuild-dashboard-only",
        action="store_true",
        help="Only regenerate the root index.html library dashboard and exit",
    )
    args = parser.parse_args(argv)

    if args.rebuild_dashboard_only:
        dash = render_root_dashboard(args.output_dir, args.queue_file)
        print(f"Updated root library dashboard: {dash}")
        return 0

    if args.urls:
        queue = [(entry, False) for entry in args.urls]
    else:
        entries = read_queue(args.queue_file)
        if not entries:
            render_root_dashboard(args.output_dir, args.queue_file)
            print(f"No URLs/IDs found in {args.queue_file}. Root index.html is up to date.")
            return 0
        queue = [(entry, True) for entry in entries]

    print(f"Processing {len(queue)} video(s)...")
    failures = 0

    for entry, from_file in queue:
        try:
            out_folder = process_single_video(entry, args.output_dir)
            if from_file:
                remove_from_queue(args.queue_file, entry)
                print(f"  [Queue] Removed '{entry}' from {args.queue_file.name}")
            render_root_dashboard(args.output_dir, args.queue_file)
            print(f"  [Done] Output folder: {out_folder}")
        except Exception as exc:
            failures += 1
            print(f"  [Error] Failed processing '{entry}': {exc}", file=sys.stderr)

    render_root_dashboard(args.output_dir, args.queue_file)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
