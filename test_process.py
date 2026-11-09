"""Unit tests for process.py local YouTube downloader & transcriber."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from process import (
    DEFAULT_QUALITY,
    QUALITY_PRESETS,
    build_markdown,
    build_markdown_files,
    build_structured_transcript,
    collect_downloaded_videos,
    extract_video_id,
    find_existing_download,
    format_timestamp,
    format_upload_date,
    group_segments_into_blocks,
    is_playlist_target,
    normalize_youtube_target,
    parse_duration_seconds,
    parse_json3_subtitles,
    read_queue,
    read_settings,
    remove_from_queue,
    render_html_viewer,
    render_root_dashboard,
    replace_playlist_in_queue,
    save_settings,
    slugify,
)
from server import update_video_watch_metadata


def test_normalize_youtube_target() -> None:
    assert (
        normalize_youtube_target("jNQXAC9IVRw")
        == "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    )
    assert (
        normalize_youtube_target("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        == "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    )
    assert (
        normalize_youtube_target("youtu.be/jNQXAC9IVRw")
        == "https://youtu.be/jNQXAC9IVRw"
    )
    with pytest.raises(ValueError):
        normalize_youtube_target("")
    with pytest.raises(ValueError):
        normalize_youtube_target("not_a_valid_id_or_url")


def test_queue_read_and_atomic_remove(tmp_path: Path) -> None:
    qfile = tmp_path / "urls.txt"
    qfile.write_text(
        "# Header comment\n\njNQXAC9IVRw\nhttps://youtu.be/dQw4w9WgXcQ\n",
        encoding="utf-8",
    )

    entries = read_queue(qfile)
    assert entries == ["jNQXAC9IVRw", "https://youtu.be/dQw4w9WgXcQ"]

    remove_from_queue(qfile, "jNQXAC9IVRw")
    remaining = read_queue(qfile)
    assert remaining == ["https://youtu.be/dQw4w9WgXcQ"]
    assert "# Header comment" in qfile.read_text(encoding="utf-8")


def test_parse_json3_and_group_blocks(tmp_path: Path) -> None:
    sample_json3 = {
        "events": [
            {"tStartMs": 0, "segs": [{"utf8": "All right,"}, {"utf8": " so here we are"}]},
            {"tStartMs": 4500, "segs": [{"utf8": "in front of the\nelephants."}]},
            {"tStartMs": 4500, "segs": [{"utf8": "in front of the elephants."}]},  # duplicate rolling line
            {"tStartMs": 35000, "segs": [{"utf8": "The cool thing about these guys"}]},
        ]
    }
    sub_path = tmp_path / "video.en.json3"
    sub_path.write_text(json.dumps(sample_json3), encoding="utf-8")

    segments = parse_json3_subtitles(sub_path)
    assert len(segments) == 3
    assert segments[0] == {"start": 0.0, "text": "All right, so here we are"}
    assert segments[1] == {"start": 4.5, "text": "in front of the elephants."}
    assert segments[2] == {"start": 35.0, "text": "The cool thing about these guys"}

    clean_text, ts_text = group_segments_into_blocks(
        segments,
        screenshots=[
            {
                "timestamp": 1.0,
                "label": "00:00:01",
                "rel_path": "screenshots/frame_00-00-01.jpg",
                "caption": "Intro (00:00:01)",
            },
            {
                "timestamp": 36.0,
                "label": "00:00:36",
                "rel_path": "screenshots/frame_00-00-36.jpg",
                "caption": "Scene @ 00:00:36",
            },
        ],
        interval_seconds=30.0,
        screenshot_prefix="../",
    )
    assert "All right, so here we are in front of the elephants." in clean_text
    assert "![Intro (00:00:01)](../screenshots/frame_00-00-01.jpg)" in ts_text
    assert "**[00:00:00]** All right, so here we are in front of the elephants." in ts_text
    assert "![Scene @ 00:00:36](../screenshots/frame_00-00-36.jpg)" in ts_text
    assert "**[00:00:35]** The cool thing about these guys" in ts_text


def test_slugify_and_formatting() -> None:
    assert slugify("Me at the zoo! (2005) / First Video") == "Me-at-the-zoo-2005-First-Video"
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(95.8) == "00:01:35"
    assert format_timestamp(3661) == "01:01:01"
    assert format_upload_date("20050423") == "2005-04-23"


def test_build_markdown() -> None:
    main_md, clean_md, ts_md = build_markdown_files(
        info={
            "title": "Me at the zoo",
            "id": "jNQXAC9IVRw",
            "webpage_url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "uploader": "jawed",
            "upload_date": "20050423",
            "duration": 19,
            "description": "The first video on YouTube.",
        },
        transcript_source="YouTube Subtitles (`en`)",
        clean_text="All right, so here we are in front of the elephants.",
        timestamped_text="**[00:00:00]** All right, so here we are in front of the elephants.",
        video_filename="video.mp4",
        audio_filename="audio.m4a",
        main_md_filename="Me-at-the-zoo.md",
    )
    assert "# Me at the zoo" in main_md
    assert "The first video on YouTube." in main_md
    assert "transcripts/transcript-clean.md" in main_md
    assert "transcripts/transcript-timestamped.md" in main_md
    assert "All right, so here we are in front of the elephants." in clean_md
    assert "**[00:00:00]** All right, so here we are in front of the elephants." in ts_md


def test_render_html_viewer() -> None:
    segments = [
        {"start": 0.0, "text": "Hello world"},
        {"start": 35.0, "text": "Second block text"},
    ]
    screenshots = [
        {
            "timestamp": 2.0,
            "label": "00:00:02",
            "rel_path": "screenshots/frame_00-00-02.jpg",
            "caption": "Scene @ 00:00:02",
        }
    ]
    blocks, clean_paras = build_structured_transcript(segments, screenshots=screenshots)
    html = render_html_viewer(
        info={
            "title": "Test HTML Video",
            "id": "jNQXAC9IVRw",
            "webpage_url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "uploader": "jawed",
            "upload_date": "20050423",
            "duration": 60,
            "description": "Sample description",
        },
        transcript_source="YouTube Subtitles (`en`)",
        blocks=blocks,
        clean_paragraphs=clean_paras,
        screenshots=screenshots,
        video_filename="video.mp4",
        audio_filename="audio.m4a",
        markdown_filename="Test-HTML-Video.md",
    )
    assert "Test HTML Video" in html
    assert "screenshots/frame_00-00-02.jpg" in html
    assert "seekTo(0.0)" in html
    assert "Second block text" in html


def test_render_root_dashboard(tmp_path: Path) -> None:
    dl_dir = tmp_path / "downloads"
    v_dir = dl_dir / "2026-11-08_Sample_jNQXAC9IVRw"
    (v_dir / "screenshots").mkdir(parents=True)
    (v_dir / "screenshots" / "frame_00-00-01.jpg").write_bytes(b"jpg")
    (v_dir / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Sample Video Title",
                "video_id": "jNQXAC9IVRw",
                "channel": "Sample Channel",
                "upload_date": "2026-11-08",
                "duration": "00:05:00",
                "details_md": "Sample.md",
            }
        ),
        encoding="utf-8",
    )
    qfile = tmp_path / "urls.txt"
    qfile.write_text("dQw4w9WgXcQ\n", encoding="utf-8")
    out_html = tmp_path / "index.html"

    render_root_dashboard(downloads_dir=dl_dir, urls_file=qfile, output_html=out_html)
    content = out_html.read_text(encoding="utf-8")
    assert "Sample Video Title" in content
    assert "Sample Channel" in content
    assert "dQw4w9WgXcQ" in content
    videos = collect_downloaded_videos(dl_dir)
    assert len(videos) == 1
    assert videos[0]["title"] == "Sample Video Title"

    # Test updating watch progress
    ok = update_video_watch_metadata(
        video_id="jNQXAC9IVRw",
        watched_seconds=150.0,
        duration_seconds=300.0,
        downloads_dir=dl_dir,
    )
    assert ok is True
    updated = collect_downloaded_videos(dl_dir)[0]
    assert updated["watched_seconds"] == 150.0
    assert updated["watched_percent"] == 50.0
    assert updated["watch_status"] == "in_progress"
    assert parse_duration_seconds("00:05:00") == 300.0


def test_quality_settings_default_720p(tmp_path: Path) -> None:
    sfile = tmp_path / "settings.json"
    res = read_settings(sfile)
    assert res["quality"] == "720p"
    assert res["cookies_browser"] == "auto"
    assert DEFAULT_QUALITY == "720p"
    assert "720p" in QUALITY_PRESETS
    assert "1080p" in QUALITY_PRESETS
    assert "480p" in QUALITY_PRESETS
    assert "audio" in QUALITY_PRESETS

    save_settings("1080p", "chrome", settings_file=sfile)
    res2 = read_settings(sfile)
    assert res2["quality"] == "1080p"
    assert res2["cookies_browser"] == "chrome"
    save_settings("invalid_preset", "invalid_browser", settings_file=sfile)
    res3 = read_settings(sfile)
    assert res3["quality"] == "1080p"
    assert res3["cookies_browser"] == "chrome"


def test_playlist_detection_and_queue_expansion(tmp_path: Path) -> None:
    assert is_playlist_target("https://www.youtube.com/playlist?list=PL1234567890abcdef") is True
    assert is_playlist_target("https://www.youtube.com/watch?v=jNQXAC9IVRw&list=PL1234567890abcdef") is True
    assert is_playlist_target("PL1234567890abcdef") is True
    assert is_playlist_target("jNQXAC9IVRw") is False
    assert is_playlist_target("https://www.youtube.com/watch?v=jNQXAC9IVRw") is False

    assert (
        normalize_youtube_target("PL1234567890abcdef")
        == "https://www.youtube.com/playlist?list=PL1234567890abcdef"
    )

    qfile = tmp_path / "urls.txt"
    qfile.write_text(
        "# Queue\nPL1234567890abcdef\ndQw4w9WgXcQ\n",
        encoding="utf-8",
    )
    replace_playlist_in_queue(
        qfile,
        "PL1234567890abcdef",
        [
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "https://www.youtube.com/watch?v=6oVzh_SAPjk",
        ],
    )
    entries = read_queue(qfile)
    assert entries == [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "https://www.youtube.com/watch?v=6oVzh_SAPjk",
        "dQw4w9WgXcQ",
    ]


def test_duplicate_detection(tmp_path: Path) -> None:
    assert extract_video_id("6oVzh_SAPjk") == "6oVzh_SAPjk"
    assert extract_video_id("https://www.youtube.com/watch?v=6oVzh_SAPjk") == "6oVzh_SAPjk"
    assert extract_video_id("https://youtu.be/6oVzh_SAPjk") == "6oVzh_SAPjk"

    dl_dir = tmp_path / "downloads"
    v_dir = dl_dir / "2026-11-08_Sample_6oVzh_SAPjk"
    v_dir.mkdir(parents=True)
    assert find_existing_download("6oVzh_SAPjk", dl_dir) is None  # incomplete folder

    (v_dir / "Sample.mp4").write_bytes(b"mp4")
    (v_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    assert find_existing_download("6oVzh_SAPjk", dl_dir) == v_dir






