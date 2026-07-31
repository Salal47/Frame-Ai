"""Source utilities: optional video download by URL, and text-file loading. (cells 10+11, unchanged logic)"""
import os
from datetime import datetime
import yt_dlp


def download_video_with_metadata(url: str, output_dir: str = "downloads") -> dict:
    """Download a video from `url` (yt-dlp supported site) and save its metadata alongside it.
    Skips re-downloading if a file with the same resolved title already exists on disk/Drive."""
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "writethumbnail": True,
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        expected_path = ydl.prepare_filename(info)
        meta_path = os.path.splitext(expected_path)[0] + "_metadata.json"

        if os.path.isfile(expected_path) and os.path.isfile(meta_path):
            print(f"\u2714 Already downloaded — skipping re-download: {expected_path}")
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)

        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)

    metadata = {
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration_seconds": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "description": info.get("description"),
        "webpage_url": info.get("webpage_url"),
        "thumbnail": info.get("thumbnail"),
        "resolution": info.get("resolution"),
        "fps": info.get("fps"),
        "ext": info.get("ext"),
        "filepath": video_path,
        "downloaded_at": datetime.now().isoformat(),
    }

    meta_path = os.path.splitext(video_path)[0] + "_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\u2705 Downloaded: {metadata['title']}")
    print(f"\U0001F4C1 Saved to: {video_path}")
    print(f"\U0001F9FE Metadata: {meta_path}")
    return metadata

from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_text(file_path: str) -> str:
    """Load text from PDF, DOCX, Markdown, or TXT files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif suffix == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .md, .txt")