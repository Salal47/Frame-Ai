"""
d2: video manifest CSV — lives at the Drive ROOT folder (config.DRIVE_ROOT,
same level as source_registry.csv), NOT inside output/. One row per video
file the pipeline actually saves — every per-chunk video AND every final
video — recorded the moment it's written to disk.

Columns: video_name, chunk_no, address, uploaded
  - video_name: the source file's stem (matches the run_dir / _final.mp4 name)
  - chunk_no:   the chunk number as a string, or "final" for the assembled video
  - address:    full absolute path to that video file
  - uploaded:   "False" by default. This pipeline NEVER sets it to True and
                NEVER resets an existing row's value — it exists purely so a
                separate/downstream project (e.g. an Instagram uploader) can
                own and flip it once it actually posts the video.
"""
import csv
import os

from . import config

MANIFEST_FIELDS = ["video_name", "chunk_no", "address", "uploaded"]


def _load(path=None):
    path = path or config.VIDEO_MANIFEST_CSV_PATH
    rows = {}
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[(row.get("video_name", ""), row.get("chunk_no", ""))] = row
    return rows


def _save(rows, path=None):
    path = path or config.VIDEO_MANIFEST_CSV_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def record_video(video_name, chunk_no, address, path=None):
    """
    Call this every time a new video is made and saved (a chunk video or the
    final assembled video) so the manifest always reflects what's actually
    on disk. Safe to call repeatedly for the same (video_name, chunk_no) —
    it just refreshes the address; an existing `uploaded` value is always
    preserved untouched (this project never sets it True, and never resets
    it back to False either).
    """
    rows = _load(path)
    key = (str(video_name), str(chunk_no))
    existing = rows.get(key)
    rows[key] = {
        "video_name": str(video_name),
        "chunk_no": str(chunk_no),
        "address": os.path.abspath(address),
        "uploaded": existing.get("uploaded", "False") if existing else "False",
    }
    _save(rows, path)
    return rows[key]
