"""
Master registry CSV (source_registry.csv on Drive) — requirement #1 & #2:

  - When loaded, it tells the run which text files still need work vs. which
    are fully complete (every chunk done + final video assembled) so
    finished files are never reprocessed.
  - Tracks a `last_touched` timestamp per file so main.py can round-robin
    fairly: whichever incomplete file was worked on longest ago goes first.

One row per source file:
    file_path, status, total_chunks, done_chunks, last_touched, last_error

status is one of: pending, in_progress, complete, error
"""
import csv
import os
import time
from pathlib import Path

from . import config

FIELDS = ["file_path", "status", "total_chunks", "done_chunks", "last_touched", "last_error"]
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}


def discover_source_files(source_dir=None):
    source_dir = source_dir or config.SOURCE_DIR
    found = []
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTS:
                found.append(os.path.join(root, fname))
    return sorted(found)


def load_registry(path=None):
    path = path or config.REGISTRY_CSV_PATH
    rows = {}
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["file_path"]] = row
    return rows


def save_registry(rows, path=None):
    path = path or config.REGISTRY_CSV_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def sync_registry(source_dir=None, path=None):
    """Add any newly-seen source files as 'pending' rows. Never touches an
    existing row (in particular, never resets a 'complete' row), so files
    that already finished stay marked complete and are skipped forever."""
    rows = load_registry(path)
    for file_path in discover_source_files(source_dir):
        if file_path not in rows:
            rows[file_path] = {
                "file_path": file_path,
                "status": "pending",
                "total_chunks": "",
                "done_chunks": "0",
                "last_touched": "",
                "last_error": "",
            }
    save_registry(rows, path)
    return rows


def incomplete_files_by_rotation(rows):
    """Incomplete files ordered oldest-touched-first, so 2+ available files
    naturally take turns across runs instead of one file hogging every run."""
    incomplete = [r for r in rows.values() if r.get("status") != "complete"]
    incomplete.sort(key=lambda r: r.get("last_touched") or "")
    return [r["file_path"] for r in incomplete]


def mark_progress(rows, file_path, done_chunks=None, total_chunks=None, status="in_progress", path=None):
    row = rows.setdefault(file_path, {"file_path": file_path})
    row["status"] = status
    if done_chunks is not None:
        row["done_chunks"] = str(done_chunks)
    if total_chunks is not None:
        row["total_chunks"] = str(total_chunks)
    row["last_touched"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    row["last_error"] = ""
    save_registry(rows, path)


def mark_complete(rows, file_path, final_path="", path=None):
    row = rows.setdefault(file_path, {"file_path": file_path})
    row["status"] = "complete"
    row["last_touched"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    row["last_error"] = final_path
    save_registry(rows, path)


def mark_error(rows, file_path, err, path=None):
    row = rows.setdefault(file_path, {"file_path": file_path})
    row["status"] = "error"
    row["last_touched"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    row["last_error"] = str(err)[:300]
    save_registry(rows, path)
