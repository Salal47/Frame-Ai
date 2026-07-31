"""Loads the b-roll and scene/stock libraries from
'My Drive/Content Gen/Media/b-roll' and '.../scene_library' — each folder's
des/*.csv lists {video, description} rows. Real clip durations are read once
via moviepy and cached in des/.duration_cache.json. cell 32, unchanged logic."""
import csv
import glob
import json
import os

from moviepy import VideoFileClip

from . import config


def _duration_cache_path(library_dir):
    return os.path.join(library_dir, "des", ".duration_cache.json")


def _load_duration_cache(library_dir):
    p = _duration_cache_path(library_dir)
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_duration_cache(library_dir, cache):
    with open(_duration_cache_path(library_dir), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _pick_col(fieldnames, candidates):
    lower = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_asset_library(library_dir, tag):
    """
    Read every des/*.csv under library_dir. Each row needs a video path
    column (video/path/file/filename/video_path) and a description column
    (description/desc/scene_description/notes). Returns a list of:
      {"id": "<tag>-<n>", "path": <resolved absolute path>,
       "description": <str>, "duration": <float seconds, real, cached>}
    Missing/unreadable video files are skipped with a warning, not fatal.
    """
    des_dir = os.path.join(library_dir, "des")
    csv_files = sorted(glob.glob(os.path.join(des_dir, "*.csv")))
    if not csv_files:
        print(f"   ⚠️ no CSV found under {des_dir} — {tag} library is empty")
        return []

    cache = _load_duration_cache(library_dir)
    assets, idx = [], 0

    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            video_col = _pick_col(reader.fieldnames, ["video", "path", "file", "filename", "video_path"])
            desc_col = _pick_col(reader.fieldnames, ["description", "desc", "scene_description", "notes"])
            if not video_col or not desc_col:
                print(f"   ⚠️ {csv_path}: couldn't find video/description columns — skipping file")
                continue

            for row in reader:
                rel = (row.get(video_col) or "").strip()
                desc = (row.get(desc_col) or "").strip()
                if not rel or not desc:
                    continue
                full_path = rel if os.path.isabs(rel) else os.path.join(library_dir, rel)
                if not os.path.isfile(full_path):
                    print(f"   ⚠️ {tag}: video not found on disk, skipping: {full_path}")
                    continue

                key = os.path.relpath(full_path, library_dir)
                if key in cache:
                    duration = cache[key]
                else:
                    try:
                        with VideoFileClip(full_path) as vc:
                            duration = round(vc.duration, 2)
                        cache[key] = duration
                    except Exception as err:
                        print(f"   ⚠️ {tag}: couldn't read duration for {full_path} ({err}) — skipping")
                        continue

                idx += 1
                assets.append({
                    "id": f"{tag}-{idx}",
                    "path": full_path,
                    "description": desc,
                    "duration": duration,
                })

    _save_duration_cache(library_dir, cache)
    print(f"   ✔ loaded {len(assets)} {tag} clip(s) from {len(csv_files)} CSV file(s)")
    return assets


def load_libraries():
    print("Loading b-roll library...")
    broll = load_asset_library(config.BROLL_DIR, "broll")
    print("Loading scene/stock library...")
    scenes = load_asset_library(config.SCENE_LIBRARY_DIR, "stock")
    return broll, scenes
