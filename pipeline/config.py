"""
Central configuration for the GitHub-Actions-deployed pipeline.

In the original Colab notebook, DRIVE_ROOT came from `drive.mount(...)` and
everything else was hardcoded in a cell you edited by hand. Here, the same
Drive folder is mounted headlessly by rclone (see .github/workflows/pipeline.yml
+ rclone_setup.sh) at MOUNT_ROOT, and every value below is overridable via
environment variables / GitHub Actions secrets so nothing is hardcoded in git.
"""
import os

# ============================================================
# PIPELINE VERSION
# ============================================================
PIPELINE_VERSION = "d2"

# ============================================================
# DRIVE MOUNT (rclone, headless — replaces `drive.mount()`)
# ============================================================
# rclone mounts the user's "My Drive" root here. Must match the mountpoint
# used in rclone_setup.sh / the workflow's "Mount Google Drive" step.
MOUNT_ROOT = os.environ.get("DRIVE_MOUNT_ROOT", "/content/drive/MyDrive")

DRIVE_ROOT = os.environ.get("DRIVE_ROOT", os.path.join(MOUNT_ROOT, "InstaUploaderPipeline"))
os.makedirs(DRIVE_ROOT, exist_ok=True)

# Where source text/PDF/DOCX files to turn into videos live.
SOURCE_DIR = os.environ.get("SOURCE_DIR", os.path.join(DRIVE_ROOT, "sources"))
os.makedirs(SOURCE_DIR, exist_ok=True)

# ============================================================
# GEMINI API KEYS  (GitHub secret: GEMINI_API_KEYS, comma-separated)
# ============================================================
_raw_keys = os.environ.get("GEMINI_API_KEYS", "")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise ValueError(
        "No Gemini API keys found. Set the GEMINI_API_KEYS secret "
        "(comma-separated, e.g. 'key1,key2,key3')."
    )

# ============================================================
# MODELS — per-task fallback chains (same as v3.3 notebook)
# ============================================================
MODEL_FALLBACKS = {
    "light": [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ],
    "standard": [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
    ],
    "creative": [
        "gemini-2.5-pro",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    ],
    "vision": [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "tts": [
        "gemini-2.5-flash-preview-tts",
        "gemini-3.1-flash-tts-preview",
    ],
}

MAX_RETRIES_PER_KEY_ROUND = 4
RATE_LIMIT_SLEEP_SECS = 20
RATE_LIMIT_BACKOFF = 1.6

# ============================================================
# OUTPUT / STATE
# ============================================================
OUTPUT_ROOT = os.path.join(DRIVE_ROOT, "output")
ASSETS_DIR = os.path.join(DRIVE_ROOT, "assets")
BG_MUSIC_PATH = os.path.join(ASSETS_DIR, "bg_music.mp3")
TRANSITION_SFX_PATH = os.path.join(ASSETS_DIR, "transition.wav")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Master registry CSV — tracks every source file's completion status so a
# finished file (all chunks done) is never re-picked-up. Lives on Drive.
REGISTRY_CSV_PATH = os.path.join(DRIVE_ROOT, "source_registry.csv")

# Instagram upload tracking CSV
UPLOAD_STATE_CSV_PATH = os.path.join(DRIVE_ROOT, "instagram_upload_state.csv")

# Video manifest CSV (root folder, i.e. DRIVE_ROOT) — one row per video file
# the pipeline has saved (every chunk video AND every final video):
# video_name, chunk_no, address (full path), uploaded.
# `uploaded` always starts False and is NEVER flipped to True by this
# project/pipeline — it's there so a separate uploader project can own it.
VIDEO_MANIFEST_CSV_PATH = os.path.join(DRIVE_ROOT, "video_manifest.csv")

# ============================================================
# VIDEO ASSEMBLY (v2 / portrait)
# ============================================================
FRAME_SIZE = (1080, 1920)
CLIP_SECONDS = 5
ZOOM_AMOUNT = 0.18
TRANSITION_SECONDS = 0.6
NARRATION_VOLUME = 1.0
BG_MUSIC_VOLUME = 0.10
TRANSITION_SFX_VOLUME = 0.55
MAX_IMAGE_ATTEMPTS = 3

# ============================================================
# V3 CONFIG — b-roll / stock library + LLM-planned edit list
# ============================================================
# "My Drive/Content Gen/Media/b-roll" per your instructions. Overridable via
# CONTENT_LIB_DIR if the folder ever moves.
CONTENT_LIB = os.environ.get("CONTENT_LIB_DIR", os.path.join(MOUNT_ROOT, "Content Gen", "Media"))
BROLL_DIR = os.path.join(CONTENT_LIB, "b-roll")
SCENE_LIBRARY_DIR = os.path.join(CONTENT_LIB, "scene_library")
for _d in (BROLL_DIR, SCENE_LIBRARY_DIR):
    os.makedirs(os.path.join(_d, "des"), exist_ok=True)

# NOTE (d2): this is now a MAX CEILING, not a minimum floor — b-roll must
# NEVER cover more than this fraction of a chunk's total duration. Enforced
# both in the planner prompt and, as a hard guarantee, in
# planner.reconcile_plan() (which demotes excess "broll" segments to
# "generate" if the LLM's plan went over).
BROLL_RATIO = float(os.environ.get("BROLL_RATIO", "0.2"))

IMAGE_MODEL_CHAIN = ["gpt-image-2", "gptimage"]
MAX_ATTEMPTS_PER_IMAGE_MODEL = 3
PLAN_DURATION_TOLERANCE_SECS = 0.20

# d2: per-model prompt guidance. Not every model in IMAGE_MODEL_CHAIN is
# equally capable — weaker/quirkier models get a simpler, more constrained
# prompt instead of the exact same prompt used for a stronger model, so the
# generation prompt is written "knowing" which model will receive it.
IMAGE_MODEL_PROMPT_HINTS = {
    "gpt-image-2": (
        "Use clear, natural-language prompts. Describe the subject, scene, "
        "action, style, lighting, composition, and any important constraints. "
        "The model follows detailed instructions well, but prioritize clarity "
        "over unnecessary length. For edits, explicitly state what should "
        "change and what must remain unchanged."
        "Subject,Primary action,Environment,Composition,Lighting,Visual style,Important details,Text to render, Constraints (optional),Negative constraints:"
    ),
    "gptimage": (
        "This model handles complex multi-subject scenes, text, and fine "
        "detail well — the full detailed description can be used as-is."
    ),
    "flux": (
        "This model is strong on photorealism and lighting but weaker on "
        "hands and readable text — keep the prompt simple, avoid asking for "
        "legible text, and avoid intricate hand poses."
    ),
    "sana": (
        "This is a fast, lower-detail model — keep the prompt short and "
        "concrete, centred on ONE clear subject and ONE clear action."
    ),
    "klein": (
        "This model tends to over-stylize — explicitly ask for "
        "'photorealistic, not illustrated/painterly' and keep composition simple."
    ),
    "zimage": (
        "This is a fast, lower-detail model — keep the prompt short and "
        "concrete, centred on ONE clear subject and ONE clear action."
    ),
    "turbo": (
        "This is a fast, low-detail model — keep the prompt short and "
        "focused on a single clear subject."
    ),
}
DEFAULT_IMAGE_MODEL_HINT = (
    "Keep the prompt concrete and unambiguous about subject, setting, and "
    "camera framing."
    "Keep the promot Generic"
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

FRAME_SIZE_V3 = (1920, 1080)
IMAGE_GEN_SIZE = (1280, 720)

# ============================================================
# SCRIPT SOURCE (new in d2)
# ============================================================
# If True: each chunk's narration script is written by rewriting the source
# text through the "creative" Gemini call (create_novel_video_script) —
# the original v3.3 notebook behaviour.
# If False (DEFAULT): skip the rewrite entirely and narrate the chunk's raw
# source text as-is, verbatim from the .txt/.pdf/.docx file — no scripting
# LLM call is made for that chunk. Override with SCRIPT_REWRITE_ENABLED=true
# if you do want the cinematic rewrite.
SCRIPT_REWRITE_ENABLED = os.environ.get("SCRIPT_REWRITE_ENABLED", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# ============================================================
# ROTATION / SCHEDULING (new — GitHub Actions specific)
# ============================================================
# Switch source file after this many chunks so 2+ available files take
# turns instead of one file hogging every run.
CHUNKS_PER_FILE_PER_TURN = int(os.environ.get("CHUNKS_PER_FILE_PER_TURN", "3"))

# Soft ceiling on total wall-clock minutes per GitHub Actions invocation,
# so a run always leaves time to save state + upload before the job's
# own timeout kills it mid-write. Leave headroom under your workflow's
# `timeout-minutes`.
MAX_RUN_MINUTES = int(os.environ.get("MAX_RUN_MINUTES", "50"))

# ============================================================
# INSTAGRAM
# ============================================================
IG_USERNAME = os.environ.get("IG_USERNAME", "")
IG_PASSWORD = os.environ.get("IG_PASSWORD", "")
IG_SESSION_PATH = os.path.join(DRIVE_ROOT, "ig_session.json")
IG_MAX_UPLOADS_PER_RUN = int(os.environ.get("IG_MAX_UPLOADS_PER_RUN", "1"))
IG_CAPTION_SUFFIX = os.environ.get("IG_CAPTION_SUFFIX", "")

print(f"✅ Config loaded (pipeline {PIPELINE_VERSION}).")
print(f"   Drive root:   {DRIVE_ROOT}")
print(f"   Sources dir:  {SOURCE_DIR}")
print(f"   Output root:  {OUTPUT_ROOT}")
print(f"   B-roll dir:   {BROLL_DIR}")
print(f"   B-roll max:   {BROLL_RATIO:.0%} (ceiling, not a floor)")
print(f"   Script rewrite: {'ON' if SCRIPT_REWRITE_ENABLED else 'OFF (using raw source text as script)'}")
